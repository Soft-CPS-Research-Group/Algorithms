"""AgentTransformerPPO — per-building Transformer + PPO over the entity interface.

Spec ``docs/specv2.md`` §11, §12, §14.

Architecture (one stack per building, indexed by ``building_idx``):
    encoded_obs ─► EntityObservationTokenizer ─► (sros, nfc, cas)
                                            └─► TransformerBackbone ─► (ca_emb, pooled)
                                                                    └─► ActorHead  ─► action
                                                                    └─► CriticHead ─► V(s)

Topology mutation (spec §12.2): the wrapper's ``_apply_entity_layout`` calls
``attach_environment(...)`` after every reset/step that increments the
topology version. Our ``attach_environment`` is idempotent: it caches the
``(observation_names, action_names)`` tuple per building and detects "this
is a topology change" by comparing those tuples. On detection it
(1) flushes the in-flight rollout buffer with a final PPO step,
(2) rebuilds the layout via the cached ``EntityTokenLayoutBuilder``,
(3) re-runs the §13.4 hard-fail rules against the new names,
(4) rejects feature-count drift on existing types (would invalidate weights).

Checkpoint resume across topology changes is explicitly out of scope
(spec §14.3) — ``load_checkpoint`` rejects on ``layout_signature`` mismatch.
"""

from __future__ import annotations

import hashlib
import json as _json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.entity_observation_tokenizer import (
    EntityObservationTokenizer,
)
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    EntityTokenLayoutBuilder,
)
from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    RolloutBuffer,
    compute_ppo_loss,
)
from algorithms.utils.transformer_backbone import TransformerBackbone
from utils.entity_tokenizer_schema import (
    EntityPayloadSample,
    EntityTokenizerConfig,
    load_entity_tokenizer_config,
    validate_against_payload,
)


_SRO_PREFIX_RE = ("storage::", "charger::", "pv::")


@dataclass
class _PerBuildingState:
    """All learning state owned by one building. Held in a list on the agent
    indexed by ``building_idx``. The ``optimizer`` is rebuilt only when the
    underlying parameter set changes (i.e. never within a stable topology)."""

    building_id: str
    tokenizer: EntityObservationTokenizer
    backbone: TransformerBackbone
    actor: ActorHead
    critic: CriticHead
    optimizer: torch.optim.Optimizer
    buffer: RolloutBuffer
    layout: BuildingTokenLayout
    obs_names_tuple: Tuple[str, ...]
    action_names_tuple: Tuple[str, ...]
    entity_specs_signature: Optional[str] = None
    # Per-building topology version. Starts at 0 on first attach and is
    # incremented each time :meth:`_handle_topology_change` succeeds. The
    # exporter records this in ``artifact_manifest.json`` so deployment
    # callers can route to the right artifact bundle for a given mutation.
    topology_version: int = 0


def _atanh_safe(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Numerically-safe inverse tanh used to recover the pre-squash sample
    from a stored ``[-1, 1]`` action."""
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class AgentTransformerPPO(BaseAgent):
    """Per-building Transformer + PPO."""

    # Spec §12.4 + §11.4: this agent rebuilds its per-building stack on
    # topology change, so it advertises capability.
    supports_dynamic_topology: ClassVar[bool] = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        algo = config["algorithm"]

        self._tokenizer_config_path: str = str(algo["tokenizer_config_path"])
        self._tokenizer_config: EntityTokenizerConfig = (
            load_entity_tokenizer_config(self._tokenizer_config_path)
        )

        transformer_cfg = dict(algo["transformer"])
        self._d_model = int(transformer_cfg["d_model"])
        self._nhead = int(transformer_cfg["nhead"])
        self._num_layers = int(transformer_cfg["num_layers"])
        self._dim_feedforward = int(transformer_cfg.get("dim_feedforward", 256))
        self._dropout = float(transformer_cfg.get("dropout", 0.1))

        h = dict(algo["hyperparameters"])
        self._lr = float(h["learning_rate"])
        self._gamma = float(h["gamma"])
        self._gae_lambda = float(h["gae_lambda"])
        self._clip_eps = float(h["clip_eps"])
        self._ppo_epochs = int(h["ppo_epochs"])
        self._minibatch_size = int(h["minibatch_size"])
        self._entropy_coeff = float(h.get("entropy_coeff", 0.0))
        self._value_coeff = float(h.get("value_coeff", 0.5))
        self._max_grad_norm = float(h.get("max_grad_norm", 0.5))
        self._actor_hidden_dim = int(
            h.get("actor_hidden_dim", max(32, self._d_model * 2))
        )
        self._critic_hidden_dim = int(
            h.get("critic_hidden_dim", max(32, self._d_model * 2))
        )

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._per_building: List[_PerBuildingState] = []

        # Tracks whether ``attach_environment`` has ever been called. The
        # very first call is *not* a topology change (spec §12.1).
        self._first_attach_done = False

    # ==========================================================================
    # BaseAgent contract
    # ==========================================================================

    def attach_environment(  # type: ignore[override]
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build (or rebuild) per-building stacks. Idempotent: identical
        ``(observation_names, action_names)`` is a no-op. Detected mutation
        triggers ``_handle_topology_change(building_idx)`` per affected
        building."""
        if not self._first_attach_done:
            # First-ever attach — fresh build for every building.
            self._build_all_per_building_states(
                observation_names, action_names, metadata
            )
            self._first_attach_done = True
            return

        if len(self._per_building) != len(observation_names):
            # Total building-count change is treated as a complete rebuild —
            # cannot resume per-building states across cardinality changes.
            self._per_building = []
            self._build_all_per_building_states(
                observation_names, action_names, metadata
            )
            return

        for b, (obs_n, act_n) in enumerate(
            zip(observation_names, action_names)
        ):
            new_obs = tuple(obs_n)
            new_act = tuple(act_n)
            state = self._per_building[b]
            if (
                state.obs_names_tuple == new_obs
                and state.action_names_tuple == new_act
            ):
                # No change for this building.
                continue
            # Spec §12.2: stash new names atomically, then run the topology
            # transition (flush PPO → rebuild layout → re-validate rules).
            state.obs_names_tuple = new_obs
            state.action_names_tuple = new_act
            state.entity_specs_signature = self._signature(metadata)
            self._handle_topology_change(b)

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        det = bool(deterministic) if deterministic is not None else False
        out: List[List[float]] = []
        for state, obs in zip(self._per_building, observations):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, _ = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                actions, _, _ = state.actor(ca_emb, deterministic=det)
            # ActorHead returns ``[B, N_ca, 1]``; the wrapper expects a flat
            # per-CA list.
            out.append(actions.squeeze(0).squeeze(-1).clamp(-1.0, 1.0).tolist())
        return out

    def update(
        self,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float64]],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        """Append the transition to each per-building rollout buffer; when
        ``update_step`` is true, run a PPO update per building and clear."""
        del update_target_step, global_learning_step, initial_exploration_done

        done = bool(terminated or truncated)
        for b, state in enumerate(self._per_building):
            obs_t = torch.as_tensor(
                np.asarray(observations[b]), dtype=torch.float
            ).unsqueeze(0)
            act_t = torch.as_tensor(
                np.asarray(actions[b]), dtype=torch.float
            )
            # Reshape actions to ``[1, N_ca, 1]`` to match ActorHead output.
            act_t = act_t.view(1, state.layout.n_ca, 1)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, pooled = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                value = state.critic(pooled).squeeze(-1)  # [1]
                log_prob = self._compute_log_prob(state.actor, ca_emb, act_t)
            state.buffer.add(
                observation=obs_t.squeeze(0),
                action=act_t.squeeze(0),
                log_prob=log_prob.squeeze(0),
                reward=float(rewards[b]),
                value=value,
                done=done,
            )

        if not update_step:
            return

        for state in self._per_building:
            self._ppo_update(state, next_observations)
            state.buffer.clear()

    def export_artifacts(  # type: ignore[override]
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write per-building ONNX artefacts + return manifest metadata.

        Spec §14.1: filename pattern ``agent_<b>__topology_v<v>.onnx``;
        per-building entry includes ``agent_index``, ``path``, ``format``,
        and a ``config`` block carrying the layout summary needed by the
        deployment side."""
        out = Path(output_dir)
        models_dir = out / "onnx_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Per-building topology version: explicit override via context wins
        # (preserves backward compat with callers that pass it), else use
        # the per-building counter maintained by ``_handle_topology_change``.
        ctx_version = (context or {}).get("topology_version")
        artifacts: List[Dict[str, Any]] = []
        agent_models: List[Dict[str, Any]] = []
        for b, state in enumerate(self._per_building):
            topology_version = (
                int(ctx_version) if ctx_version is not None
                else int(state.topology_version)
            )
            obs_dim = self._infer_obs_dim(state.layout)
            relpath = (
                f"onnx_models/agent_{b}__topology_v{topology_version}.onnx"
            )
            self._export_onnx(state, out / relpath, obs_dim)
            sro_types = [
                s.type_name for s in state.layout.segments if s.family == "sro"
            ]
            ca_types = [
                s.type_name for s in state.layout.segments if s.family == "ca"
            ]
            cfg = {
                "building_id": state.building_id,
                "topology_version": topology_version,
                "obs_dim": obs_dim,
                "n_sro": state.layout.n_sro,
                "n_ca": state.layout.n_ca,
                "sro_types": sro_types,
                "ca_types": ca_types,
                "ca_action_names": list(state.layout.ca_action_names),
            }
            artifacts.append(
                {
                    "agent_index": b,
                    "path": relpath,
                    "format": "onnx",
                    "config": cfg,
                }
            )
            agent_models.append(
                {
                    "building_index": b,
                    "building_id": state.building_id,
                    "topology_version": topology_version,
                    "model_path": relpath,
                    **{
                        k: cfg[k]
                        for k in (
                            "obs_dim",
                            "n_sro",
                            "n_ca",
                            "sro_types",
                            "ca_types",
                            "ca_action_names",
                        )
                    },
                }
            )
        return {
            "format": "onnx",
            "artifacts": artifacts,
            "tokenizer_config_path": self._tokenizer_config_path,
            "supports_dynamic_topology": True,
            "agent_models": agent_models,
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        out = Path(output_dir) / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"transformer_ppo_step{step}.pt"
        payload = {
            "step": int(step),
            "config": dict(self.config["algorithm"]),
            "agents": [
                {
                    "building_id": s.building_id,
                    "tokenizer_state": s.tokenizer.state_dict(),
                    "backbone_state": s.backbone.state_dict(),
                    "actor_state": s.actor.state_dict(),
                    "critic_state": s.critic.state_dict(),
                    "optimizer_state": s.optimizer.state_dict(),
                    "layout_signature": tuple(sorted(s.obs_names_tuple)),
                    "action_names": list(s.action_names_tuple),
                }
                for s in self._per_building
            ],
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        agents = payload["agents"]
        if len(agents) != len(self._per_building):
            raise ValueError(
                f"Checkpoint has {len(agents)} per-building entries; current "
                f"agent has {len(self._per_building)}. Cross-cardinality "
                "resume is not supported (spec §14.3)."
            )
        for state, saved in zip(self._per_building, agents):
            sig_now = tuple(sorted(state.obs_names_tuple))
            sig_saved = tuple(saved["layout_signature"])
            if sig_now != sig_saved:
                raise ValueError(
                    "Checkpoint layout_signature mismatch for building "
                    f"{state.building_id!r}: cannot resume across topology "
                    "changes (spec §14.3)."
                )
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.critic.load_state_dict(saved["critic_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    def _build_all_per_building_states(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        sig = self._signature(metadata)
        building_names = (
            (metadata or {}).get("building_names")
            if metadata is not None
            else None
        )
        for b, (obs_n, act_n) in enumerate(
            zip(observation_names, action_names)
        ):
            building_id = (
                building_names[b]
                if building_names and b < len(building_names) and building_names[b]
                else f"building_{b}"
            )
            state = self._build_one_per_building_state(
                building_id, list(obs_n), list(act_n), sig
            )
            self._per_building.append(state)

    def _build_one_per_building_state(
        self,
        building_id: str,
        observation_names: List[str],
        action_names: List[str],
        entity_specs_signature: Optional[str],
    ) -> _PerBuildingState:
        layout = self._layout_builder.build(
            building_id, observation_names, action_names
        )
        # Spec §10.1 post-condition: CA order matches simulator action order.
        # Match is exact OR ``action_field`` is a prefix of the simulator
        # action name (CityLearn appends a charger-id suffix when multiple
        # CAs of the same type are present, e.g.
        # ``electric_vehicle_storage_charger_1_1``).
        for af, an in zip(layout.ca_action_names, action_names):
            if an == af or an.startswith(af + "_"):
                continue
            raise ValueError(
                f"BuildingTokenLayout.ca_action_names "
                f"{layout.ca_action_names!r} does not match action_names "
                f"{tuple(action_names)!r} for building {building_id!r}."
            )
        type_input_dims = self._compute_type_input_dims(layout)
        tokenizer = EntityObservationTokenizer(
            tokenizer_config=self._tokenizer_config,
            d_model=self._d_model,
            type_input_dims=type_input_dims,
        )
        backbone = TransformerBackbone(
            d_model=self._d_model,
            nhead=self._nhead,
            num_layers=self._num_layers,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
        )
        actor = ActorHead(d_model=self._d_model, hidden_dim=self._actor_hidden_dim)
        critic = CriticHead(
            d_model=self._d_model, hidden_dim=self._critic_hidden_dim
        )
        params = (
            list(tokenizer.parameters())
            + list(backbone.parameters())
            + list(actor.parameters())
            + list(critic.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self._lr)
        buffer = RolloutBuffer(gamma=self._gamma, gae_lambda=self._gae_lambda)
        return _PerBuildingState(
            building_id=building_id,
            tokenizer=tokenizer,
            backbone=backbone,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            buffer=buffer,
            layout=layout,
            obs_names_tuple=tuple(observation_names),
            action_names_tuple=tuple(action_names),
            entity_specs_signature=entity_specs_signature,
        )

    def _handle_topology_change(self, building_idx: int) -> None:
        """Spec §12.2: flush PPO → rebuild layout → re-validate rules."""
        state = self._per_building[building_idx]

        # 1. Flush in-flight rollout (best-effort) before discarding the old
        #    layout. ``_ppo_update`` swallows empty-buffer no-ops.
        if len(state.buffer) > 0:
            try:
                # Use a zero "next value" — last transition is treated as
                # terminal because the topology under it is gone. This is
                # the conservative choice; a future improvement could
                # forward through the new layout's critic.
                self._run_ppo_update_with_last_value(
                    state, last_value=torch.zeros(1)
                )
            finally:
                state.buffer.clear()

        # 2. Rebuild the layout from the cached ``names``.
        new_layout = self._layout_builder.build(
            state.building_id,
            list(state.obs_names_tuple),
            list(state.action_names_tuple),
        )

        # 3. Re-run the §13.4 hard-fail rules against a synthetic single-table
        #    sample reconstructed from the current observation_names.
        synthetic_sample = _synthetic_sample_from_obs_names(
            list(state.obs_names_tuple)
        )
        validate_against_payload(
            self._tokenizer_config,
            synthetic_sample,
            [list(state.action_names_tuple)],
            # Rule 5 (action-field coverage) is a startup-only sanity check
            # against the simulator schema. After a runtime topology
            # mutation the set of active assets may legitimately become a
            # strict subset of the configured CA types (e.g. last EV
            # charger was removed); skipping it avoids false positives.
            include_rule_5=False,
        )

        # 4. Reject feature-count drift on existing types — would invalidate
        #    learned weights. New instances of an existing type are fine
        #    (per-type weight sharing); a different feature COUNT for a type
        #    that already has weights is a hard fail.
        new_dims = self._compute_type_input_dims(new_layout)
        for tname, dim in new_dims.items():
            if tname not in state.tokenizer.projections:
                # New type appearing — that's an unrecoverable schema change.
                raise ValueError(
                    f"Topology change for building {state.building_id!r}: "
                    f"new type {tname!r} appeared in layout; current "
                    "tokenizer has no projection for it. Restart from "
                    "scratch with a tokenizer config that declares this type."
                )
            existing_proj = state.tokenizer.projections[tname]
            if int(existing_proj.in_features) != int(dim):
                raise ValueError(
                    f"Topology change for building {state.building_id!r}: "
                    f"feature count for type {tname!r} changed "
                    f"{existing_proj.in_features} -> {dim}; weights cannot "
                    "be preserved across feature-schema changes."
                )

        # 5. Replace the layout. Per-building NN weights and optimizer state
        #    are preserved (spec §11.4 — per-type weight sharing).
        state.layout = new_layout
        state.topology_version += 1

        # 6. Spec §10.1 post-condition. Mirror the startup-side prefix
        #    tolerance from :meth:`_build_one_per_building_state`: CityLearn
        #    suffixes ``electric_vehicle_storage_<charger_id>`` when there
        #    are multiple chargers, so we accept exact match OR
        #    ``action_field`` being a prefix of the simulator action name.
        if len(state.layout.ca_action_names) != len(state.action_names_tuple):
            raise ValueError(
                "Post-rebuild CA order mismatch for building "
                f"{state.building_id!r}: layout has "
                f"{state.layout.ca_action_names!r}, action_names "
                f"{state.action_names_tuple!r}"
            )
        for af, an in zip(state.layout.ca_action_names, state.action_names_tuple):
            if an == af or an.startswith(af + "_"):
                continue
            raise ValueError(
                "Post-rebuild CA order mismatch for building "
                f"{state.building_id!r}: layout has "
                f"{state.layout.ca_action_names!r}, action_names "
                f"{state.action_names_tuple!r}"
            )

    def _compute_type_input_dims(
        self, layout: BuildingTokenLayout
    ) -> Dict[str, int]:
        """Spec §8.5: per-type input dim derived from segment widths.

        NFC is hard-coded to 1. Declared types absent from the layout get a
        placeholder dim equal to their declared ``input_dim_fallback`` so
        the per-type projection is sized correctly from the start. This
        matters under dynamic topology: when a previously-empty type later
        gains its first instance (e.g. a topology event adds the first EV
        charger to a building), the new segment width will equal the
        fallback and the existing projection will accept it without the
        feature-count-drift fail-fast in :meth:`_handle_topology_change`.

        If the placeholder were always 1, any later real instance with
        ``input_dim_fallback > 1`` would force a hard failure, even though
        no learning has yet happened on that type's weights.
        """
        nfc_name = self._tokenizer_config.nfc.type_name
        dims: Dict[str, int] = {nfc_name: 1}
        for seg in layout.segments:
            if seg.family == "nfc":
                continue
            existing = dims.get(seg.type_name)
            new = len(seg.feature_indices)
            if existing is not None and existing != new:
                raise ValueError(
                    f"Inconsistent input dim for type {seg.type_name!r}: "
                    f"{existing} vs {new}"
                )
            dims[seg.type_name] = new
        for tname, ca_cfg in self._tokenizer_config.ca_types.items():
            dims.setdefault(tname, int(ca_cfg.input_dim_fallback))
        for tname, sro_cfg in self._tokenizer_config.sro_types.items():
            dims.setdefault(tname, int(sro_cfg.input_dim_fallback))
        return dims

    @staticmethod
    def _signature(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata or "entity_specs" not in metadata or not metadata["entity_specs"]:
            return None
        s = _json.dumps(metadata["entity_specs"], sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @staticmethod
    def _infer_obs_dim(layout: BuildingTokenLayout) -> int:
        return max(max(seg.feature_indices) for seg in layout.segments) + 1

    # ----- PPO loss helpers ---------------------------------------------------

    @staticmethod
    def _compute_log_prob(
        actor: ActorHead,
        ca_embeddings: torch.Tensor,  # [B, N_ca, d_model]
        actions_tanh: torch.Tensor,  # [B, N_ca, 1] in [-1, 1]
    ) -> torch.Tensor:
        """Compute log-prob of a stored squashed action under the actor's
        current Normal(mean, std)+tanh distribution. Returns ``[B, N_ca]``.
        """
        means = actor.mlp(ca_embeddings)  # [B, N_ca, 1]
        std = torch.exp(actor.log_std).expand_as(means)
        pre_tanh = _atanh_safe(actions_tanh)
        # log p(pre_tanh) - log(1 - tanh(pre_tanh)^2)
        normal = torch.distributions.Normal(means, std)
        log_prob_pre = normal.log_prob(pre_tanh)
        log_prob = log_prob_pre - torch.log(
            1.0 - actions_tanh.pow(2) + 1.0e-6
        )
        return log_prob.squeeze(-1)

    def _ppo_update(
        self,
        state: _PerBuildingState,
        next_observations: List[npt.NDArray[np.float64]],
    ) -> None:
        if len(state.buffer) == 0:
            return
        # Bootstrap value from the next observation of this building.
        try:
            b_idx = self._per_building.index(state)
            next_obs = next_observations[b_idx]
        except (ValueError, IndexError):
            next_obs = None
        if next_obs is None:
            last_value = torch.zeros(1)
        else:
            last_value = self._critic_value(
                state, np.asarray(next_obs, dtype=np.float64)
            )
        self._run_ppo_update_with_last_value(state, last_value)

    def _critic_value(
        self,
        state: _PerBuildingState,
        obs: npt.NDArray[np.float64],
    ) -> torch.Tensor:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0)
            tokenized = state.tokenizer(obs_t, state.layout)
            _, pooled = state.backbone(
                tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
            )
            return state.critic(pooled).squeeze(-1)

    def _run_ppo_update_with_last_value(
        self, state: _PerBuildingState, last_value: torch.Tensor
    ) -> None:
        state.buffer.compute_returns_and_advantages(last_value)
        for _epoch in range(self._ppo_epochs):
            for batch in state.buffer.get_batches(self._minibatch_size):
                state.optimizer.zero_grad()
                obs_b = batch.observations  # [B, obs_dim]
                act_b = batch.actions  # [B, N_ca, 1]
                # Forward through tokenizer + backbone with grads on.
                tokenized = state.tokenizer(obs_b, state.layout)
                ca_emb, pooled = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                log_probs_new = self._compute_log_prob(
                    state.actor, ca_emb, act_b
                )  # [B, N_ca]
                # Sum over CA actions per step → scalar per step (matches
                # log_probs_old shape stored in buffer).
                log_probs_new_sum = log_probs_new.sum(dim=-1)
                log_probs_old_sum = batch.log_probs.sum(dim=-1)
                values = state.critic(pooled).squeeze(-1)  # [B]
                loss, _metrics = compute_ppo_loss(
                    log_probs_new=log_probs_new_sum,
                    log_probs_old=log_probs_old_sum,
                    advantages=batch.advantages,
                    values=values,
                    returns=batch.returns,
                    clip_eps=self._clip_eps,
                    value_coeff=self._value_coeff,
                    entropy_coeff=self._entropy_coeff,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(state.tokenizer.parameters())
                    + list(state.backbone.parameters())
                    + list(state.actor.parameters())
                    + list(state.critic.parameters()),
                    self._max_grad_norm,
                )
                state.optimizer.step()

    # ----- ONNX export --------------------------------------------------------

    def _export_onnx(
        self,
        state: _PerBuildingState,
        path: Path,
        obs_dim: int,
    ) -> None:
        """Save the actor pipeline as ONNX. The layout indices are baked
        into the wrapper as Python constants. Pure ``index_select`` +
        Linear + Transformer + ActorHead → traceable."""
        layout = state.layout
        sros_idx_per_seg = [
            torch.tensor(list(seg.feature_indices), dtype=torch.long)
            for seg in layout.segments
            if seg.family == "sro"
        ]
        ca_idx_per_seg = [
            torch.tensor(list(seg.feature_indices), dtype=torch.long)
            for seg in layout.segments
            if seg.family == "ca"
        ]
        nfc_seg = next(s for s in layout.segments if s.family == "nfc")
        nfc_idx = torch.tensor(
            list(nfc_seg.feature_indices), dtype=torch.long
        )
        nfc_l = nfc_seg.derived.left_index_in_segment
        nfc_r = nfc_seg.derived.right_index_in_segment

        sro_types = [s.type_name for s in layout.segments if s.family == "sro"]
        ca_types = [s.type_name for s in layout.segments if s.family == "ca"]

        tokenizer = state.tokenizer
        backbone = state.backbone
        actor = state.actor

        class _ExportWrapper(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.tokenizer = tokenizer
                self_inner.backbone = backbone
                self_inner.actor = actor

            def forward(self_inner, encoded_obs: torch.Tensor) -> torch.Tensor:
                sro_tokens_list = []
                for seg_idx, idx in zip(range(len(sros_idx_per_seg)), sros_idx_per_seg):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    proj = self_inner.tokenizer.projections[sro_types[seg_idx]]
                    sro_tokens_list.append(proj(g).unsqueeze(1))
                ca_tokens_list = []
                for seg_idx, idx in zip(range(len(ca_idx_per_seg)), ca_idx_per_seg):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    proj = self_inner.tokenizer.projections[ca_types[seg_idx]]
                    ca_tokens_list.append(proj(g).unsqueeze(1))
                nfc_grp = encoded_obs.index_select(dim=1, index=nfc_idx)
                scalar = (nfc_grp[:, nfc_l] - nfc_grp[:, nfc_r]).unsqueeze(1)
                nfc_tok = self_inner.tokenizer.projections[
                    "building_nfc"
                ](scalar).unsqueeze(1)
                if sro_tokens_list:
                    sros = torch.cat(sro_tokens_list, dim=1)
                else:
                    sros = encoded_obs.new_zeros(
                        encoded_obs.shape[0], 0, self_inner.backbone.d_model
                    )
                if ca_tokens_list:
                    cas = torch.cat(ca_tokens_list, dim=1)
                else:
                    cas = encoded_obs.new_zeros(
                        encoded_obs.shape[0], 0, self_inner.backbone.d_model
                    )
                ca_emb, _ = self_inner.backbone(sros, nfc_tok, cas)
                # ActorHead.forward returns (actions, log_probs, means);
                # for export we want the deterministic mean ∈ [-1, 1].
                means = self_inner.actor.mlp(ca_emb)
                return torch.tanh(means).squeeze(-1)

        wrapper = _ExportWrapper().eval()
        dummy = torch.zeros(1, obs_dim)
        with torch.no_grad():
            # The legacy TorchScript-based ONNX exporter (default
            # ``torch.onnx.export``) does not support PyTorch's fast-path
            # ``aten::_transformer_encoder_layer_fwd`` operator. Disable the
            # MHA fastpath for the duration of the trace so the standard
            # decomposed encoder ops (matmul, softmax, etc.) are emitted.
            try:
                from torch.backends.mha import set_fastpath_enabled

                _restore_to: Optional[bool] = True
                set_fastpath_enabled(False)
            except ImportError:  # pragma: no cover
                _restore_to = None
            try:
                torch.onnx.export(
                    wrapper,
                    (dummy,),
                    str(path),
                    input_names=["encoded_obs"],
                    output_names=["actions"],
                    dynamic_axes={
                        "encoded_obs": {0: "batch"},
                        "actions": {0: "batch"},
                    },
                    opset_version=17,
                )
            finally:
                if _restore_to is not None:
                    from torch.backends.mha import set_fastpath_enabled

                    set_fastpath_enabled(True)


# ==========================================================================
# Free helpers
# ==========================================================================


def _synthetic_sample_from_obs_names(
    observation_names: List[str],
) -> EntityPayloadSample:
    """Reconstruct an ``EntityPayloadSample`` (per-table feature lists) from
    the agent-level observation_names. The validator only needs to know
    which features exist per table — not their values — so this is enough
    to re-run the §13.4 rules at runtime after a topology change.

    Naming is mirrored from ``utils/entity_adapter.py:213-329``:

    - ``district__<feat>``  → ``district`` table feature ``<feat>``
    - ``storage::<id>::<feat>``  → ``storage`` table feature ``<feat>``
    - ``pv::<id>::<feat>``  → ``pv``
    - ``charger::<id>::<feat>``  → ``charger``  (excluding the EV nested
      forms; those go to the ``ev`` table)
    - ``charger::<id>::connected_ev::<feat>``  / ``::incoming_ev::<feat>``
      → ``ev``
    - everything else (no ``::`` and no ``district__``)  → ``building``
    """
    by_table: Dict[str, set[str]] = {
        "district": set(),
        "building": set(),
        "storage": set(),
        "pv": set(),
        "charger": set(),
        "ev": set(),
    }
    for name in observation_names:
        if name.startswith("district__"):
            # Validator's per-table feature lists for the ``district`` table
            # are emitted with the ``district__`` prefix preserved (see
            # ``utils/entity_tokenizer_schema.py:_load_default_payload_sample``
            # — it applies the ``table_prefix`` map). Mirror that here so
            # the §13.4 SRO ``feature_patterns`` matchers (which expect
            # full prefixed names) line up.
            by_table["district"].add(name)
            continue
        if "::" in name:
            head, *rest = name.split("::")
            if head not in {"storage", "pv", "charger"}:
                # Unknown prefix — silently skip (validator covers misses).
                continue
            if head == "charger" and len(rest) >= 3 and rest[1] in {
                "connected_ev",
                "incoming_ev",
            }:
                # charger::<id>::(connected_ev|incoming_ev)::<feat>
                by_table["ev"].add(rest[2])
            else:
                # <head>::<id>::<feat>
                by_table[head].add(rest[1] if len(rest) >= 2 else rest[0])
            continue
        # plain top-level: building feature
        by_table["building"].add(name)
    return EntityPayloadSample(
        feature_names_per_table={k: sorted(v) for k, v in by_table.items()}
    )
