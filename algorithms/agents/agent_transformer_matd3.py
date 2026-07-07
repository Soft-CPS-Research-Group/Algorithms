"""AgentTransformerMATD3 - per-building Transformer actor + centralized twin critics.

Plan A scope: actor stack only (predict + export). Critics, replay, and
training are added in Plans B/C/D.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    EntityTokenLayoutBuilder,
)
from algorithms.utils.matd3_actor_head import DeterministicActorHead
from algorithms.utils.transformer_backbone import TransformerBackbone
from utils.entity_tokenizer_schema import (
    load_entity_tokenizer_config,
    EntityTokenizerConfig,
)


@dataclass
class _ActorState:
    """Per-building actor stack (deployable)."""
    building_id: str
    tokenizer: EntityObservationTokenizer
    backbone: TransformerBackbone
    actor: DeterministicActorHead
    target_actor: DeterministicActorHead
    optimizer: torch.optim.Optimizer
    layout: BuildingTokenLayout
    obs_names_tuple: Tuple[str, ...]
    action_names_tuple: Tuple[str, ...]
    topology_version: int = 0


class AgentTransformerMATD3(BaseAgent):
    """Per-building Transformer actor + centralized twin TD3 critics.

    Plan A implements actor stack only. Training (critics, replay, residual,
    BC) will be wired in Plans B/C/D.
    """

    supports_dynamic_topology: ClassVar[bool] = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        algo = config["algorithm"]

        self._tokenizer_config_path: str = str(algo["tokenizer_config_path"])
        self._tokenizer_config: EntityTokenizerConfig = load_entity_tokenizer_config(
            self._tokenizer_config_path
        )

        actor_cfg = dict(algo["transformer_actor"])
        self._actor_d_model = int(actor_cfg["d_model"])
        self._actor_nhead = int(actor_cfg["nhead"])
        self._actor_num_layers = int(actor_cfg["num_layers"])
        self._actor_dim_feedforward = int(actor_cfg.get("dim_feedforward", 256))
        self._actor_dropout = float(actor_cfg.get("dropout", 0.1))

        h = dict(algo["hyperparameters"])
        self._actor_lr = float(h["actor_lr"])
        self._actor_hidden_dim = int(h.get("actor_hidden_dim", max(32, self._actor_d_model * 2)))

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._actors: List[_ActorState] = []
        self._first_attach_done = False

    # ==========================================================================
    # BaseAgent contract
    # ==========================================================================

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._first_attach_done:
            self._build_all_actor_states(observation_names, action_names, metadata)
            self._first_attach_done = True
            return

        if len(self._actors) != len(observation_names):
            raise ValueError(
                f"AgentTransformerMATD3: building count changed from "
                f"{len(self._actors)} to {len(observation_names)}. "
                "Building-count changes are not supported."
            )

        for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names)):
            new_obs = tuple(obs_n)
            new_act = tuple(act_n)
            state = self._actors[b]
            if state.obs_names_tuple == new_obs and state.action_names_tuple == new_act:
                continue
            state.obs_names_tuple = new_obs
            state.action_names_tuple = new_act
            self._handle_topology_change(b)

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        out: List[List[float]] = []
        for state, obs in zip(self._actors, observations):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, _ = state.backbone(
                    tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
                )
                actions = state.actor(ca_emb)
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
        # Plan A: no-op. Training wired in Plan B/C/D.
        pass

    def export_artifacts(
        self, output_dir: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        models_dir = out / "onnx_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[Dict[str, Any]] = []
        for b, state in enumerate(self._actors):
            obs_dim = self._infer_obs_dim(state.layout)
            relpath = f"onnx_models/agent_{b}__topology_v{state.topology_version}.onnx"
            self._export_actor_onnx(state, out / relpath, obs_dim)
            cfg = {
                "building_id": state.building_id,
                "topology_version": state.topology_version,
                "obs_dim": obs_dim,
                "n_sro": state.layout.n_sro,
                "n_ca": state.layout.n_ca,
                "sro_types": [s.type_name for s in state.layout.segments if s.family == "sro"],
                "ca_types": [s.type_name for s in state.layout.segments if s.family == "ca"],
                "ca_action_names": list(state.layout.ca_action_names),
                "action_low": [-1.0] * state.layout.n_ca,
                "action_high": [1.0] * state.layout.n_ca,
            }
            artifacts.append({
                "agent_index": b,
                "path": relpath,
                "format": "onnx",
                "config": cfg,
            })
        return {
            "format": "onnx",
            "artifacts": artifacts,
            "tokenizer_config_path": self._tokenizer_config_path,
            "supports_dynamic_topology": True,
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        # Plan A: minimal checkpoint.
        out = Path(output_dir) / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"transformer_matd3_step{step}.pt"
        payload = {
            "step": step,
            "actors": [
                {
                    "building_id": s.building_id,
                    "tokenizer_state": s.tokenizer.state_dict(),
                    "backbone_state": s.backbone.state_dict(),
                    "actor_state": s.actor.state_dict(),
                    "target_actor_state": s.target_actor.state_dict(),
                    "optimizer_state": s.optimizer.state_dict(),
                    "obs_names": list(s.obs_names_tuple),
                    "action_names": list(s.action_names_tuple),
                    "topology_version": s.topology_version,
                }
                for s in self._actors
            ],
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        actors_data = payload["actors"]
        if len(actors_data) != len(self._actors):
            raise ValueError(
                f"Checkpoint has {len(actors_data)} buildings; current agent "
                f"has {len(self._actors)}. Building-count mismatch."
            )
        for state, saved in zip(self._actors, actors_data):
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.target_actor.load_state_dict(saved["target_actor_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])

    # ==========================================================================
    # Internal
    # ==========================================================================

    def _build_all_actor_states(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        building_names = (metadata or {}).get("building_names")
        for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names)):
            building_id = (
                building_names[b]
                if building_names and b < len(building_names)
                else f"building_{b}"
            )
            state = self._build_one_actor_state(building_id, list(obs_n), list(act_n))
            self._actors.append(state)

    def _build_one_actor_state(
        self, building_id: str, observation_names: List[str], action_names: List[str]
    ) -> _ActorState:
        layout = self._layout_builder.build(building_id, observation_names, action_names)
        # Validate CA order (prefix tolerance like TransformerPPO)
        for af, an in zip(layout.ca_action_names, action_names):
            if an == af or an.startswith(af + "_"):
                continue
            raise ValueError(
                f"BuildingTokenLayout.ca_action_names {layout.ca_action_names!r} "
                f"does not match action_names {tuple(action_names)!r} "
                f"for building {building_id!r}."
            )
        type_input_dims = self._compute_type_input_dims(layout)
        tokenizer = EntityObservationTokenizer(
            tokenizer_config=self._tokenizer_config,
            d_model=self._actor_d_model,
            type_input_dims=type_input_dims,
        )
        backbone = TransformerBackbone(
            d_model=self._actor_d_model,
            nhead=self._actor_nhead,
            num_layers=self._actor_num_layers,
            dim_feedforward=self._actor_dim_feedforward,
            dropout=self._actor_dropout,
        )
        actor = DeterministicActorHead(
            d_model=self._actor_d_model, hidden_dim=self._actor_hidden_dim
        )
        target_actor = DeterministicActorHead(
            d_model=self._actor_d_model, hidden_dim=self._actor_hidden_dim
        )
        target_actor.load_state_dict(actor.state_dict())
        params = list(tokenizer.parameters()) + list(backbone.parameters()) + list(actor.parameters())
        optimizer = torch.optim.Adam(params, lr=self._actor_lr)
        return _ActorState(
            building_id=building_id,
            tokenizer=tokenizer,
            backbone=backbone,
            actor=actor,
            target_actor=target_actor,
            optimizer=optimizer,
            layout=layout,
            obs_names_tuple=tuple(observation_names),
            action_names_tuple=tuple(action_names),
        )

    def _handle_topology_change(self, building_idx: int) -> None:
        state = self._actors[building_idx]
        new_layout = self._layout_builder.build(
            state.building_id,
            list(state.obs_names_tuple),
            list(state.action_names_tuple),
        )
        new_dims = self._compute_type_input_dims(new_layout)
        for tname, dim in new_dims.items():
            if tname not in state.tokenizer.projections:
                raise ValueError(
                    f"Topology change for {state.building_id!r}: new type "
                    f"{tname!r} has no projection. Restart required."
                )
            if int(state.tokenizer.projections[tname].in_features) != int(dim):
                raise ValueError(
                    f"Topology change for {state.building_id!r}: feature count "
                    f"for type {tname!r} changed. Weights cannot be preserved."
                )
        state.layout = new_layout
        state.topology_version += 1
        logger.info(
            "Topology change: {} v{} — n_ca={}",
            state.building_id, state.topology_version, new_layout.n_ca,
        )

    def _compute_type_input_dims(self, layout: BuildingTokenLayout) -> Dict[str, int]:
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
    def _infer_obs_dim(layout: BuildingTokenLayout) -> int:
        return max(max(seg.feature_indices) for seg in layout.segments) + 1

    def _export_actor_onnx(self, state: _ActorState, path: Path, obs_dim: int) -> None:
        layout = state.layout
        sros_idx = [
            torch.tensor(list(s.feature_indices), dtype=torch.long)
            for s in layout.segments if s.family == "sro"
        ]
        ca_idx = [
            torch.tensor(list(s.feature_indices), dtype=torch.long)
            for s in layout.segments if s.family == "ca"
        ]
        nfc_seg = next(s for s in layout.segments if s.family == "nfc")
        nfc_idx = torch.tensor(list(nfc_seg.feature_indices), dtype=torch.long)
        nfc_l = nfc_seg.derived.left_index_in_segment
        nfc_r = nfc_seg.derived.right_index_in_segment
        sro_types = [s.type_name for s in layout.segments if s.family == "sro"]
        ca_types = [s.type_name for s in layout.segments if s.family == "ca"]
        tokenizer = state.tokenizer
        backbone = state.backbone
        actor = state.actor

        class _Wrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.tokenizer = tokenizer
                self_inner.backbone = backbone
                self_inner.actor = actor

            def forward(self_inner, encoded_obs: torch.Tensor) -> torch.Tensor:
                sro_list = []
                for i, idx in enumerate(sros_idx):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    sro_list.append(self_inner.tokenizer.projections[sro_types[i]](g).unsqueeze(1))
                ca_list = []
                for i, idx in enumerate(ca_idx):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    ca_list.append(self_inner.tokenizer.projections[ca_types[i]](g).unsqueeze(1))
                nfc_grp = encoded_obs.index_select(dim=1, index=nfc_idx)
                scalar = (nfc_grp[:, nfc_l] - nfc_grp[:, nfc_r]).unsqueeze(1)
                nfc_tok = self_inner.tokenizer.projections["building_nfc"](scalar).unsqueeze(1)
                sros = torch.cat(sro_list, dim=1) if sro_list else encoded_obs.new_zeros(encoded_obs.shape[0], 0, backbone.d_model)
                cas = torch.cat(ca_list, dim=1) if ca_list else encoded_obs.new_zeros(encoded_obs.shape[0], 0, backbone.d_model)
                ca_emb, _ = self_inner.backbone(sros, nfc_tok, cas)
                return self_inner.actor(ca_emb).squeeze(-1)

        wrapper = _Wrapper().eval()
        dummy = torch.zeros(1, obs_dim)
        with torch.no_grad():
            try:
                from torch.backends.mha import set_fastpath_enabled
                set_fastpath_enabled(False)
                _restore = True
            except ImportError:
                _restore = False
            try:
                torch.onnx.export(
                    wrapper, (dummy,), str(path),
                    input_names=["encoded_obs"], output_names=["actions"],
                    dynamic_axes={"encoded_obs": {0: "batch"}, "actions": {0: "batch"}},
                    opset_version=17,
                )
            finally:
                if _restore:
                    from torch.backends.mha import set_fastpath_enabled
                    set_fastpath_enabled(True)
