"""AgentTransformerPPO — per-building Transformer + PPO over the entity interface.

Architecture (one stack per building, indexed by ``building_idx``):
    encoded_obs ─► EntityObservationTokenizer ─► (sros, nfc, cas)
                                            └─► TransformerBackbone ─► (ca_emb, pooled)
                                                                    └─► ActorHead  ─► action
                                                                    └─► CriticHead ─► V(s)

Topology mutation: the wrapper's ``_apply_entity_layout`` calls
``attach_environment(...)`` after every reset/step that increments the
topology version. ``attach_environment`` is idempotent: it caches the
``(observation_names, action_names)`` tuple per building and detects a
topology change by comparing those tuples. On detection it
(1) flushes the in-flight rollout buffer with a final PPO step,
(2) rebuilds the layout via the cached ``EntityTokenLayoutBuilder``,
(3) re-runs the hard-fail tokenizer rules against the new names,
(4) rejects feature-count drift on existing types (would invalidate weights).

Checkpoint resume across topology changes is out of scope —
``load_checkpoint`` rejects on ``layout_signature`` mismatch.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import _log_torch_runtime, _select_torch_device
from algorithms.exceptions import DeferredCheckpointError
from algorithms.utils.behavior_cloning import BehaviorCloningRegularizer
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
    RunningValueNormalizer,
    compute_ppo_loss,
)
from algorithms.utils.transformer_backbone import TransformerBackbone
from utils.entity_tokenizer_schema import (
    EntityPayloadSample,
    EntityTokenizerConfig,
    load_entity_tokenizer_config,
    validate_against_payload,
)


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
    # BC updates the tokenizer/backbone/actor only.  Keep a separate optimizer
    # so it never moves critic parameters or PPO optimizer moments.
    bc_optimizer: torch.optim.Optimizer
    buffer: RolloutBuffer
    value_normalizer: RunningValueNormalizer
    layout: BuildingTokenLayout
    obs_names_tuple: Tuple[str, ...]
    action_names_tuple: Tuple[str, ...]
    # Per-building topology version. Starts at 0 on first attach and is
    # incremented each time :meth:`_handle_topology_change` succeeds. The
    # exporter records this in ``artifact_manifest.json`` so deployment
    # callers can route to the right artifact bundle for a given mutation.
    topology_version: int = 0
    last_next_observation: Optional[torch.Tensor] = None
    last_transition_terminated: bool = False
    raw_rewards: List[float] = field(default_factory=list)


@dataclass
class _PendingDecision:
    """Exact policy statistics collected for one environment decision."""

    observation: torch.Tensor
    action: torch.Tensor
    pre_tanh_action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class _TransitionCandidate:
    """Validated transition ready to commit to one rollout buffer."""

    pending: _PendingDecision
    raw_reward: float
    reward: float
    terminated: bool
    truncated: bool
    next_observation: Optional[torch.Tensor]


@dataclass
class _UpdateRuntimeSnapshot:
    """Mutable PPO and BC state that an ordinary update may change."""

    buffers: List[RolloutBuffer]
    last_next_observations: List[Optional[torch.Tensor]]
    last_transition_terminated: List[bool]
    raw_rewards: List[List[float]]
    pending_decisions: List[Optional[_PendingDecision]]
    latest_global_learning_step: int
    latest_training_metrics: Dict[str, float]
    ppo_update_count: int
    tokenizer_states: List[Dict[str, torch.Tensor]]
    backbone_states: List[Dict[str, torch.Tensor]]
    actor_states: List[Dict[str, torch.Tensor]]
    critic_states: List[Dict[str, torch.Tensor]]
    optimizer_states: List[Dict[str, Any]]
    bc_optimizer_states: List[Dict[str, Any]]
    value_normalizer_states: List[Dict[str, Any]]
    behavior_cloning_state: Optional[Dict[str, Any]]
    python_rng_state: object
    numpy_rng_state: tuple
    torch_rng_state: torch.Tensor
    cuda_rng_state: Optional[torch.Tensor]


@dataclass
class _CollectionSnapshot:
    """Small rollback state for a transition that does not train PPO."""

    buffer_lengths: List[int]
    last_next_observations: List[Optional[torch.Tensor]]
    last_transition_terminated: List[bool]
    raw_reward_lengths: List[int]
    pending_decisions: List[Optional[_PendingDecision]]
    latest_global_learning_step: int


@dataclass
class _TopologyStateSnapshot:
    """Complete agent state required to roll back a topology transaction."""

    agent_state: Dict[str, Any]
    python_rng_state: object
    numpy_rng_state: tuple
    torch_rng_state: torch.Tensor
    cuda_rng_state: Optional[torch.Tensor]


class AgentTransformerPPO(BaseAgent):
    """Per-building Transformer + PPO."""

    supports_dynamic_topology: ClassVar[bool] = True
    requires_final_pipeline_stage: ClassVar[bool] = True

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
        self._dropout = float(transformer_cfg.get("dropout", 0.0))

        h = dict(algo["hyperparameters"])
        self.require_cuda = bool(h.get("require_cuda", False))
        self.device = _select_torch_device(
            require_cuda=self.require_cuda,
            algorithm_name="AgentTransformerPPO",
        )
        _log_torch_runtime(self.device)
        torch.backends.cudnn.benchmark = self.device.type == "cuda"
        self._lr = float(h["learning_rate"])
        self._gamma = float(h["gamma"])
        self._gae_lambda = float(h["gae_lambda"])
        self._clip_eps = float(h["clip_eps"])
        self._ppo_epochs = int(h["ppo_epochs"])
        self._minibatch_size = int(h["minibatch_size"])
        self._entropy_coeff = float(h.get("entropy_coeff", 0.0))
        self._value_coeff = float(h.get("value_coeff", 0.5))
        self._max_grad_norm = float(h.get("max_grad_norm", 0.5))
        self._reward_clip = float(h.get("reward_clip", 10.0))  # floor rewards at -clip
        self._actor_hidden_dim = int(
            h.get("actor_hidden_dim", max(32, self._d_model * 2))
        )
        self._critic_hidden_dim = int(
            h.get("critic_hidden_dim", max(32, self._d_model * 2))
        )
        self._actor_log_std_init = float(h.get("actor_log_std_init", -0.5))

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._per_building: List[_PerBuildingState] = []
        self._pending_decisions: List[Optional[_PendingDecision]] = []
        self._action_bounds: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._bc = BehaviorCloningRegularizer.from_config(algo, self.config)
        self.requires_raw_observation_context = bool(self._bc is not None)
        self._latest_raw_observations: Optional[List[npt.NDArray[np.float64]]] = None
        self._latest_encoded_observations: Optional[List[npt.NDArray[np.float64]]] = None
        self._latest_global_learning_step = 0
        self._latest_training_metrics: Dict[str, float] = {}
        self._current_episode = 0
        # No collection phase is active until the wrapper starts a training
        # episode. This keeps evaluation actor-only, including evaluation that
        # begins at episode zero.
        self._current_episode_is_training = False
        self._bc_pretraining_complete = False
        self._ppo_update_count = 0

        # Tracks whether ``attach_environment`` has ever been called. The
        # very first call is not a topology change.
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
        action_space, observation_space = self._validate_environment_metadata(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
        )
        if not self._first_attach_done:
            # First-ever attach — fresh build for every building.
            self._per_building = self._build_per_building_states(
                observation_names, action_names, metadata
            )
            self._pending_decisions = [None] * len(self._per_building)
            self._set_action_bounds(self._prepare_action_bounds(action_space, action_names))
            self._attach_bc_environment(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
            )
            self._first_attach_done = True
            for st in self._per_building:
                logger.info(
                    "Initial attach: {} — obs_dim={}, n_sro={}, n_ca={}, "
                    "actions={}",
                    st.building_id,
                    len(st.obs_names_tuple),
                    st.layout.n_sro,
                    st.layout.n_ca,
                    list(st.action_names_tuple),
                )
            return

        if len(self._per_building) != len(observation_names):
            # Total building-count change is treated as a complete rebuild —
            # cannot resume per-building states across cardinality changes.
            for building_idx, state in enumerate(self._per_building):
                self._flush_rollout_boundary(building_idx, state)
            self._per_building = self._build_per_building_states(
                observation_names, action_names, metadata
            )
            self._pending_decisions = [None] * len(self._per_building)
            self._set_action_bounds(self._prepare_action_bounds(action_space, action_names))
            self._notify_bc_topology_change(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
            )
            return

        changed_buildings: List[int] = []
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
            # Stash new names then run the topology transition
            # (flush PPO → rebuild layout → re-validate rules).
            state.obs_names_tuple = new_obs
            state.action_names_tuple = new_act
            self._handle_topology_change(b)
            changed_buildings.append(b)
        self._set_action_bounds(self._prepare_action_bounds(action_space, action_names))
        if changed_buildings:
            self._notify_bc_topology_change(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
                changed_buildings=changed_buildings,
            )

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[npt.NDArray[np.float64]]] = None,
        encoded_observations: Optional[List[npt.NDArray[np.float64]]] = None,
    ) -> None:
        """Receive raw observations for the teacher without changing PPO input."""
        self._latest_raw_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in raw_observations]
            if raw_observations is not None
            else None
        )
        self._latest_encoded_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in encoded_observations]
            if encoded_observations is not None
            else None
        )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        building_count = len(self._per_building)
        if len(observations) != building_count:
            raise ValueError(
                f"TPPO predict observations has {len(observations)} rows; expected {building_count}."
            )
        det = bool(deterministic) if deterministic is not None else False
        if self._in_demonstration_phase():
            assert self._bc is not None
            teacher_observations = (
                self._latest_raw_observations
                if self._latest_raw_observations is not None
                else observations
            )
            self._pending_decisions = [None] * building_count
            return self._bc.compute_teacher_actions(teacher_observations)
        out: List[List[float]] = []
        pending_decisions: List[_PendingDecision] = []
        for building_idx, (state, obs) in enumerate(zip(self._per_building, observations)):
            obs_t = torch.as_tensor(
                np.asarray(obs), dtype=torch.float, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, pooled = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                tanh_actions, raw_log_prob, _, pre_tanh_action = state.actor(
                    ca_emb, deterministic=det, return_pre_tanh=True
                )
                low, high = self._action_bounds[building_idx]
                actions = self._affine_action(tanh_actions, low, high)
                log_prob = raw_log_prob - torch.log((high - low) / 2.0).squeeze(-1)
                value = state.value_normalizer.denormalize(state.critic(pooled).squeeze(-1))
            pending_decisions.append(_PendingDecision(
                observation=obs_t.squeeze(0).detach().cpu(),
                action=actions.squeeze(0).detach().cpu(),
                pre_tanh_action=pre_tanh_action.squeeze(0).detach().cpu(),
                log_prob=log_prob.squeeze(0).detach().cpu(),
                value=value.detach().cpu(),
            ))
            # ActorHead returns ``[B, N_ca, 1]``; the wrapper expects a flat
            # per-CA list.
            out.append(
                actions.squeeze(0)
                .squeeze(-1)
                .detach()
                .cpu()
                .tolist()
            )
        self._pending_decisions = pending_decisions
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
        del update_target_step, initial_exploration_done
        building_count = len(self._per_building)
        for name, rows in (("observations", observations), ("actions", actions),
                           ("rewards", rewards), ("next_observations", next_observations)):
            if len(rows) != building_count:
                raise ValueError(f"TPPO update {name} has {len(rows)} rows; expected {building_count}.")
        for name, value in (("terminated", terminated), ("truncated", truncated)):
            if not isinstance(value, (bool, np.bool_)) and len(value) != building_count:
                raise ValueError(f"TPPO update {name} has {len(value)} rows; expected {building_count}.")

        if self._in_demonstration_phase():
            assert self._bc is not None
            for building_idx, state in enumerate(self._per_building):
                teacher_action = np.asarray(actions[building_idx], dtype=np.float32)
                if teacher_action.shape != (state.layout.n_ca,):
                    raise ValueError(
                        f"Teacher action for building {state.building_id!r} has invalid shape."
                    )
                low, high = self._action_bounds[building_idx]
                low_values = low.squeeze(-1).detach().cpu().numpy()
                high_values = high.squeeze(-1).detach().cpu().numpy()
                teacher_tanh_action = (
                    2.0 * (teacher_action - low_values) / (high_values - low_values)
                    - 1.0
                )
                self._bc.record_demonstration(
                    building_idx,
                    np.asarray(observations[building_idx]),
                    state.layout,
                    teacher_tanh_action.tolist(),
                )
            self._pending_decisions = [None] * building_count
            self._latest_global_learning_step = int(global_learning_step)
            self._latest_training_metrics.update(
                {"episode_training": 1.0, "teacher_action_execution": 1.0}
            )
            self._latest_training_metrics.update(self._bc.snapshot_metrics())
            return

        candidates: List[_TransitionCandidate] = []
        for b, state in enumerate(self._per_building):
            pending = self._pending_decisions[b] if b < len(self._pending_decisions) else None
            if pending is None:
                raise ValueError(
                    f"TPPO update for building {state.building_id!r} has no pending decision. "
                    "Call predict() and pass its returned action before update()."
                )
            executed_action = torch.as_tensor(
                np.asarray(actions[b]), dtype=pending.action.dtype, device=pending.action.device
            ).view(state.layout.n_ca, 1)
            if not torch.equal(executed_action, pending.action):
                raise ValueError(
                    f"Executed action for building {state.building_id!r} does not match the "
                    "pending TPPO action from predict(). Pass predict()'s returned action unchanged, "
                    "or call predict() again before update()."
                )
            next_observation = next_observations[b]
            next_observation_t = None if next_observation is None else torch.as_tensor(
                np.asarray(next_observation), dtype=torch.float, device=self.device
            )
            candidates.append(_TransitionCandidate(
                pending=pending,
                raw_reward=float(rewards[b]),
                reward=max(float(rewards[b]), -self._reward_clip),
                terminated=bool(terminated) if isinstance(terminated, (bool, np.bool_)) else bool(terminated[b]),
                truncated=bool(truncated) if isinstance(truncated, (bool, np.bool_)) else bool(truncated[b]),
                next_observation=next_observation_t,
            ))

        collection_snapshot = self._snapshot_collection_state()
        try:
            for state, candidate in zip(self._per_building, candidates):
                state.buffer.add(
                    observation=candidate.pending.observation,
                    action=candidate.pending.action,
                    pre_tanh_action=candidate.pending.pre_tanh_action,
                    log_prob=candidate.pending.log_prob,
                    reward=candidate.reward,
                    value=candidate.pending.value,
                    terminated=candidate.terminated,
                    truncated=candidate.truncated,
                )
                state.last_next_observation = candidate.next_observation
                state.last_transition_terminated = candidate.terminated
                state.raw_rewards.append(candidate.raw_reward)
            self._pending_decisions = [None] * building_count
            self._latest_global_learning_step = int(global_learning_step)
            if not update_step:
                return
            optimizer_snapshot = (
                self._snapshot_update_runtime_state()
                if any(len(state.buffer) >= self._minibatch_size for state in self._per_building)
                else None
            )
            for building_idx, state in enumerate(self._per_building):
                candidate = candidates[building_idx]
                last_value = torch.zeros(1, device=self.device) if candidate.terminated or candidate.next_observation is None else self._critic_value(state, candidate.next_observation)
                if self._ppo_update(building_idx, state, last_value):
                    self._clear_rollout(building_idx, state)
        except Exception:
            if "optimizer_snapshot" in locals() and optimizer_snapshot is not None:
                self._restore_update_runtime_state(optimizer_snapshot)
            else:
                self._restore_collection_state(collection_snapshot)
            raise

    def on_episode_start(self, *, episode: int, training: bool) -> None:
        self._current_episode = episode
        self._current_episode_is_training = bool(training)

    def on_episode_end(self, *, episode: int, training: bool) -> None:
        self._current_episode = episode
        if not training:
            self._pending_decisions = [None] * len(self._per_building)
            return
        self._pending_decisions = [None] * len(self._per_building)
        if self._in_demonstration_phase() and self._bc is not None:
            if episode + 1 == self._bc.demonstration_episodes:
                self._run_bc_pretraining()
                self._bc_pretraining_complete = True
        for building_idx, state in enumerate(self._per_building):
            self._flush_rollout_boundary(building_idx, state)

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        return {} if self._bc is None else self._bc.snapshot_metrics()

    def record_topology_transition(
        self,
        *,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        terminated: bool,
        truncated: bool,
        global_learning_step: int,
    ) -> None:
        """Commit the old-layout transition before its successor is replaced."""
        self.update(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=[None] * len(self._per_building),
            terminated=terminated,
            truncated=truncated,
            update_target_step=False,
            global_learning_step=global_learning_step,
            update_step=False,
            initial_exploration_done=True,
        )

    def snapshot_topology_state(self) -> _TopologyStateSnapshot:
        """Capture all mutable agent state before a topology transaction."""
        return _TopologyStateSnapshot(
            agent_state=deepcopy(self.__dict__),
            python_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=self._capture_cuda_rng_state(),
        )

    def restore_topology_state(self, snapshot: _TopologyStateSnapshot) -> None:
        """Restore a snapshot after topology adaptation fails."""
        if not isinstance(snapshot, _TopologyStateSnapshot):
            raise TypeError("Invalid TPPO topology snapshot.")
        restored_state = deepcopy(snapshot.agent_state)
        self.__dict__.clear()
        self.__dict__.update(restored_state)
        random.setstate(snapshot.python_rng_state)
        np.random.set_state(snapshot.numpy_rng_state)
        torch.set_rng_state(snapshot.torch_rng_state)
        self._restore_cuda_rng_state(snapshot.cuda_rng_state)

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = {f"TPPO/{name}": value for name, value in self._latest_training_metrics.items()}
        self._latest_training_metrics = {}
        return metrics

    def _snapshot_update_runtime_state(self) -> _UpdateRuntimeSnapshot:
        states = self._per_building
        return _UpdateRuntimeSnapshot(
            buffers=[deepcopy(state.buffer) for state in states],
            last_next_observations=[
                None if state.last_next_observation is None else state.last_next_observation.detach().clone()
                for state in states
            ],
            last_transition_terminated=[state.last_transition_terminated for state in states],
            raw_rewards=[list(state.raw_rewards) for state in states],
            pending_decisions=deepcopy(self._pending_decisions),
            latest_global_learning_step=self._latest_global_learning_step,
            latest_training_metrics=dict(self._latest_training_metrics),
            ppo_update_count=self._ppo_update_count,
            tokenizer_states=[deepcopy(state.tokenizer.state_dict()) for state in states],
            backbone_states=[deepcopy(state.backbone.state_dict()) for state in states],
            actor_states=[deepcopy(state.actor.state_dict()) for state in states],
            critic_states=[deepcopy(state.critic.state_dict()) for state in states],
            optimizer_states=[deepcopy(state.optimizer.state_dict()) for state in states],
            bc_optimizer_states=[deepcopy(state.bc_optimizer.state_dict()) for state in states],
            value_normalizer_states=[state.value_normalizer.state_dict() for state in states],
            behavior_cloning_state=(
                self._bc.state_dict() if self._bc is not None else None
            ),
            python_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=self._capture_cuda_rng_state(),
        )

    def _snapshot_collection_state(self) -> _CollectionSnapshot:
        return _CollectionSnapshot(
            buffer_lengths=[len(state.buffer) for state in self._per_building],
            last_next_observations=[state.last_next_observation for state in self._per_building],
            last_transition_terminated=[state.last_transition_terminated for state in self._per_building],
            raw_reward_lengths=[len(state.raw_rewards) for state in self._per_building],
            pending_decisions=list(self._pending_decisions),
            latest_global_learning_step=self._latest_global_learning_step,
        )

    def _restore_collection_state(self, snapshot: _CollectionSnapshot) -> None:
        buffer_fields = (
            "observations", "actions", "pre_tanh_actions", "log_probs", "rewards",
            "values", "terminated", "truncated",
        )
        for index, state in enumerate(self._per_building):
            for name in buffer_fields:
                del getattr(state.buffer, name)[snapshot.buffer_lengths[index]:]
            state.buffer.advantages = None
            state.buffer.returns = None
            state.last_next_observation = snapshot.last_next_observations[index]
            state.last_transition_terminated = snapshot.last_transition_terminated[index]
            del state.raw_rewards[snapshot.raw_reward_lengths[index]:]
        self._pending_decisions = list(snapshot.pending_decisions)
        self._latest_global_learning_step = snapshot.latest_global_learning_step

    def _restore_update_runtime_state(self, snapshot: _UpdateRuntimeSnapshot) -> None:
        for index, state in enumerate(self._per_building):
            state.buffer = deepcopy(snapshot.buffers[index])
            state.last_next_observation = snapshot.last_next_observations[index]
            state.last_transition_terminated = snapshot.last_transition_terminated[index]
            state.raw_rewards = list(snapshot.raw_rewards[index])
            state.tokenizer.load_state_dict(snapshot.tokenizer_states[index])
            state.backbone.load_state_dict(snapshot.backbone_states[index])
            state.actor.load_state_dict(snapshot.actor_states[index])
            state.critic.load_state_dict(snapshot.critic_states[index])
            state.optimizer.load_state_dict(snapshot.optimizer_states[index])
            state.bc_optimizer.load_state_dict(snapshot.bc_optimizer_states[index])
            state.value_normalizer.load_state_dict(snapshot.value_normalizer_states[index])
        self._pending_decisions = deepcopy(snapshot.pending_decisions)
        self._latest_global_learning_step = snapshot.latest_global_learning_step
        self._latest_training_metrics = dict(snapshot.latest_training_metrics)
        self._ppo_update_count = snapshot.ppo_update_count
        if self._bc is not None and snapshot.behavior_cloning_state is not None:
            self._bc.load_state_dict(snapshot.behavior_cloning_state)
        random.setstate(snapshot.python_rng_state)
        np.random.set_state(snapshot.numpy_rng_state)
        torch.set_rng_state(snapshot.torch_rng_state)
        self._restore_cuda_rng_state(snapshot.cuda_rng_state)

    def export_artifacts(  # type: ignore[override]
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write per-building ONNX artefacts + return manifest metadata.

        Filename pattern: ``agent_<b>__topology_v<v>.onnx``. Per-building
        entry includes ``agent_index``, ``path``, ``format``, and a
        ``config`` block carrying the layout summary needed by the
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
            self._export_onnx(state, out / relpath, obs_dim, *self._action_bounds[b])
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
        self.preflight_checkpoint(step)
        out = Path(output_dir) / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"transformer_ppo_step{step}.pt"
        payload = {
            "checkpoint_format_version": 3,
            "step": int(step),
            "config": dict(self.config["algorithm"]),
            "global_learning_step": self._latest_global_learning_step,
            "ppo_update_count": self._ppo_update_count,
            "current_episode": self._current_episode,
            "bc_pretraining_complete": self._bc_pretraining_complete,
            "latest_training_metrics": dict(self._latest_training_metrics),
            "behavior_cloning_state": (
                self._bc.state_dict() if self._bc is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": self._capture_cuda_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "agents": [
                {
                    "building_id": s.building_id,
                    "tokenizer_state": s.tokenizer.state_dict(),
                    "backbone_state": s.backbone.state_dict(),
                    "actor_state": s.actor.state_dict(),
                    "critic_state": s.critic.state_dict(),
                    "optimizer_state": s.optimizer.state_dict(),
                    "bc_optimizer_state": s.bc_optimizer.state_dict(),
                    "layout_signature": tuple(sorted(s.obs_names_tuple)),
                    "action_names": list(s.action_names_tuple),
                    "action_bounds": [bound.detach().cpu() for bound in self._action_bounds[self._per_building.index(s)]],
                    "value_normalizer": s.value_normalizer.state_dict(),
                }
                for s in self._per_building
            ],
        }
        torch.save(payload, path)
        return str(path)

    def preflight_checkpoint(self, step: int) -> None:
        """Reject a deferred save before composite stages write checkpoint data."""
        del step
        if any(len(state.buffer) for state in self._per_building):
            raise DeferredCheckpointError(
                "TPPO cannot save a checkpoint with a nonempty rollout. Save at a completed optimizer or episode boundary."
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agents = payload["agents"]
        if len(agents) != len(self._per_building):
            raise ValueError(
                f"Checkpoint has {len(agents)} per-building entries; current "
                f"agent has {len(self._per_building)}. Cross-cardinality "
                "resume is not supported."
            )
        behavior_cloning_state = payload.get("behavior_cloning_state")
        if behavior_cloning_state is not None and self._bc is None:
            raise RuntimeError(
                "Checkpoint contains behavior-cloning state but the BC-disabled "
                "target cannot restore it."
            )
        if behavior_cloning_state is not None:
            BehaviorCloningRegularizer.validate_state_dict(
                behavior_cloning_state,
                max_samples_per_building=(
                    self._bc.max_samples_per_building if self._bc is not None else None
                ),
            )
            self._validate_checkpoint_bc_tokenizer_compatibility(
                behavior_cloning_state
            )
        checkpoint_version = payload.get("checkpoint_format_version", 1)
        if (
            not isinstance(checkpoint_version, int)
            or isinstance(checkpoint_version, bool)
            or checkpoint_version not in {1, 2, 3}
        ):
            raise RuntimeError(
                "Unsupported TPPO checkpoint_format_version "
                f"{checkpoint_version!r}; expected one of 1, 2, or 3."
            )
        if checkpoint_version == 3:
            bc_pretraining_complete = payload.get("bc_pretraining_complete")
            if not isinstance(bc_pretraining_complete, bool):
                raise RuntimeError(
                    "Checkpoint bc_pretraining_complete must be a boolean."
                )
        else:
            bc_pretraining_complete = bool(
                behavior_cloning_state is not None
                and float(
                    behavior_cloning_state.get("latest_pretraining_epochs", 0.0)
                )
                > 0.0
            )
        if bc_pretraining_complete and self._bc is None:
            raise RuntimeError(
                "Checkpoint completed BC pretraining but the target has BC disabled."
            )
        for building_idx, (state, saved) in enumerate(zip(self._per_building, agents)):
            sig_now = tuple(sorted(state.obs_names_tuple))
            sig_saved = tuple(saved["layout_signature"])
            if sig_now != sig_saved:
                raise ValueError(
                    "Checkpoint layout_signature mismatch for building "
                    f"{state.building_id!r}: cannot resume across topology "
                    "changes."
                )
            if "action_names" in saved and list(saved["action_names"]) != list(state.action_names_tuple):
                raise ValueError(f"Checkpoint action_names mismatch for building {state.building_id!r}.")
            if "action_bounds" in saved:
                try:
                    saved_low, saved_high = (
                        torch.as_tensor(bound, dtype=torch.float64).reshape(-1, 1)
                        for bound in saved["action_bounds"]
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Checkpoint action_bounds are invalid for building {state.building_id!r}."
                    ) from error
                current_low, current_high = self._action_bounds[building_idx]
                if not (
                    torch.equal(saved_low, current_low.detach().cpu().to(dtype=torch.float64))
                    and torch.equal(saved_high, current_high.detach().cpu().to(dtype=torch.float64))
                ):
                    raise ValueError(
                        f"Checkpoint action_bounds mismatch for building {state.building_id!r}; "
                        "resume requires the same environment action bounds."
                    )
        for state, saved in zip(self._per_building, agents):
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.critic.load_state_dict(saved["critic_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])
            if "bc_optimizer_state" in saved:
                state.bc_optimizer.load_state_dict(saved["bc_optimizer_state"])
            for optimizer in (state.optimizer, state.bc_optimizer):
                for parameter_state in optimizer.state.values():
                    for key, value in parameter_state.items():
                        if isinstance(value, torch.Tensor):
                            parameter_state[key] = value.to(self.device)
            if "value_normalizer" in saved:
                state.value_normalizer.load_state_dict(saved["value_normalizer"])
        self._latest_global_learning_step = int(payload.get("global_learning_step", 0))
        self._ppo_update_count = int(payload.get("ppo_update_count", 0))
        self._current_episode = int(payload.get("current_episode", 0))
        self._bc_pretraining_complete = bc_pretraining_complete
        self._latest_training_metrics = dict(payload.get("latest_training_metrics", {}))
        if self._bc is not None and behavior_cloning_state is not None:
            self._bc.load_state_dict(behavior_cloning_state)
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
        self._restore_cuda_rng_state(payload.get("cuda_rng_state"))
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])
        if "python_rng_state" in payload:
            random.setstate(payload["python_rng_state"])

    def _validate_checkpoint_bc_tokenizer_compatibility(
        self, behavior_cloning_state: Dict[str, Any]
    ) -> None:
        """Reject stored demos that cannot be represented by the attached model."""
        nfc_type = self._tokenizer_config.nfc.type_name
        sro_types = set(self._tokenizer_config.sro_types)
        ca_types = set(self._tokenizer_config.ca_types)
        for building_idx, demonstrations in behavior_cloning_state["demonstrations"].items():
            if not isinstance(building_idx, int) or not 0 <= building_idx < len(self._per_building):
                raise RuntimeError(
                    "Checkpoint BC layout/tokenizer compatibility failed: invalid "
                    f"building index {building_idx!r}."
                )
            projections = self._per_building[building_idx].tokenizer.projections
            for demonstration in demonstrations:
                for segment in demonstration.layout.segments:
                    valid_type = (
                        (segment.family == "nfc" and segment.type_name == nfc_type)
                        or (segment.family == "sro" and segment.type_name in sro_types)
                        or (segment.family == "ca" and segment.type_name in ca_types)
                    )
                    if not valid_type:
                        raise RuntimeError(
                            "Checkpoint BC layout/tokenizer compatibility failed: "
                            f"invalid {segment.family} type {segment.type_name!r}."
                        )
                    if segment.type_name not in projections:
                        raise RuntimeError(
                            "Checkpoint BC layout/tokenizer compatibility failed: "
                            f"building {building_idx} has no projection for "
                            f"type {segment.type_name!r}."
                        )
                    expected_width = 1 if segment.family == "nfc" else len(segment.feature_indices)
                    actual_width = int(projections[segment.type_name].in_features)
                    if actual_width != expected_width:
                        raise RuntimeError(
                            "Checkpoint BC layout/tokenizer compatibility failed: "
                            f"building {building_idx} projection for "
                            f"{segment.type_name!r} expects {actual_width} features, "
                            f"but stored layout provides {expected_width}."
                        )

    # ==========================================================================
    # Internal helpers
    # ==========================================================================

    def _capture_cuda_rng_state(self) -> Optional[torch.Tensor]:
        if self.device.type != "cuda":
            return None
        return torch.cuda.get_rng_state(self.device).cpu()

    def _restore_cuda_rng_state(self, state: Optional[torch.Tensor]) -> None:
        if self.device.type == "cuda" and state is not None:
            torch.cuda.set_rng_state(state.cpu(), self.device)

    @staticmethod
    def _validate_environment_metadata(
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: Any,
        observation_space: Any,
    ) -> Tuple[List[Any], List[Any]]:
        expected_count = len(observation_names)
        if len(action_names) != expected_count:
            raise ValueError(
                "observation_names and action_names must have equal counts; "
                f"got {expected_count} and {len(action_names)}."
            )

        def normalize_spaces(name: str, spaces: Any) -> List[Any]:
            if isinstance(spaces, (list, tuple)):
                if len(spaces) != expected_count:
                    raise ValueError(
                        f"{name} has {len(spaces)} per-building entries; "
                        f"expected {expected_count}."
                    )
                return list(spaces)
            return [spaces] * expected_count

        return normalize_spaces("action_space", action_space), normalize_spaces(
            "observation_space", observation_space
        )

    def _prepare_action_bounds(
        self, action_space: List[Any], action_names: List[List[str]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bounds: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for building_idx, names in enumerate(action_names):
            action_count = len(names)
            space = action_space[building_idx]
            has_low, has_high = hasattr(space, "low"), hasattr(space, "high")
            if has_low != has_high:
                raise ValueError(
                    f"Action-space object for building {building_idx} must expose both low and high attributes."
                )
            if has_low:
                low = np.asarray(space.low, dtype=np.float64).reshape(-1)
                high = np.asarray(space.high, dtype=np.float64).reshape(-1)
                if low.shape[0] != action_count or high.shape[0] != action_count:
                    raise ValueError(
                        f"Action-space bounds for building {building_idx} have shape "
                        f"({low.shape[0]}, {high.shape[0]}), expected exactly {action_count}."
                    )
                if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low >= high):
                    raise ValueError(
                        f"Action-space bounds for building {building_idx} must be finite and satisfy low < high."
                    )
            else:
                low = np.full(action_count, -1.0, dtype=np.float64)
                high = np.full(action_count, 1.0, dtype=np.float64)
            bounds.append((
                torch.as_tensor(low, dtype=torch.float64, device=self.device).view(action_count, 1),
                torch.as_tensor(high, dtype=torch.float64, device=self.device).view(action_count, 1),
            ))
        return bounds

    def _set_action_bounds(self, bounds: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self._action_bounds = bounds

    def _attach_bc_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self._bc is not None:
            self._bc.attach_environment(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
            )

    def _notify_bc_topology_change(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
        changed_buildings: Optional[List[int]] = None,
    ) -> None:
        if self._bc is not None:
            self._bc.on_topology_change(
                observation_names=observation_names,
                action_names=action_names,
                action_space=action_space,
                observation_space=observation_space,
                metadata=metadata,
                changed_buildings=changed_buildings,
            )

    def _build_per_building_states(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> List[_PerBuildingState]:
        building_names = (metadata or {}).get("building_names")
        return [
            self._build_one_per_building_state(
                building_names[b] if building_names and b < len(building_names) and building_names[b] else f"building_{b}",
                list(obs_n), list(act_n),
            )
            for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names))
        ]

    def _build_one_per_building_state(
        self,
        building_id: str,
        observation_names: List[str],
        action_names: List[str],
    ) -> _PerBuildingState:
        layout = self._layout_builder.build(
            building_id, observation_names, action_names
        )
        # post-condition: CA order matches simulator action order.
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
        ).to(self.device)
        backbone = TransformerBackbone(
            d_model=self._d_model,
            nhead=self._nhead,
            num_layers=self._num_layers,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
        ).to(self.device)
        actor = ActorHead(
            d_model=self._d_model, hidden_dim=self._actor_hidden_dim
        ).to(self.device)
        actor.log_std.data.fill_(self._actor_log_std_init)
        critic = CriticHead(
            d_model=self._d_model, hidden_dim=self._critic_hidden_dim
        ).to(self.device)
        params = (
            list(tokenizer.parameters())
            + list(backbone.parameters())
            + list(actor.parameters())
            + list(critic.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=self._lr)
        bc_optimizer = torch.optim.Adam(
            list(tokenizer.parameters())
            + list(backbone.parameters())
            + list(actor.parameters()),
            lr=self._lr,
        )
        buffer = RolloutBuffer(gamma=self._gamma, gae_lambda=self._gae_lambda)
        return _PerBuildingState(
            building_id=building_id,
            tokenizer=tokenizer,
            backbone=backbone,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            bc_optimizer=bc_optimizer,
            buffer=buffer,
            value_normalizer=RunningValueNormalizer(),
            layout=layout,
            obs_names_tuple=tuple(observation_names),
            action_names_tuple=tuple(action_names),
        )

    def _handle_topology_change(self, building_idx: int) -> None:
        """Flush PPO → rebuild layout → re-validate rules."""
        state = self._per_building[building_idx]

        old_n_ca = state.layout.n_ca
        old_n_sro = state.layout.n_sro
        old_actions = list(state.layout.ca_action_names)
        old_obs_dim = len(state.layout.segments)

        # 1. Flush in-flight rollout (best-effort) before discarding the old
        #    layout. ``_ppo_update`` swallows empty-buffer no-ops.
        if len(state.buffer) > 0:
            try:
                # Use a zero "next value" — last transition is treated as
                # terminal because the topology under it is gone. This is
                # the conservative choice; a future improvement could
                # forward through the new layout's critic.
                self._run_ppo_update_with_last_value(
                    state, last_value=torch.zeros(1, device=self.device), building_idx=building_idx
                )
            finally:
                self._clear_rollout(building_idx, state)

        # 2. Rebuild the layout from the cached ``names``.
        new_layout = self._layout_builder.build(
            state.building_id,
            list(state.obs_names_tuple),
            list(state.action_names_tuple),
        )

        # 3. Re-run the hard-fail rules against a synthetic single-table
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
        #    are preserved (per-type weight sharing).
        state.layout = new_layout
        state.topology_version += 1

        logger.info(
            "Topology change: {} v{} — "
            "n_ca: {} → {}, n_sro: {} → {}, "
            "actions: {} → {}",
            state.building_id,
            state.topology_version,
            old_n_ca,
            new_layout.n_ca,
            old_n_sro,
            new_layout.n_sro,
            old_actions,
            list(new_layout.ca_action_names),
        )

        # 6. post-condition. Mirror the startup-side prefix
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
        """Per-type input dim derived from segment widths.

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
    def _infer_obs_dim(layout: BuildingTokenLayout) -> int:
        return BehaviorCloningRegularizer.full_representation_width(layout)

    def _in_demonstration_phase(self) -> bool:
        return (
            self._bc is not None
            and not self._bc_pretraining_complete
            and self._current_episode_is_training
            and self._current_episode < self._bc.demonstration_episodes
        )

    def _run_bc_pretraining(self) -> None:
        """Fit the shared representation and actor to recorded teacher samples."""
        assert self._bc is not None
        logger.info("event=bc_pretraining_start buildings={}", len(self._per_building))
        failure_logged = False
        try:
            grouped_by_building = [
                self._bc.demonstrations_for_building_by_signature(building_idx)
                for building_idx in range(len(self._per_building))
            ]
            missing_buildings = [
                state.building_id
                for state, grouped in zip(self._per_building, grouped_by_building)
                if not grouped
            ]
            if missing_buildings:
                logger.info(
                    "event=bc_pretraining_failure reason=zero_usable_demonstrations "
                    "missing_building_count={} total_buildings={}",
                    len(missing_buildings), len(self._per_building),
                )
                failure_logged = True
                raise RuntimeError(
                    "Behavior-cloning pretraining has zero compatible demonstrations for "
                    f"building(s): {', '.join(missing_buildings)}."
                )
            total_usable_samples = 0
            total_trained_batches = 0
            metrics: Dict[str, float] = {}
            for state, grouped in zip(self._per_building, grouped_by_building):
                usable_samples = 0
                trained_batches = 0
                for group_index, demonstrations in enumerate(grouped.values(), start=1):
                    layout = demonstrations[0].layout
                    group_samples = len(demonstrations)
                    group_trained_batches = 0
                    usable_samples += group_samples
                    for _ in range(self._bc.pretraining_epochs):
                        for start in range(0, group_samples, self._bc.batch_size):
                            batch = demonstrations[start : start + self._bc.batch_size]
                            observations = torch.as_tensor(
                                np.stack([demo.observation for demo in batch]),
                                dtype=torch.float, device=self.device,
                            )
                            state.bc_optimizer.zero_grad()
                            tokenized = state.tokenizer(observations, layout)
                            ca_embeddings, _ = state.backbone(
                                tokenized.sro_tokens,
                                tokenized.nfc_token,
                                tokenized.ca_tokens,
                            )
                            loss = self._bc.demonstration_loss(
                                layout=layout,
                                demonstrations=list(batch),
                                predicted_means=torch.tanh(state.actor.mlp(ca_embeddings)),
                                global_learning_step=0,
                                apply_weight=False,
                            )
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(state.tokenizer.parameters())
                                + list(state.backbone.parameters())
                                + list(state.actor.parameters()),
                                self._max_grad_norm,
                            )
                            state.bc_optimizer.step()
                            trained_batches += 1
                            group_trained_batches += 1
                    logger.info(
                        "event=bc_pretraining_group building_id={} group_index={} "
                        "group_count={} group_samples={} usable_samples={} trained_batches={}",
                        state.building_id,
                        group_index,
                        len(grouped),
                        group_samples,
                        group_samples,
                        group_trained_batches,
                    )
                metrics[f"behavior_cloning_building_{state.building_id}_usable_samples"] = float(usable_samples)
                metrics[f"behavior_cloning_building_{state.building_id}_trained_batches"] = float(trained_batches)
                total_usable_samples += usable_samples
                total_trained_batches += trained_batches
            self._bc.set_pretraining_epochs(self._bc.pretraining_epochs)
            self._bc.set_incompatible_demonstration_samples(0)
            metrics["behavior_cloning_pretraining_batches"] = float(total_trained_batches)
            self._latest_training_metrics.update(self._bc.snapshot_metrics())
            self._latest_training_metrics.update(metrics)
            logger.info(
                "event=bc_pretraining_complete buildings={} usable_samples={} trained_batches={}",
                len(self._per_building), total_usable_samples, total_trained_batches,
            )
        except Exception as error:
            if not failure_logged:
                logger.info(
                    "event=bc_pretraining_failure reason=pretraining_error error_type={}",
                    type(error).__name__,
                )
            raise

    def _run_auxiliary_bc_update(
        self, building_idx: int, state: _PerBuildingState
    ) -> None:
        """Apply one actor-only BC step after all PPO epochs are complete."""
        assert self._bc is not None
        demonstrations = self._bc.sample_demonstrations(
            building_idx, state.layout, self._bc.batch_size
        )
        if not demonstrations:
            return
        observations = torch.as_tensor(
            np.stack([demo.observation for demo in demonstrations]),
            dtype=torch.float, device=self.device,
        )
        state.bc_optimizer.zero_grad()
        tokenized = state.tokenizer(observations, state.layout)
        ca_embeddings, _ = state.backbone(
            tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
        )
        loss = self._bc.demonstration_loss(
            layout=state.layout,
            demonstrations=demonstrations,
            predicted_means=torch.tanh(state.actor.mlp(ca_embeddings)),
            global_learning_step=self._latest_global_learning_step,
        )
        if not loss.requires_grad:
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(state.tokenizer.parameters())
            + list(state.backbone.parameters())
            + list(state.actor.parameters()),
            self._max_grad_norm,
        )
        state.bc_optimizer.step()

    # ----- PPO loss helpers ---------------------------------------------------

    @staticmethod
    def _compute_log_prob(
        actor: ActorHead,
        ca_embeddings: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """Score retained samples under the affine squashed policy."""
        return actor.log_prob_from_pre_tanh(ca_embeddings, pre_tanh_actions) - torch.log(
            (high - low) / 2.0
        ).squeeze(-1)

    @staticmethod
    def _affine_action(tanh_actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return low + (tanh_actions + 1.0) * ((high - low) / 2.0)

    def _ppo_update(
        self,
        building_idx: int,
        state: _PerBuildingState,
        last_value: Optional[torch.Tensor],
    ) -> bool:
        if len(state.buffer) == 0:
            return False
        # A short update cadence retains the rollout until it reaches one
        # minibatch. Episode and topology boundaries still train a short final
        # rollout through ``_flush_rollout_boundary``.
        if len(state.buffer) < self._minibatch_size:
            return False
        assert last_value is not None
        return self._run_ppo_update_with_last_value(state, last_value, building_idx=building_idx)

    def _clear_rollout(self, building_idx: int, state: _PerBuildingState) -> None:
        del building_idx
        state.buffer.clear()
        state.last_next_observation = None
        state.last_transition_terminated = False
        state.raw_rewards.clear()

    def _flush_rollout_boundary(
        self, building_idx: int, state: _PerBuildingState,
        last_value: Optional[torch.Tensor] = None,
    ) -> None:
        if len(state.buffer) == 0:
            return
        if last_value is None:
            last_value = torch.zeros(1, device=self.device) if state.last_transition_terminated or state.last_next_observation is None else self._critic_value(state, state.last_next_observation)
        if self._run_ppo_update_with_last_value(state, last_value, building_idx=building_idx):
            self._clear_rollout(building_idx, state)

    def _critic_value(
        self,
        state: _PerBuildingState,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            obs_t = obs.to(self.device).unsqueeze(0)
            tokenized = state.tokenizer(obs_t, state.layout)
            _, pooled = state.backbone(
                tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
            )
            return state.value_normalizer.denormalize(state.critic(pooled).squeeze(-1))

    def _run_ppo_update_with_last_value(
        self, state: _PerBuildingState, last_value: torch.Tensor, *, building_idx: int
    ) -> bool:
        if len(state.buffer) == 0:
            return False
        rollout_size = len(state.buffer)
        raw_rewards = np.asarray(state.raw_rewards, dtype=np.float64)
        clipped_rewards = np.asarray(state.buffer.rewards, dtype=np.float64)
        batch_size = min(self._minibatch_size, len(state.buffer))
        state.buffer.compute_returns_and_advantages(last_value.detach().cpu())
        assert state.buffer.returns is not None
        state.value_normalizer.update(state.buffer.returns)
        all_metrics: dict = {"policy_loss": [], "value_loss": [], "entropy": [], "actor_grad_norm": [], "critic_grad_norm": []}
        for _epoch in range(self._ppo_epochs):
            for batch in state.buffer.get_batches(batch_size):
                state.optimizer.zero_grad()
                obs_b = batch.observations.to(self.device)
                # Forward through tokenizer + backbone with grads on.
                tokenized = state.tokenizer(obs_b, state.layout)
                ca_emb, pooled = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                log_probs_new = self._compute_log_prob(
                    state.actor, ca_emb, batch.pre_tanh_actions.to(self.device), *self._action_bounds[building_idx]
                )  # [B, N_ca]
                # Sum over CA actions per step → scalar per step (matches
                # log_probs_old shape stored in buffer).
                log_probs_new_sum = log_probs_new.sum(dim=-1)
                log_probs_old_sum = batch.log_probs.to(self.device).sum(dim=-1)
                values = state.critic(pooled).squeeze(-1)
                loss, _metrics = compute_ppo_loss(
                    log_probs_new=log_probs_new_sum,
                    log_probs_old=log_probs_old_sum,
                    advantages=batch.advantages.to(self.device),
                    values=values,
                    returns=state.value_normalizer.normalize(batch.returns).to(self.device),
                    clip_eps=self._clip_eps,
                    value_coeff=self._value_coeff,
                    entropy_coeff=self._entropy_coeff,
                )
                loss.backward()
                actor_parameters = list(state.tokenizer.parameters()) + list(state.backbone.parameters()) + list(state.actor.parameters())
                critic_parameters = list(state.critic.parameters())
                all_metrics["actor_grad_norm"].append(self._gradient_norm(actor_parameters))
                all_metrics["critic_grad_norm"].append(self._gradient_norm(critic_parameters))
                torch.nn.utils.clip_grad_norm_(actor_parameters + critic_parameters, self._max_grad_norm)
                state.optimizer.step()
                for k, v in _metrics.items():
                    all_metrics.setdefault(k, []).append(v)
        # Do not alter the policy between PPO minibatches: the rollout was
        # collected by a single policy.  BC is therefore deliberately a
        # separate post-PPO actor-only step.
        if self._bc is not None:
            self._run_auxiliary_bc_update(building_idx, state)
        averaged = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        returns = state.buffer.returns.detach().cpu().numpy()
        values = torch.stack(state.buffer.values).squeeze(-1).cpu().numpy()
        residuals = np.abs(returns - values)
        averaged.update({
            "update_count": float(self._ppo_update_count + 1), "rollout_size": float(rollout_size),
            "raw_reward_mean": float(raw_rewards.mean()), "raw_reward_min": float(raw_rewards.min()), "raw_reward_max": float(raw_rewards.max()),
            "clipped_reward_mean": float(clipped_rewards.mean()), "clipped_reward_min": float(clipped_rewards.min()), "clipped_reward_max": float(clipped_rewards.max()),
            "value_residual_p50": float(np.percentile(residuals, 50)), "value_residual_p90": float(np.percentile(residuals, 90)), "value_residual_p99": float(np.percentile(residuals, 99)),
            "episode_training": 1.0,
            "teacher_action_execution": 0.0,
        })
        self._ppo_update_count += 1
        self._latest_training_metrics.update(
            {f"building_{building_idx}/{name}": value for name, value in averaged.items()}
        )
        if self._bc is not None:
            self._latest_training_metrics.update(self._bc.snapshot_metrics())
        building_id = getattr(state, "building_id", "?")
        logger.info(
            "PPO update [{}]: policy_loss={:.4f}, value_loss={:.4f}, entropy={:.4f}, clip_frac={:.3f}",
            building_id,
            averaged.get("policy_loss", 0.0),
            averaged.get("value_loss", 0.0),
            averaged.get("entropy", 0.0),
            averaged.get("clip_fraction", 0.0),
        )
        return True

    @staticmethod
    def _gradient_norm(parameters: List[torch.nn.Parameter]) -> float:
        squared_norm = sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in parameters
            if parameter.grad is not None
        )
        return float(squared_norm.sqrt().cpu()) if isinstance(squared_norm, torch.Tensor) else 0.0

    # ----- ONNX export --------------------------------------------------------

    def _export_onnx(
        self,
        state: _PerBuildingState,
        path: Path,
        obs_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
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

        tokenizer = deepcopy(state.tokenizer).to("cpu").eval()
        backbone = deepcopy(state.backbone).to("cpu").eval()
        actor = deepcopy(state.actor).to("cpu").eval()

        class _ExportWrapper(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.tokenizer = tokenizer
                self_inner.backbone = backbone
                self_inner.actor = actor
                self_inner.register_buffer("action_low", action_low.detach().cpu())
                self_inner.register_buffer("action_high", action_high.detach().cpu())

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
                actions = self_inner.action_low + (torch.tanh(means) + 1.0) * (
                    (self_inner.action_high - self_inner.action_low) / 2.0
                )
                return actions.squeeze(-1)

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
    the agent-level observation_names, so the validator can re-run its rules
    at runtime after a topology change without needing the env.

    Naming mirrors ``utils/entity_adapter.py``:

    - ``district__<feat>``                         → ``district`` (prefix kept)
    - ``storage::<id>::<feat>``                    → ``storage``
    - ``pv::<id>::<feat>``                         → ``pv``
    - ``charger::<id>::<feat>``                    → ``charger``
    - ``charger::<id>::(connected_ev|incoming_ev)::<feat>`` → ``ev``
    - everything else (no ``::`` and no ``district__``) → ``building``
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
            by_table["district"].add(name)
            continue
        if "::" in name:
            head, *rest = name.split("::")
            if head not in {"storage", "pv", "charger"}:
                continue
            if head == "charger" and len(rest) >= 3 and rest[1] in {
                "connected_ev",
                "incoming_ev",
            }:
                by_table["ev"].add(rest[2])
            else:
                by_table[head].add(rest[1] if len(rest) >= 2 else rest[0])
            continue
        by_table["building"].add(name)
    return EntityPayloadSample(
        feature_names_per_table={k: sorted(v) for k, v in by_table.items()}
    )
