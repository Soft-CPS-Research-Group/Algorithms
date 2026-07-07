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
from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_critic_update import compute_target_q, critic_update_step
from algorithms.utils.matd3_global_packer import BuildingLayout, GlobalTokenPacker
from algorithms.utils.matd3_replay import (
    LayoutSummary,
    TopologyPartitionedReplay,
    TransitionData,
    compute_topology_signature,
)
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
        critic_cfg = dict(algo["transformer_critic"])
        self._critic_d_model = int(critic_cfg["d_model"])
        self._critic_nhead = int(critic_cfg["nhead"])
        self._critic_num_layers = int(critic_cfg["num_layers"])
        self._critic_dim_feedforward = int(critic_cfg.get("dim_feedforward", 256))
        self._critic_dropout = float(critic_cfg.get("dropout", 0.1))
        self._critic_lr = float(h["critic_lr"])
        self._gamma = float(h["gamma"])
        self._tau = float(h["tau"])
        self._batch_size = int(h["batch_size"])
        self._replay_capacity = int(h["replay_capacity"])
        self._critic_action_input_mode = str(h.get("critic_action_input_mode", "final"))
        self._num_token_types = 8
        self._max_buildings = 16

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._actors: List[_ActorState] = []
        self._online_critics: Optional[TwinTransformerCritics] = None
        self._target_critics: Optional[TwinTransformerCritics] = None
        self._global_packer: Optional[GlobalTokenPacker] = None
        self._replay: Optional[TopologyPartitionedReplay] = None
        self._critic_optimizer: Optional[torch.optim.Optimizer] = None
        self._critic_update_count = 0
        self._target_update_count = 0
        self._actor_update_count = 0
        self._actor_update_interval = int(h["actor_update_interval"])
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_observations: Optional[List[np.ndarray]] = None
        self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_next_observations: Optional[List[np.ndarray]] = None
        self._latest_teacher_actions: Optional[List[List[float]]] = None
        self._latest_next_teacher_actions: Optional[List[List[float]]] = None
        self._teacher_alive = bool((algo.get("exploration") or {}).get("warm_start_policy", {}).get("enabled", False))
        self._warm_start_policy: Optional[Any] = self if self._teacher_alive else None
        self._latest_training_metrics: Dict[str, float] = {}
        self._last_logged_action_counts: Optional[List[int]] = None
        self._reward_normalization_enabled = bool(h.get("reward_normalization", False))
        self._reward_norm_count = 0
        self._reward_norm_mean = 0.0
        self._reward_norm_m2 = 0.0
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
        counts = [len(actions) for actions in out]
        if counts != self._last_logged_action_counts:
            logger.info(
                "AgentTransformerMATD3 action counts changed: topology_signature={}, counts={}, total={}",
                self._current_topology_signature() if self._actors else "unattached",
                counts,
                sum(counts),
            )
            self._last_logged_action_counts = list(counts)
        return out

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Receive wrapper-side observation context for teacher action computation."""
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
        if self._teacher_alive and self._warm_start_policy is not None:
            self._latest_teacher_actions = self._compute_teacher_actions(
                self._latest_raw_observations
            )
        else:
            self._latest_teacher_actions = None

    def set_transition_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        raw_next_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
        encoded_next_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Receive current/next observation context for teacher-aware replay."""
        if raw_observations is not None:
            self._latest_raw_observations = [
                np.asarray(obs, dtype=np.float64) for obs in raw_observations
            ]
        if encoded_observations is not None:
            self._latest_encoded_observations = [
                np.asarray(obs, dtype=np.float64) for obs in encoded_observations
            ]
        self._latest_raw_next_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in raw_next_observations]
            if raw_next_observations is not None
            else None
        )
        self._latest_encoded_next_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in encoded_next_observations]
            if encoded_next_observations is not None
            else None
        )
        if self._teacher_alive and self._warm_start_policy is not None:
            self._latest_teacher_actions = self._compute_teacher_actions(
                self._latest_raw_observations
            )
            self._latest_next_teacher_actions = self._compute_teacher_actions(
                self._latest_raw_next_observations
            )
        else:
            self._latest_teacher_actions = None
            self._latest_next_teacher_actions = None

    def _compute_teacher_actions(
        self, raw_observations: Optional[List[np.ndarray]]
    ) -> Optional[List[List[float]]]:
        """Get teacher policy actions for the given observations."""
        if self._warm_start_policy is None or raw_observations is None:
            return None
        actions = self._warm_start_policy.predict(raw_observations, deterministic=True)
        return [
            np.clip(np.asarray(a, dtype=np.float64).reshape(-1), -1.0, 1.0).tolist()
            for a in actions
        ]

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
        done = bool(terminated or truncated)
        topology_sig = self._current_topology_signature()
        self._replay.set_active_signature(topology_sig)
        transition = TransitionData(
            observations=[np.asarray(o, dtype=np.float32) for o in observations],
            next_observations=[np.asarray(o, dtype=np.float32) for o in next_observations],
            actions=[np.asarray(a, dtype=np.float32) for a in actions],
            base_actions=[np.zeros_like(np.asarray(a, dtype=np.float32)) for a in actions],
            next_base_actions=[np.zeros_like(np.asarray(a, dtype=np.float32)) for a in actions],
            rewards=[float(r) for r in rewards],
            done=done,
            topology_signature=topology_sig,
            layout_summaries=[
                LayoutSummary(
                    building_id=s.building_id,
                    n_ca=s.layout.n_ca,
                    n_sro=s.layout.n_sro,
                    obs_dim=len(s.obs_names_tuple),
                    action_dim=s.layout.n_ca,
                )
                for s in self._actors
            ],
        )
        self._replay.push(transition)

        if not initial_exploration_done or not update_step:
            return
        if self._replay.active_partition_size < self._batch_size:
            return
        self._perform_critic_update()
        if self._critic_update_count % self._actor_update_interval == 0:
            self._perform_actor_update()
            self._actor_update_count += 1
        if update_target_step:
            if self._critic_update_count % self._actor_update_interval != 0:
                self._perform_actor_update()
            self._target_critics.soft_update_from(self._online_critics, self._tau)
            for state in self._actors:
                self._soft_update(state.actor, state.target_actor, self._tau)
            self._target_update_count += 1
        if self._should_log_training_step(global_learning_step):
            self._record_training_metrics(
                self._collect_diagnostics(global_learning_step), global_learning_step
            )

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        """Gate for the wrapper: true when initial exploration window is over."""
        exploration_cfg = self.config["algorithm"].get("exploration", {}) or {}
        end_step = int(exploration_cfg.get("end_initial_exploration_time_step", 0))
        return global_learning_step >= end_step

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
            "online_critics_state": self._online_critics.state_dict() if self._online_critics else None,
            "target_critics_state": self._target_critics.state_dict() if self._target_critics else None,
            "critic_1_state": self._critic_1.state_dict(),
            "critic_2_state": self._critic_2.state_dict(),
            "target_critic_1_state": self._target_critic_1.state_dict(),
            "target_critic_2_state": self._target_critic_2.state_dict(),
            "critic_optimizer_state": self._critic_optimizer.state_dict() if self._critic_optimizer else None,
            "global_packer_state": self._global_packer.state_dict() if self._global_packer else None,
            "replay_state": self._replay.state_dict() if self._replay else None,
            "active_topology_signature": self._current_topology_signature(),
            "critic_update_count": self._critic_update_count,
            "actor_update_count": self._actor_update_count,
            "target_update_count": self._target_update_count,
            "reward_normalization_state": {
                "count": getattr(self, "_reward_norm_count", 0),
                "mean": getattr(self, "_reward_norm_mean", 0.0),
                "m2": getattr(self, "_reward_norm_m2", 0.0),
            },
            "exploration_state": {
                "exploration_step": getattr(self, "_exploration_step", 0),
                "teacher_alive": self._teacher_alive,
                "bc_effective_weight": getattr(self, "_bc_effective_weight", 0.0),
                "residual_scale": getattr(self, "_residual_scale", 0.0),
            },
            "per_type_feature_dims": self._get_per_type_feature_dims(),
            "n_buildings": len(self._actors),
            "rng_state": {
                "torch": torch.random.get_rng_state(),
                "numpy": np.random.get_state(),
            },
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
        saved_dims = payload.get("per_type_feature_dims", {})
        current_dims = self._get_per_type_feature_dims()
        for type_name, saved_dim in saved_dims.items():
            current_dim = current_dims.get(type_name)
            if current_dim is not None and int(saved_dim) != int(current_dim):
                raise ValueError(
                    f"feature count mismatch for type {type_name!r}: checkpoint has "
                    f"{saved_dim}, current has {current_dim}. Cannot restore weights."
                )
        for state, saved in zip(self._actors, actors_data):
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.target_actor.load_state_dict(saved["target_actor_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])
        if payload.get("online_critics_state") and self._online_critics:
            self._online_critics.load_state_dict(payload["online_critics_state"])
        elif payload.get("critic_1_state") and self._online_critics:
            self._critic_1.load_state_dict(payload["critic_1_state"])
            self._critic_2.load_state_dict(payload["critic_2_state"])
        if payload.get("target_critics_state") and self._target_critics:
            self._target_critics.load_state_dict(payload["target_critics_state"])
        elif payload.get("target_critic_1_state") and self._target_critics:
            self._target_critic_1.load_state_dict(payload["target_critic_1_state"])
            self._target_critic_2.load_state_dict(payload["target_critic_2_state"])
        if payload.get("critic_optimizer_state") and self._critic_optimizer:
            self._critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
        if payload.get("global_packer_state") and self._global_packer:
            self._global_packer.load_state_dict(payload["global_packer_state"])
        if payload.get("replay_state") and self._replay:
            self._replay.load_state_dict(payload["replay_state"])
        self._critic_update_count = int(payload.get("critic_update_count", 0))
        self._actor_update_count = int(payload.get("actor_update_count", 0))
        self._target_update_count = int(payload.get("target_update_count", 0))
        rn = payload.get("reward_normalization_state", {})
        self._reward_norm_count = int(rn.get("count", 0))
        self._reward_norm_mean = float(rn.get("mean", 0.0))
        self._reward_norm_m2 = float(rn.get("m2", 0.0))
        es = payload.get("exploration_state", {})
        self._exploration_step = int(es.get("exploration_step", 0))
        self._teacher_alive = bool(es.get("teacher_alive", self._teacher_alive))
        self._bc_effective_weight = float(es.get("bc_effective_weight", 0.0))
        self._residual_scale = float(es.get("residual_scale", 0.0))
        rng = payload.get("rng_state")
        if rng is not None:
            torch.random.set_rng_state(rng["torch"])
            np.random.set_state(rng["numpy"])

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
        self._initialize_critic_infrastructure()

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
            "Topology change: {} v{} - n_ca={}",
            state.building_id, state.topology_version, new_layout.n_ca,
        )
        if self._replay is not None:
            self._replay.set_active_signature(self._current_topology_signature())

    def _initialize_critic_infrastructure(self) -> None:
        max_buildings = max(self._max_buildings, len(self._actors))
        self._online_critics = TwinTransformerCritics(
            d_model=self._critic_d_model,
            nhead=self._critic_nhead,
            num_layers=self._critic_num_layers,
            dim_feedforward=self._critic_dim_feedforward,
            dropout=self._critic_dropout,
            num_token_types=self._num_token_types,
            max_buildings=max_buildings,
        )
        self._target_critics = TwinTransformerCritics(
            d_model=self._critic_d_model,
            nhead=self._critic_nhead,
            num_layers=self._critic_num_layers,
            dim_feedforward=self._critic_dim_feedforward,
            dropout=self._critic_dropout,
            num_token_types=self._num_token_types,
            max_buildings=max_buildings,
        )
        self._target_critics.load_state_dict(self._online_critics.state_dict())
        self._critic_1 = self._online_critics.critic_1
        self._critic_2 = self._online_critics.critic_2
        self._target_critic_1 = self._target_critics.critic_1
        self._target_critic_2 = self._target_critics.critic_2
        self._critic_optimizer = torch.optim.Adam(
            self._online_critics.parameters(), lr=self._critic_lr
        )
        self._global_packer = GlobalTokenPacker(
            d_model=self._critic_d_model,
            num_token_types=self._num_token_types,
            max_buildings=max_buildings,
            action_input_mode=self._critic_action_input_mode,
        )
        self._replay = TopologyPartitionedReplay(
            capacity=self._replay_capacity,
            batch_size=self._batch_size,
        )
        self._replay.set_active_signature(self._current_topology_signature())

    def _current_topology_signature(self) -> str:
        return compute_topology_signature(
            building_ids=[s.building_id for s in self._actors],
            observation_names=[list(s.obs_names_tuple) for s in self._actors],
            action_names=[list(s.action_names_tuple) for s in self._actors],
            ca_action_names=[list(s.layout.ca_action_names) for s in self._actors],
            per_type_feature_dims=self._get_per_type_feature_dims(),
        )

    def _get_per_type_feature_dims(self) -> Dict[str, int]:
        dims: Dict[str, int] = {}
        for state in self._actors:
            for seg in state.layout.segments:
                if seg.family == "nfc":
                    continue
                dims.setdefault(seg.type_name, len(seg.feature_indices))
        return dims

    def _get_building_layouts(self) -> List[BuildingLayout]:
        return [
            BuildingLayout(
                building_index=i,
                n_sro=s.layout.n_sro,
                n_nfc=1,
                n_ca=s.layout.n_ca,
                is_controlled=s.layout.n_ca > 0,
            )
            for i, s in enumerate(self._actors)
        ]

    def _perform_critic_update(self) -> None:
        batch = self._replay.sample()
        if batch is None:
            return
        layouts = self._get_building_layouts()
        obs_tokens_current = self._tokenize_batch(batch.observations)
        obs_tokens_next = self._tokenize_batch(batch.next_observations)
        n_buildings = len(batch.observations)
        action_tensors = [
            torch.as_tensor(batch.actions[b], dtype=torch.float32)
            for b in range(n_buildings)
        ]
        packed_current = self._global_packer.pack(
            obs_tokens_current, action_tensors, layouts
        )
        packed_next = self._global_packer.pack(
            obs_tokens_next, action_tensors, layouts
        )
        controlled = [b for b, layout in enumerate(layouts) if layout.is_controlled]
        rewards_t = torch.stack(
            [torch.as_tensor(batch.rewards[b], dtype=torch.float32) for b in controlled],
            dim=1,
        )
        done_t = torch.as_tensor(batch.done, dtype=torch.float32).unsqueeze(1).expand_as(rewards_t)
        target_q = compute_target_q(
            target_critics=self._target_critics,
            packed_next_state=packed_next,
            rewards=rewards_t,
            done=done_t,
            gamma=self._gamma,
        )
        result = critic_update_step(
            online_critics=self._online_critics,
            optimizer=self._critic_optimizer,
            packed_current_state=packed_current,
            target_q=target_q,
        )
        self._last_q1_loss = result.critic_1_loss
        self._last_q2_loss = result.critic_2_loss
        self._last_target_q_mean = result.mean_target_q
        self._critic_update_count += 1

    def _perform_actor_update(self) -> None:
        for state in self._actors:
            if state.layout.n_ca < 1:
                continue
            for param in self._critic_1.parameters():
                param.requires_grad_(False)
            try:
                # Minimal deterministic actor update for Plan D integration:
                # move actor outputs toward zero while critic gradients are frozen.
                obs_dim = len(state.obs_names_tuple)
                obs_t = torch.zeros(self._batch_size, obs_dim, dtype=torch.float32)
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, _ = state.backbone(
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                    tokenized.ca_tokens,
                )
                actions = state.actor(ca_emb).squeeze(-1)
                loss = actions.pow(2).mean()
                state.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(state.actor.parameters(), 10.0)
                state.optimizer.step()
                self._last_actor_loss = float(loss.detach().item())
                self._last_actor_grad_norm = float(grad_norm)
            finally:
                for param in self._critic_1.parameters():
                    param.requires_grad_(True)

    @staticmethod
    def _soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(target.parameters(), online.parameters()):
                target_param.data.lerp_(online_param.data, tau)

    def _tokenize_batch(
        self, observations_per_building: List[npt.NDArray[np.float32]]
    ) -> List[torch.Tensor]:
        tokens: List[torch.Tensor] = []
        for b, state in enumerate(self._actors):
            obs_t = torch.as_tensor(observations_per_building[b], dtype=torch.float32)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                all_tokens = torch.cat(
                    [tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens], dim=1
                )
            tokens.append(all_tokens)
        return tokens

    def _reward_normalization_std(self) -> float:
        if self._reward_norm_count < 2:
            return 0.0
        return float(np.sqrt(self._reward_norm_m2 / max(self._reward_norm_count - 1, 1)))

    def _collect_diagnostics(self, global_learning_step: int) -> Dict[str, float]:
        """Collect all training diagnostics under TransformerMATD3/ namespace."""
        sig = self._current_topology_signature()
        critic_mode = self.config["algorithm"]["hyperparameters"].get(
            "critic_action_input_mode", "final"
        )
        metrics: Dict[str, float] = {
            "TransformerMATD3/replay_size": float(self._replay.total_size),
            "TransformerMATD3/active_partition_size": float(self._replay.partition_size(sig)),
            "TransformerMATD3/partition_count": float(self._replay.partition_count),
            "TransformerMATD3/critic_q1_loss": float(getattr(self, "_last_q1_loss", 0.0)),
            "TransformerMATD3/critic_q2_loss": float(getattr(self, "_last_q2_loss", 0.0)),
            "TransformerMATD3/critic_q_gap": float(getattr(self, "_last_q_gap", 0.0)),
            "TransformerMATD3/target_q_mean": float(getattr(self, "_last_target_q_mean", 0.0)),
            "TransformerMATD3/target_q_std": float(getattr(self, "_last_target_q_std", 0.0)),
            "TransformerMATD3/actor_loss": float(getattr(self, "_last_actor_loss", 0.0)),
            "TransformerMATD3/actor_grad_norm": float(getattr(self, "_last_actor_grad_norm", 0.0)),
            "TransformerMATD3/teacher_alive": float(self._teacher_alive),
            "TransformerMATD3/residual_scale": float(getattr(self, "_residual_scale", 0.0)),
            "TransformerMATD3/phaseout_probability": float(getattr(self, "_last_phaseout_probability", 0.0)),
            "TransformerMATD3/bc_loss": float(getattr(self, "_last_bc_loss", 0.0)),
            "TransformerMATD3/bc_effective_weight": float(getattr(self, "_bc_effective_weight", 0.0)),
            "TransformerMATD3/reward_norm_mean": float(getattr(self, "_reward_norm_mean", 0.0)),
            "TransformerMATD3/reward_norm_std": float(
                self._reward_normalization_std() if self._reward_normalization_enabled else 0.0
            ),
            "TransformerMATD3/reward_norm_count": float(getattr(self, "_reward_norm_count", 0)),
            "TransformerMATD3/critic_action_input_mode_final": float(critic_mode == "final"),
            "TransformerMATD3/critic_action_input_mode_delta": float(
                critic_mode in ("final_base_delta", "final_base_delta_normalized")
            ),
            "TransformerMATD3/critic_update_count": float(self._critic_update_count),
            "TransformerMATD3/actor_update_count": float(self._actor_update_count),
            "TransformerMATD3/global_learning_step": float(global_learning_step),
        }
        return metrics

    def _should_log_training_step(self, global_learning_step: int) -> bool:
        interval = self.config["algorithm"]["hyperparameters"].get("log_interval", 10)
        return global_learning_step % max(int(interval), 1) == 0

    def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        del step
        if metrics:
            self._latest_training_metrics = dict(metrics)

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
