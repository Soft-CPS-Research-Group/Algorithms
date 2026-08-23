from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import _log_torch_runtime, _select_torch_device
from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
)
from algorithms.utils.price_multiplier_adapter import (
    PriceMultiplierObservationAdapter,
    normalize_price_multiplier_contexts,
    price_feature_bounds_from_metadata,
    price_observation_names_from_metadata,
)
from algorithms.transformer_matd3.components import (
    CentralizedCritic,
    DeterministicActorHead,
)
from algorithms.transformer_matd3 import behavior_cloning as matd3_bc
from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer
from algorithms.transformer_matd3.types import LayoutSignature, ReplayBatch
from algorithms.transformer_shared.behavior_cloning import (
    BehaviorCloningRegularizer,
    Demonstration,
)
from algorithms.transformer_shared.entity_observation_tokenizer import (
    EntityObservationTokenizer,
)
from algorithms.transformer_shared.entity_token_layout import (
    BuildingTokenLayout,
    EntityTokenLayoutBuilder,
)
from algorithms.transformer_shared.transformer_backbone import TransformerBackbone
from utils.entity_tokenizer_schema import (
    EntityTokenizerConfig,
    load_entity_tokenizer_config,
)


_METRIC_PREFIX = "TransformerMATD3/"


@dataclass
class _PerBuildingState:
    building_id: str
    tokenizer: EntityObservationTokenizer
    backbone: TransformerBackbone
    actor: DeterministicActorHead
    tokenizer_target: EntityObservationTokenizer
    backbone_target: TransformerBackbone
    actor_target: DeterministicActorHead
    critic_1: CentralizedCritic
    critic_2: CentralizedCritic
    critic_1_target: CentralizedCritic
    critic_2_target: CentralizedCritic
    actor_optimizer: torch.optim.Optimizer
    critic_1_optimizer: torch.optim.Optimizer
    critic_2_optimizer: torch.optim.Optimizer
    bc_a_optimizer: Optional[torch.optim.Optimizer]
    bc_b_optimizer: Optional[torch.optim.Optimizer]
    layout: BuildingTokenLayout
    action_names: Tuple[str, ...]
    action_low: torch.Tensor
    action_high: torch.Tensor
    topology_version: int = 0


@dataclass
class _TopologyStateSnapshot:
    agent_state: Dict[str, Any]
    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_rng_state: torch.Tensor
    cuda_rng_state: Optional[List[torch.Tensor]]


class AgentTransformerMATD3(BaseAgent):
    """Transformer MATD3 learner with transactional topology adaptation."""

    supports_dynamic_topology: ClassVar[bool] = True
    requires_final_pipeline_stage: ClassVar[bool] = True
    checkpoint_version: ClassVar[int] = 5
    onnx_opset_version: ClassVar[int] = 17

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_mode = str(
            (config.get("checkpointing") or {}).get("checkpoint_mode", "full")
            or "full"
        ).strip().lower()
        if self.checkpoint_mode not in {"full", "inference"}:
            raise ValueError(
                "Transformer MATD3 checkpoint_mode must be 'full' or 'inference'"
            )
        algorithm = config["algorithm"]
        self._tokenizer_config_path = str(algorithm["tokenizer_config_path"])
        self._tokenizer_config: EntityTokenizerConfig = load_entity_tokenizer_config(
            self._tokenizer_config_path
        )
        transformer = dict(algorithm["transformer"])
        self._d_model = int(transformer["d_model"])
        self._nhead = int(transformer["nhead"])
        self._num_layers = int(transformer["num_layers"])
        self._dim_feedforward = int(transformer.get("dim_feedforward", 256))
        self._dropout = float(transformer.get("dropout", 0.0))

        hyperparameters = dict(algorithm["hyperparameters"])
        self.require_cuda = bool(hyperparameters.get("require_cuda", False))
        self.device = _select_torch_device(
            require_cuda=self.require_cuda,
            algorithm_name="AgentTransformerMATD3",
        )
        _log_torch_runtime(self.device)
        self.learning_rate = float(hyperparameters["learning_rate"])
        self.gamma = float(hyperparameters["gamma"])
        self.tau = float(hyperparameters["tau"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.buffer_capacity = int(hyperparameters["buffer_capacity"])
        self.max_grad_norm = float(hyperparameters.get("max_grad_norm", 1.0))
        self.actor_hidden_dim = int(
            hyperparameters.get("actor_hidden_dim", max(32, self._d_model * 2))
        )
        self.critic_hidden_dim = int(
            hyperparameters.get("critic_hidden_dim", max(32, self._d_model * 2))
        )
        self.n_step_returns = int(hyperparameters.get("n_step_returns", 1))
        self.n_step_gamma = float(
            hyperparameters.get("n_step_gamma", self.gamma) or self.gamma
        )
        self.critic_team_reward_mix = float(
            hyperparameters.get("critic_team_reward_mix", 0.0)
        )
        self.critic_target_clip_abs = float(
            hyperparameters.get("critic_target_clip_abs", 0.0)
        )
        self.reward_normalization_enabled = bool(
            hyperparameters.get("reward_normalization_enabled", False)
        )
        self.reward_normalization_clip = float(
            hyperparameters.get("reward_normalization_clip", 10.0)
        )
        self.target_policy_smoothing = bool(
            hyperparameters.get("target_policy_smoothing", True)
        )
        self.target_policy_noise = float(
            hyperparameters.get("target_policy_noise", 0.0)
        )
        self.target_policy_noise_clip = float(
            hyperparameters.get("target_policy_noise_clip", 0.0)
        )
        self.actor_update_interval = int(
            hyperparameters.get("actor_update_interval", 2)
        )
        self.sigma = float(hyperparameters.get("sigma", 0.0))
        self.sigma_decay = float(hyperparameters.get("sigma_decay", 1.0))
        self.min_sigma = float(hyperparameters.get("min_sigma", 0.0))
        self.bias = float(hyperparameters.get("bias", 0.0))
        raw_noise_clip = hyperparameters.get("noise_clip")
        self.noise_clip = (
            None if raw_noise_clip is None else float(raw_noise_clip)
        )
        self.random_exploration_steps = int(
            hyperparameters.get("random_exploration_steps", 0)
        )
        self.end_initial_exploration_time_step = int(
            hyperparameters.get("end_initial_exploration_time_step", 0)
        )
        self.storage_exploration_noise_multiplier = float(
            hyperparameters.get("storage_exploration_noise_multiplier", 1.0)
        )
        self.ev_negative_exploration_noise_multiplier = float(
            hyperparameters.get("ev_negative_exploration_noise_multiplier", 1.0)
        )
        self.deferrable_trigger_threshold = float(
            hyperparameters.get("deferrable_trigger_threshold", 0.0)
        )
        self.deferrable_on_probability = float(
            hyperparameters.get("deferrable_on_probability", 0.0)
        )
        self.residual_policy_enabled = bool(
            hyperparameters.get("residual_policy_enabled", False)
        )
        self.residual_policy_runtime_only_export = bool(
            hyperparameters.get("residual_policy_runtime_only_export", False)
        )
        self.warm_start_policy_name = self._optional_string(
            hyperparameters.get("warm_start_policy_name")
        )
        self.warm_start_policy_hyperparameters = dict(
            hyperparameters.get("warm_start_policy_hyperparameters") or {}
        )
        self.residual_action_scale = float(
            hyperparameters.get("residual_action_scale", 1.0)
        )
        self.residual_action_final_scale = float(
            hyperparameters.get(
                "residual_action_final_scale", self.residual_action_scale
            )
        )
        self.residual_action_scale_start_step = int(
            hyperparameters.get("residual_action_scale_start_step", 0)
        )
        self.residual_action_scale_growth_steps = int(
            hyperparameters.get("residual_action_scale_growth_steps", 0)
        )
        self.residual_storage_action_scale_multiplier = float(
            hyperparameters.get("residual_storage_action_scale_multiplier", 1.0)
        )
        self.residual_ev_action_scale_multiplier = float(
            hyperparameters.get("residual_ev_action_scale_multiplier", 1.0)
        )
        self.residual_deferrable_action_scale_multiplier = float(
            hyperparameters.get("residual_deferrable_action_scale_multiplier", 1.0)
        )
        self.critic_action_input_mode = str(
            hyperparameters.get("critic_action_input_mode", "final")
        )
        self._local_action_safety_enabled = bool(
            hyperparameters.get("local_action_safety_enabled", False)
        )
        self._local_action_safety_runtime_only_export = bool(
            hyperparameters.get("local_action_safety_runtime_only_export", False)
        )
        self._local_action_safety_config = CityLearnSafetyConfig(
            fail_on_infeasible=bool(
                hyperparameters.get("local_action_safety_fail_on_infeasible", False)
            ),
            protect_ev_minimum=bool(
                hyperparameters.get("local_action_safety_protect_ev_minimum", True)
            ),
            ev_minimum_mode=str(
                hyperparameters.get("local_action_safety_ev_minimum_mode", "average")
            ),
            protect_ev_service_target=bool(
                hyperparameters.get(
                    "local_action_safety_protect_ev_service_target", False
                )
            ),
            allow_ev_service_target_to_use_reserved_headroom=bool(
                hyperparameters.get(
                    "local_action_safety_allow_ev_service_target_to_use_reserved_headroom",
                    False,
                )
            ),
            protect_deferrable_must_start=bool(
                hyperparameters.get(
                    "local_action_safety_protect_deferrable_must_start", True
                )
            ),
            allow_discretionary_deferrable_start=bool(
                hyperparameters.get(
                    "local_action_safety_allow_discretionary_deferrable_start", False
                )
            ),
            headroom_reserve_kw=float(
                hyperparameters.get("local_action_safety_headroom_reserve_kw", 0.0)
            ),
        )
        self._local_price_conditioning_enabled = bool(
            hyperparameters.get("local_price_conditioning_enabled", False)
        )
        self._local_price_conditioning_runtime_only_export = bool(
            hyperparameters.get(
                "local_price_conditioning_runtime_only_export", False
            )
        )
        self._local_price_forecast_mode = str(
            hyperparameters.get("local_price_forecast_mode", "real_unmodified")
        )
        replay_bc = dict(
            (algorithm.get("behavior_cloning") or {}).get("replay_based") or {}
        )
        self.bc_a_enabled = bool(replay_bc.get("enabled", False))
        self.bc_a_teacher = str(replay_bc.get("teacher", "warm_start"))
        self.bc_a_weight = float(replay_bc.get("weight", 0.0))
        self.bc_a_min_weight = float(replay_bc.get("min_weight", 0.0))
        self.bc_a_decay_start_step = int(replay_bc.get("decay_start_step", 0))
        self.bc_a_decay_steps = int(replay_bc.get("decay_steps", 0))
        self.bc_a_ev_multiplier = float(replay_bc.get("ev_multiplier", 1.0))
        self.bc_a_storage_multiplier = float(
            replay_bc.get("storage_multiplier", 1.0)
        )
        self.bc_a_deferrable_multiplier = float(
            replay_bc.get("deferrable_multiplier", 1.0)
        )
        has_extra_window = bool(
            int(replay_bc.get("extra_update_start_step", 0))
            or int(replay_bc.get("extra_update_end_step", 0))
        )
        self.bc_a_extra_updates = int(
            replay_bc.get("extra_updates", 1 if has_extra_window else 0)
        )
        self.bc_a_extra_update_start_step = int(
            replay_bc.get("extra_update_start_step", 0)
        )
        self.bc_a_extra_update_end_step = int(
            replay_bc.get("extra_update_end_step", 0)
        )
        self.bc_a_clip_target_to_residual_authority = bool(
            replay_bc.get("clip_target_to_residual_authority", False)
        )
        self.bc_a_offline_pretrain_steps = int(
            replay_bc.get("offline_pretrain_steps", 0)
        )
        self.bc_a_offline_pretrain_completed_steps = 0
        demonstration_bc = dict(
            (algorithm.get("behavior_cloning") or {}).get("demonstration_based")
            or {}
        )
        self._bc_b = (
            BehaviorCloningRegularizer.from_config(
                {"behavior_cloning": demonstration_bc},
                self.config,
            )
            if bool(demonstration_bc.get("enabled", False))
            else None
        )
        self._validate_hyperparameters()

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._per_building: List[_PerBuildingState] = []
        self._layout_signature: Optional[LayoutSignature] = None
        self.replay_buffer: Optional[SignatureBucketedReplayBuffer] = None
        self._attached_names: Optional[
            Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...]
        ] = None
        self._n_step_queue: deque[Dict[str, Any]] = deque()
        self.exploration_sigma = self.sigma
        self.exploration_step = 0
        self.reward_norm_count = 0
        self.reward_norm_mean = 0.0
        self.reward_norm_m2 = 0.0
        self._latest_training_metrics: Dict[str, float] = {}
        tracking_cfg = config.get("tracking", {}) if isinstance(config, dict) else {}
        self.runtime_profiling_enabled = bool(
            tracking_cfg.get("runtime_profiling_enabled", False)
        )
        try:
            self.runtime_profiling_interval = int(
                tracking_cfg.get("runtime_profiling_interval", 512) or 512
            )
        except (TypeError, ValueError):
            self.runtime_profiling_interval = 512
        if self.runtime_profiling_interval < 1:
            self.runtime_profiling_interval = 512
        self._last_train_rewards: Optional[torch.Tensor] = None
        self._warm_start_policy: Optional[BaseAgent] = None
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
        self._latest_conditioned_observations: Optional[List[np.ndarray]] = None
        self._latest_conditioned_next_observations: Optional[List[np.ndarray]] = None
        self._latest_price_contexts: Optional[List[Mapping[str, Any] | None]] = None
        self._transition_conditioned_observations: Optional[List[np.ndarray]] = None
        self._transition_conditioned_next_observations: Optional[List[np.ndarray]] = None
        self._last_warm_start_policy_actions: Optional[List[List[float]]] = None
        self._last_warm_start_next_policy_actions: Optional[List[List[float]]] = None
        self._latest_external_cloning_actions: Optional[List[np.ndarray]] = None
        self._local_action_safety_adapters: List[
            CityLearnLocalSafetyAdapter
        ] = []
        self._local_action_safety_projection_count = 0
        self._local_action_safety_intervention_count = 0
        self._local_action_safety_infeasible_count = 0
        self._local_action_safety_reason_counts: Dict[str, int] = {}
        self._local_price_adapters: List[PriceMultiplierObservationAdapter] = []
        self._local_price_context_non_neutral = False
        self._local_price_clipping_count = 0
        self._last_residual_action_scale = 0.0
        self._current_episode = 0
        self._current_episode_is_training = False
        self._bc_b_pretraining_complete = False
        self._bc_b_actor_training_step = 0
        self.requires_raw_observation_context = bool(
            self.residual_policy_enabled
            or self._local_action_safety_enabled
            or (self.bc_a_enabled and self.bc_a_teacher == "warm_start")
            or self._bc_b is not None
        )

    def attach_environment(  # type: ignore[override]
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        count = len(observation_names)
        if count == 0:
            raise ValueError("Transformer MATD3 requires at least one building")
        if len(action_names) != count:
            raise ValueError("observation_names and action_names counts must match")
        spaces = self._normalize_spaces(action_space, count)
        observation_spaces = self._normalize_spaces(
            observation_space, count, name="observation_space"
        )
        names_key = tuple(
            (tuple(observation), tuple(actions))
            for observation, actions in zip(observation_names, action_names)
        )
        snapshot = self.snapshot_topology_state()
        try:
            building_names = (metadata or {}).get("building_names") or ()
            layouts = [
                self._layout_builder.build(
                    str(building_names[index])
                    if index < len(building_names) and building_names[index]
                    else (
                        self._per_building[index].building_id
                        if index < len(self._per_building)
                        else f"building_{index}"
                    ),
                    observation_names[index],
                    action_names[index],
                )
                for index in range(count)
            ]
            for index, (layout, names) in enumerate(zip(layouts, action_names)):
                self._validate_ca_order(index, layout, names)
            type_input_dims = self._community_type_input_dims(layouts)
            bounds = [
                self._action_bounds(index, names, space)
                for index, (names, space) in enumerate(zip(action_names, spaces))
            ]
            candidate_signature = self._build_layout_signature(layouts)
            was_attached = self._attached_names is not None
            if (
                was_attached
                and count == len(self._per_building)
                and candidate_signature == self._layout_signature
                and all(
                    torch.equal(state.action_low, low)
                    and torch.equal(state.action_high, high)
                    for state, (low, high) in zip(self._per_building, bounds)
                )
            ):
                return
            if self._attached_names is None:
                self._install_fresh_environment(
                    layouts=layouts,
                    action_names=action_names,
                    bounds=bounds,
                    type_input_dims=type_input_dims,
                )
            elif count != len(self._per_building):
                self._reset_full_for_building_count_change(
                    layouts=layouts,
                    action_names=action_names,
                    bounds=bounds,
                    type_input_dims=type_input_dims,
                )
            else:
                self._adapt_compatible_topology(
                    layouts=layouts,
                    action_names=action_names,
                    bounds=bounds,
                    type_input_dims=type_input_dims,
                    candidate_signature=candidate_signature,
                )
            self._attached_names = names_key
            self._layout_signature = candidate_signature
            self._attach_warm_start_policy(
                observation_names=observation_names,
                action_names=action_names,
                action_space=spaces,
                observation_space=observation_spaces,
                metadata=metadata,
            )
            self._attach_local_action_safety(
                observation_names=observation_names,
                action_names=action_names,
                metadata=metadata,
            )
            self._attach_local_price_conditioning(
                observation_names=observation_names,
                metadata=metadata,
            )
            self._attach_bc_b_environment(
                observation_names=observation_names,
                action_names=action_names,
                action_space=spaces,
                observation_space=observation_spaces,
                metadata=metadata,
                topology_change=was_attached,
            )
        except Exception:
            self.restore_topology_state(snapshot)
            raise

    def _install_fresh_environment(
        self,
        *,
        layouts: Sequence[BuildingTokenLayout],
        action_names: Sequence[Sequence[str]],
        bounds: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        type_input_dims: Mapping[str, int],
    ) -> None:
        self._per_building = [
            self._build_state(
                layout=layout,
                action_names=tuple(names),
                action_low=low,
                action_high=high,
                type_input_dims=type_input_dims,
            )
            for layout, names, (low, high) in zip(layouts, action_names, bounds)
        ]
        self.replay_buffer = SignatureBucketedReplayBuffer(
            capacity=self.buffer_capacity,
            num_agents=len(layouts),
            batch_size=self.batch_size,
        )

    def _adapt_compatible_topology(
        self,
        *,
        layouts: Sequence[BuildingTokenLayout],
        action_names: Sequence[Sequence[str]],
        bounds: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        type_input_dims: Mapping[str, int],
        candidate_signature: LayoutSignature,
    ) -> None:
        self._validate_compatible_layout_signature(candidate_signature)
        for index, (state, layout) in enumerate(zip(self._per_building, layouts)):
            if layout.building_id != state.building_id:
                raise ValueError(
                    f"topology change building {index} changed identity from "
                    f"{state.building_id!r} to {layout.building_id!r}"
                )
            for type_name, width in type_input_dims.items():
                if type_name not in state.tokenizer.projections:
                    raise ValueError(
                        f"topology change introduced unsupported type {type_name!r}"
                    )
                projection = state.tokenizer.projections[type_name]
                if int(projection.in_features) != int(width):
                    raise ValueError(
                        f"topology change feature width for type {type_name!r} "
                        f"changed {projection.in_features} -> {width}"
                    )
        self._flush_n_step_topology_boundary()
        for index, (state, layout, names, (low, high)) in enumerate(
            zip(self._per_building, layouts, action_names, bounds)
        ):
            changed = (
                candidate_signature[index] != self._layout_signature[index]
                or not torch.equal(state.action_low, low)
                or not torch.equal(state.action_high, high)
            )
            state.layout = layout
            state.action_names = tuple(names)
            state.action_low = low
            state.action_high = high
            if changed:
                state.topology_version += 1

    def _validate_compatible_layout_signature(
        self,
        candidate_signature: LayoutSignature,
    ) -> None:
        """Reject schema drift while allowing new controllable assets."""
        assert self._layout_signature is not None
        for building_index, (previous, candidate) in enumerate(
            zip(self._layout_signature, candidate_signature)
        ):
            previous_segments = previous[3]
            candidate_segments = candidate[3]
            previous_keys = [
                (segment[0], segment[1], segment[2])
                for segment in previous_segments
            ]
            previous_key_set = set(previous_keys)
            candidate_existing_keys = [
                (segment[0], segment[1], segment[2])
                for segment in candidate_segments
                if (segment[0], segment[1], segment[2]) in previous_key_set
            ]
            if candidate_existing_keys != previous_keys:
                raise ValueError(
                    "topology schema drift: ordered segments changed for "
                    f"building {building_index}"
                )
            candidate_by_key = {
                (segment[0], segment[1], segment[2]): segment
                for segment in candidate_segments
            }
            for segment in previous_segments:
                key = (segment[0], segment[1], segment[2])
                if candidate_by_key.get(key) != segment:
                    candidate_segment = candidate_by_key.get(key)
                    if (
                        candidate_segment is not None
                        and len(candidate_segment[3]) != len(segment[3])
                    ):
                        raise ValueError(
                            "topology schema drift: feature width changed for "
                            f"building {building_index}, segment {key!r}"
                        )
                    raise ValueError(
                        "topology schema drift: segment feature names or NFC "
                        f"expression changed for building {building_index}, segment {key!r}"
                    )

    def _reset_full_for_building_count_change(
        self,
        *,
        layouts: Sequence[BuildingTokenLayout],
        action_names: Sequence[Sequence[str]],
        bounds: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        type_input_dims: Mapping[str, int],
    ) -> None:
        self._flush_n_step_topology_boundary()
        demonstration = dict(
            (self.config["algorithm"].get("behavior_cloning") or {}).get(
                "demonstration_based"
            )
            or {}
        )
        self._bc_b = (
            BehaviorCloningRegularizer.from_config(
                {"behavior_cloning": demonstration}, self.config
            )
            if bool(demonstration.get("enabled", False))
            else None
        )
        self._install_fresh_environment(
            layouts=layouts,
            action_names=action_names,
            bounds=bounds,
            type_input_dims=type_input_dims,
        )
        self._n_step_queue.clear()
        self.exploration_sigma = self.sigma
        self.exploration_step = 0
        self.reward_norm_count = 0
        self.reward_norm_mean = 0.0
        self.reward_norm_m2 = 0.0
        self.bc_a_offline_pretrain_completed_steps = 0
        self._bc_b_pretraining_complete = False
        self._bc_b_actor_training_step = 0
        self._latest_training_metrics = {}
        self._last_train_rewards = None
        self._latest_raw_observations = None
        self._latest_raw_next_observations = None
        self._last_warm_start_policy_actions = None
        self._last_warm_start_next_policy_actions = None
        self._latest_external_cloning_actions = None

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[npt.NDArray[np.float64]]] = None,
        encoded_observations: Optional[List[npt.NDArray[np.float64]]] = None,
    ) -> None:
        del encoded_observations
        if raw_observations is not None:
            self._validate_vector_count("raw_observations", raw_observations)
        self._latest_raw_observations = self._copied_optional_vectors(
            raw_observations
        )
        self._last_warm_start_policy_actions = None

    def set_transition_context(
        self,
        *,
        raw_observations: Optional[List[npt.NDArray[np.float64]]] = None,
        raw_next_observations: Optional[List[npt.NDArray[np.float64]]] = None,
        encoded_observations: Optional[List[npt.NDArray[np.float64]]] = None,
        encoded_next_observations: Optional[
            List[npt.NDArray[np.float64]]
        ] = None,
        cloning_actions: Optional[List[npt.NDArray[np.float64]]] = None,
        price_context: Any = None,
        next_price_context: Any = None,
    ) -> None:
        if raw_observations is not None:
            self._validate_vector_count("raw_observations", raw_observations)
            self._latest_raw_observations = self._copied_optional_vectors(
                raw_observations
            )
        if raw_next_observations is not None:
            self._validate_vector_count(
                "raw_next_observations", raw_next_observations
            )
        self._latest_raw_next_observations = self._copied_optional_vectors(
            raw_next_observations
        )
        self._last_warm_start_next_policy_actions = (
            self._predict_warm_start_policy_for_observations(
                self._latest_raw_next_observations
            )
        )
        self._latest_external_cloning_actions = self._copied_optional_vectors(
            cloning_actions
        )
        if encoded_observations is not None:
            if price_context is None and self._latest_conditioned_observations is not None:
                self._transition_conditioned_observations = self._copied_vectors(
                    self._latest_conditioned_observations
                )
            else:
                self._transition_conditioned_observations = self._apply_local_price_context(
                    [np.asarray(value, dtype=np.float64) for value in encoded_observations],
                    price_context,
                )
        else:
            self._transition_conditioned_observations = None
        if encoded_next_observations is not None:
            successor_context = (
                next_price_context
                if next_price_context is not None
                else price_context
                if price_context is not None
                else self._latest_price_contexts
            )
            self._transition_conditioned_next_observations = self._apply_local_price_context(
                [
                    np.asarray(value, dtype=np.float64)
                    for value in encoded_next_observations
                ],
                successor_context,
            )
        else:
            self._transition_conditioned_next_observations = None

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        self._require_attached()
        self._validate_vector_count("predict observations", observations)
        if self._in_bc_b_demonstration_phase():
            assert self._bc_b is not None
            teacher_observations = (
                self._latest_raw_observations
                if self._latest_raw_observations is not None
                else observations
            )
            teacher_actions = self._bc_b.compute_teacher_actions(
                teacher_observations
            )
            return [
                self._apply_local_action_safety(index, action)
                for index, action in enumerate(teacher_actions)
            ]
        observations = self._apply_local_price_context(observations, context)
        self._latest_conditioned_observations = self._copied_vectors(observations)
        self._latest_price_contexts = (
            normalize_price_multiplier_contexts(
                context,
                num_agents=len(self._per_building),
            )
            if self._local_price_conditioning_enabled
            else [None for _ in self._per_building]
        )
        use_deterministic = bool(deterministic)
        base_actions = self._predict_warm_start_policy_for_observations(
            self._latest_raw_observations
        )
        if self.residual_policy_enabled and base_actions is None:
            raise RuntimeError(
                "Transformer MATD3 residual policy requires raw observation "
                "context before predict"
            )
        self._last_warm_start_policy_actions = deepcopy(base_actions)
        result: List[List[float]] = []
        for index, (state, observation) in enumerate(
            zip(self._per_building, observations)
        ):
            observation_tensor = self._tensor(observation).unsqueeze(0)
            actor_modules = self._actor_modules(state)
            prior_modes = [module.training for module in actor_modules]
            actor_modules.eval()
            try:
                with torch.no_grad():
                    unit_action = self._actor_unit_action(
                        state, observation_tensor, target=False
                    )
                    if not use_deterministic:
                        unit_action = self._explore_unit_action(
                            unit_action, state, index
                        )
                    action = self._compose_policy_action(
                        index,
                        unit_action,
                        base_action=(
                            None
                            if base_actions is None
                            else self._tensor(base_actions[index]).unsqueeze(0)
                        ),
                    )
            finally:
                for module, training in zip(actor_modules, prior_modes):
                    module.train(training)
            executed = action.squeeze(0).cpu().tolist()
            executed = self._apply_local_action_safety(index, executed)
            result.append(executed)
        if not use_deterministic:
            self.exploration_step += 1
            self.exploration_sigma = max(
                self.min_sigma,
                self.exploration_sigma * self.sigma_decay,
            )
        return result

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
        self._require_attached()
        should_profile = self._should_runtime_profile_step(global_learning_step)
        profile_metrics: Dict[str, float] = {}
        profile_start = time.perf_counter() if should_profile else 0.0
        for name, values in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
            ("next_observations", next_observations),
        ):
            self._validate_vector_count(name, values)
        if self._in_bc_b_demonstration_phase():
            self._record_bc_b_demonstrations(observations, actions)
            self._latest_training_metrics = {
                f"{_METRIC_PREFIX}episode_training": 1.0,
                f"{_METRIC_PREFIX}teacher_action_execution": 1.0,
                **self._bc_b_metrics(),
            }
            return
        if should_profile:
            profile_metrics[f"{_METRIC_PREFIX}runtime_update_prepare_seconds"] = (
                time.perf_counter() - profile_start
            )
            profile_start = time.perf_counter()
        if self._bc_b is not None and not self._bc_b_pretraining_complete:
            self._run_bc_b_pretraining()
            self._bc_b_pretraining_complete = True
            self._bc_b_actor_training_step = 0
        self._update_reward_normalizer(rewards)
        behavior_actions = self._transition_behavior_actions(actions)
        next_behavior_actions = self._transition_next_behavior_actions(
            behavior_actions
        )
        cloning_actions = self._transition_cloning_actions(
            actions,
            base_actions=behavior_actions,
        )
        conditioned_observations = self._transition_conditioned_observations
        if conditioned_observations is None and self._latest_conditioned_observations is not None:
            conditioned_observations = self._latest_conditioned_observations
        conditioned_next_observations = self._transition_conditioned_next_observations
        if conditioned_next_observations is None:
            conditioned_next_observations = self._apply_local_price_context(
                [np.asarray(value, dtype=np.float64) for value in next_observations],
                self._latest_price_contexts,
            )
        transition = {
            "observations": self._copied_vectors(
                conditioned_observations
                if conditioned_observations is not None
                else observations
            ),
            "actions": self._copied_vectors(actions),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1).copy(),
            "next_observations": self._copied_vectors(conditioned_next_observations),
            "terminated": self._done_vector(terminated),
            "truncated": self._done_vector(truncated),
            "behavior_actions": self._optional_replay_actions(behavior_actions),
            "next_behavior_actions": self._optional_replay_actions(
                next_behavior_actions
            ),
            "cloning_actions": self._distinct_cloning_actions(
                cloning_actions,
                behavior_actions,
            ),
            "layout_signature": self._layout_signature,
        }
        self._validate_transition_vectors(transition)
        self._store_transition(transition)
        if should_profile:
            profile_metrics[f"{_METRIC_PREFIX}runtime_replay_push_seconds"] = (
                time.perf_counter() - profile_start
            )
            profile_start = time.perf_counter()
        if self._bc_b is not None and self._bc_b_pretraining_complete:
            self._bc_b_actor_training_step += 1

        assert self.replay_buffer is not None
        assert self._layout_signature is not None
        bucket_size = self.replay_buffer.bucket_size(self._layout_signature)
        if bucket_size < self.batch_size:
            self._record_skip("replay_underfull", bucket_size)
            if should_profile:
                profile_metrics[f"{_METRIC_PREFIX}runtime_update_skip_replay_warmup"] = 1.0
                self._latest_training_metrics.update(profile_metrics)
            return
        if not initial_exploration_done:
            self._record_skip("initial_exploration", bucket_size)
            if should_profile:
                profile_metrics[f"{_METRIC_PREFIX}runtime_update_skip_initial_exploration"] = 1.0
                self._latest_training_metrics.update(profile_metrics)
            return
        if not update_step:
            self._record_skip("schedule", bucket_size)
            if should_profile:
                profile_metrics[f"{_METRIC_PREFIX}runtime_update_skip_schedule"] = 1.0
                self._latest_training_metrics.update(profile_metrics)
            return
        if should_profile:
            profile_start = time.perf_counter()
        batch = self.replay_buffer.sample(self._layout_signature, self.batch_size)
        if should_profile:
            profile_metrics[f"{_METRIC_PREFIX}runtime_replay_sample_seconds"] = (
                time.perf_counter() - profile_start
            )
        self._learn(
            batch,
            update_target_step=update_target_step,
            global_learning_step=global_learning_step,
            runtime_profile_metrics=profile_metrics if should_profile else None,
        )

    def on_episode_start(self, *, episode: int, training: bool) -> None:
        self._current_episode = int(episode)
        self._current_episode_is_training = bool(training)
        if (
            training
            and self._bc_b is not None
            and not self._bc_b_pretraining_complete
            and episode >= self._bc_b.demonstration_episodes
        ):
            self._run_bc_b_pretraining()
            self._bc_b_pretraining_complete = True
            self._bc_b_actor_training_step = 0

    def on_episode_end(self, *, episode: int, training: bool) -> None:
        self._current_episode = int(episode)
        if not training or self._bc_b is None:
            return
        if (
            self._in_bc_b_demonstration_phase()
            and episode + 1 == self._bc_b.demonstration_episodes
        ):
            self._run_bc_b_pretraining()
            self._bc_b_pretraining_complete = True
            self._bc_b_actor_training_step = 0

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
        del truncated, global_learning_step
        self._require_attached()
        for name, values in (
            ("observations", observations),
            ("actions", actions),
            ("rewards", rewards),
        ):
            self._validate_vector_count(name, values)
        if self._in_bc_b_demonstration_phase():
            self._record_bc_b_demonstrations(observations, actions)
            return
        if self._bc_b is not None and not self._bc_b_pretraining_complete:
            self._run_bc_b_pretraining()
            self._bc_b_pretraining_complete = True
            self._bc_b_actor_training_step = 0
        self._update_reward_normalizer(rewards)
        behavior_actions = self._transition_behavior_actions(actions)
        cloning_actions = self._transition_cloning_actions(
            actions, base_actions=behavior_actions
        )
        transition = {
            "observations": self._copied_vectors(observations),
            "actions": self._copied_vectors(actions),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1).copy(),
            # A topology boundary has no shape-compatible successor. The old
            # observation is a storage placeholder and cannot bootstrap.
            "next_observations": self._copied_vectors(observations),
            "terminated": self._done_vector(terminated),
            "truncated": np.ones(len(self._per_building), dtype=np.bool_),
            "behavior_actions": self._optional_replay_actions(behavior_actions),
            "next_behavior_actions": self._optional_replay_actions(
                behavior_actions
            ),
            "cloning_actions": self._distinct_cloning_actions(
                cloning_actions, behavior_actions
            ),
            "layout_signature": self._layout_signature,
        }
        self._validate_transition_vectors(transition)
        self._store_transition(transition)

    def snapshot_topology_state(self) -> _TopologyStateSnapshot:
        return _TopologyStateSnapshot(
            agent_state=deepcopy(self.__dict__),
            python_rng_state=random.getstate(),
            numpy_rng_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=self._capture_cuda_rng_state(),
        )

    def restore_topology_state(self, snapshot: _TopologyStateSnapshot) -> None:
        if not isinstance(snapshot, _TopologyStateSnapshot):
            raise TypeError("invalid Transformer MATD3 topology snapshot")
        restored = deepcopy(snapshot.agent_state)
        self.__dict__.clear()
        self.__dict__.update(restored)
        random.setstate(snapshot.python_rng_state)
        np.random.set_state(snapshot.numpy_rng_state)
        torch.set_rng_state(snapshot.torch_rng_state)
        self._restore_cuda_rng_state(snapshot.cuda_rng_state)

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del context
        guard_requirements = (
            (
                self.residual_policy_enabled,
                self.residual_policy_runtime_only_export,
                "residual_policy_runtime_only_export",
            ),
            (
                self._local_action_safety_enabled,
                self._local_action_safety_runtime_only_export,
                "local_action_safety_runtime_only_export",
            ),
            (
                self._local_price_conditioning_enabled,
                self._local_price_conditioning_runtime_only_export,
                "local_price_conditioning_runtime_only_export",
            ),
        )
        missing_opt_ins = [
            name for enabled, opted_in, name in guard_requirements
            if enabled and not opted_in
        ]
        if missing_opt_ins:
            raise RuntimeError(
                "Transformer MATD3 ONNX export excludes enabled runtime "
                "dependencies. Set these flags only for non-deployable "
                f"experiment evidence: {', '.join(missing_opt_ins)}."
            )

        self._require_attached()
        assert self._attached_names is not None
        export_root = Path(output_dir)
        models_dir = export_root / "onnx_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        requires_runtime_residual = bool(self.residual_policy_enabled)
        requires_runtime_safety = bool(self._local_action_safety_enabled)
        requires_runtime_price = bool(self._local_price_conditioning_enabled)
        deployable = not any(
            (
                requires_runtime_residual,
                requires_runtime_safety,
                requires_runtime_price,
            )
        )
        artifacts: List[Dict[str, Any]] = []
        agent_models: List[Dict[str, Any]] = []
        for index, state in enumerate(self._per_building):
            topology_version = int(state.topology_version)
            observation_dimension = len(self._attached_names[index][0])
            relative_path = (
                f"onnx_models/agent_{index}__topology_v"
                f"{topology_version}.onnx"
            )
            self._export_onnx(
                state=state,
                path=export_root / relative_path,
                observation_dimension=observation_dimension,
            )
            sro_types = [
                segment.type_name
                for segment in state.layout.segments
                if segment.family == "sro"
            ]
            ca_types = [
                segment.type_name
                for segment in state.layout.segments
                if segment.family == "ca"
            ]
            layout_metadata = {
                "building_id": state.building_id,
                "topology_version": topology_version,
                "obs_dim": observation_dimension,
                "n_sro": state.layout.n_sro,
                "n_ca": state.layout.n_ca,
                "sro_types": sro_types,
                "ca_types": ca_types,
                "ca_action_names": list(state.layout.ca_action_names),
            }
            artifact_config = {
                **layout_metadata,
                "deployable": deployable,
                "requires_runtime_residual": requires_runtime_residual,
                "requires_runtime_local_action_safety": requires_runtime_safety,
                "requires_runtime_local_price_conditioning": requires_runtime_price,
            }
            artifacts.append(
                {
                    "agent_index": index,
                    "path": relative_path,
                    "format": "onnx",
                    "config": artifact_config,
                }
            )
            agent_models.append(
                {
                    "model_path": relative_path,
                    "building_index": index,
                    **layout_metadata,
                }
            )
        return {
            "format": "onnx",
            "artifacts": artifacts,
            "tokenizer_config_path": self._tokenizer_config_path,
            "supports_dynamic_topology": True,
            "agent_models": agent_models,
        }

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        self._require_attached()
        mode = self.checkpoint_mode
        payload: Dict[str, Any] = {
            "checkpoint_version": self.checkpoint_version,
            "algorithm": "AgentTransformerMATD3",
            "checkpoint_mode": mode,
            "step": int(step),
            "num_agents": len(self._per_building),
            "building_names": [state.building_id for state in self._per_building],
        }
        for index, state in enumerate(self._per_building):
            payload[f"tokenizer_state_dict_{index}"] = state.tokenizer.state_dict()
            payload[f"backbone_state_dict_{index}"] = state.backbone.state_dict()
            payload[f"actor_state_dict_{index}"] = state.actor.state_dict()
            payload[f"layout_signature_{index}"] = self._layout_signature[index]
            payload[f"action_names_{index}"] = state.action_names
            payload[f"action_bounds_{index}"] = (
                state.action_low.detach().cpu().numpy().copy(),
                state.action_high.detach().cpu().numpy().copy(),
            )
            payload[f"topology_version_{index}"] = state.topology_version
            if mode == "inference":
                continue
            payload[f"tokenizer_target_state_dict_{index}"] = (
                state.tokenizer_target.state_dict()
            )
            payload[f"backbone_target_state_dict_{index}"] = (
                state.backbone_target.state_dict()
            )
            payload[f"actor_target_state_dict_{index}"] = (
                state.actor_target.state_dict()
            )
            for critic_index in (1, 2):
                critic = getattr(state, f"critic_{critic_index}")
                critic_target = getattr(state, f"critic_{critic_index}_target")
                optimizer = getattr(state, f"critic_{critic_index}_optimizer")
                payload[f"critic_{critic_index}_state_dict_{index}"] = (
                    critic.state_dict()
                )
                payload[f"critic_{critic_index}_target_state_dict_{index}"] = (
                    critic_target.state_dict()
                )
                payload[
                    f"critic_{critic_index}_optimizer_state_dict_{index}"
                ] = optimizer.state_dict()
            payload[f"actor_optimizer_state_dict_{index}"] = (
                state.actor_optimizer.state_dict()
            )
            if state.bc_a_optimizer is not None:
                payload[f"bc_a_optimizer_state_dict_{index}"] = (
                    state.bc_a_optimizer.state_dict()
                )
            if state.bc_b_optimizer is not None:
                payload[f"bc_b_optimizer_state_dict_{index}"] = (
                    state.bc_b_optimizer.state_dict()
                )
        if mode == "inference":
            payload["inference_policy_state"] = {
                "exploration_step": int(self.exploration_step)
            }
        else:
            assert self.replay_buffer is not None
            payload.update(
                {
                    "replay_buffer": self.replay_buffer.get_state(),
                    "n_step_queue": deepcopy(list(self._n_step_queue)),
                    "current_layout_signature": self._layout_signature,
                    "exploration_state": {
                        "sigma": float(self.exploration_sigma),
                        "exploration_step": int(self.exploration_step),
                    },
                    "reward_normalization_state": {
                        "enabled": self.reward_normalization_enabled,
                        "count": int(self.reward_norm_count),
                        "mean": float(self.reward_norm_mean),
                        "m2": float(self.reward_norm_m2),
                    },
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "torch_cuda": self._capture_cuda_rng_state(),
                    },
                    "bc_state": {
                        "bc_a_state": (
                            {
                                "offline_pretrain_completed_steps": int(
                                    self.bc_a_offline_pretrain_completed_steps
                                )
                            }
                            if self.bc_a_enabled
                            else None
                        ),
                        "bc_b_state": (
                            {
                                "regularizer": self._bc_b.state_dict(),
                                "pretraining_complete": self._bc_b_pretraining_complete,
                                "actor_training_step": self._bc_b_actor_training_step,
                            }
                            if self._bc_b is not None
                            else None
                        ),
                    },
                }
            )
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"transformer_matd3_step{int(step)}.pt"
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._require_attached()
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        prepared_replay = self._validate_checkpoint_payload(payload)
        snapshot = self.snapshot_topology_state()
        try:
            mode = payload["checkpoint_mode"]
            for index, state in enumerate(self._per_building):
                state.tokenizer.load_state_dict(
                    payload[f"tokenizer_state_dict_{index}"]
                )
                state.backbone.load_state_dict(
                    payload[f"backbone_state_dict_{index}"]
                )
                state.actor.load_state_dict(payload[f"actor_state_dict_{index}"])
                low, high = payload[f"action_bounds_{index}"]
                state.action_low = self._tensor(low)
                state.action_high = self._tensor(high)
                state.topology_version = int(payload[f"topology_version_{index}"])
                if mode == "inference":
                    continue
                state.tokenizer_target.load_state_dict(
                    payload[f"tokenizer_target_state_dict_{index}"]
                )
                state.backbone_target.load_state_dict(
                    payload[f"backbone_target_state_dict_{index}"]
                )
                state.actor_target.load_state_dict(
                    payload[f"actor_target_state_dict_{index}"]
                )
                for critic_index in (1, 2):
                    getattr(state, f"critic_{critic_index}").load_state_dict(
                        payload[f"critic_{critic_index}_state_dict_{index}"]
                    )
                    getattr(
                        state, f"critic_{critic_index}_target"
                    ).load_state_dict(
                        payload[
                            f"critic_{critic_index}_target_state_dict_{index}"
                        ]
                    )
                    optimizer = getattr(
                        state, f"critic_{critic_index}_optimizer"
                    )
                    optimizer.load_state_dict(
                        payload[
                            f"critic_{critic_index}_optimizer_state_dict_{index}"
                        ]
                    )
                    self._move_optimizer_state_to_device(optimizer)
                state.actor_optimizer.load_state_dict(
                    payload[f"actor_optimizer_state_dict_{index}"]
                )
                self._move_optimizer_state_to_device(state.actor_optimizer)
                for bc_name in ("bc_a", "bc_b"):
                    optimizer = getattr(state, f"{bc_name}_optimizer")
                    if optimizer is not None:
                        optimizer.load_state_dict(
                            payload[f"{bc_name}_optimizer_state_dict_{index}"]
                        )
                        self._move_optimizer_state_to_device(optimizer)
            if mode == "inference":
                self.exploration_step = int(
                    payload["inference_policy_state"]["exploration_step"]
                )
                return
            assert prepared_replay is not None
            self.replay_buffer = prepared_replay
            self._n_step_queue = deque(deepcopy(payload["n_step_queue"]))
            exploration = payload["exploration_state"]
            self.exploration_sigma = float(exploration["sigma"])
            self.exploration_step = int(exploration["exploration_step"])
            reward = payload["reward_normalization_state"]
            self.reward_norm_count = int(reward["count"])
            self.reward_norm_mean = float(reward["mean"])
            self.reward_norm_m2 = float(reward["m2"])
            bc_state = payload["bc_state"]
            if self.bc_a_enabled:
                self.bc_a_offline_pretrain_completed_steps = int(
                    bc_state["bc_a_state"]["offline_pretrain_completed_steps"]
                )
            if self._bc_b is not None:
                saved_bc_b = bc_state["bc_b_state"]
                self._bc_b.load_state_dict(saved_bc_b["regularizer"])
                self._bc_b_pretraining_complete = bool(
                    saved_bc_b["pretraining_complete"]
                )
                self._bc_b_actor_training_step = int(
                    saved_bc_b["actor_training_step"]
                )
            rng = payload["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            self._restore_cuda_rng_state(rng["torch_cuda"])
        except Exception:
            self.restore_topology_state(snapshot)
            raise

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.end_initial_exploration_time_step

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = dict(self._latest_training_metrics)
        self._latest_training_metrics = {}
        return metrics

    def _should_runtime_profile_step(self, global_learning_step: int) -> bool:
        return bool(self.runtime_profiling_enabled) and (
            global_learning_step % self.runtime_profiling_interval == 0
        )

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        buffer_size = self.replay_buffer.total_size() if self.replay_buffer else 0
        bucket_size = 0
        bucket_count = 0
        if self.replay_buffer is not None and self._layout_signature is not None:
            bucket_size = self.replay_buffer.bucket_size(self._layout_signature)
            bucket_count = len(tuple(self.replay_buffer.signatures()))
        metrics = {
            f"{_METRIC_PREFIX}enabled": 1.0,
            f"{_METRIC_PREFIX}exploration_sigma": float(self.exploration_sigma),
            f"{_METRIC_PREFIX}exploration_step": float(self.exploration_step),
            f"{_METRIC_PREFIX}replay_buffer_size": float(buffer_size),
            f"{_METRIC_PREFIX}replay_bucket_size_current": float(bucket_size),
            f"{_METRIC_PREFIX}replay_bucket_count": float(bucket_count),
            f"{_METRIC_PREFIX}n_step_queue_size": float(len(self._n_step_queue)),
            f"{_METRIC_PREFIX}target_policy_smoothing": float(
                self.target_policy_smoothing
            ),
            f"{_METRIC_PREFIX}actor_update_interval": float(
                self.actor_update_interval
            ),
            f"{_METRIC_PREFIX}residual_policy_enabled": float(
                self.residual_policy_enabled
            ),
            f"{_METRIC_PREFIX}residual_action_scale_effective": float(
                self._residual_action_effective_scale()
            ),
            f"{_METRIC_PREFIX}local_action_safety_enabled": float(
                self._local_action_safety_enabled
            ),
            f"{_METRIC_PREFIX}local_price_conditioning_enabled": float(
                self._local_price_conditioning_enabled
            ),
        }
        if self._local_action_safety_enabled:
            metrics.update(
                {
                    f"{_METRIC_PREFIX}local_action_safety_projections": float(
                        self._local_action_safety_projection_count
                    ),
                    f"{_METRIC_PREFIX}local_action_safety_interventions": float(
                        self._local_action_safety_intervention_count
                    ),
                    f"{_METRIC_PREFIX}local_action_safety_infeasible": float(
                        self._local_action_safety_infeasible_count
                    ),
                }
            )
            metrics.update(
                {
                    f"{_METRIC_PREFIX}local_action_safety_reason_{reason}": float(
                        count
                    )
                    for reason, count in self._local_action_safety_reason_counts.items()
                }
            )
        if self._local_price_conditioning_enabled:
            metrics.update(
                {
                    f"{_METRIC_PREFIX}local_price_context_non_neutral": float(
                        self._local_price_context_non_neutral
                    ),
                    f"{_METRIC_PREFIX}local_price_clipping_count": float(
                        self._local_price_clipping_count
                    ),
                }
            )
        metrics.update(self._bc_b_metrics())
        metrics.update(self._latest_training_metrics)
        return metrics

    def _learn(
        self,
        batch: ReplayBatch,
        *,
        update_target_step: bool,
        global_learning_step: int,
        runtime_profile_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        started = time.perf_counter()
        should_profile = runtime_profile_metrics is not None
        phase_start = time.perf_counter() if should_profile else 0.0
        observations = [self._tensor(value) for value in batch.observations]
        next_observations = [
            self._tensor(value) for value in batch.next_observations
        ]
        actions = [self._tensor(value) for value in batch.actions]
        behavior_actions = (
            None
            if batch.behavior_actions is None
            else [self._tensor(value) for value in batch.behavior_actions]
        )
        next_behavior_actions = (
            None
            if batch.next_behavior_actions is None
            else [self._tensor(value) for value in batch.next_behavior_actions]
        )
        if batch.cloning_actions is not None:
            cloning_actions = [
                self._tensor(value) for value in batch.cloning_actions
            ]
        elif self.bc_a_teacher == "warm_start":
            cloning_actions = behavior_actions
        else:
            cloning_actions = actions
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_tensor_prepare_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()
        if self.residual_policy_enabled and (
            behavior_actions is None or next_behavior_actions is None
        ):
            raise RuntimeError("residual replay batch is missing base actions")
        offline_losses, offline_grad_norms = self._run_bc_a_offline_pretraining(
            observations=observations,
            behavior_actions=behavior_actions,
            cloning_actions=cloning_actions,
            global_learning_step=global_learning_step,
        )
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_bc_a_offline_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()
        raw_rewards = self._tensor(batch.rewards)
        individual_rewards = self._normalize_reward_tensor(raw_rewards)
        train_rewards = self._team_rewards(individual_rewards)
        self._last_train_rewards = train_rewards.detach().cpu()
        done = self._tensor(batch.done.astype(np.float32))

        with torch.no_grad():
            next_actions = []
            for index, (state, observation) in enumerate(
                zip(self._per_building, next_observations)
            ):
                next_actions.append(
                    self._target_action(
                        state,
                        observation,
                        index=index,
                        base_action=(
                            None
                            if next_behavior_actions is None
                            else next_behavior_actions[index]
                        ),
                    )
                )
            targets = []
            for index, state in enumerate(self._per_building):
                q1_next = state.critic_1_target(
                    next_observations, self._layouts, next_actions
                )
                q2_next = state.critic_2_target(
                    next_observations, self._layouts, next_actions
                )
                q_min = torch.minimum(q1_next, q2_next)
                target = train_rewards[:, index : index + 1] + (
                    self.n_step_gamma ** self.n_step_returns
                ) * q_min * (1.0 - done[:, index : index + 1])
                if self.critic_target_clip_abs > 0.0:
                    target = target.clamp(
                        -self.critic_target_clip_abs,
                        self.critic_target_clip_abs,
                    )
                targets.append(target)
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_target_compute_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()

        critic_1_losses = []
        critic_2_losses = []
        expected_1_values = []
        expected_2_values = []
        td_values = []
        gap_values = []
        critic_grad_norms = []
        for state, target in zip(self._per_building, targets):
            expected_1 = state.critic_1(observations, self._layouts, actions)
            expected_2 = state.critic_2(observations, self._layouts, actions)
            loss_1 = nn.functional.mse_loss(expected_1, target)
            loss_2 = nn.functional.mse_loss(expected_2, target)
            state.critic_1_optimizer.zero_grad(set_to_none=True)
            loss_1.backward()
            grad_1 = clip_grad_norm_(state.critic_1.parameters(), self.max_grad_norm)
            state.critic_1_optimizer.step()
            state.critic_2_optimizer.zero_grad(set_to_none=True)
            loss_2.backward()
            grad_2 = clip_grad_norm_(state.critic_2.parameters(), self.max_grad_norm)
            state.critic_2_optimizer.step()
            critic_1_losses.append(float(loss_1.detach()))
            critic_2_losses.append(float(loss_2.detach()))
            expected_1_values.append(expected_1.detach())
            expected_2_values.append(expected_2.detach())
            td_values.append(float((expected_1.detach() - target).abs().mean()))
            gap_values.append(
                float((expected_1.detach() - expected_2.detach()).abs().mean())
            )
            critic_grad_norms.extend((float(grad_1), float(grad_2)))
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_critic_update_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()

        actor_update_due = global_learning_step % self.actor_update_interval == 0
        actor_losses: List[float] = []
        actor_policy_losses: List[float] = []
        actor_bc_losses: List[float] = []
        actor_bc_type_losses: Dict[str, List[float]] = {
            "ev": [],
            "storage": [],
            "deferrable": [],
            "other": [],
        }
        actor_q_abs: List[float] = []
        actor_grad_norms: List[float] = []
        bc_weight = self._bc_a_effective_weight(global_learning_step)
        if actor_update_due:
            with torch.no_grad():
                detached_actions = []
                for index, (state, observation) in enumerate(
                    zip(self._per_building, observations)
                ):
                    detached_actions.append(
                        self._policy_action(
                            index,
                            state,
                            observation,
                            target=False,
                            base_action=(
                                None
                                if behavior_actions is None
                                else behavior_actions[index]
                            ),
                        ).detach()
                    )
            for index, state in enumerate(self._per_building):
                if state.layout.n_ca == 0:
                    continue
                joint_actions = list(detached_actions)
                joint_actions[index] = self._policy_action(
                    index,
                    state,
                    observations[index],
                    target=False,
                    base_action=(
                        None
                        if behavior_actions is None
                        else behavior_actions[index]
                    ),
                )
                self._set_requires_grad(state.critic_1, False)
                try:
                    q_policy = state.critic_1(
                        observations, self._layouts, joint_actions
                    )
                    policy_loss = -q_policy.mean()
                    bc_loss = policy_loss.new_tensor(0.0)
                    if (
                        bc_weight > 0.0
                        and cloning_actions is not None
                    ):
                        bc_loss = self._actor_behavior_cloning_loss(
                            index,
                            joint_actions[index],
                            cloning_actions[index],
                            base_action=(
                                None
                                if behavior_actions is None
                                else behavior_actions[index]
                            ),
                        )
                        type_losses = self._actor_behavior_cloning_type_losses(
                            index,
                            joint_actions[index],
                            cloning_actions[index],
                        )
                        for label, value in type_losses.items():
                            actor_bc_type_losses[label].append(float(value.detach()))
                    actor_loss = policy_loss + bc_weight * bc_loss
                    state.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad = clip_grad_norm_(
                        self._actor_modules(state).parameters(),
                        self.max_grad_norm,
                    )
                    state.actor_optimizer.step()
                finally:
                    self._set_requires_grad(state.critic_1, True)
                actor_losses.append(float(actor_loss.detach()))
                actor_policy_losses.append(float(policy_loss.detach()))
                actor_bc_losses.append(float(bc_loss.detach()))
                actor_q_abs.append(float(q_policy.detach().abs().mean()))
                actor_grad_norms.append(float(actor_grad))
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_actor_update_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()
        extra_losses, extra_grad_norms = self._run_bc_a_extra_updates(
            observations=observations,
            behavior_actions=behavior_actions,
            cloning_actions=cloning_actions,
            effective_weight=bc_weight,
            global_learning_step=global_learning_step,
        )
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_bc_a_extra_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()
        bc_b_losses, bc_b_grad_norms = self._run_bc_b_auxiliary_updates(
            global_learning_step=self._bc_b_actor_training_step
        )
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_bc_b_auxiliary_seconds"] = (
                time.perf_counter() - phase_start
            )
            phase_start = time.perf_counter()
        if actor_update_due and update_target_step:
            for state in self._per_building:
                self._soft_update(state.tokenizer, state.tokenizer_target)
                self._soft_update(state.backbone, state.backbone_target)
                self._soft_update(state.actor, state.actor_target)
                self._soft_update(state.critic_1, state.critic_1_target)
                self._soft_update(state.critic_2, state.critic_2_target)
        if should_profile:
            runtime_profile_metrics[f"{_METRIC_PREFIX}runtime_target_update_seconds"] = (
                time.perf_counter() - phase_start
            )

        expected_1_flat = torch.cat(
            [value.reshape(-1) for value in expected_1_values]
        )
        expected_2_flat = torch.cat(
            [value.reshape(-1) for value in expected_2_values]
        )
        target_flat = torch.cat([value.reshape(-1) for value in targets])
        self._latest_training_metrics = {
            f"{_METRIC_PREFIX}critic_1_loss_mean": float(np.mean(critic_1_losses)),
            f"{_METRIC_PREFIX}critic_2_loss_mean": float(np.mean(critic_2_losses)),
            f"{_METRIC_PREFIX}critic_loss_mean": float(
                np.mean(critic_1_losses) + np.mean(critic_2_losses)
            ),
            f"{_METRIC_PREFIX}critic_td_abs_mean": float(np.mean(td_values)),
            f"{_METRIC_PREFIX}critic_gap_abs_mean": float(np.mean(gap_values)),
            f"{_METRIC_PREFIX}critic_grad_norm_mean": float(
                np.mean(critic_grad_norms)
            ),
            f"{_METRIC_PREFIX}q1_expected_mean": float(expected_1_flat.mean()),
            f"{_METRIC_PREFIX}q2_expected_mean": float(expected_2_flat.mean()),
            f"{_METRIC_PREFIX}q_min_expected_mean": float(
                torch.minimum(expected_1_flat, expected_2_flat).mean()
            ),
            f"{_METRIC_PREFIX}q_target_mean": float(target_flat.mean()),
            f"{_METRIC_PREFIX}actor_update_performed": float(bool(actor_losses)),
            f"{_METRIC_PREFIX}actor_loss_mean": self._mean_or_zero(actor_losses),
            f"{_METRIC_PREFIX}actor_policy_loss_mean": self._mean_or_zero(
                actor_policy_losses
            ),
            f"{_METRIC_PREFIX}actor_policy_q_abs_mean": self._mean_or_zero(
                actor_q_abs
            ),
            f"{_METRIC_PREFIX}actor_grad_norm_mean": self._mean_or_zero(
                actor_grad_norms
            ),
            f"{_METRIC_PREFIX}reward_raw_mean": float(raw_rewards.mean()),
            f"{_METRIC_PREFIX}reward_train_mean": float(train_rewards.mean()),
            f"{_METRIC_PREFIX}reward_train_std": float(
                train_rewards.std(unbiased=False)
            ),
            f"{_METRIC_PREFIX}replay_buffer_size": float(
                self.replay_buffer.total_size() if self.replay_buffer else 0
            ),
            f"{_METRIC_PREFIX}replay_bucket_size_current": float(
                self.replay_buffer.bucket_size(self._layout_signature)
                if self.replay_buffer and self._layout_signature
                else 0
            ),
            f"{_METRIC_PREFIX}replay_bucket_count": float(
                len(tuple(self.replay_buffer.signatures()))
                if self.replay_buffer
                else 0
            ),
            f"{_METRIC_PREFIX}n_step_returns": float(self.n_step_returns),
            f"{_METRIC_PREFIX}n_step_queue_size": float(len(self._n_step_queue)),
            f"{_METRIC_PREFIX}target_policy_smoothing": float(
                self.target_policy_smoothing
            ),
            f"{_METRIC_PREFIX}target_policy_noise": self.target_policy_noise,
            f"{_METRIC_PREFIX}target_policy_noise_clip": (
                self.target_policy_noise_clip
            ),
            f"{_METRIC_PREFIX}actor_update_interval": float(
                self.actor_update_interval
            ),
            f"{_METRIC_PREFIX}exploration_sigma": float(self.exploration_sigma),
            f"{_METRIC_PREFIX}exploration_step": float(self.exploration_step),
            f"{_METRIC_PREFIX}training_step_time": time.perf_counter() - started,
        }
        if self.bc_a_enabled:
            self._latest_training_metrics.update(
                {
                    f"{_METRIC_PREFIX}actor_behavior_cloning_loss_mean": (
                        self._mean_or_zero(actor_bc_losses)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_effective_weight": (
                        bc_weight
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_extra_updates": float(
                        len(extra_losses)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_extra_loss_mean": (
                        self._mean_or_zero(extra_losses)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_extra_grad_norm_mean": (
                        self._mean_or_zero(extra_grad_norms)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_offline_updates": float(
                        len(offline_losses)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_offline_loss_mean": (
                        self._mean_or_zero(offline_losses)
                    ),
                    f"{_METRIC_PREFIX}actor_behavior_cloning_offline_grad_norm_mean": (
                        self._mean_or_zero(offline_grad_norms)
                    ),
                }
            )
            for label, values in actor_bc_type_losses.items():
                self._latest_training_metrics[
                    f"{_METRIC_PREFIX}actor_behavior_cloning_{label}_loss_mean"
                ] = self._mean_or_zero(values)
        if self._bc_b is not None:
            self._latest_training_metrics.update(self._bc_b_metrics())
            self._latest_training_metrics.update(
                {
                    f"{_METRIC_PREFIX}behavior_cloning_auxiliary_updates": float(
                        len(bc_b_losses)
                    ),
                    f"{_METRIC_PREFIX}behavior_cloning_auxiliary_loss_mean": (
                        self._mean_or_zero(bc_b_losses)
                    ),
                    f"{_METRIC_PREFIX}behavior_cloning_auxiliary_grad_norm_mean": (
                        self._mean_or_zero(bc_b_grad_norms)
                    ),
                }
            )
        if runtime_profile_metrics is not None:
            runtime_profile_metrics[
                f"{_METRIC_PREFIX}runtime_training_step_seconds"
            ] = time.perf_counter() - started
            self._latest_training_metrics.update(runtime_profile_metrics)

    @property
    def _layouts(self) -> List[BuildingTokenLayout]:
        return [state.layout for state in self._per_building]

    def _export_onnx(
        self,
        *,
        state: _PerBuildingState,
        path: Path,
        observation_dimension: int,
    ) -> None:
        layout = state.layout
        sro_indices = [
            torch.tensor(segment.feature_indices, dtype=torch.long)
            for segment in layout.segments
            if segment.family == "sro"
        ]
        ca_indices = [
            torch.tensor(segment.feature_indices, dtype=torch.long)
            for segment in layout.segments
            if segment.family == "ca"
        ]
        nfc_segment = next(
            segment for segment in layout.segments if segment.family == "nfc"
        )
        assert nfc_segment.derived is not None
        nfc_indices = torch.tensor(
            nfc_segment.feature_indices,
            dtype=torch.long,
        )
        nfc_left = nfc_segment.derived.left_index_in_segment
        nfc_right = nfc_segment.derived.right_index_in_segment
        sro_types = [
            segment.type_name
            for segment in layout.segments
            if segment.family == "sro"
        ]
        ca_types = [
            segment.type_name
            for segment in layout.segments
            if segment.family == "ca"
        ]
        tokenizer = deepcopy(state.tokenizer).to("cpu").eval()
        backbone = deepcopy(state.backbone).to("cpu").eval()
        actor = deepcopy(state.actor).to("cpu").eval()
        action_low = state.action_low.detach().cpu()
        action_high = state.action_high.detach().cpu()

        class _ExportWrapper(nn.Module):
            def __init__(self_inner) -> None:
                super().__init__()
                self_inner.tokenizer = tokenizer
                self_inner.backbone = backbone
                self_inner.actor = actor
                self_inner.register_buffer("action_low", action_low)
                self_inner.register_buffer("action_high", action_high)

            def forward(self_inner, encoded_obs: torch.Tensor) -> torch.Tensor:
                if layout.n_ca == 0:
                    return encoded_obs[:, :0]
                sro_tokens = [
                    self_inner.tokenizer.projections[type_name](
                        encoded_obs.index_select(1, indices)
                    ).unsqueeze(1)
                    for type_name, indices in zip(sro_types, sro_indices)
                ]
                ca_tokens = [
                    self_inner.tokenizer.projections[type_name](
                        encoded_obs.index_select(1, indices)
                    ).unsqueeze(1)
                    for type_name, indices in zip(ca_types, ca_indices)
                ]
                nfc_group = encoded_obs.index_select(1, nfc_indices)
                nfc_value = (
                    nfc_group[:, nfc_left] - nfc_group[:, nfc_right]
                ).unsqueeze(1)
                nfc_token = self_inner.tokenizer.projections[
                    nfc_segment.type_name
                ](nfc_value).unsqueeze(1)
                sro_stack = (
                    torch.cat(sro_tokens, dim=1)
                    if sro_tokens
                    else encoded_obs.new_zeros(
                        encoded_obs.shape[0],
                        0,
                        self_inner.backbone.d_model,
                    )
                )
                ca_stack = (
                    torch.cat(ca_tokens, dim=1)
                    if ca_tokens
                    else encoded_obs.new_zeros(
                        encoded_obs.shape[0],
                        0,
                        self_inner.backbone.d_model,
                    )
                )
                ca_embeddings, _ = self_inner.backbone(
                    sro_stack,
                    nfc_token,
                    ca_stack,
                )
                unit_actions = torch.tanh(
                    self_inner.actor(ca_embeddings)
                ).squeeze(-1)
                actions = self_inner.action_low + (unit_actions + 1.0) * (
                    (self_inner.action_high - self_inner.action_low) / 2.0
                )
                return actions.reshape(encoded_obs.shape[0], layout.n_ca)

        wrapper = _ExportWrapper().eval()
        dummy = torch.zeros(1, observation_dimension)
        previous_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    (dummy,),
                    str(path),
                    export_params=True,
                    do_constant_folding=True,
                    input_names=["encoded_obs"],
                    output_names=["actions"],
                    dynamic_axes={
                        "encoded_obs": {0: "batch"},
                        "actions": {0: "batch"},
                    },
                    opset_version=self.onnx_opset_version,
                )
                import onnx

                exported_model = onnx.load(path)
                output_width = (
                    exported_model.graph.output[0]
                    .type.tensor_type.shape.dim[1]
                )
                output_width.dim_value = layout.n_ca
                onnx.save(exported_model, path)
        finally:
            torch.backends.mha.set_fastpath_enabled(previous_fastpath)

    def _actor_unit_action(
        self,
        state: _PerBuildingState,
        observations: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        tokenizer = state.tokenizer_target if target else state.tokenizer
        backbone = state.backbone_target if target else state.backbone
        actor = state.actor_target if target else state.actor
        tokens = tokenizer(observations, state.layout)
        ca_embeddings, _ = backbone(
            tokens.sro_tokens,
            tokens.nfc_token,
            tokens.ca_tokens,
        )
        return torch.tanh(actor(ca_embeddings)).squeeze(-1)

    def _actor_action(
        self,
        state: _PerBuildingState,
        observations: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        unit_action = self._actor_unit_action(state, observations, target=target)
        return self._affine_action(unit_action, state)

    @staticmethod
    def _affine_action(
        unit_action: torch.Tensor,
        state: _PerBuildingState,
    ) -> torch.Tensor:
        return state.action_low + (unit_action + 1.0) * (
            state.action_high - state.action_low
        ) / 2.0

    def _policy_action(
        self,
        index: int,
        state: _PerBuildingState,
        observations: torch.Tensor,
        *,
        target: bool,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        unit_action = self._actor_unit_action(state, observations, target=target)
        return self._compose_policy_action(index, unit_action, base_action)

    def _compose_policy_action(
        self,
        index: int,
        unit_action: torch.Tensor,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        state = self._per_building[index]
        if not self.residual_policy_enabled:
            return self._affine_action(unit_action, state)
        if base_action is None:
            raise RuntimeError("residual policy action requires a base action")
        if base_action.shape[-1] != state.layout.n_ca:
            raise ValueError(
                f"warm-start action width for building {index} is "
                f"{base_action.shape[-1]}; expected {state.layout.n_ca}"
            )
        base = base_action.to(dtype=unit_action.dtype, device=unit_action.device)
        if base.ndim == 1 and unit_action.ndim == 2:
            base = base.unsqueeze(0).expand(unit_action.shape[0], -1)
        span = state.action_high - state.action_low
        authority = self._residual_action_effective_scale()
        mask = self._residual_action_scale_mask(index, unit_action)
        action = base + 0.5 * span * authority * mask * unit_action
        return torch.maximum(
            torch.minimum(action, state.action_high), state.action_low
        )

    def _target_action(
        self,
        state: _PerBuildingState,
        observations: torch.Tensor,
        *,
        index: Optional[int] = None,
        base_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        building_index = (
            self._per_building.index(state) if index is None else index
        )
        action = self._policy_action(
            building_index,
            state,
            observations,
            target=True,
            base_action=base_action,
        )
        if not self.target_policy_smoothing or self.target_policy_noise <= 0.0:
            return action
        span = state.action_high - state.action_low
        authority = torch.ones_like(span)
        if self.residual_policy_enabled:
            authority = self._residual_action_effective_scale() * (
                self._residual_action_scale_mask(building_index, action)
            )
        noise = torch.randn_like(action) * (
            self.target_policy_noise * span * authority
        )
        limit = self.target_policy_noise_clip * span * authority
        noise = torch.maximum(torch.minimum(noise, limit), -limit)
        return torch.maximum(
            torch.minimum(action + noise, state.action_high), state.action_low
        )

    def _explore_unit_action(
        self,
        unit_action: torch.Tensor,
        state: _PerBuildingState,
        index: int,
    ) -> torch.Tensor:
        if self.exploration_step < self.random_exploration_steps:
            return torch.rand_like(unit_action) * 2.0 - 1.0
        noise = 2.0 * (
            torch.randn_like(unit_action) * self.exploration_sigma + self.bias
        )
        multipliers = torch.ones_like(noise)
        for action_index, action_name in enumerate(state.action_names):
            if self._is_storage_action_name(action_name):
                multipliers[..., action_index] *= (
                    self.storage_exploration_noise_multiplier
                )
            if (
                self._is_ev_action_name(action_name)
                and state.action_low[action_index] < 0.0
            ):
                negative = noise[..., action_index] < 0.0
                multipliers[..., action_index] = torch.where(
                    negative,
                    multipliers[..., action_index]
                    * self.ev_negative_exploration_noise_multiplier,
                    multipliers[..., action_index],
                )
        noise = noise * multipliers
        if self.noise_clip is not None:
            noise = noise.clamp(-2.0 * self.noise_clip, 2.0 * self.noise_clip)
        explored = (unit_action + noise).clamp(-1.0, 1.0)
        self._apply_deferrable_exploration(explored, state, index)
        return explored

    def _apply_deferrable_exploration(
        self,
        unit_action: torch.Tensor,
        state: _PerBuildingState,
        index: int,
    ) -> None:
        del index
        if self.deferrable_on_probability <= 0.0:
            return
        threshold = 2.0 * self.deferrable_trigger_threshold - 1.0
        for action_index, action_name in enumerate(state.action_names):
            if not self._is_deferrable_action_name(action_name):
                continue
            if float(torch.rand((), device=unit_action.device)) < (
                self.deferrable_on_probability
            ):
                unit_action[..., action_index] = torch.empty(
                    (), device=unit_action.device
                ).uniform_(threshold, 1.0)

    def _residual_action_effective_scale(
        self,
        global_learning_step: Optional[int] = None,
    ) -> float:
        if not self.residual_policy_enabled:
            self._last_residual_action_scale = 0.0
            return 0.0
        step = self.exploration_step if global_learning_step is None else int(
            global_learning_step
        )
        if step < self.residual_action_scale_start_step:
            self._last_residual_action_scale = 0.0
            return 0.0
        if self.residual_action_scale_growth_steps <= 0:
            scale = self.residual_action_final_scale
        else:
            progress = min(
                max(
                    (step - self.residual_action_scale_start_step)
                    / self.residual_action_scale_growth_steps,
                    0.0,
                ),
                1.0,
            )
            scale = self.residual_action_scale + (
                self.residual_action_final_scale - self.residual_action_scale
            ) * progress
        self._last_residual_action_scale = float(np.clip(scale, 0.0, 1.0))
        return self._last_residual_action_scale

    def _residual_action_scale_mask(
        self,
        index: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.ones(
            self._per_building[index].layout.n_ca,
            dtype=like.dtype,
            device=like.device,
        )
        for action_index, action_name in enumerate(
            self._per_building[index].action_names
        ):
            if self._is_storage_action_name(action_name):
                values[action_index] *= (
                    self.residual_storage_action_scale_multiplier
                )
            if self._is_ev_action_name(action_name):
                values[action_index] *= self.residual_ev_action_scale_multiplier
            if self._is_deferrable_action_name(action_name):
                values[action_index] *= (
                    self.residual_deferrable_action_scale_multiplier
                )
        return values

    def _transition_behavior_actions(
        self,
        actions: Sequence[Any],
    ) -> List[Any]:
        needs_warm_start = self.residual_policy_enabled or (
            self.bc_a_enabled and self.bc_a_teacher == "warm_start"
        )
        if not needs_warm_start:
            return list(actions)
        if self._last_warm_start_policy_actions is None:
            predicted = self._predict_warm_start_policy_for_observations(
                self._latest_raw_observations
            )
            if predicted is None:
                if not self.residual_policy_enabled:
                    return list(actions)
                raise RuntimeError("residual replay requires warm-start base actions")
            return predicted
        return deepcopy(self._last_warm_start_policy_actions)

    def _transition_next_behavior_actions(
        self,
        fallback_actions: Sequence[Any],
    ) -> List[Any]:
        needs_warm_start = self.residual_policy_enabled or (
            self.bc_a_enabled and self.bc_a_teacher == "warm_start"
        )
        if not needs_warm_start:
            return list(fallback_actions)
        if self._last_warm_start_next_policy_actions is not None:
            return deepcopy(self._last_warm_start_next_policy_actions)
        predicted = self._predict_warm_start_policy_for_observations(
            self._latest_raw_next_observations
        )
        if predicted is None:
            if not self.residual_policy_enabled:
                return list(fallback_actions)
            raise RuntimeError(
                "residual replay requires next warm-start base actions"
            )
        return predicted

    def _transition_cloning_actions(
        self,
        fallback_actions: Sequence[Any],
        *,
        base_actions: Sequence[Any],
    ) -> List[Any]:
        if not self.bc_a_enabled or self.bc_a_teacher == "replay_action":
            return list(fallback_actions)
        if self.bc_a_teacher == "warm_start":
            return list(base_actions)
        if self.bc_a_teacher == "external":
            if self._latest_external_cloning_actions is None:
                return list(fallback_actions)
            self._validate_action_vector_group(
                "external BC-A cloning actions",
                self._latest_external_cloning_actions,
            )
            return deepcopy(self._latest_external_cloning_actions)
        raise ValueError(f"unsupported BC-A teacher {self.bc_a_teacher!r}")

    def _optional_replay_actions(
        self,
        values: Sequence[Any],
    ) -> Optional[List[Any]]:
        if self.residual_policy_enabled or (
            self.bc_a_enabled and self.bc_a_teacher == "warm_start"
        ):
            return list(values)
        return None

    def _distinct_cloning_actions(
        self,
        cloning_actions: Sequence[Any],
        behavior_actions: Sequence[Any],
    ) -> Optional[List[Any]]:
        if not self.bc_a_enabled:
            return None
        if all(
            np.array_equal(
                np.asarray(cloning), np.asarray(behavior)
            )
            for cloning, behavior in zip(cloning_actions, behavior_actions)
        ):
            return None
        return list(cloning_actions)

    def _bc_a_effective_weight(self, global_learning_step: int) -> float:
        return matd3_bc.effective_weight(self, global_learning_step)

    def _actor_behavior_cloning_loss(
        self,
        index: int,
        predicted_action: torch.Tensor,
        cloning_action: torch.Tensor,
        *,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return matd3_bc.actor_loss(
            self,
            index,
            predicted_action,
            cloning_action,
            base_action=base_action,
        )

    def _actor_behavior_cloning_type_losses(
        self,
        index: int,
        predicted_action: torch.Tensor,
        cloning_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return matd3_bc.actor_type_losses(
            self,
            index,
            predicted_action,
            cloning_action,
        )

    def _reachable_behavior_cloning_target(
        self,
        index: int,
        cloning_action: torch.Tensor,
        *,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return matd3_bc.reachable_target(
            self,
            index,
            cloning_action,
            base_action=base_action,
        )

    def _normalize_action(
        self,
        index: int,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = self._per_building[index]
        span = (state.action_high - state.action_low).clamp_min(1.0e-6)
        return (2.0 * (action - state.action_low) / span - 1.0).clamp(-1.0, 1.0)

    def _bc_a_action_weights(
        self,
        index: int,
        like: torch.Tensor,
    ) -> torch.Tensor:
        return matd3_bc.action_weights(self, index, like)

    def _run_bc_a_extra_updates(
        self,
        *,
        observations: Sequence[torch.Tensor],
        behavior_actions: Optional[Sequence[torch.Tensor]],
        cloning_actions: Optional[Sequence[torch.Tensor]],
        effective_weight: float,
        global_learning_step: int,
        update_count: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        count = self.bc_a_extra_updates if update_count is None else update_count
        if not matd3_bc.extra_updates_are_due(
            self,
            effective_weight_value=effective_weight,
            cloning_actions=cloning_actions,
            global_learning_step=global_learning_step,
            update_count=update_count,
        ):
            return [], []
        losses: List[float] = []
        gradients: List[float] = []
        for index, state in enumerate(self._per_building):
            if state.bc_a_optimizer is None:
                raise RuntimeError("BC-A optimizer is not initialized")
            for _ in range(count):
                predicted = self._policy_action(
                    index,
                    state,
                    observations[index],
                    target=False,
                    base_action=(
                        None
                        if behavior_actions is None
                        else behavior_actions[index]
                    ),
                )
                loss = effective_weight * self._actor_behavior_cloning_loss(
                    index,
                    predicted,
                    cloning_actions[index],
                    base_action=(
                        None
                        if behavior_actions is None
                        else behavior_actions[index]
                    ),
                )
                state.bc_a_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = clip_grad_norm_(
                    self._actor_modules(state).parameters(),
                    self.max_grad_norm,
                )
                state.bc_a_optimizer.step()
                losses.append(float(loss.detach()))
                gradients.append(float(gradient))
        return losses, gradients

    def _run_bc_a_offline_pretraining(
        self,
        *,
        observations: Sequence[torch.Tensor],
        behavior_actions: Optional[Sequence[torch.Tensor]],
        cloning_actions: Optional[Sequence[torch.Tensor]],
        global_learning_step: int,
    ) -> Tuple[List[float], List[float]]:
        remaining = max(
            self.bc_a_offline_pretrain_steps
            - self.bc_a_offline_pretrain_completed_steps,
            0,
        )
        effective_weight = self._bc_a_effective_weight(global_learning_step)
        if remaining <= 0 or effective_weight <= 0.0:
            return [], []
        losses, gradients = self._run_bc_a_extra_updates(
            observations=observations,
            behavior_actions=behavior_actions,
            cloning_actions=cloning_actions,
            effective_weight=effective_weight,
            global_learning_step=global_learning_step,
            update_count=remaining,
        )
        if losses:
            self.bc_a_offline_pretrain_completed_steps += remaining
        return losses, gradients

    def _in_bc_b_demonstration_phase(self) -> bool:
        return (
            self._bc_b is not None
            and not self._bc_b_pretraining_complete
            and self._current_episode_is_training
            and self._current_episode < self._bc_b.demonstration_episodes
        )

    def _record_bc_b_demonstrations(
        self,
        observations: Sequence[Any],
        actions: Sequence[Any],
    ) -> None:
        assert self._bc_b is not None
        for building_idx, (state, observation, action) in enumerate(
            zip(self._per_building, observations, actions)
        ):
            teacher_action = np.asarray(action, dtype=np.float32).reshape(-1)
            if teacher_action.shape != (state.layout.n_ca,):
                raise ValueError(
                    f"BC-B teacher action for building {state.building_id!r} "
                    f"has width {teacher_action.size}; expected {state.layout.n_ca}"
                )
            if not np.isfinite(teacher_action).all():
                raise ValueError("BC-B teacher actions must be finite")
            low = state.action_low.detach().cpu().numpy()
            high = state.action_high.detach().cpu().numpy()
            normalized_target = 2.0 * (teacher_action - low) / (high - low) - 1.0
            self._bc_b.record_demonstration(
                building_idx,
                np.asarray(observation, dtype=np.float32),
                state.layout,
                normalized_target.tolist(),
            )

    def _run_bc_b_pretraining(self) -> None:
        """Train every compatible stored signature before the first RL update."""
        assert self._bc_b is not None
        logger.info(
            "operation=matd3_bc_b_pretraining_start, message='pretraining started', buildings={}",
            len(self._per_building),
        )
        prepared_groups: List[List[Tuple[Demonstration, ...]]] = []
        incompatible_samples = 0
        missing_buildings: List[str] = []
        zero_action_samples_by_building: List[int] = []
        for building_idx, state in enumerate(self._per_building):
            grouped = self._bc_b.demonstrations_for_building_by_signature(
                building_idx
            )
            usable_groups: List[Tuple[Demonstration, ...]] = []
            zero_action_samples = 0
            for demonstrations in grouped.values():
                layout = demonstrations[0].layout
                if layout.n_ca == 0:
                    zero_action_samples += len(demonstrations)
                    logger.info(
                        "operation=matd3_bc_b_pretraining_group_skipped, "
                        "message='no controllable actions', building_id={} group_samples={}",
                        state.building_id,
                        len(demonstrations),
                    )
                    continue
                if not self._bc_b_layout_is_compatible(state, layout):
                    incompatible_samples += len(demonstrations)
                    continue
                usable_groups.append(demonstrations)
            prepared_groups.append(usable_groups)
            zero_action_samples_by_building.append(zero_action_samples)
            if state.layout.n_ca > 0 and not usable_groups:
                missing_buildings.append(state.building_id)
        self._bc_b.set_incompatible_demonstration_samples(incompatible_samples)
        if missing_buildings:
            raise RuntimeError(
                "Behavior-cloning pretraining has zero usable demonstrations for "
                f"building(s): {', '.join(missing_buildings)}."
            )

        total_batches = 0
        metrics: Dict[str, float] = {}
        for state, groups, zero_action_samples in zip(
            self._per_building, prepared_groups, zero_action_samples_by_building
        ):
            usable_samples = sum(len(group) for group in groups)
            trained_batches = 0
            for demonstrations in groups:
                layout = demonstrations[0].layout
                for _ in range(self._bc_b.pretraining_epochs):
                    for start in range(0, len(demonstrations), self._bc_b.batch_size):
                        batch = demonstrations[start : start + self._bc_b.batch_size]
                        self._apply_bc_b_gradient_step(
                            state=state,
                            layout=layout,
                            demonstrations=batch,
                            global_learning_step=0,
                            apply_weight=False,
                        )
                        trained_batches += 1
            total_batches += trained_batches
            metrics[
                f"{_METRIC_PREFIX}behavior_cloning_building_"
                f"{state.building_id}_usable_samples"
            ] = float(usable_samples)
            metrics[
                f"{_METRIC_PREFIX}behavior_cloning_building_"
                f"{state.building_id}_trained_batches"
            ] = float(trained_batches)
            metrics[
                f"{_METRIC_PREFIX}behavior_cloning_building_"
                f"{state.building_id}_zero_action_samples"
            ] = float(zero_action_samples)
        self._bc_b.set_pretraining_epochs(self._bc_b.pretraining_epochs)
        metrics[f"{_METRIC_PREFIX}behavior_cloning_pretraining_batches"] = float(
            total_batches
        )
        self._latest_training_metrics.update(self._bc_b_metrics())
        self._latest_training_metrics.update(metrics)
        logger.info(
            "operation=matd3_bc_b_pretraining_complete, message='pretraining complete', "
            "buildings={} trained_batches={}",
            len(self._per_building),
            total_batches,
        )

    def _bc_b_layout_is_compatible(
        self,
        state: _PerBuildingState,
        layout: BuildingTokenLayout,
    ) -> bool:
        assert self._bc_b is not None
        if layout.n_ca == 0:
            return False
        for segment in layout.segments:
            if segment.type_name not in state.tokenizer.projections:
                return False
            projection = state.tokenizer.projections[segment.type_name]
            expected_width = 1 if segment.family == "nfc" else len(
                segment.feature_indices
            )
            if projection.in_features != expected_width:
                return False
        weights = self._bc_b.ca_type_weights(
            layout, dtype=torch.float32, device=self.device
        )
        if weights.numel() != layout.n_ca:
            raise RuntimeError(
                "BC-B action-weight count does not match the stored layout for "
                f"building {state.building_id!r}"
            )
        if weights.sum().item() <= 0.0:
            raise ValueError(
                "BC-B has no active action weights for building "
                f"{state.building_id!r}"
            )
        return True

    def _apply_bc_b_gradient_step(
        self,
        *,
        state: _PerBuildingState,
        layout: BuildingTokenLayout,
        demonstrations: Sequence[Demonstration],
        global_learning_step: int,
        apply_weight: bool,
    ) -> Tuple[float, float]:
        assert self._bc_b is not None
        assert state.bc_b_optimizer is not None
        observations = self._tensor(
            np.stack([demonstration.observation for demonstration in demonstrations])
        )
        state.bc_b_optimizer.zero_grad(set_to_none=True)
        tokenized = state.tokenizer(observations, layout)
        ca_embeddings, _ = state.backbone(
            tokenized.sro_tokens,
            tokenized.nfc_token,
            tokenized.ca_tokens,
        )
        predicted_means = torch.tanh(state.actor(ca_embeddings))
        loss = self._bc_b.demonstration_loss(
            layout=layout,
            demonstrations=list(demonstrations),
            predicted_means=predicted_means,
            global_learning_step=global_learning_step,
            apply_weight=apply_weight,
        )
        if not loss.requires_grad:
            return 0.0, 0.0
        loss.backward()
        gradient = clip_grad_norm_(
            self._actor_modules(state).parameters(), self.max_grad_norm
        )
        state.bc_b_optimizer.step()
        return float(loss.detach()), float(gradient)

    def _run_bc_b_auxiliary_updates(
        self, *, global_learning_step: int
    ) -> Tuple[List[float], List[float]]:
        if self._bc_b is None or not self._bc_b_pretraining_complete:
            return [], []
        if self._bc_b.effective_weight(global_learning_step) <= 0.0:
            return [], []
        losses: List[float] = []
        gradients: List[float] = []
        for building_idx, state in enumerate(self._per_building):
            demonstrations = self._bc_b.sample_demonstrations(
                building_idx, state.layout, self._bc_b.batch_size
            )
            if not demonstrations:
                continue
            loss, gradient = self._apply_bc_b_gradient_step(
                state=state,
                layout=state.layout,
                demonstrations=demonstrations,
                global_learning_step=global_learning_step,
                apply_weight=True,
            )
            losses.append(loss)
            gradients.append(gradient)
        return losses, gradients

    def _bc_b_metrics(self) -> Dict[str, float]:
        if self._bc_b is None:
            return {}
        return {
            f"{_METRIC_PREFIX}{name}": value
            for name, value in self._bc_b.snapshot_metrics().items()
        }

    def _store_transition(self, transition: Dict[str, Any]) -> None:
        if self.n_step_returns == 1:
            self._push_transition(transition)
            return
        self._n_step_queue.append(transition)
        if len(self._n_step_queue) >= self.n_step_returns:
            self._push_oldest_n_step(force=False)
        if np.logical_or(transition["terminated"], transition["truncated"]).any():
            while self._n_step_queue:
                self._push_oldest_n_step(force=True)

    def _push_oldest_n_step(
        self, *, force: bool, topology_boundary: bool = False
    ) -> None:
        if not self._n_step_queue:
            return
        if not force and len(self._n_step_queue) < self.n_step_returns:
            return
        transitions = list(self._n_step_queue)[: self.n_step_returns]
        first = transitions[0]
        last = transitions[-1]
        rewards = np.zeros(len(self._per_building), dtype=np.float32)
        discount = 1.0
        for item in transitions:
            rewards += discount * item["rewards"]
            if np.logical_or(item["terminated"], item["truncated"]).any():
                break
            discount *= self.n_step_gamma
        last_terminated = np.asarray(last["terminated"], dtype=np.bool_).copy()
        last_truncated = np.asarray(last["truncated"], dtype=np.bool_).copy()
        if topology_boundary:
            last_truncated[:] = True
        self._push_transition(
            {
                "observations": first["observations"],
                "actions": first["actions"],
                "rewards": rewards,
                "next_observations": last["next_observations"],
                "terminated": last_terminated,
                "truncated": last_truncated,
                "behavior_actions": first.get("behavior_actions"),
                "next_behavior_actions": last.get("next_behavior_actions"),
                "cloning_actions": first.get("cloning_actions"),
                "layout_signature": first["layout_signature"],
            }
        )
        self._n_step_queue.popleft()

    def _push_transition(self, transition: Dict[str, Any]) -> None:
        assert self.replay_buffer is not None
        signature = transition.get("layout_signature", self._layout_signature)
        assert signature is not None
        self.replay_buffer.push(
            encoded_obs=transition["observations"],
            next_encoded_obs=transition["next_observations"],
            actions=transition["actions"],
            reward=transition["rewards"],
            terminated=transition["terminated"],
            truncated=transition["truncated"],
            layout_signature=signature,
            behavior_actions=transition.get("behavior_actions"),
            next_behavior_actions=transition.get("next_behavior_actions"),
            cloning_actions=transition.get("cloning_actions"),
        )

    def _flush_n_step_topology_boundary(self) -> None:
        while self._n_step_queue:
            self._push_oldest_n_step(force=True, topology_boundary=True)

    def _attach_warm_start_policy(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self.warm_start_policy_name is None:
            return
        from algorithms.agents.baseline_policies import (
            NormalNoBatteryPolicy,
            NormalPolicy,
            RBCBasicPolicy,
            RBCCommunityPolicy,
            RBCSmartLocalPolicy,
            RBCSmartPolicy,
            RandomPolicy,
        )
        from algorithms.agents.oracle_replay_policy import (
            FixedServiceOracleReplayPolicy,
        )
        from algorithms.agents.rbc_agent import RuleBasedPolicy
        from algorithms.agents.total_home_oracle_replay_policy import (
            TotalHomeOracleReplayPolicy,
        )
        from algorithms.agents.total_oracle_replay_policy import (
            TotalOracleReplayPolicy,
        )

        policy_classes = {
            "RuleBasedPolicy": RuleBasedPolicy,
            "RandomPolicy": RandomPolicy,
            "NormalNoBatteryPolicy": NormalNoBatteryPolicy,
            "NormalPolicy": NormalPolicy,
            "RBCBasicPolicy": RBCBasicPolicy,
            "RBCCommunityPolicy": RBCCommunityPolicy,
            "RBCSmartLocalPolicy": RBCSmartLocalPolicy,
            "RBCSmartPolicy": RBCSmartPolicy,
            "FixedServiceOracleReplayPolicy": FixedServiceOracleReplayPolicy,
            "TotalHomeOracleReplayPolicy": TotalHomeOracleReplayPolicy,
            "TotalOracleReplayPolicy": TotalOracleReplayPolicy,
        }
        policy_class = policy_classes.get(self.warm_start_policy_name)
        if policy_class is None:
            supported = ", ".join(sorted(policy_classes))
            raise ValueError(
                f"unsupported warm_start_policy_name "
                f"{self.warm_start_policy_name!r}; supported: {supported}"
            )
        policy_config = deepcopy(self.config)
        policy_config["algorithm"] = {
            "name": self.warm_start_policy_name,
            "hyperparameters": deepcopy(self.warm_start_policy_hyperparameters),
        }
        self._warm_start_policy = policy_class(policy_config)
        self._warm_start_policy.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def _attach_bc_b_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
        topology_change: bool = False,
    ) -> None:
        """Build the live teacher without storing it in MATD3 state."""
        if self._bc_b is None:
            return
        attach = (
            self._bc_b.on_topology_change
            if topology_change
            else self._bc_b.attach_environment
        )
        attach(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

    def _predict_warm_start_policy_for_observations(
        self,
        observations: Optional[List[np.ndarray]],
    ) -> Optional[List[List[float]]]:
        if self._warm_start_policy is None or observations is None:
            return None
        predict_at_step = getattr(self._warm_start_policy, "predict_at_step", None)
        if callable(predict_at_step):
            actions = predict_at_step(
                observations,
                schedule_step=self.exploration_step,
                deterministic=True,
            )
        else:
            actions = self._warm_start_policy.predict(
                observations,
                deterministic=True,
            )
        if len(actions) != len(self._per_building):
            raise ValueError(
                "warm-start policy returned an action group count that does not "
                "match the building count"
            )
        result: List[List[float]] = []
        for index, (action, state) in enumerate(zip(actions, self._per_building)):
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.shape != (state.layout.n_ca,):
                raise ValueError(
                    f"warm-start action width for building {index} is "
                    f"{values.size}; expected {state.layout.n_ca}"
                )
            if not np.isfinite(values).all():
                raise ValueError("warm-start actions must be finite")
            low = state.action_low.detach().cpu().numpy()
            high = state.action_high.detach().cpu().numpy()
            result.append(np.clip(values, low, high).tolist())
        return result

    def _attach_local_action_safety(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self._local_action_safety_adapters = []
        if not self._local_action_safety_enabled:
            return
        for state, names, actions in zip(
            self._per_building, observation_names, action_names
        ):
            self._local_action_safety_adapters.append(
                CityLearnLocalSafetyAdapter(
                    observation_names=names,
                    action_names=actions,
                    action_low=state.action_low.detach().cpu().numpy(),
                    action_high=state.action_high.detach().cpu().numpy(),
                    metadata=metadata,
                    config=self._local_action_safety_config,
                )
            )

    def _apply_local_action_safety(
        self,
        index: int,
        proposed_action: Sequence[float],
    ) -> List[float]:
        if not self._local_action_safety_enabled:
            return [float(value) for value in proposed_action]
        if self._latest_raw_observations is None:
            raise RuntimeError(
                "Transformer MATD3 local action safety requires raw observation "
                "context before predict"
            )
        projection = self._local_action_safety_adapters[index].project(
            self._latest_raw_observations[index],
            proposed_action,
        )
        self._local_action_safety_projection_count += 1
        self._local_action_safety_intervention_count += len(
            projection.interventions
        )
        self._local_action_safety_infeasible_count += len(
            projection.infeasible_reasons
        )
        for intervention in projection.interventions:
            for reason in intervention.reason_codes:
                label = str(reason.value)
                self._local_action_safety_reason_counts[label] = (
                    self._local_action_safety_reason_counts.get(label, 0) + 1
                )
        for reason in projection.infeasible_reasons:
            label = str(reason.code.value)
            self._local_action_safety_reason_counts[label] = (
                self._local_action_safety_reason_counts.get(label, 0) + 1
            )
        return [float(value) for value in projection.executed_actions]

    def _attach_local_price_conditioning(
        self,
        *,
        observation_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self._local_price_adapters = []
        if not self._local_price_conditioning_enabled:
            return
        for index, fallback_names in enumerate(observation_names):
            feature_low, feature_high = price_feature_bounds_from_metadata(
                metadata=metadata,
                agent_index=index,
            )
            encoded_names = price_observation_names_from_metadata(
                metadata=metadata,
                agent_index=index,
                fallback_observation_names=fallback_names,
            )
            self._local_price_adapters.append(
                PriceMultiplierObservationAdapter(
                    observation_names=encoded_names,
                    feature_low=feature_low,
                    feature_high=feature_high,
                    forecast_mode=self._local_price_forecast_mode,
                    require_strict_local=False,
                )
            )

    def _apply_local_price_context(
        self,
        observations: List[npt.NDArray[np.float64]],
        context: Any,
    ) -> List[npt.NDArray[np.float64]]:
        self._local_price_context_non_neutral = False
        self._local_price_clipping_count = 0
        if not self._local_price_conditioning_enabled:
            return observations
        if len(self._local_price_adapters) != len(self._per_building):
            raise RuntimeError(
                "Transformer MATD3 local price conditioning requires an attached environment"
            )
        contexts = normalize_price_multiplier_contexts(
            context,
            num_agents=len(self._per_building),
        )
        transformed: List[npt.NDArray[np.float64]] = []
        for adapter, observation, price_context in zip(
            self._local_price_adapters, observations, contexts
        ):
            if price_context is None:
                transformed.append(np.asarray(observation).copy())
                continue
            conditioned, diagnostics = adapter.transform(
                observation,
                price_context,
            )
            transformed.append(conditioned)
            self._local_price_context_non_neutral |= not diagnostics.neutral_noop
            self._local_price_clipping_count += diagnostics.clipping_count
        return transformed

    def _build_state(
        self,
        *,
        layout: BuildingTokenLayout,
        action_names: Tuple[str, ...],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        type_input_dims: Mapping[str, int],
    ) -> _PerBuildingState:
        tokenizer = EntityObservationTokenizer(
            self._tokenizer_config, self._d_model, type_input_dims
        ).to(self.device)
        backbone = self._new_backbone()
        actor = DeterministicActorHead(
            self._d_model, self.actor_hidden_dim
        ).to(self.device)
        tokenizer_target = deepcopy(tokenizer)
        backbone_target = deepcopy(backbone)
        actor_target = deepcopy(actor)
        critic_arguments = {
            "d_model": self._d_model,
            "nhead": self._nhead,
            "num_layers": self._num_layers,
            "dim_feedforward": self._dim_feedforward,
            "hidden_dim": self.critic_hidden_dim,
            "dropout": self._dropout,
            "tokenizer_config": self._tokenizer_config,
            "type_input_dims": type_input_dims,
        }
        critic_1 = CentralizedCritic(**critic_arguments).to(self.device)
        critic_2 = CentralizedCritic(**critic_arguments).to(self.device)
        critic_1_target = deepcopy(critic_1)
        critic_2_target = deepcopy(critic_2)
        tokenizer_target.eval()
        backbone_target.eval()
        actor_target.eval()
        critic_1_target.eval()
        critic_2_target.eval()
        actor_modules = nn.ModuleList((tokenizer, backbone, actor))
        actor_parameters = list(actor_modules.parameters())
        return _PerBuildingState(
            building_id=layout.building_id,
            tokenizer=tokenizer,
            backbone=backbone,
            actor=actor,
            tokenizer_target=tokenizer_target,
            backbone_target=backbone_target,
            actor_target=actor_target,
            critic_1=critic_1,
            critic_2=critic_2,
            critic_1_target=critic_1_target,
            critic_2_target=critic_2_target,
            actor_optimizer=torch.optim.Adam(actor_parameters, lr=self.learning_rate),
            critic_1_optimizer=torch.optim.Adam(
                critic_1.parameters(), lr=self.learning_rate
            ),
            critic_2_optimizer=torch.optim.Adam(
                critic_2.parameters(), lr=self.learning_rate
            ),
            bc_a_optimizer=(
                torch.optim.Adam(actor_parameters, lr=self.learning_rate)
                if self.bc_a_enabled
                else None
            ),
            bc_b_optimizer=(
                torch.optim.Adam(actor_parameters, lr=self.learning_rate)
                if self._bc_b is not None
                else None
            ),
            layout=layout,
            action_names=action_names,
            action_low=action_low,
            action_high=action_high,
        )

    def _new_backbone(self) -> TransformerBackbone:
        return TransformerBackbone(
            d_model=self._d_model,
            nhead=self._nhead,
            num_layers=self._num_layers,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
        ).to(self.device)

    def _community_type_input_dims(
        self, layouts: Sequence[BuildingTokenLayout]
    ) -> Dict[str, int]:
        nfc_name = self._tokenizer_config.nfc.type_name
        dimensions: Dict[str, int] = {nfc_name: 1}
        for layout in layouts:
            for segment in layout.segments:
                if segment.family == "nfc":
                    continue
                width = len(segment.feature_indices)
                existing = dimensions.get(segment.type_name)
                if existing is not None and existing != width:
                    raise ValueError(
                        f"inconsistent input width for type {segment.type_name!r}: "
                        f"{existing} and {width}"
                    )
                dimensions[segment.type_name] = width
        for type_name, config in self._tokenizer_config.ca_types.items():
            dimensions.setdefault(type_name, int(config.input_dim_fallback))
        for type_name, config in self._tokenizer_config.sro_types.items():
            dimensions.setdefault(type_name, int(config.input_dim_fallback))
        return dimensions

    @staticmethod
    def _build_layout_signature(
        layouts: Sequence[BuildingTokenLayout],
    ) -> LayoutSignature:
        buildings = []
        for layout in layouts:
            widths: Dict[str, int] = {}
            segments = []
            for segment in layout.segments:
                segments.append(
                    (
                        segment.family,
                        segment.type_name,
                        segment.instance_id,
                        tuple(segment.feature_names),
                        (
                            (
                                segment.derived.op,
                                segment.feature_names[
                                    segment.derived.left_index_in_segment
                                ],
                                segment.feature_names[
                                    segment.derived.right_index_in_segment
                                ],
                            )
                            if segment.derived is not None
                            else None
                        ),
                    )
                )
                width = 1 if segment.family == "nfc" else len(
                    segment.feature_indices
                )
                existing = widths.setdefault(segment.type_name, width)
                if existing != width:
                    raise ValueError(
                        f"layout type {segment.type_name!r} has mixed widths"
                    )
            buildings.append(
                (
                    layout.n_sro,
                    layout.n_ca,
                    tuple(layout.ca_action_names),
                    tuple(segments),
                    tuple(layout.excluded_feature_names),
                    tuple(sorted(widths.items())),
                )
            )
        return tuple(buildings)

    def _action_bounds(
        self,
        building_index: int,
        action_names: Sequence[str],
        space: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count = len(action_names)
        if space is None:
            low = np.full(count, -1.0, dtype=np.float32)
            high = np.full(count, 1.0, dtype=np.float32)
        else:
            if not hasattr(space, "low") or not hasattr(space, "high"):
                raise ValueError(
                    f"action space {building_index} must expose low and high"
                )
            low = np.asarray(space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        if low.shape != (count,) or high.shape != (count,):
            raise ValueError(
                f"action bounds for building {building_index} must have width {count}"
            )
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("action bounds must be finite")
        if np.any(low >= high):
            raise ValueError("action bounds must satisfy low < high")
        return self._tensor(low), self._tensor(high)

    @staticmethod
    def _validate_ca_order(
        building_index: int,
        layout: BuildingTokenLayout,
        action_names: Sequence[str],
    ) -> None:
        if len(layout.ca_action_names) != len(action_names):
            raise ValueError(
                f"building {building_index} CA count does not match action count"
            )
        for action_field, action_name in zip(layout.ca_action_names, action_names):
            if action_name == action_field or action_name.startswith(
                action_field + "_"
            ):
                continue
            raise ValueError(
                f"building {building_index} CA order does not match action order"
            )

    def _update_reward_normalizer(self, rewards: Sequence[float]) -> None:
        if not self.reward_normalization_enabled:
            return
        for value in np.asarray(rewards, dtype=np.float64).reshape(-1):
            if not np.isfinite(value):
                continue
            self.reward_norm_count += 1
            delta = float(value) - self.reward_norm_mean
            self.reward_norm_mean += delta / self.reward_norm_count
            self.reward_norm_m2 += delta * (float(value) - self.reward_norm_mean)

    def _normalize_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.reward_normalization_enabled or self.reward_norm_count < 2:
            return rewards
        variance = max(
            self.reward_norm_m2 / (self.reward_norm_count - 1), 0.0
        )
        standard_deviation = max(float(np.sqrt(variance)), 1.0e-8)
        return ((rewards - self.reward_norm_mean) / standard_deviation).clamp(
            -self.reward_normalization_clip,
            self.reward_normalization_clip,
        )

    def _team_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.critic_team_reward_mix <= 0.0 or rewards.shape[1] <= 1:
            return rewards
        team = rewards.mean(dim=1, keepdim=True)
        return (
            (1.0 - self.critic_team_reward_mix) * rewards
            + self.critic_team_reward_mix * team
        )

    def _record_skip(self, reason: str, bucket_size: int) -> None:
        assert self.replay_buffer is not None
        self._latest_training_metrics = {
            f"{_METRIC_PREFIX}update_skipped": 1.0,
            f"{_METRIC_PREFIX}update_skip_replay_underfull": float(
                reason == "replay_underfull"
            ),
            f"{_METRIC_PREFIX}update_skip_initial_exploration": float(
                reason == "initial_exploration"
            ),
            f"{_METRIC_PREFIX}update_skip_schedule": float(reason == "schedule"),
            f"{_METRIC_PREFIX}replay_buffer_size": float(
                self.replay_buffer.total_size()
            ),
            f"{_METRIC_PREFIX}replay_bucket_size_current": float(bucket_size),
        }

    def _validate_checkpoint_payload(
        self, payload: Any
    ) -> Optional[SignatureBucketedReplayBuffer]:
        if not isinstance(payload, Mapping):
            raise ValueError("Transformer MATD3 checkpoint must be a mapping")
        checkpoint_version = payload.get("checkpoint_version")
        if (
            isinstance(checkpoint_version, bool)
            or not isinstance(checkpoint_version, int)
            or checkpoint_version != self.checkpoint_version
        ):
            raise ValueError("Transformer MATD3 checkpoint_version must be exactly 5")
        if payload.get("algorithm") != "AgentTransformerMATD3":
            raise ValueError("checkpoint algorithm must be AgentTransformerMATD3")
        mode = payload.get("checkpoint_mode")
        if mode not in {"full", "inference"}:
            raise ValueError("checkpoint_mode must be 'full' or 'inference'")
        if mode == "inference" and not bool(getattr(self, "frozen", False)):
            raise RuntimeError(
                "Transformer MATD3 inference checkpoints may be loaded only "
                "into a frozen pipeline stage"
            )
        num_agents = payload.get("num_agents")
        if (
            isinstance(num_agents, bool)
            or not isinstance(num_agents, int)
            or num_agents != len(self._per_building)
        ):
            raise ValueError(
                "checkpoint num_agents does not match the live building count"
            )
        if payload.get("building_names") != [
            state.building_id for state in self._per_building
        ]:
            raise ValueError("checkpoint building_names do not match the live layout")
        for index, state in enumerate(self._per_building):
            if (
                payload.get(f"layout_signature_{index}")
                != self._layout_signature[index]
            ):
                raise ValueError(
                    f"checkpoint layout signature mismatch for building {index}"
                )
            if payload.get(f"action_names_{index}") != state.action_names:
                raise ValueError(
                    f"checkpoint action names mismatch for building {index}"
                )
            bounds = payload.get(f"action_bounds_{index}")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(
                    f"checkpoint action bounds are invalid for building {index}"
                )
            saved_low = np.asarray(bounds[0], dtype=np.float32).reshape(-1)
            saved_high = np.asarray(bounds[1], dtype=np.float32).reshape(-1)
            live_low = state.action_low.detach().cpu().numpy()
            live_high = state.action_high.detach().cpu().numpy()
            if (
                saved_low.shape != live_low.shape
                or saved_high.shape != live_high.shape
                or not np.isfinite(saved_low).all()
                or not np.isfinite(saved_high).all()
                or not np.allclose(saved_low, live_low, rtol=0.0, atol=1.0e-6)
                or not np.allclose(saved_high, live_high, rtol=0.0, atol=1.0e-6)
            ):
                raise ValueError(
                    f"checkpoint action bounds mismatch for building {index}"
                )
            topology_version = payload.get(f"topology_version_{index}")
            if (
                isinstance(topology_version, bool)
                or not isinstance(topology_version, int)
                or topology_version < 0
            ):
                raise ValueError("checkpoint topology_version must be non-negative")
            for module_name in ("tokenizer", "backbone", "actor"):
                self._validate_module_state_dict(
                    getattr(state, module_name),
                    payload.get(f"{module_name}_state_dict_{index}"),
                    f"{module_name} building {index}",
                )
            if mode == "inference":
                continue
            for module_name in (
                "tokenizer_target", "backbone_target", "actor_target",
                "critic_1", "critic_1_target", "critic_2", "critic_2_target",
            ):
                self._validate_module_state_dict(
                    getattr(state, module_name),
                    payload.get(f"{module_name}_state_dict_{index}"),
                    f"{module_name} building {index}",
                )
            for optimizer_name in (
                "actor_optimizer", "critic_1_optimizer", "critic_2_optimizer",
            ):
                self._validate_optimizer_state_dict(
                    getattr(state, optimizer_name),
                    payload.get(f"{optimizer_name}_state_dict_{index}"),
                    f"{optimizer_name} building {index}",
                )
            for bc_name in ("bc_a", "bc_b"):
                optimizer = getattr(state, f"{bc_name}_optimizer")
                saved_optimizer = payload.get(f"{bc_name}_optimizer_state_dict_{index}")
                if optimizer is None and saved_optimizer is not None:
                    raise ValueError(
                        f"checkpoint contains disabled {bc_name} optimizer state"
                    )
                if optimizer is not None:
                    self._validate_optimizer_state_dict(
                        optimizer,
                        saved_optimizer,
                        f"{bc_name} optimizer building {index}",
                    )
        if mode == "inference":
            inference = payload.get("inference_policy_state")
            if not isinstance(inference, Mapping):
                raise ValueError("checkpoint inference_policy_state is invalid")
            step = inference.get("exploration_step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("checkpoint inference exploration_step is invalid")
            return None

        required_global = (
            "replay_buffer", "n_step_queue", "current_layout_signature",
            "exploration_state", "reward_normalization_state", "rng_state", "bc_state",
        )
        for key in required_global:
            if key not in payload:
                raise ValueError(f"checkpoint is missing required field {key!r}")
        if payload["current_layout_signature"] != self._layout_signature:
            raise ValueError("checkpoint current layout signature mismatch")
        replay = SignatureBucketedReplayBuffer(
            capacity=self.buffer_capacity,
            num_agents=len(self._per_building),
            batch_size=self.batch_size,
        )
        replay.set_state(payload["replay_buffer"])
        self._validate_checkpoint_n_step_queue(payload["n_step_queue"], replay)
        exploration = payload["exploration_state"]
        if not isinstance(exploration, Mapping):
            raise ValueError("checkpoint exploration_state is invalid")
        sigma = exploration.get("sigma")
        exploration_step = exploration.get("exploration_step")
        if (
            isinstance(exploration_step, bool)
            or not isinstance(exploration_step, int)
            or exploration_step < 0
            or isinstance(sigma, bool)
            or not isinstance(sigma, (int, float))
            or not np.isfinite(float(sigma))
        ):
            raise ValueError("checkpoint exploration_state is invalid")
        reward = payload["reward_normalization_state"]
        if not isinstance(reward, Mapping):
            raise ValueError("checkpoint reward_normalization_state is invalid")
        if reward.get("enabled") is not self.reward_normalization_enabled:
            raise ValueError("checkpoint reward normalization mode mismatch")
        count, mean, m2 = reward.get("count"), reward.get("mean"), reward.get("m2")
        if (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            or not isinstance(mean, (int, float))
            or not isinstance(m2, (int, float))
            or not np.isfinite(float(mean))
            or not np.isfinite(float(m2))
            or float(m2) < 0.0
        ):
            raise ValueError("checkpoint reward normalization state is invalid")
        self._validate_checkpoint_bc_state(payload["bc_state"])
        self._validate_rng_state(payload["rng_state"])
        return replay

    @staticmethod
    def _validate_module_state_dict(
        module: nn.Module, saved: Any, label: str
    ) -> None:
        if not isinstance(saved, Mapping):
            raise ValueError(f"checkpoint {label} state is invalid")
        live = module.state_dict()
        if tuple(saved.keys()) != tuple(live.keys()):
            raise ValueError(f"checkpoint {label} parameter keys mismatch")
        for key, live_value in live.items():
            saved_value = saved[key]
            if (
                not isinstance(saved_value, torch.Tensor)
                or saved_value.shape != live_value.shape
                or saved_value.dtype != live_value.dtype
            ):
                raise ValueError(
                    f"checkpoint {label} parameter {key!r} is incompatible"
                )

    @staticmethod
    def _validate_optimizer_state_dict(
        optimizer: torch.optim.Optimizer, saved: Any, label: str
    ) -> None:
        if not isinstance(saved, Mapping):
            raise ValueError(f"checkpoint {label} state is invalid")
        candidate = deepcopy(optimizer)
        try:
            candidate.load_state_dict(saved)
            for group in candidate.param_groups:
                for parameter in group["params"]:
                    for key, value in candidate.state.get(parameter, {}).items():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim > 0
                            and value.shape != parameter.shape
                        ):
                            raise ValueError(
                                f"optimizer tensor {key!r} has incompatible shape"
                            )
        except Exception as exc:
            raise ValueError(f"checkpoint {label} state is incompatible") from exc

    def _validate_checkpoint_n_step_queue(
        self,
        queue: Any,
        replay: SignatureBucketedReplayBuffer,
    ) -> None:
        if not isinstance(queue, list) or len(queue) >= self.n_step_returns:
            raise ValueError("checkpoint n_step_queue length is invalid")
        expected_optional_presence: Optional[Tuple[bool, bool, bool]] = None
        current_bucket = replay.get_state()["buckets"].get(self._layout_signature)
        if current_bucket:
            first = current_bucket[0]
            expected_optional_presence = (
                first.behavior_actions is not None,
                first.next_behavior_actions is not None,
                first.cloning_actions is not None,
            )
        for transition in deepcopy(queue):
            if not isinstance(transition, dict):
                raise ValueError("checkpoint n_step_queue entry is invalid")
            if transition.get("layout_signature") != self._layout_signature:
                raise ValueError("checkpoint n_step_queue signature mismatch")
            required = (
                "observations", "actions", "rewards", "next_observations",
                "terminated", "truncated",
            )
            if any(key not in transition for key in required):
                raise ValueError("checkpoint n_step_queue entry is incomplete")
            for field in ("observations", "next_observations", "actions"):
                vectors = transition[field]
                if not isinstance(vectors, (list, tuple)) or len(vectors) != len(
                    self._per_building
                ):
                    raise ValueError(
                        f"checkpoint n_step_queue {field} group is invalid"
                    )
                for index, vector in enumerate(vectors):
                    expected_width = (
                        len(self._attached_names[index][0])
                        if field != "actions"
                        else self._per_building[index].layout.n_ca
                    )
                    if (
                        not isinstance(vector, np.ndarray)
                        or vector.dtype != np.dtype(np.float32)
                        or vector.shape != (expected_width,)
                        or not np.isfinite(vector).all()
                    ):
                        raise ValueError(
                            f"checkpoint n_step_queue {field}[{index}] is invalid"
                        )
            for field in (
                "behavior_actions", "next_behavior_actions", "cloning_actions"
            ):
                vectors = transition.get(field)
                if vectors is None:
                    continue
                if not isinstance(vectors, (list, tuple)) or len(vectors) != len(
                    self._per_building
                ):
                    raise ValueError(
                        f"checkpoint n_step_queue {field} group is invalid"
                    )
                for index, vector in enumerate(vectors):
                    if (
                        not isinstance(vector, np.ndarray)
                        or vector.dtype != np.dtype(np.float32)
                        or vector.shape != (self._per_building[index].layout.n_ca,)
                        or not np.isfinite(vector).all()
                    ):
                        raise ValueError(
                            f"checkpoint n_step_queue {field}[{index}] is invalid"
                        )
            optional_presence = tuple(
                transition.get(field) is not None
                for field in (
                    "behavior_actions",
                    "next_behavior_actions",
                    "cloning_actions",
                )
            )
            if expected_optional_presence is None:
                expected_optional_presence = optional_presence
            elif optional_presence != expected_optional_presence:
                raise ValueError(
                    "checkpoint n_step_queue optional action presence is unstable"
                )
            rewards = transition["rewards"]
            if (
                not isinstance(rewards, np.ndarray)
                or rewards.dtype != np.dtype(np.float32)
                or rewards.shape != (len(self._per_building),)
                or not np.isfinite(rewards).all()
            ):
                raise ValueError("checkpoint n_step_queue rewards are invalid")
            for field in ("terminated", "truncated"):
                values = transition[field]
                if (
                    not isinstance(values, np.ndarray)
                    or values.dtype != np.dtype(np.bool_)
                    or values.shape != (len(self._per_building),)
                ):
                    raise ValueError(
                        f"checkpoint n_step_queue {field} values are invalid"
                    )

    def _validate_checkpoint_bc_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint bc_state is invalid")
        bc_a_state = state.get("bc_a_state")
        if self.bc_a_enabled:
            if not isinstance(bc_a_state, Mapping):
                raise ValueError("checkpoint BC-A state is missing")
            completed = bc_a_state.get("offline_pretrain_completed_steps")
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed < 0
            ):
                raise ValueError("checkpoint BC-A state is invalid")
        elif bc_a_state is not None:
            raise ValueError("checkpoint contains BC-A state but BC-A is disabled")
        bc_b_state = state.get("bc_b_state")
        if self._bc_b is None:
            if bc_b_state is not None:
                raise ValueError("checkpoint contains BC-B state but BC-B is disabled")
            return
        if not isinstance(bc_b_state, Mapping):
            raise ValueError("checkpoint BC-B state is missing")
        BehaviorCloningRegularizer.validate_state_dict(
            bc_b_state.get("regularizer"),
            max_samples_per_building=self._bc_b.max_samples_per_building,
        )
        if not isinstance(bc_b_state.get("pretraining_complete"), bool):
            raise ValueError("checkpoint BC-B pretraining state is invalid")
        actor_step = bc_b_state.get("actor_training_step")
        if (
            isinstance(actor_step, bool)
            or not isinstance(actor_step, int)
            or actor_step < 0
        ):
            raise ValueError("checkpoint BC-B actor training step is invalid")

    @staticmethod
    def _validate_rng_state(state: Any) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint rng_state is invalid")
        try:
            python_rng = random.Random()
            python_rng.setstate(state["python"])
            numpy_rng = np.random.RandomState()
            numpy_rng.set_state(state["numpy"])
            torch_rng = torch.Generator(device="cpu")
            torch_rng.set_state(state["torch"].cpu())
            cuda_state = state["torch_cuda"]
            if cuda_state is not None and not isinstance(cuda_state, list):
                raise TypeError("CUDA RNG state must be a list or None")
        except Exception as exc:
            raise ValueError("checkpoint rng_state is invalid") from exc

    @staticmethod
    def _capture_cuda_rng_state() -> Optional[List[torch.Tensor]]:
        if not torch.cuda.is_available():
            return None
        return [state.clone() for state in torch.cuda.get_rng_state_all()]

    @staticmethod
    def _restore_cuda_rng_state(state: Optional[List[torch.Tensor]]) -> None:
        if state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state])

    def _move_optimizer_state_to_device(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(self.device)

    def _validate_hyperparameters(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.n_step_gamma <= 1.0:
            raise ValueError("gamma values must be in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.batch_size <= 0 or self.buffer_capacity < self.batch_size:
            raise ValueError("buffer_capacity must be at least batch_size")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.n_step_returns <= 0:
            raise ValueError("n_step_returns must be positive")
        if not 0.0 <= self.critic_team_reward_mix <= 1.0:
            raise ValueError("critic_team_reward_mix must be in [0, 1]")
        if self.actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive")
        if self.sigma < 0.0 or not 0.0 <= self.min_sigma <= self.sigma:
            raise ValueError("exploration sigma values are invalid")
        if not 0.0 < self.sigma_decay <= 1.0:
            raise ValueError("sigma_decay must be in (0, 1]")
        if self.target_policy_noise < 0.0 or self.target_policy_noise_clip < 0.0:
            raise ValueError("target policy noise values must be non-negative")
        if self.critic_target_clip_abs < 0.0:
            raise ValueError("critic_target_clip_abs must be non-negative")
        if self.reward_normalization_clip <= 0.0:
            raise ValueError("reward_normalization_clip must be positive")
        if self.noise_clip is not None and self.noise_clip < 0.0:
            raise ValueError("noise_clip must be non-negative")
        if self.random_exploration_steps < 0:
            raise ValueError("random_exploration_steps must be non-negative")
        if self.end_initial_exploration_time_step < 0:
            raise ValueError(
                "end_initial_exploration_time_step must be non-negative"
            )
        if self.storage_exploration_noise_multiplier < 0.0:
            raise ValueError("storage exploration multiplier must be non-negative")
        if self.ev_negative_exploration_noise_multiplier < 0.0:
            raise ValueError("EV exploration multiplier must be non-negative")
        if not 0.0 <= self.deferrable_trigger_threshold <= 1.0:
            raise ValueError("deferrable_trigger_threshold must be in [0, 1]")
        if not 0.0 <= self.deferrable_on_probability <= 1.0:
            raise ValueError("deferrable_on_probability must be in [0, 1]")
        if self.residual_policy_enabled and self.warm_start_policy_name is None:
            raise ValueError(
                "residual_policy_enabled requires warm_start_policy_name"
            )
        if self.critic_action_input_mode != "final":
            raise ValueError("critic_action_input_mode must be 'final'")
        for label, value in (
            ("residual_action_scale", self.residual_action_scale),
            ("residual_action_final_scale", self.residual_action_final_scale),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if (
            self.residual_action_scale_start_step < 0
            or self.residual_action_scale_growth_steps < 0
        ):
            raise ValueError("residual schedule steps must be non-negative")
        for value in (
            self.residual_storage_action_scale_multiplier,
            self.residual_ev_action_scale_multiplier,
            self.residual_deferrable_action_scale_multiplier,
            self.bc_a_ev_multiplier,
            self.bc_a_storage_multiplier,
            self.bc_a_deferrable_multiplier,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("action type multipliers must be non-negative")
        if self.bc_a_teacher not in {"warm_start", "replay_action", "external"}:
            raise ValueError("BC-A teacher is invalid")
        if self.bc_a_enabled and self.bc_a_teacher == "external":
            raise ValueError(
                "BC-A teacher='external' is not supported by "
                "AgentTransformerMATD3"
            )
        if (
            self.bc_a_enabled
            and self.bc_a_teacher == "warm_start"
            and self.warm_start_policy_name is None
        ):
            raise ValueError("BC-A warm_start teacher requires warm_start_policy_name")
        if not 0.0 <= self.bc_a_min_weight <= self.bc_a_weight:
            raise ValueError("BC-A min_weight must be between zero and weight")
        if min(
            self.bc_a_decay_start_step,
            self.bc_a_decay_steps,
            self.bc_a_extra_updates,
            self.bc_a_extra_update_start_step,
            self.bc_a_extra_update_end_step,
            self.bc_a_offline_pretrain_steps,
        ) < 0:
            raise ValueError("BC-A schedule values must be non-negative")
        if self._bc_b is not None:
            if (
                self._bc_b.demonstration_episodes <= 0
                or self._bc_b.max_samples_per_building <= 0
                or self._bc_b.pretraining_epochs <= 0
                or self._bc_b.batch_size <= 0
            ):
                raise ValueError("BC-B collection and training sizes must be positive")
            if not 0.0 <= self._bc_b.min_weight <= self._bc_b.weight:
                raise ValueError("BC-B min_weight must be between zero and weight")
            if self._bc_b.decay_start_step < 0 or self._bc_b.decay_steps < 0:
                raise ValueError("BC-B schedule values must be non-negative")
            if any(
                not np.isfinite(value) or value < 0.0
                for value in (
                    self._bc_b.ev_multiplier,
                    self._bc_b.storage_multiplier,
                )
            ):
                raise ValueError("BC-B action type multipliers must be non-negative")

    def _validate_transition_vectors(self, transition: Dict[str, Any]) -> None:
        for index, state in enumerate(self._per_building):
            for field in ("observations", "next_observations"):
                values = transition[field][index]
                if values.ndim != 1 or not np.isfinite(values).all():
                    raise ValueError(
                        f"{field}[{index}] must be a finite one-dimensional vector"
                    )
            actions = transition["actions"][index]
            if actions.shape != (state.layout.n_ca,):
                raise ValueError(
                    f"actions[{index}] width is {actions.size}; "
                    f"expected {state.layout.n_ca}"
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"actions[{index}] must contain finite values")
        if not np.isfinite(transition["rewards"]).all():
            raise ValueError("rewards must contain finite values")

    def _validate_action_vector_group(
        self,
        label: str,
        values: Sequence[Any],
    ) -> None:
        self._validate_vector_count(label, values)
        for index, (value, state) in enumerate(zip(values, self._per_building)):
            vector = np.asarray(value, dtype=np.float32).reshape(-1)
            if vector.shape != (state.layout.n_ca,):
                raise ValueError(
                    f"{label}[{index}] width is {vector.size}; "
                    f"expected {state.layout.n_ca}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"{label}[{index}] must contain finite values")

    def _require_attached(self) -> None:
        if not self._per_building or self.replay_buffer is None:
            raise RuntimeError("attach_environment must be called first")

    def _validate_vector_count(self, name: str, values: Sequence[Any]) -> None:
        if len(values) != len(self._per_building):
            raise ValueError(
                f"{name} has {len(values)} entries; expected {len(self._per_building)}"
            )

    def _done_vector(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.bool_).reshape(-1)
        if array.shape == (1,):
            array = np.repeat(array, len(self._per_building))
        if array.shape != (len(self._per_building),):
            raise ValueError("done flag width must match building count")
        return array

    @staticmethod
    def _copied_vectors(values: Sequence[Any]) -> Tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(value, dtype=np.float32).reshape(-1).copy()
            for value in values
        )

    @staticmethod
    def _copied_optional_vectors(
        values: Optional[Sequence[Any]],
    ) -> Optional[List[np.ndarray]]:
        if values is None:
            return None
        return [
            np.asarray(value, dtype=np.float64).reshape(-1).copy()
            for value in values
        ]

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        parsed = str(value).strip()
        return parsed or None

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        normalized = str(action_name or "").lower()
        return "electrical_storage" in normalized or normalized in {
            "battery",
            "storage",
        }

    @staticmethod
    def _is_ev_action_name(action_name: str) -> bool:
        normalized = str(action_name or "").lower()
        return (
            "electric_vehicle" in normalized
            or "charger" in normalized
            or normalized.startswith("ev_")
            or normalized in {"ev", "v2g"}
        )

    @staticmethod
    def _is_deferrable_action_name(action_name: str) -> bool:
        normalized = str(action_name or "")
        return (
            normalized.startswith("deferrable_appliance")
            or normalized.endswith("::start")
            or normalized == "start"
        )

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _normalize_spaces(
        spaces: Any,
        count: int,
        *,
        name: str = "action_space",
    ) -> List[Any]:
        if isinstance(spaces, (list, tuple)):
            if len(spaces) != count:
                raise ValueError(f"{name} count must match building count")
            return list(spaces)
        return [spaces] * count

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def _actor_modules(state: _PerBuildingState) -> nn.ModuleList:
        return nn.ModuleList((state.tokenizer, state.backbone, state.actor))

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters()
            ):
                target_parameter.mul_(1.0 - self.tau)
                target_parameter.add_(online_parameter, alpha=self.tau)

    @staticmethod
    def _mean_or_zero(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0
