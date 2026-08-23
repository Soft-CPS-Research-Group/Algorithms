from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
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
from algorithms.transformer_matd3.components import (
    CentralizedCritic,
    DeterministicActorHead,
)
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


class AgentTransformerMATD3(BaseAgent):
    """Static-layout Transformer MATD3 learner.

    Dynamic topology, persistence, price conditioning, and export belong to
    later implementation stages.
    """

    supports_dynamic_topology: ClassVar[bool] = False
    requires_final_pipeline_stage: ClassVar[bool] = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
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
        self._last_train_rewards: Optional[torch.Tensor] = None
        self._warm_start_policy: Optional[BaseAgent] = None
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
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
        self._normalize_spaces(observation_space, count, name="observation_space")
        names_key = tuple(
            (tuple(observation), tuple(actions))
            for observation, actions in zip(observation_names, action_names)
        )
        if self._attached_names is not None:
            if names_key != self._attached_names:
                raise RuntimeError(
                    "Transformer MATD3 PR 3 supports a static entity layout only"
                )
            return

        building_names = (metadata or {}).get("building_names") or ()
        layouts = [
            self._layout_builder.build(
                str(building_names[index])
                if index < len(building_names) and building_names[index]
                else f"building_{index}",
                observation_names[index],
                action_names[index],
            )
            for index in range(count)
        ]
        for index, (layout, names) in enumerate(zip(layouts, action_names)):
            self._validate_ca_order(index, layout, names)

        type_input_dims = self._community_type_input_dims(layouts)
        states = []
        for index, (layout, names, space) in enumerate(
            zip(layouts, action_names, spaces)
        ):
            low, high = self._action_bounds(index, names, space)
            states.append(
                self._build_state(
                    layout=layout,
                    action_names=tuple(names),
                    action_low=low,
                    action_high=high,
                    type_input_dims=type_input_dims,
                )
            )
        self._per_building = states
        self._layout_signature = self._build_layout_signature(layouts)
        self.replay_buffer = SignatureBucketedReplayBuffer(
            capacity=self.buffer_capacity,
            num_agents=count,
            batch_size=self.batch_size,
        )
        self._attached_names = names_key
        self._attach_warm_start_policy(
            observation_names=observation_names,
            action_names=action_names,
            action_space=spaces,
            observation_space=self._normalize_spaces(observation_space, count),
            metadata=metadata,
        )
        self._attach_local_action_safety(
            observation_names=observation_names,
            action_names=action_names,
            metadata=metadata,
        )
        self._attach_bc_b_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=spaces,
            observation_space=self._normalize_spaces(
                observation_space, count, name="observation_space"
            ),
            metadata=metadata,
        )

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
    ) -> None:
        del encoded_observations, encoded_next_observations
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

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        del context
        self._require_attached()
        self._validate_vector_count("predict observations", observations)
        if self._in_bc_b_demonstration_phase():
            assert self._bc_b is not None
            teacher_observations = (
                self._latest_raw_observations
                if self._latest_raw_observations is not None
                else observations
            )
            return self._bc_b.compute_teacher_actions(teacher_observations)
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
        transition = {
            "observations": self._copied_vectors(observations),
            "actions": self._copied_vectors(actions),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1).copy(),
            "next_observations": self._copied_vectors(next_observations),
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
        }
        self._validate_transition_vectors(transition)
        self._store_transition(transition)
        if self._bc_b is not None and self._bc_b_pretraining_complete:
            self._bc_b_actor_training_step += 1

        assert self.replay_buffer is not None
        assert self._layout_signature is not None
        bucket_size = self.replay_buffer.bucket_size(self._layout_signature)
        if bucket_size < self.batch_size:
            self._record_skip("replay_underfull", bucket_size)
            return
        if not initial_exploration_done:
            self._record_skip("initial_exploration", bucket_size)
            return
        if not update_step:
            self._record_skip("schedule", bucket_size)
            return
        batch = self.replay_buffer.sample(self._layout_signature, self.batch_size)
        self._learn(
            batch,
            update_target_step=update_target_step,
            global_learning_step=global_learning_step,
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

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del output_dir, context
        raise NotImplementedError("Transformer MATD3 export is outside PR 3 scope")

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.random_exploration_steps

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = dict(self._latest_training_metrics)
        self._latest_training_metrics = {}
        return metrics

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
        metrics.update(self._bc_b_metrics())
        return metrics

    def _learn(
        self,
        batch: ReplayBatch,
        *,
        update_target_step: bool,
        global_learning_step: int,
    ) -> None:
        started = time.perf_counter()
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
        extra_losses, extra_grad_norms = self._run_bc_a_extra_updates(
            observations=observations,
            behavior_actions=behavior_actions,
            cloning_actions=cloning_actions,
            effective_weight=bc_weight,
            global_learning_step=global_learning_step,
        )
        bc_b_losses, bc_b_grad_norms = self._run_bc_b_auxiliary_updates(
            global_learning_step=self._bc_b_actor_training_step
        )
        if actor_update_due and update_target_step:
            for state in self._per_building:
                self._soft_update(state.tokenizer, state.tokenizer_target)
                self._soft_update(state.backbone, state.backbone_target)
                self._soft_update(state.actor, state.actor_target)
                self._soft_update(state.critic_1, state.critic_1_target)
                self._soft_update(state.critic_2, state.critic_2_target)

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
            f"{_METRIC_PREFIX}actor_update_performed": float(actor_update_due),
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

    @property
    def _layouts(self) -> List[BuildingTokenLayout]:
        return [state.layout for state in self._per_building]

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
        if not self.bc_a_enabled or self.bc_a_weight <= 0.0:
            return 0.0
        if self.bc_a_decay_steps <= 0:
            return self.bc_a_weight
        if global_learning_step <= self.bc_a_decay_start_step:
            return self.bc_a_weight
        progress = min(
            max(
                (global_learning_step - self.bc_a_decay_start_step)
                / self.bc_a_decay_steps,
                0.0,
            ),
            1.0,
        )
        return self.bc_a_weight + (
            self.bc_a_min_weight - self.bc_a_weight
        ) * progress

    def _actor_behavior_cloning_loss(
        self,
        index: int,
        predicted_action: torch.Tensor,
        cloning_action: torch.Tensor,
        *,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target = self._reachable_behavior_cloning_target(
            index,
            cloning_action.detach(),
            base_action=base_action,
        )
        predicted = self._normalize_action(index, predicted_action)
        normalized_target = self._normalize_action(index, target)
        weights = self._bc_a_action_weights(index, predicted)
        return (
            (predicted - normalized_target).square() * weights
        ).sum() / weights.expand_as(predicted).sum().clamp_min(1.0)

    def _actor_behavior_cloning_type_losses(
        self,
        index: int,
        predicted_action: torch.Tensor,
        cloning_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        error = (
            self._normalize_action(index, predicted_action)
            - self._normalize_action(index, cloning_action.detach())
        ).square()
        result: Dict[str, torch.Tensor] = {}
        predicates = {
            "ev": self._is_ev_action_name,
            "storage": self._is_storage_action_name,
            "deferrable": self._is_deferrable_action_name,
        }
        known = torch.zeros(error.shape[-1], dtype=torch.bool, device=error.device)
        for label, predicate in predicates.items():
            mask = torch.as_tensor(
                [
                    predicate(name)
                    for name in self._per_building[index].action_names
                ],
                dtype=torch.bool,
                device=error.device,
            )
            known |= mask
            result[label] = (
                error[..., mask].mean() if mask.any() else error.new_tensor(0.0)
            )
        result["other"] = (
            error[..., ~known].mean() if (~known).any() else error.new_tensor(0.0)
        )
        return result

    def _reachable_behavior_cloning_target(
        self,
        index: int,
        cloning_action: torch.Tensor,
        *,
        base_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            not self.bc_a_clip_target_to_residual_authority
            or not self.residual_policy_enabled
            or base_action is None
        ):
            return cloning_action
        base = base_action.detach().to(cloning_action)
        if base.shape != cloning_action.shape:
            raise ValueError("BC-A base and cloning action shapes must match")
        state = self._per_building[index]
        authority = self._residual_action_effective_scale() * (
            self._residual_action_scale_mask(index, cloning_action)
        )
        maximum_delta = 0.5 * (state.action_high - state.action_low) * authority
        return torch.maximum(
            torch.minimum(cloning_action, base + maximum_delta),
            base - maximum_delta,
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
        values = []
        for action_name in self._per_building[index].action_names:
            multiplier = 1.0
            if self._is_ev_action_name(action_name):
                multiplier *= self.bc_a_ev_multiplier
            if self._is_storage_action_name(action_name):
                multiplier *= self.bc_a_storage_multiplier
            if self._is_deferrable_action_name(action_name):
                multiplier *= self.bc_a_deferrable_multiplier
            values.append(multiplier)
        return torch.as_tensor(values, dtype=like.dtype, device=like.device).view(1, -1)

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
        if (
            effective_weight <= 0.0
            or cloning_actions is None
            or count <= 0
            or (
                update_count is None
                and global_learning_step < self.bc_a_extra_update_start_step
            )
            or (
                update_count is None
                and self.bc_a_extra_update_end_step > 0
                and global_learning_step > self.bc_a_extra_update_end_step
            )
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
            "event=matd3_bc_b_pretraining_start buildings={}",
            len(self._per_building),
        )
        prepared_groups: List[List[Tuple[Demonstration, ...]]] = []
        incompatible_samples = 0
        missing_buildings: List[str] = []
        for building_idx, state in enumerate(self._per_building):
            grouped = self._bc_b.demonstrations_for_building_by_signature(
                building_idx
            )
            usable_groups: List[Tuple[Demonstration, ...]] = []
            for demonstrations in grouped.values():
                layout = demonstrations[0].layout
                if not self._bc_b_layout_is_compatible(state, layout):
                    incompatible_samples += len(demonstrations)
                    continue
                usable_groups.append(demonstrations)
            prepared_groups.append(usable_groups)
            if not usable_groups:
                missing_buildings.append(state.building_id)
        self._bc_b.set_incompatible_demonstration_samples(incompatible_samples)
        if missing_buildings:
            raise RuntimeError(
                "Behavior-cloning pretraining has zero usable demonstrations for "
                f"building(s): {', '.join(missing_buildings)}."
            )

        total_batches = 0
        metrics: Dict[str, float] = {}
        for state, groups in zip(self._per_building, prepared_groups):
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
        self._bc_b.set_pretraining_epochs(self._bc_b.pretraining_epochs)
        metrics[f"{_METRIC_PREFIX}behavior_cloning_pretraining_batches"] = float(
            total_batches
        )
        self._latest_training_metrics.update(self._bc_b_metrics())
        self._latest_training_metrics.update(metrics)
        logger.info(
            "event=matd3_bc_b_pretraining_complete buildings={} trained_batches={}",
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

    def _push_oldest_n_step(self, *, force: bool) -> None:
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
        self._push_transition(
            {
                "observations": first["observations"],
                "actions": first["actions"],
                "rewards": rewards,
                "next_observations": last["next_observations"],
                "terminated": last["terminated"],
                "truncated": last["truncated"],
                "behavior_actions": first.get("behavior_actions"),
                "next_behavior_actions": last.get("next_behavior_actions"),
                "cloning_actions": first.get("cloning_actions"),
            }
        )
        self._n_step_queue.popleft()

    def _push_transition(self, transition: Dict[str, Any]) -> None:
        assert self.replay_buffer is not None
        assert self._layout_signature is not None
        self.replay_buffer.push(
            encoded_obs=transition["observations"],
            next_encoded_obs=transition["next_observations"],
            actions=transition["actions"],
            reward=transition["rewards"],
            terminated=transition["terminated"],
            truncated=transition["truncated"],
            layout_signature=self._layout_signature,
            behavior_actions=transition.get("behavior_actions"),
            next_behavior_actions=transition.get("next_behavior_actions"),
            cloning_actions=transition.get("cloning_actions"),
        )

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
                    (segment.family, segment.type_name, segment.instance_id)
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
