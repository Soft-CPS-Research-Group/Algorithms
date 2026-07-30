from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import ActionScaledActor, _log_torch_runtime, _select_torch_device
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
    preserve_teacher_service_with_storage_fallback,
    replace_service_actions_with_teacher,
)
from algorithms.utils.networks import GaussianActor, ValueNetwork
from algorithms.utils.price_multiplier_adapter import (
    ForecastMode,
    PriceMultiplierObservationAdapter,
    normalize_price_multiplier_context,
    price_feature_bounds_from_metadata,
    price_observation_names_from_metadata,
)
from utils.artifact_config_builder import build_auto_artifact_config


class _PPOBase(BaseAgent):
    """Shared implementation for IPPO and MAPPO.

    Actors are decentralized and consume per-agent observations. The value
    function input is selected by subclasses:

    - IPPO: local observation per agent.
    - MAPPO: concatenated global observation per agent value network.
    """

    value_scope = "local"
    metric_prefix = "PPO"

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        algorithm_cfg = self.config["algorithm"]
        hyperparams = algorithm_cfg.get("hyperparameters", {})
        exploration_cfg = (algorithm_cfg.get("exploration") or {}).get("params", {})
        network_cfg = algorithm_cfg["networks"]
        rollout_cfg = algorithm_cfg.get("replay_buffer", {})
        training_cfg = self.config.get("training", {})
        tracking_cfg = self.config.get("tracking", {})
        checkpoint_cfg = self.config.get("checkpointing", {})
        topology = self.config.get("topology", {})

        self.require_cuda = bool(exploration_cfg.get("require_cuda", hyperparams.get("require_cuda", False)))
        self.device = _select_torch_device(require_cuda=self.require_cuda)
        _log_torch_runtime(self.device)
        torch.backends.cudnn.benchmark = self.device.type == "cuda"

        self.gamma = float(hyperparams.get("gamma", exploration_cfg.get("gamma", 0.99)))
        self.gae_lambda = float(np.clip(float(exploration_cfg.get("gae_lambda", 0.95)), 0.0, 1.0))
        self.clip_ratio = float(max(exploration_cfg.get("clip_ratio", 0.2), 1.0e-6))
        self.entropy_coef = float(max(exploration_cfg.get("entropy_coef", 0.01), 0.0))
        self.value_loss_coef = float(max(exploration_cfg.get("value_loss_coef", 0.5), 0.0))
        self.max_grad_norm = float(max(exploration_cfg.get("max_grad_norm", 0.5), 0.0))
        self.ppo_epochs = max(1, int(exploration_cfg.get("ppo_epochs", 4) or 4))
        self.rollout_length = max(
            1,
            int(exploration_cfg.get("rollout_length", rollout_cfg.get("capacity", 256)) or 256),
        )
        self.minibatch_size = max(
            1,
            int(exploration_cfg.get("minibatch_size", rollout_cfg.get("batch_size", 64)) or 64),
        )
        self.initial_log_std = float(exploration_cfg.get("initial_log_std", -0.5))
        self.min_log_std = float(exploration_cfg.get("min_log_std", -5.0))
        self.max_log_std = float(exploration_cfg.get("max_log_std", 1.0))
        self.end_initial_exploration_time_step = max(
            0,
            int(exploration_cfg.get("end_initial_exploration_time_step", 0) or 0),
        )
        self.random_exploration_steps = max(
            0,
            int(exploration_cfg.get("random_exploration_steps", self.end_initial_exploration_time_step) or 0),
        )
        self.warm_start_policy_name = self._optional_string(exploration_cfg.get("warm_start_policy"))
        self.initial_exploration_strategy = str(
            exploration_cfg.get(
                "initial_exploration_strategy",
                "policy" if self.warm_start_policy_name else "uniform_full_range",
            )
            or "uniform_full_range"
        ).strip().lower()
        if self.initial_exploration_strategy not in {"uniform_full_range", "policy"}:
            raise ValueError("PPO initial_exploration_strategy must be 'uniform_full_range' or 'policy'.")
        if self.initial_exploration_strategy == "policy" and not self.warm_start_policy_name:
            raise ValueError("PPO initial_exploration_strategy='policy' requires warm_start_policy.")
        self.warm_start_policy_deterministic = bool(exploration_cfg.get("warm_start_policy_deterministic", True))
        self.warm_start_policy_noise_scale = max(
            0.0,
            float(exploration_cfg.get("warm_start_policy_noise_scale", 0.0) or 0.0),
        )
        self.warm_start_policy_phaseout_steps = max(
            0,
            int(exploration_cfg.get("warm_start_policy_phaseout_steps", 0) or 0),
        )
        self.warm_start_policy_phaseout_mode = str(
            exploration_cfg.get("warm_start_policy_phaseout_mode", "probability") or "probability"
        ).strip().lower()
        if self.warm_start_policy_phaseout_mode not in {"probability", "blend"}:
            raise ValueError("PPO warm_start_policy_phaseout_mode must be 'probability' or 'blend'.")
        self.actor_behavior_cloning_weight = max(
            0.0,
            float(exploration_cfg.get("actor_behavior_cloning_weight", 0.0) or 0.0),
        )
        self.actor_ev_behavior_cloning_multiplier = max(
            0.0,
            float(exploration_cfg.get("actor_ev_behavior_cloning_multiplier", 1.0) or 0.0),
        )
        self.actor_ev_behavior_cloning_positive_target_weight = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_ev_behavior_cloning_positive_target_weight",
                    0.0,
                )
                or 0.0
            ),
        )
        self.actor_ev_behavior_cloning_positive_target_power = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_ev_behavior_cloning_positive_target_power",
                    1.0,
                )
                or 0.0
            ),
        )
        self.actor_ev_behavior_cloning_zero_target_weight = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_ev_behavior_cloning_zero_target_weight",
                    0.0,
                )
                or 0.0
            ),
        )
        self.actor_ev_behavior_cloning_zero_target_threshold = float(
            np.clip(
                float(
                    exploration_cfg.get(
                        "actor_ev_behavior_cloning_zero_target_threshold",
                        0.05,
                    )
                    or 0.05
                ),
                0.0,
                1.0,
            )
        )
        self.actor_storage_behavior_cloning_multiplier = max(
            0.0,
            float(exploration_cfg.get("actor_storage_behavior_cloning_multiplier", 1.0) or 0.0),
        )
        self.actor_deferrable_behavior_cloning_multiplier = max(
            0.0,
            float(exploration_cfg.get("actor_deferrable_behavior_cloning_multiplier", 1.0) or 0.0),
        )
        self.actor_deferrable_behavior_cloning_positive_target_weight = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_deferrable_behavior_cloning_positive_target_weight",
                    0.0,
                )
                or 0.0
            ),
        )
        self.actor_deferrable_behavior_cloning_positive_target_power = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_deferrable_behavior_cloning_positive_target_power",
                    1.0,
                )
                or 0.0
            ),
        )
        self.actor_other_behavior_cloning_multiplier = max(
            0.0,
            float(exploration_cfg.get("actor_other_behavior_cloning_multiplier", 1.0) or 0.0),
        )
        self.local_action_safety_enabled = bool(
            exploration_cfg.get("local_action_safety_enabled", False)
        )
        self.local_action_safety_fail_on_infeasible = bool(
            exploration_cfg.get("local_action_safety_fail_on_infeasible", False)
        )
        self.local_action_safety_headroom_reserve_kw = max(
            0.0,
            float(exploration_cfg.get("local_action_safety_headroom_reserve_kw", 0.0) or 0.0),
        )
        self.local_action_safety_allow_discretionary_deferrable_start = bool(
            exploration_cfg.get(
                "local_action_safety_allow_discretionary_deferrable_start",
                False,
            )
        )
        self.local_action_safety_runtime_only_export = bool(
            exploration_cfg.get("local_action_safety_runtime_only_export", False)
        )
        self.local_action_safety_protect_ev_minimum = bool(
            exploration_cfg.get("local_action_safety_protect_ev_minimum", True)
        )
        self.local_action_safety_ev_minimum_mode = str(
            exploration_cfg.get("local_action_safety_ev_minimum_mode", "average")
            or "average"
        )
        self.local_action_safety_protect_ev_service_target = bool(
            exploration_cfg.get(
                "local_action_safety_protect_ev_service_target",
                False,
            )
        )
        self.local_action_safety_service_teacher_enabled = bool(
            exploration_cfg.get("local_action_safety_service_teacher_enabled", False)
        )
        self.local_action_safety_service_teacher_eval_enabled = bool(
            exploration_cfg.get(
                "local_action_safety_service_teacher_eval_enabled",
                self.local_action_safety_service_teacher_enabled,
            )
        )
        self.local_price_conditioning_enabled = bool(
            exploration_cfg.get("local_price_conditioning_enabled", False)
        )
        self.local_price_forecast_mode = ForecastMode(
            str(
                exploration_cfg.get(
                    "local_price_forecast_mode",
                    ForecastMode.REAL_UNMODIFIED.value,
                )
            )
        )
        self.actor_behavior_cloning_min_weight = max(
            0.0,
            float(exploration_cfg.get("actor_behavior_cloning_min_weight", 0.0) or 0.0),
        )
        self.actor_behavior_cloning_decay_start_step = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_decay_start_step", 0) or 0),
        )
        self.actor_behavior_cloning_decay_steps = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_decay_steps", 0) or 0),
        )
        self.actor_behavior_cloning_extra_updates = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_extra_updates", 0) or 0),
        )
        self.actor_behavior_cloning_extra_update_start_step = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_extra_update_start_step", 0) or 0),
        )
        self.actor_behavior_cloning_extra_update_end_step = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_extra_update_end_step", 0) or 0),
        )
        self.actor_behavior_cloning_replay_capacity = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_replay_capacity", 0) or 0),
        )
        self.actor_action_l2_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_action_l2_penalty", 0.0) or 0.0),
        )
        self.actor_storage_action_l2_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_storage_action_l2_penalty", 0.0) or 0.0),
        )
        self.actor_ev_v2g_action_l2_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_ev_v2g_action_l2_penalty", 0.0) or 0.0),
        )
        self.actor_action_saturation_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_action_saturation_penalty", 0.0) or 0.0),
        )
        self.actor_action_saturation_threshold = float(
            np.clip(float(exploration_cfg.get("actor_action_saturation_threshold", 0.85) or 0.85), 0.0, 1.0)
        )
        self.train_during_initial_exploration = bool(
            exploration_cfg.get("train_during_initial_exploration", False)
        )
        self.initial_exploration_training_start_step = max(
            0,
            int(exploration_cfg.get("initial_exploration_training_start_step", 0) or 0),
        )
        self.target_kl = exploration_cfg.get("target_kl")
        self.target_kl = None if self.target_kl is None else float(max(self.target_kl, 0.0))
        self.agent_update_order = str(
            exploration_cfg.get("agent_update_order", "fixed") or "fixed"
        ).strip().lower()
        if self.agent_update_order not in {"fixed", "random"}:
            raise ValueError("PPO agent_update_order must be 'fixed' or 'random'.")

        try:
            self.mlflow_step_sample_interval = int(tracking_cfg.get("mlflow_step_sample_interval", 10) or 10)
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10
        self.mlflow_step_sample_interval = max(1, self.mlflow_step_sample_interval)
        self.training_diagnostics_enabled = bool(tracking_cfg.get("training_diagnostics_enabled", True))

        self.seed = int(training_cfg.get("seed", 22))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.checkpoint_artifact = checkpoint_cfg.get("checkpoint_artifact", "latest_checkpoint.pth")
        self.reset_replay_buffer = bool(checkpoint_cfg.get("reset_replay_buffer", False))
        self.fine_tune = bool(checkpoint_cfg.get("fine_tune", False))
        restore_optimizers = checkpoint_cfg.get("restore_optimizers")
        restore_replay_buffer = checkpoint_cfg.get("restore_replay_buffer")
        self.restore_optimizers = (
            not self.fine_tune
            if restore_optimizers is None
            else bool(restore_optimizers)
        )
        self.restore_replay_buffer = (
            not self.reset_replay_buffer
            if restore_replay_buffer is None
            else bool(restore_replay_buffer)
        )
        self.restore_exploration_state = bool(
            checkpoint_cfg.get("restore_exploration_state", True)
        )

        self.num_agents = topology.get("num_agents") or hyperparams.get("num_agents")
        self.observation_dimension = topology.get("observation_dimensions") or hyperparams.get("observation_dimensions")
        self.action_dimension = topology.get("action_dimensions") or hyperparams.get("action_dimensions")
        if self.num_agents is None or self.observation_dimension is None or self.action_dimension is None:
            raise ValueError(
                f"Topology information (num_agents / observation_dimensions / action_dimensions) is required for {self.metric_prefix}."
            )

        self.action_low, self.action_high = self._default_action_bounds()
        self.action_names: List[List[str]] = [[] for _ in range(int(self.num_agents))]
        self.observation_names: List[List[str]] = [[] for _ in range(int(self.num_agents))]
        self.observation_space: List[Any] = []
        self.exploration_step = 0
        self._latest_training_metrics: Dict[str, float] = {}
        self.rollout: List[Dict[str, Any]] = []
        self.behavior_cloning_replay = deque(
            maxlen=(
                self.actor_behavior_cloning_replay_capacity
                if self.actor_behavior_cloning_replay_capacity > 0
                else None
            )
        )
        self._warm_start_policy = None
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_observations: Optional[List[np.ndarray]] = None
        self._episode_clock_is_explicit = False
        self._episode_schedule_step: Optional[int] = None
        self._last_warm_start_policy_actions: Optional[List[List[float]]] = None
        self._last_warm_start_phaseout_probability = 0.0
        self._last_warm_start_phaseout_used = False
        self._last_policy_action_eligible = [True for _ in range(int(self.num_agents))]
        # Ephemeral, one-step cache populated by ``predict`` and consumed by
        # ``update``. PPO's behaviour probability must be the probability of
        # the exact latent sample that produced the environment action. It is
        # unsafe to reconstruct it later from a clipped/scaled action after
        # the actor may have changed, and it will be impossible once action
        # projection is introduced.
        self._last_policy_samples: Optional[List[Dict[str, Any]]] = None
        self._local_action_safety_adapters: List[CityLearnLocalSafetyAdapter] = []
        self._last_local_action_projections: List[Any] = []
        self._last_service_teacher_applied = False
        self._local_price_adapters: List[PriceMultiplierObservationAdapter] = []
        self._last_local_price_diagnostics: List[Any] = []
        self._last_local_price_context_non_neutral = False
        if self.local_action_safety_enabled:
            self.requires_raw_observation_context = True

        actor_layers = network_cfg["actor"]["layers"]
        value_layers = network_cfg["critic"]["layers"]
        self.lr_actor = float(network_cfg["actor"]["lr"])
        self.lr_value = float(network_cfg["critic"]["lr"])

        self.actors = [
            GaussianActor(
                self.observation_dimension[agent_idx],
                self.action_dimension[agent_idx],
                self.seed + agent_idx,
                actor_layers,
                initial_log_std=self.initial_log_std,
                min_log_std=self.min_log_std,
                max_log_std=self.max_log_std,
            ).to(self.device)
            for agent_idx in range(int(self.num_agents))
        ]
        self.value_nets = [
            ValueNetwork(
                self._value_input_dimension(agent_idx),
                self.seed + 1009 + agent_idx,
                value_layers,
            ).to(self.device)
            for agent_idx in range(int(self.num_agents))
        ]
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors
        ]
        self.value_optimizers = [
            torch.optim.Adam(value_net.parameters(), lr=self.lr_value) for value_net in self.value_nets
        ]

        logger.info("{} initialization complete on {}.", self.metric_prefix, self.device)

    def _default_action_bounds(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        lows = [
            np.full(int(self.action_dimension[agent_idx]), -1.0, dtype=np.float32)
            for agent_idx in range(int(self.num_agents))
        ]
        highs = [
            np.full(int(self.action_dimension[agent_idx]), 1.0, dtype=np.float32)
            for agent_idx in range(int(self.num_agents))
        ]
        return lows, highs

    def _value_input_dimension(self, agent_idx: int) -> int:
        if self.value_scope == "global":
            return int(sum(self.observation_dimension))
        return int(self.observation_dimension[agent_idx])

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ = metadata
        self.observation_names = [list(names) for names in observation_names]
        self.action_names = [list(names) for names in action_names]
        self.observation_space = list(observation_space)
        lows, highs = self._default_action_bounds()
        for agent_idx, space in enumerate(action_space[: int(self.num_agents)]):
            if not hasattr(space, "low") or not hasattr(space, "high"):
                continue
            low = np.asarray(space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(space.high, dtype=np.float32).reshape(-1)
            if low.shape[0] == int(self.action_dimension[agent_idx]) and high.shape[0] == int(self.action_dimension[agent_idx]):
                lows[agent_idx] = low
                highs[agent_idx] = high
        self.action_low = lows
        self.action_high = highs
        self._initialize_warm_start_policy(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self._local_action_safety_adapters = []
        self._last_local_action_projections = []
        if self.local_action_safety_enabled:
            for agent_idx in range(int(self.num_agents)):
                self._local_action_safety_adapters.append(
                    CityLearnLocalSafetyAdapter(
                        observation_names=observation_names[agent_idx],
                        action_names=action_names[agent_idx],
                        action_low=self._action_low_for_agent(agent_idx),
                        action_high=self._action_high_for_agent(agent_idx),
                        metadata=metadata,
                        config=CityLearnSafetyConfig(
                            fail_on_infeasible=self.local_action_safety_fail_on_infeasible,
                            protect_ev_minimum=self.local_action_safety_protect_ev_minimum,
                            ev_minimum_mode=self.local_action_safety_ev_minimum_mode,
                            protect_ev_service_target=(
                                self.local_action_safety_protect_ev_service_target
                            ),
                            headroom_reserve_kw=self.local_action_safety_headroom_reserve_kw,
                            allow_discretionary_deferrable_start=(
                                self.local_action_safety_allow_discretionary_deferrable_start
                            ),
                        ),
                    )
                )

        self._local_price_adapters = []
        self._last_local_price_diagnostics = []
        if self.local_price_conditioning_enabled:
            for agent_idx in range(int(self.num_agents)):
                feature_low, feature_high = price_feature_bounds_from_metadata(
                    metadata=metadata,
                    agent_index=agent_idx,
                )
                actor_observation_names = price_observation_names_from_metadata(
                    metadata=metadata,
                    agent_index=agent_idx,
                    fallback_observation_names=observation_names[agent_idx],
                )
                self._local_price_adapters.append(
                    PriceMultiplierObservationAdapter(
                        observation_names=actor_observation_names,
                        feature_low=feature_low,
                        feature_high=feature_high,
                        forecast_mode=self.local_price_forecast_mode,
                    )
                )

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        return text

    def _initialize_warm_start_policy(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if not self.warm_start_policy_name:
            return

        from algorithms.agents.baseline_policies import (  # Local import avoids registry cycles.
            NormalNoBatteryPolicy,
            NormalPolicy,
            RBCBasicPolicy,
            RBCSmartLocalPolicy,
            RBCSmartPolicy,
            RandomPolicy,
        )
        from algorithms.agents.rbc_agent import RuleBasedPolicy
        from algorithms.agents.oracle_replay_policy import FixedServiceOracleReplayPolicy
        from algorithms.agents.total_home_oracle_replay_policy import (
            TotalHomeOracleReplayPolicy,
        )
        from algorithms.agents.total_oracle_replay_policy import (
            TotalOracleReplayPolicy,
        )

        policy_registry = {
            "RuleBasedPolicy": RuleBasedPolicy,
            "RandomPolicy": RandomPolicy,
            "NormalNoBatteryPolicy": NormalNoBatteryPolicy,
            "NormalPolicy": NormalPolicy,
            "RBCBasicPolicy": RBCBasicPolicy,
            "RBCSmartLocalPolicy": RBCSmartLocalPolicy,
            "RBCSmartPolicy": RBCSmartPolicy,
            "FixedServiceOracleReplayPolicy": FixedServiceOracleReplayPolicy,
            "TotalHomeOracleReplayPolicy": TotalHomeOracleReplayPolicy,
            "TotalOracleReplayPolicy": TotalOracleReplayPolicy,
        }
        policy_cls = policy_registry.get(str(self.warm_start_policy_name))
        if policy_cls is None:
            supported = ", ".join(sorted(policy_registry))
            raise ValueError(
                f"Unsupported PPO warm_start_policy '{self.warm_start_policy_name}'. "
                f"Supported policies: {supported}."
            )

        exploration_cfg = self.config["algorithm"]["exploration"]["params"]
        policy_hyperparams = exploration_cfg.get("warm_start_policy_hyperparameters") or {}
        if not isinstance(policy_hyperparams, dict):
            raise ValueError("PPO warm_start_policy_hyperparameters must be an object when provided.")

        policy_config = deepcopy(self.config)
        policy_config["algorithm"] = {
            "name": str(self.warm_start_policy_name),
            "hyperparameters": dict(policy_hyperparams),
        }
        self._warm_start_policy = policy_cls(policy_config)
        self._warm_start_policy.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        logger.info("{} warm-start policy enabled: {}", self.metric_prefix, self.warm_start_policy_name)

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
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
        self._last_warm_start_policy_actions = None

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        del next_episode_step
        self._episode_clock_is_explicit = episode_step is not None
        self._episode_schedule_step = (
            None if episode_step is None else max(int(episode_step), 0)
        )

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.end_initial_exploration_time_step

    def _should_train_on_step(self, initial_exploration_done: bool, global_learning_step: int) -> bool:
        if initial_exploration_done:
            return True
        return self.train_during_initial_exploration and (
            global_learning_step >= self.initial_exploration_training_start_step
        )

    def predict(
        self,
        observations,
        deterministic: bool | None = False,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        observations = self._apply_local_price_context(observations, context)
        deterministic = bool(deterministic)
        self.exploration_step += 1
        self._last_service_teacher_applied = False
        self._last_warm_start_policy_actions = None
        self._last_policy_action_eligible = [False for _ in range(int(self.num_agents))]
        self._last_policy_samples = None
        if not deterministic and self.exploration_step <= self.random_exploration_steps:
            # Teacher/random actions are deliberately ineligible for the PPO
            # ratio, but the value prediction still belongs to this state and
            # is cached at action-selection time for a coherent rollout.
            self._cache_value_predictions(observations)
            if self.initial_exploration_strategy == "policy":
                actions = self._predict_warm_start_policy()
            else:
                actions = self._predict_random()
            actions = self._apply_service_teacher(actions, deterministic=deterministic)
            return self._apply_local_action_safety(actions)

        actions = self._predict_actor(observations, deterministic=deterministic)
        if not deterministic:
            actions = self._apply_warm_start_phaseout(actions)
        actions = self._apply_service_teacher(actions, deterministic=deterministic)
        return self._apply_local_action_safety(actions)

    def _apply_local_price_context(
        self,
        observations: List[Any],
        context: Any,
    ) -> List[Any]:
        self._last_local_price_diagnostics = []
        self._last_local_price_context_non_neutral = False
        if not self.local_price_conditioning_enabled:
            return observations
        if len(self._local_price_adapters) != int(self.num_agents):
            raise RuntimeError(
                "PPO local price conditioning is enabled but the environment is not attached."
            )
        parsed_context = normalize_price_multiplier_context(context)
        if parsed_context is None:
            return observations

        transformed: List[np.ndarray] = []
        for adapter, observation in zip(self._local_price_adapters, observations):
            conditioned, diagnostics = adapter.transform(observation, parsed_context)
            transformed.append(conditioned)
            self._last_local_price_diagnostics.append(diagnostics)
            self._last_local_price_context_non_neutral |= not diagnostics.neutral_noop
        return transformed

    def _apply_service_teacher(
        self,
        actions: List[List[float]],
        *,
        deterministic: bool,
    ) -> List[List[float]]:
        if not self.local_action_safety_service_teacher_enabled:
            return actions
        if deterministic and not self.local_action_safety_service_teacher_eval_enabled:
            return actions
        if self._warm_start_policy is None:
            raise RuntimeError(
                "PPO service-teacher safety requires a configured warm_start_policy."
            )
        teacher_actions = self._predict_warm_start_policy()
        self._last_service_teacher_applied = True
        return replace_service_actions_with_teacher(
            action_names=self.action_names,
            proposed_actions=actions,
            teacher_actions=teacher_actions,
        )

    def _apply_local_action_safety(
        self,
        actions: List[List[float]],
    ) -> List[List[float]]:
        if not self.local_action_safety_enabled:
            return actions
        if len(self._local_action_safety_adapters) != int(self.num_agents):
            raise RuntimeError(
                "PPO local action safety is enabled but the environment is not attached."
            )
        if self._latest_raw_observations is None or len(self._latest_raw_observations) != int(
            self.num_agents
        ):
            raise RuntimeError(
                "PPO local action safety requires raw observation context before predict."
            )

        projected: List[List[float]] = []
        projections = []
        for agent_idx, (adapter, raw_observation, proposed) in enumerate(
            zip(
                self._local_action_safety_adapters,
                self._latest_raw_observations,
                actions,
            )
        ):
            result = adapter.project(raw_observation, proposed)
            executed = list(result.executed_actions)
            if self._last_service_teacher_applied:
                executed = preserve_teacher_service_with_storage_fallback(
                    action_names=self.action_names[agent_idx],
                    teacher_merged_actions=proposed,
                    projected_actions=executed,
                )
            projected.append(executed)
            projections.append(result)
            if self._last_policy_samples is not None and agent_idx < len(
                self._last_policy_samples
            ):
                self._last_policy_samples[agent_idx]["executed_action"] = torch.as_tensor(
                    executed,
                    dtype=torch.float32,
                )
        self._last_local_action_projections = projections
        return projected

    def _predict_actor(self, observations, *, deterministic: bool) -> List[List[float]]:
        actions: List[List[float]] = []
        obs_tensors = [
            torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(-1)
            for obs in observations
        ]
        policy_samples: List[Dict[str, Any]] = []
        # Cached tensors are later inputs to an autograd-enabled log-prob
        # calculation. ``no_grad`` produces ordinary detached tensors;
        # inference tensors cannot safely be saved for backward.
        with torch.no_grad():
            for agent_idx, obs_tensor_flat in enumerate(obs_tensors):
                obs_tensor = obs_tensor_flat.view(1, -1)
                distribution = self.actors[agent_idx].distribution(obs_tensor)
                if deterministic:
                    raw_action = distribution.mean
                    normalized = torch.tanh(raw_action)
                    log_prob = None
                else:
                    raw_action = distribution.rsample()
                    normalized = torch.tanh(raw_action)
                    log_prob = self._squashed_log_prob_from_latent(
                        distribution,
                        raw_action,
                        normalized,
                    )
                scaled = self._scale_action_tensor(agent_idx, normalized)
                value_input = self._value_input_for_agent(agent_idx, obs_tensors).view(1, -1)
                value = self.value_nets[agent_idx](value_input).squeeze(-1)
                actions.append(scaled.squeeze(0).cpu().numpy().tolist())
                policy_samples.append(
                    {
                        "observation": obs_tensor_flat.detach().cpu().clone(),
                        "raw_action": raw_action.squeeze(0).detach().cpu().clone(),
                        "normalized_action": normalized.squeeze(0).detach().cpu().clone(),
                        "scaled_action": scaled.squeeze(0).detach().cpu().clone(),
                        "log_prob": (
                            log_prob.squeeze(0).detach().cpu().clone()
                            if log_prob is not None
                            else None
                        ),
                        "value": value.squeeze(0).detach().cpu().clone(),
                        "stochastic": not deterministic,
                    }
                )
        self._last_policy_samples = policy_samples
        return actions

    def _cache_value_predictions(self, observations: List[Any]) -> None:
        """Cache value estimates for non-policy behaviour actions.

        Warm-start and uniform exploration rows are never PPO-policy eligible,
        so they intentionally have no latent action or behaviour log-prob.
        Their critic target still uses the value estimate made for the state
        at action-selection time.
        """

        obs_tensors = [
            torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(-1)
            for obs in observations
        ]
        policy_samples: List[Dict[str, Any]] = []
        with torch.no_grad():
            for agent_idx, obs_tensor in enumerate(obs_tensors):
                value_input = self._value_input_for_agent(agent_idx, obs_tensors).view(1, -1)
                value = self.value_nets[agent_idx](value_input).squeeze(-1)
                policy_samples.append(
                    {
                        "observation": obs_tensor.detach().cpu().clone(),
                        "raw_action": None,
                        "normalized_action": None,
                        "scaled_action": None,
                        "log_prob": None,
                        "value": value.squeeze(0).detach().cpu().clone(),
                        "stochastic": False,
                    }
                )
        self._last_policy_samples = policy_samples

    def _predict_warm_start_policy(self) -> List[List[float]]:
        if self._warm_start_policy is None:
            return self._predict_random()
        observations = self._latest_raw_observations or self._latest_encoded_observations
        if observations is None:
            return self._predict_random()
        predict_at_step = getattr(self._warm_start_policy, "predict_at_step", None)
        if callable(predict_at_step):
            schedule_step = (
                self._episode_schedule_step
                if self._episode_clock_is_explicit
                else max(int(self.exploration_step) - 1, 0)
            )
            if schedule_step is None:
                return self._predict_random()
            actions = predict_at_step(
                observations,
                schedule_step=schedule_step,
                deterministic=self.warm_start_policy_deterministic,
            )
        else:
            actions = self._warm_start_policy.predict(
                observations,
                deterministic=self.warm_start_policy_deterministic,
            )
        actions = self._add_warm_start_noise(actions)
        self._last_warm_start_policy_actions = actions
        return actions

    def _add_warm_start_noise(self, actions: List[List[float]]) -> List[List[float]]:
        if self.warm_start_policy_noise_scale <= 0.0:
            return [[float(value) for value in agent_actions] for agent_actions in actions]

        noisy: List[List[float]] = []
        for agent_idx, agent_actions in enumerate(actions):
            low = self._action_low_for_agent(agent_idx)
            high = self._action_high_for_agent(agent_idx)
            span = np.maximum(high - low, 1.0e-6)
            values = np.asarray(agent_actions, dtype=np.float32)
            noise = np.random.normal(0.0, self.warm_start_policy_noise_scale, size=values.shape) * span
            noisy.append(np.clip(values + noise, low, high).astype(np.float32).tolist())
        return noisy

    def _warm_start_probability(self) -> float:
        if self._warm_start_policy is None or self.warm_start_policy_phaseout_steps <= 0:
            return 0.0
        progress = min(
            max(float(self.exploration_step) / float(self.warm_start_policy_phaseout_steps), 0.0),
            1.0,
        )
        return float(1.0 - progress)

    def _apply_warm_start_phaseout(self, actor_actions: List[List[float]]) -> List[List[float]]:
        probability = self._warm_start_probability()
        self._last_warm_start_phaseout_probability = probability
        self._last_warm_start_phaseout_used = False
        if probability <= 0.0 or self._warm_start_policy is None:
            self._last_policy_action_eligible = [True for _ in range(int(self.num_agents))]
            # Once the teacher no longer controls behavior, it may still label
            # the actor's on-policy states for an auxiliary BC loss.  Querying
            # it here does not change the action sent to the environment.
            if self._warm_start_policy is not None and self.actor_behavior_cloning_weight > 0.0:
                self._predict_warm_start_policy()
            return actor_actions

        # While a teacher can still affect the behavior trajectory, samples
        # are not on-policy for the PPO actor.  They remain useful for value
        # learning and behavior cloning, but must not enter the PPO ratio,
        # entropy bonus, or KL early-stop calculation.  This also covers the
        # actor branch of probability phase-out: its state distribution still
        # comes from the mixed behavior policy.
        self._last_policy_action_eligible = [False for _ in range(int(self.num_agents))]
        teacher_actions = (
            self._last_warm_start_policy_actions
            if self._last_warm_start_policy_actions is not None
            else self._predict_warm_start_policy()
        )
        if self.warm_start_policy_phaseout_mode == "probability":
            if random.random() < probability:
                self._last_warm_start_phaseout_used = True
                return teacher_actions
            return actor_actions

        blended: List[List[float]] = []
        for agent_idx, (actor_agent, teacher_agent) in enumerate(zip(actor_actions, teacher_actions)):
            low = self._action_low_for_agent(agent_idx)
            high = self._action_high_for_agent(agent_idx)
            actor_array = np.asarray(actor_agent, dtype=np.float32)
            teacher_array = np.asarray(teacher_agent, dtype=np.float32)
            blended_array = probability * teacher_array + (1.0 - probability) * actor_array
            blended.append(np.clip(blended_array, low, high).astype(np.float32).tolist())
        self._last_warm_start_phaseout_used = True
        return blended

    def _predict_random(self) -> List[List[float]]:
        return [
            np.random.uniform(
                low=self._action_low_for_agent(agent_idx),
                high=self._action_high_for_agent(agent_idx),
                size=(int(self.action_dimension[agent_idx]),),
            ).tolist()
            for agent_idx in range(int(self.num_agents))
        ]

    def update(
        self,
        observations: List[Any],
        actions: List[Any],
        rewards: List[float],
        next_observations: List[Any],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        if self._last_local_price_context_non_neutral:
            raise RuntimeError(
                "PPO received a non-neutral local price context during learning. "
                "Price-conditioned leaves are currently inference-only and must be frozen "
                "under the community coordinator."
            )
        _ = update_target_step
        done = bool(terminated or truncated)
        self._append_rollout_transition(observations, actions, rewards, next_observations, done)

        if not self._should_train_on_step(initial_exploration_done, global_learning_step):
            return
        if not update_step:
            return
        if not done and len(self.rollout) < self.rollout_length:
            return

        self._train_from_rollout(global_learning_step=global_learning_step)
        self.rollout.clear()

    def _append_rollout_transition(
        self,
        observations: List[Any],
        actions: List[Any],
        rewards: List[float],
        next_observations: List[Any],
        done: bool,
    ) -> None:
        obs_tensors = [
            torch.as_tensor(observations[agent_idx], dtype=torch.float32).view(-1)
            for agent_idx in range(int(self.num_agents))
        ]
        next_obs_tensors = [
            torch.as_tensor(next_observations[agent_idx], dtype=torch.float32).view(-1)
            for agent_idx in range(int(self.num_agents))
        ]
        normalized_actions = []
        policy_actions = []
        latent_actions = []
        teacher_actions = []
        old_log_probs = []
        values = []
        with torch.no_grad():
            for agent_idx in range(int(self.num_agents)):
                action_tensor = torch.as_tensor(actions[agent_idx], dtype=torch.float32, device=self.device).view(1, -1)
                normalized = self._normalize_scaled_action_tensor(agent_idx, action_tensor)
                policy_eligible = bool(self._last_policy_action_eligible[agent_idx])
                cached_sample = (
                    self._last_policy_samples[agent_idx]
                    if self._last_policy_samples is not None
                    and agent_idx < len(self._last_policy_samples)
                    else None
                )
                cache_matches_observation = bool(
                    cached_sample is not None
                    and cached_sample.get("observation") is not None
                    and torch.allclose(
                        cached_sample["observation"].view(-1),
                        obs_tensors[agent_idx].view(-1),
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    )
                )
                if policy_eligible:
                    if (
                        not cache_matches_observation
                        or not bool(cached_sample.get("stochastic", False))
                        or cached_sample.get("raw_action") is None
                        or cached_sample.get("normalized_action") is None
                        or cached_sample.get("scaled_action") is None
                        or cached_sample.get("log_prob") is None
                    ):
                        raise RuntimeError(
                            "PPO policy-eligible transition has no matching stochastic "
                            "sample cache. Call predict exactly once before update and "
                            "pass back the same observations/actions."
                        )
                    cached_executed = cached_sample.get(
                        "executed_action",
                        cached_sample["scaled_action"],
                    )
                    cached_scaled = cached_executed.to(
                        device=self.device,
                        dtype=action_tensor.dtype,
                    ).view(1, -1)
                    if not torch.allclose(
                        cached_scaled,
                        action_tensor,
                        rtol=1.0e-5,
                        atol=1.0e-6,
                    ):
                        raise RuntimeError(
                            "PPO policy-eligible action differs from the action returned "
                            "by predict; refusing to construct an invalid on-policy ratio."
                        )
                    policy_normalized = cached_sample["normalized_action"].to(
                        device=self.device,
                        dtype=normalized.dtype,
                    ).view(1, -1)
                    latent_action = cached_sample["raw_action"].to(
                        device=self.device,
                        dtype=normalized.dtype,
                    ).view(1, -1)
                    log_prob = cached_sample["log_prob"].to(
                        device=self.device,
                        dtype=normalized.dtype,
                    ).view(-1)
                else:
                    # A finite placeholder keeps mixed minibatches numerically
                    # stable. The policy mask prevents this inferred latent and
                    # zero log-prob from ever entering ratio/entropy/KL terms.
                    policy_normalized = normalized
                    bounded = policy_normalized.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
                    latent_action = torch.atanh(bounded)
                    log_prob = torch.zeros(1, dtype=normalized.dtype, device=self.device)
                teacher_action = None
                if self._last_warm_start_policy_actions is not None and agent_idx < len(self._last_warm_start_policy_actions):
                    teacher_tensor = torch.as_tensor(
                        self._last_warm_start_policy_actions[agent_idx],
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, -1)
                    if teacher_tensor.shape[-1] == normalized.shape[-1]:
                        teacher_action = self._normalize_scaled_action_tensor(agent_idx, teacher_tensor)
                if cache_matches_observation and cached_sample.get("value") is not None:
                    value = cached_sample["value"].to(
                        device=self.device,
                        dtype=normalized.dtype,
                    ).view(-1)
                else:
                    # Backward-compatible critic fallback for manually supplied
                    # teacher transitions. Policy-eligible rows never use it.
                    value_input = self._value_input_for_agent(agent_idx, obs_tensors).to(self.device).view(1, -1)
                    value = self.value_nets[agent_idx](value_input).squeeze(-1)
                normalized_actions.append(normalized.squeeze(0).cpu())
                policy_actions.append(policy_normalized.squeeze(0).cpu())
                latent_actions.append(latent_action.squeeze(0).cpu())
                if teacher_action is None:
                    teacher_actions.append(torch.full_like(normalized.squeeze(0).cpu(), float("nan")))
                else:
                    teacher_actions.append(teacher_action.squeeze(0).cpu())
                old_log_probs.append(log_prob.squeeze(0).cpu())
                values.append(value.squeeze(0).cpu())

        transition = {
            "observations": obs_tensors,
            "next_observations": next_obs_tensors,
            "actions": normalized_actions,
            "policy_actions": policy_actions,
            "latent_actions": latent_actions,
            "teacher_actions": teacher_actions,
            "rewards": torch.as_tensor(rewards, dtype=torch.float32).view(-1),
            "done": bool(done),
            "old_log_probs": torch.stack(old_log_probs),
            "values": torch.stack(values),
            "policy_eligible": torch.as_tensor(
                self._last_policy_action_eligible,
                dtype=torch.bool,
            ),
        }
        self.rollout.append(transition)
        # A behaviour sample is a single-use hand-off from predict to update.
        # Consuming it prevents accidental reuse when update is called twice.
        self._last_policy_samples = None
        self._last_policy_action_eligible = [False for _ in range(int(self.num_agents))]
        if self.actor_behavior_cloning_replay_capacity > 0 and any(
            bool(torch.isfinite(action).all()) for action in teacher_actions
        ):
            self.behavior_cloning_replay.append(
                {
                    "observations": [observation.detach().clone() for observation in obs_tensors],
                    "teacher_actions": [action.detach().clone() for action in teacher_actions],
                }
            )

    def _train_from_rollout(self, *, global_learning_step: int) -> None:
        if not self.rollout:
            return

        rollout_size = len(self.rollout)
        rewards = torch.stack([transition["rewards"] for transition in self.rollout]).to(self.device)
        dones = torch.as_tensor(
            [float(transition["done"]) for transition in self.rollout],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.stack([transition["values"] for transition in self.rollout]).to(self.device)
        old_log_probs = torch.stack([transition["old_log_probs"] for transition in self.rollout]).to(self.device)
        policy_eligible = torch.stack(
            [
                transition.get(
                    "policy_eligible",
                    torch.ones(int(self.num_agents), dtype=torch.bool),
                )
                for transition in self.rollout
            ]
        ).to(self.device, dtype=torch.bool)

        with torch.no_grad():
            next_obs_tensors = self.rollout[-1]["next_observations"]
            next_values = []
            for agent_idx in range(int(self.num_agents)):
                value_input = self._value_input_for_agent(agent_idx, next_obs_tensors).to(self.device).view(1, -1)
                next_values.append(self.value_nets[agent_idx](value_input).squeeze())
            bootstrap_values = torch.stack(next_values)

            advantages = torch.zeros_like(rewards)
            last_gae = torch.zeros(int(self.num_agents), dtype=torch.float32, device=self.device)
            for step_idx in reversed(range(rollout_size)):
                next_value = bootstrap_values if step_idx == rollout_size - 1 else old_values[step_idx + 1]
                nonterminal = 1.0 - dones[step_idx]
                delta = rewards[step_idx] + self.gamma * next_value * nonterminal - old_values[step_idx]
                last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
                advantages[step_idx] = last_gae
            returns = advantages + old_values

        eligible_advantages = advantages[policy_eligible]
        flat_advantages = (
            eligible_advantages
            if eligible_advantages.numel() > 0
            else advantages.reshape(-1)
        )
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std(unbiased=False).clamp_min(1.0e-8)
        advantages = (advantages - adv_mean) / adv_std

        indices = torch.arange(rollout_size, device=self.device)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        behavior_cloning_losses: List[float] = []
        actor_regularization_losses: List[float] = []
        entropy_values: List[float] = []
        approx_kl_values: List[float] = []
        grad_norm_values: List[float] = []
        behavior_cloning_weight = self._actor_behavior_cloning_effective_weight(global_learning_step)
        behavior_cloning_extra_updates = self._actor_behavior_cloning_extra_updates_for_step(
            global_learning_step,
            behavior_cloning_weight,
        )
        behavior_cloning_extra_losses: List[float] = []
        behavior_cloning_extra_grad_norms: List[float] = []

        for _epoch in range(self.ppo_epochs):
            shuffled = indices[torch.randperm(rollout_size, device=self.device)]
            stop_epoch = False
            for start in range(0, rollout_size, self.minibatch_size):
                batch_idx = shuffled[start : start + self.minibatch_size]
                for agent_idx in self._ppo_agent_update_order():
                    obs_batch = self._stack_agent_observations(agent_idx, batch_idx)
                    action_batch = self._stack_agent_actions(agent_idx, batch_idx)
                    latent_action_batch = self._stack_agent_latent_actions(agent_idx, batch_idx)
                    value_input_batch = self._stack_value_inputs(agent_idx, batch_idx)

                    distribution = self.actors[agent_idx].distribution(obs_batch)
                    log_prob = self._squashed_log_prob_from_latent(
                        distribution,
                        latent_action_batch,
                        action_batch,
                    )
                    eligible_mask = policy_eligible[batch_idx, agent_idx]
                    if bool(eligible_mask.any()):
                        eligible_log_prob = log_prob[eligible_mask]
                        eligible_old_log_prob = old_log_probs[batch_idx, agent_idx][eligible_mask]
                        entropy = distribution.entropy().sum(dim=-1)[eligible_mask].mean()
                        ratio = torch.exp(eligible_log_prob - eligible_old_log_prob)
                        advantage_batch = advantages[batch_idx, agent_idx][eligible_mask]
                        unclipped_loss = ratio * advantage_batch
                        clipped_loss = torch.clamp(
                            ratio,
                            1.0 - self.clip_ratio,
                            1.0 + self.clip_ratio,
                        ) * advantage_batch
                        policy_loss = -torch.minimum(unclipped_loss, clipped_loss).mean()
                        approx_kl = (eligible_old_log_prob - eligible_log_prob).mean().abs()
                    else:
                        # Preserve an actor-connected zero so BC/regularization
                        # can still be combined with this loss safely.
                        policy_loss = log_prob.sum() * 0.0
                        entropy = distribution.entropy().sum() * 0.0
                        approx_kl = log_prob.detach().sum() * 0.0

                    value_pred = self.value_nets[agent_idx](value_input_batch).squeeze(-1)
                    value_loss = mse_loss(value_pred, returns[batch_idx, agent_idx])
                    behavior_cloning_loss = self._actor_behavior_cloning_loss(
                        agent_idx,
                        obs_batch,
                        batch_idx,
                    )
                    actor_regularization_loss = self._actor_action_regularization_loss(agent_idx, obs_batch)
                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy
                        + behavior_cloning_weight * behavior_cloning_loss
                        + actor_regularization_loss
                    )

                    self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                    self.value_optimizers[agent_idx].zero_grad(set_to_none=True)
                    loss.backward()
                    parameters = [
                        *self.actors[agent_idx].parameters(),
                        *self.value_nets[agent_idx].parameters(),
                    ]
                    if self.max_grad_norm > 0.0:
                        grad_norm = clip_grad_norm_(parameters, self.max_grad_norm)
                    else:
                        grad_norm = torch.as_tensor(0.0)
                    self.actor_optimizers[agent_idx].step()
                    self.value_optimizers[agent_idx].step()

                    policy_losses.append(float(policy_loss.detach().item()))
                    value_losses.append(float(value_loss.detach().item()))
                    behavior_cloning_losses.append(float(behavior_cloning_loss.detach().item()))
                    actor_regularization_losses.append(float(actor_regularization_loss.detach().item()))
                    entropy_values.append(float(entropy.detach().item()))
                    approx_kl_values.append(float(approx_kl.detach().item()))
                    grad_norm_values.append(float(grad_norm))

                    if (
                        self.target_kl is not None
                        and bool(eligible_mask.any())
                        and approx_kl.item() > self.target_kl
                    ):
                        stop_epoch = True
                        break
                if stop_epoch:
                    break
            if stop_epoch:
                break

        # Keep the actor unchanged between rollout collection and the PPO
        # ratio calculation. Running auxiliary BC first makes the stored
        # old_log_probs stale and can produce an artificial KL spike before
        # the first PPO minibatch. Extra imitation updates are safe only after
        # every PPO objective that references the rollout policy is complete.
        behavior_cloning_extra_losses, behavior_cloning_extra_grad_norms = (
            self._run_actor_behavior_cloning_extra_updates(
                indices,
                behavior_cloning_weight=behavior_cloning_weight,
                extra_updates=behavior_cloning_extra_updates,
            )
        )

        if self.training_diagnostics_enabled and global_learning_step % self.mlflow_step_sample_interval == 0:
            metrics = {
                f"{self.metric_prefix}/rollout_size": float(rollout_size),
                f"{self.metric_prefix}/policy_eligible_fraction": float(
                    policy_eligible.float().mean().item()
                ),
                f"{self.metric_prefix}/policy_loss_mean": float(np.mean(policy_losses) if policy_losses else 0.0),
                f"{self.metric_prefix}/value_loss_mean": float(np.mean(value_losses) if value_losses else 0.0),
                f"{self.metric_prefix}/behavior_cloning_loss_mean": float(
                    np.mean(behavior_cloning_losses) if behavior_cloning_losses else 0.0
                ),
                f"{self.metric_prefix}/behavior_cloning_effective_weight": float(behavior_cloning_weight),
                f"{self.metric_prefix}/behavior_cloning_extra_updates": float(behavior_cloning_extra_updates),
                f"{self.metric_prefix}/behavior_cloning_extra_loss_mean": float(
                    np.mean(behavior_cloning_extra_losses) if behavior_cloning_extra_losses else 0.0
                ),
                f"{self.metric_prefix}/behavior_cloning_extra_grad_norm_mean": float(
                    np.mean(behavior_cloning_extra_grad_norms) if behavior_cloning_extra_grad_norms else 0.0
                ),
                f"{self.metric_prefix}/actor_regularization_loss_mean": float(
                    np.mean(actor_regularization_losses) if actor_regularization_losses else 0.0
                ),
                f"{self.metric_prefix}/entropy_mean": float(np.mean(entropy_values) if entropy_values else 0.0),
                f"{self.metric_prefix}/approx_kl_mean": float(np.mean(approx_kl_values) if approx_kl_values else 0.0),
                f"{self.metric_prefix}/grad_norm_mean": float(np.mean(grad_norm_values) if grad_norm_values else 0.0),
                f"{self.metric_prefix}/reward_mean": float(rewards.mean().item()),
                f"{self.metric_prefix}/reward_std": float(rewards.std(unbiased=False).item()),
                f"{self.metric_prefix}/advantage_mean": float(advantages.mean().item()),
                f"{self.metric_prefix}/advantage_std": float(advantages.std(unbiased=False).item()),
                f"{self.metric_prefix}/value_scope_global": float(self.value_scope == "global"),
                f"{self.metric_prefix}/agent_update_order_random": float(self.agent_update_order == "random"),
            }
            self._record_training_metrics(metrics, global_learning_step)

    def _ppo_agent_update_order(self) -> List[int]:
        order = list(range(int(self.num_agents)))
        if self.agent_update_order == "random":
            random.shuffle(order)
        return order

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        normalized_action: torch.Tensor,
        epsilon: float = 1.0e-6,
    ) -> torch.Tensor:
        """Log-probability under a Normal policy transformed by ``tanh``.

        PPO stores the bounded action sent to the simulator. Recovering its
        pre-tanh sample and applying the Jacobian correction keeps rollout and
        update probabilities consistent, including near action bounds.
        """
        bounded = normalized_action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        raw_action = torch.atanh(bounded)
        correction = torch.log(torch.clamp(1.0 - bounded.pow(2), min=epsilon))
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    @staticmethod
    def _squashed_log_prob_from_latent(
        distribution: torch.distributions.Normal,
        raw_action: torch.Tensor,
        normalized_action: torch.Tensor,
        epsilon: float = 1.0e-6,
    ) -> torch.Tensor:
        """Evaluate an exact cached pre-tanh policy sample.

        Keeping the latent avoids the lossy ``atanh`` reconstruction used by
        the compatibility helper above, especially for saturated actions.
        """

        correction = torch.log(
            torch.clamp(1.0 - normalized_action.pow(2), min=epsilon)
        )
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def _stack_agent_observations(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [self.rollout[int(idx.item())]["observations"][agent_idx] for idx in indices]
        return torch.stack(selected).to(self.device)

    def _stack_agent_actions(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [
            self.rollout[int(idx.item())].get(
                "policy_actions",
                self.rollout[int(idx.item())]["actions"],
            )[agent_idx]
            for idx in indices
        ]
        return torch.stack(selected).to(self.device)

    def _stack_agent_latent_actions(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = []
        for idx in indices:
            transition = self.rollout[int(idx.item())]
            latent_actions = transition.get("latent_actions")
            if latent_actions is not None:
                selected.append(latent_actions[agent_idx])
                continue
            # Checkpoints created before atomic policy sampling only contain
            # bounded normalized actions. Preserve resume compatibility by
            # reconstructing their latent exactly as the old implementation.
            policy_actions = transition.get("policy_actions", transition["actions"])
            bounded = policy_actions[agent_idx].clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
            selected.append(torch.atanh(bounded))
        return torch.stack(selected).to(self.device)

    def _stack_agent_teacher_actions(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [self.rollout[int(idx.item())]["teacher_actions"][agent_idx] for idx in indices]
        return torch.stack(selected).to(self.device)

    def _stack_value_inputs(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [
            self._value_input_for_agent(agent_idx, self.rollout[int(idx.item())]["observations"])
            for idx in indices
        ]
        return torch.stack(selected).to(self.device)

    def _actor_behavior_cloning_effective_weight(self, global_learning_step: int) -> float:
        base_weight = float(getattr(self, "actor_behavior_cloning_weight", 0.0))
        if base_weight <= 0.0:
            return 0.0
        min_weight = min(float(getattr(self, "actor_behavior_cloning_min_weight", 0.0)), base_weight)
        decay_steps = int(getattr(self, "actor_behavior_cloning_decay_steps", 0) or 0)
        decay_start = int(getattr(self, "actor_behavior_cloning_decay_start_step", 0) or 0)
        if global_learning_step < decay_start or decay_steps <= 0:
            return base_weight
        progress = min(max((global_learning_step - decay_start) / float(decay_steps), 0.0), 1.0)
        return float(base_weight + progress * (min_weight - base_weight))

    def _actor_behavior_cloning_extra_updates_for_step(
        self,
        global_learning_step: int,
        behavior_cloning_weight: float,
    ) -> int:
        if behavior_cloning_weight <= 0.0:
            return 0
        extra_updates = int(getattr(self, "actor_behavior_cloning_extra_updates", 0) or 0)
        if extra_updates <= 0:
            return 0
        start_step = int(getattr(self, "actor_behavior_cloning_extra_update_start_step", 0) or 0)
        if global_learning_step < start_step:
            return 0
        end_step = int(getattr(self, "actor_behavior_cloning_extra_update_end_step", 0) or 0)
        if end_step > 0 and global_learning_step > end_step:
            return 0
        return extra_updates

    def _run_actor_behavior_cloning_extra_updates(
        self,
        indices: torch.Tensor,
        *,
        behavior_cloning_weight: float,
        extra_updates: int,
    ) -> tuple[List[float], List[float]]:
        if extra_updates <= 0 or behavior_cloning_weight <= 0.0:
            return [], []

        losses: List[float] = []
        grad_norms: List[float] = []
        use_demonstration_replay = bool(
            self.actor_behavior_cloning_replay_capacity > 0
            and len(self.behavior_cloning_replay) > 0
        )
        source_size = (
            len(self.behavior_cloning_replay)
            if use_demonstration_replay
            else int(indices.numel())
        )
        sample_size = min(source_size, int(indices.numel()))
        for _update in range(extra_updates):
            if use_demonstration_replay:
                shuffled = torch.randperm(source_size, device=self.device)[:sample_size]
            else:
                shuffled = indices[torch.randperm(source_size, device=self.device)]
            for start in range(0, sample_size, self.minibatch_size):
                batch_idx = shuffled[start : start + self.minibatch_size]
                for agent_idx in self._ppo_agent_update_order():
                    if use_demonstration_replay:
                        obs_batch = self._stack_behavior_cloning_replay_observations(
                            agent_idx,
                            batch_idx,
                        )
                        teacher_actions = self._stack_behavior_cloning_replay_teacher_actions(
                            agent_idx,
                            batch_idx,
                        )
                        behavior_cloning_loss = self._actor_behavior_cloning_loss_for_targets(
                            agent_idx,
                            obs_batch,
                            teacher_actions,
                        )
                    else:
                        obs_batch = self._stack_agent_observations(agent_idx, batch_idx)
                        behavior_cloning_loss = self._actor_behavior_cloning_loss(
                            agent_idx,
                            obs_batch,
                            batch_idx,
                        )
                    if not torch.isfinite(behavior_cloning_loss) or behavior_cloning_loss.detach().item() <= 0.0:
                        continue
                    weighted_loss = behavior_cloning_weight * behavior_cloning_loss
                    self.actor_optimizers[agent_idx].zero_grad(set_to_none=True)
                    weighted_loss.backward()
                    if self.max_grad_norm > 0.0:
                        grad_norm = clip_grad_norm_(self.actors[agent_idx].parameters(), self.max_grad_norm)
                    else:
                        grad_norm = torch.as_tensor(0.0)
                    self.actor_optimizers[agent_idx].step()
                    losses.append(float(behavior_cloning_loss.detach().item()))
                    grad_norms.append(float(grad_norm))
        return losses, grad_norms

    def _stack_behavior_cloning_replay_observations(
        self,
        agent_idx: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        selected = [
            self.behavior_cloning_replay[int(idx.item())]["observations"][agent_idx]
            for idx in indices
        ]
        return torch.stack(selected).to(self.device)

    def _stack_behavior_cloning_replay_teacher_actions(
        self,
        agent_idx: int,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        selected = [
            self.behavior_cloning_replay[int(idx.item())]["teacher_actions"][agent_idx]
            for idx in indices
        ]
        return torch.stack(selected).to(self.device)

    def _actor_behavior_cloning_loss(
        self,
        agent_idx: int,
        obs_batch: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if float(getattr(self, "actor_behavior_cloning_weight", 0.0)) <= 0.0:
            return torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        teacher_actions = self._stack_agent_teacher_actions(agent_idx, indices)
        return self._actor_behavior_cloning_loss_for_targets(
            agent_idx,
            obs_batch,
            teacher_actions,
        )

    def _actor_behavior_cloning_loss_for_targets(
        self,
        agent_idx: int,
        obs_batch: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = torch.isfinite(teacher_actions).all(dim=1)
        if not torch.any(valid_mask):
            return torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        predicted = self.actors[agent_idx](obs_batch[valid_mask])
        target = teacher_actions[valid_mask]
        squared_error = (predicted - target).pow(2)
        weights = self._actor_behavior_cloning_action_weights(
            agent_idx,
            action_dim=squared_error.shape[-1],
            dtype=squared_error.dtype,
            device=squared_error.device,
        )
        sample_weights = self._actor_behavior_cloning_sample_weights(
            agent_idx,
            base_weights=weights,
            normalized_target=target,
        )
        denominator = torch.clamp(sample_weights.sum(), min=1.0)
        return (squared_error * sample_weights).sum() / denominator

    def _actor_behavior_cloning_sample_weights(
        self,
        agent_idx: int,
        *,
        base_weights: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> torch.Tensor:
        sample_weights = base_weights.view(1, -1).expand_as(normalized_target)
        names = self._action_names_for_agent(agent_idx)
        action_dim = int(normalized_target.shape[-1])
        ev_positive_weight = float(
            getattr(self, "actor_ev_behavior_cloning_positive_target_weight", 0.0) or 0.0
        )
        ev_zero_weight = float(
            getattr(self, "actor_ev_behavior_cloning_zero_target_weight", 0.0) or 0.0
        )
        deferrable_positive_weight = float(
            getattr(
                self,
                "actor_deferrable_behavior_cloning_positive_target_weight",
                0.0,
            )
            or 0.0
        )
        if (
            ev_positive_weight <= 0.0
            and ev_zero_weight <= 0.0
            and deferrable_positive_weight <= 0.0
        ):
            return sample_weights

        multiplier = torch.ones_like(normalized_target)
        for action_idx in range(action_dim):
            action_name = names[action_idx] if action_idx < len(names) else ""
            if self._is_ev_action_name(action_name):
                if ev_positive_weight > 0.0:
                    positive = torch.clamp(normalized_target[:, action_idx], 0.0, 1.0)
                    power = float(
                        getattr(self, "actor_ev_behavior_cloning_positive_target_power", 1.0)
                        or 1.0
                    )
                    if power != 1.0:
                        positive = positive.pow(power)
                    multiplier[:, action_idx] += ev_positive_weight * positive
                if ev_zero_weight > 0.0:
                    zero_threshold = float(
                        getattr(self, "actor_ev_behavior_cloning_zero_target_threshold", 0.05)
                        or 0.05
                    )
                    zero_target = (
                        normalized_target[:, action_idx].abs() <= zero_threshold
                    ).to(dtype=normalized_target.dtype)
                    multiplier[:, action_idx] += ev_zero_weight * zero_target
            elif self._is_deferrable_action_name(action_name) and deferrable_positive_weight > 0.0:
                # Deferrable actions use [0, 1] in the simulator and therefore
                # [-1, 1] after normalization. Map the target back to an
                # off/on fraction before upweighting the rare start commands.
                positive = torch.clamp(
                    (normalized_target[:, action_idx] + 1.0) * 0.5,
                    0.0,
                    1.0,
                )
                power = float(
                    getattr(
                        self,
                        "actor_deferrable_behavior_cloning_positive_target_power",
                        1.0,
                    )
                    or 1.0
                )
                if power != 1.0:
                    positive = positive.pow(power)
                multiplier[:, action_idx] += deferrable_positive_weight * positive
        return sample_weights * multiplier

    def _actor_behavior_cloning_action_weights(
        self,
        agent_idx: int,
        *,
        action_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        names = self._action_names_for_agent(agent_idx)
        weights: List[float] = []
        for action_idx in range(int(action_dim)):
            action_name = names[action_idx] if action_idx < len(names) else ""
            if self._is_ev_action_name(action_name):
                multiplier = self.actor_ev_behavior_cloning_multiplier
            elif self._is_storage_action_name(action_name):
                multiplier = self.actor_storage_behavior_cloning_multiplier
            elif self._is_deferrable_action_name(action_name):
                multiplier = self.actor_deferrable_behavior_cloning_multiplier
            else:
                multiplier = self.actor_other_behavior_cloning_multiplier
            weights.append(float(multiplier))
        return torch.as_tensor(weights, dtype=dtype, device=device)

    def _actor_action_regularization_loss(self, agent_idx: int, obs_batch: torch.Tensor) -> torch.Tensor:
        if (
            self.actor_action_l2_penalty <= 0.0
            and self.actor_storage_action_l2_penalty <= 0.0
            and self.actor_ev_v2g_action_l2_penalty <= 0.0
            and self.actor_action_saturation_penalty <= 0.0
        ):
            return torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        normalized_action = self.actors[agent_idx](obs_batch)
        scaled_action = self._scale_action_tensor(agent_idx, normalized_action)
        loss = torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        if self.actor_action_l2_penalty > 0.0:
            loss = loss + float(self.actor_action_l2_penalty) * torch.mean(normalized_action.pow(2))

        action_names = self._action_names_for_agent(agent_idx)
        if self.actor_storage_action_l2_penalty > 0.0:
            mask = self._action_mask(action_names, scaled_action.shape[-1], self._is_storage_action_name)
            if mask is not None:
                mask = mask.to(device=scaled_action.device)
                storage_actions = scaled_action[:, mask]
                loss = loss + float(self.actor_storage_action_l2_penalty) * torch.mean(storage_actions.pow(2))

        if self.actor_ev_v2g_action_l2_penalty > 0.0:
            mask = self._action_mask(action_names, scaled_action.shape[-1], self._is_ev_action_name)
            if mask is not None:
                mask = mask.to(device=scaled_action.device)
                ev_actions = scaled_action[:, mask]
                ev_discharge = torch.clamp(-ev_actions, min=0.0)
                loss = loss + float(self.actor_ev_v2g_action_l2_penalty) * torch.mean(ev_discharge.pow(2))

        if self.actor_action_saturation_penalty > 0.0:
            excess = torch.clamp(normalized_action.abs() - self.actor_action_saturation_threshold, min=0.0)
            loss = loss + float(self.actor_action_saturation_penalty) * torch.mean(excess.pow(2))

        return loss

    def _action_names_for_agent(self, agent_idx: int) -> List[str]:
        if hasattr(self, "action_names") and agent_idx < len(self.action_names):
            return list(self.action_names[agent_idx])
        return []

    @staticmethod
    def _action_mask(action_names: List[str], action_dim: int, predicate) -> Optional[torch.Tensor]:
        mask = [
            bool(action_idx < len(action_names) and predicate(action_names[action_idx]))
            for action_idx in range(action_dim)
        ]
        if not any(mask):
            return None
        return torch.as_tensor(mask, dtype=torch.bool)

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        lowered = str(action_name).lower()
        return "battery" in lowered or "storage" in lowered

    @staticmethod
    def _is_ev_action_name(action_name: str) -> bool:
        lowered = str(action_name).lower()
        return "charger" in lowered or "electric_vehicle" in lowered or lowered.startswith("ev")

    @staticmethod
    def _is_deferrable_action_name(action_name: str) -> bool:
        lowered = str(action_name).lower()
        return (
            lowered.startswith("deferrable_appliance")
            or lowered.endswith("::start")
            or lowered == "start"
        )

    def _value_input_for_agent(self, agent_idx: int, observations: List[torch.Tensor]) -> torch.Tensor:
        if self.value_scope == "global":
            return torch.cat([obs.view(-1) for obs in observations], dim=0)
        return observations[agent_idx].view(-1)

    def _action_low_for_agent(self, agent_idx: int) -> np.ndarray:
        if hasattr(self, "action_low") and agent_idx < len(self.action_low):
            return np.asarray(self.action_low[agent_idx], dtype=np.float32)
        return np.full(int(self.action_dimension[agent_idx]), -1.0, dtype=np.float32)

    def _action_high_for_agent(self, agent_idx: int) -> np.ndarray:
        if hasattr(self, "action_high") and agent_idx < len(self.action_high):
            return np.asarray(self.action_high[agent_idx], dtype=np.float32)
        return np.full(int(self.action_dimension[agent_idx]), 1.0, dtype=np.float32)

    def _scale_action_tensor(self, agent_idx: int, normalized_action: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(
            self._action_low_for_agent(agent_idx),
            dtype=normalized_action.dtype,
            device=normalized_action.device,
        )
        high = torch.as_tensor(
            self._action_high_for_agent(agent_idx),
            dtype=normalized_action.dtype,
            device=normalized_action.device,
        )
        scaled = low + 0.5 * (normalized_action + 1.0) * (high - low)
        return torch.max(torch.min(scaled, high), low)

    def _normalize_scaled_action_tensor(self, agent_idx: int, action: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(
            self._action_low_for_agent(agent_idx),
            dtype=action.dtype,
            device=action.device,
        )
        high = torch.as_tensor(
            self._action_high_for_agent(agent_idx),
            dtype=action.dtype,
            device=action.device,
        )
        span = torch.clamp(high - low, min=1.0e-6)
        return torch.clamp(2.0 * (action - low) / span - 1.0, -1.0, 1.0)

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        return {
            f"{self.metric_prefix}/rollout_buffer_size": float(len(self.rollout)),
            f"{self.metric_prefix}/rollout_length": float(self.rollout_length),
            f"{self.metric_prefix}/minibatch_size": float(self.minibatch_size),
            f"{self.metric_prefix}/ppo_epochs": float(self.ppo_epochs),
            f"{self.metric_prefix}/clip_ratio": float(self.clip_ratio),
            f"{self.metric_prefix}/entropy_coef": float(self.entropy_coef),
            f"{self.metric_prefix}/value_loss_coef": float(self.value_loss_coef),
            f"{self.metric_prefix}/gae_lambda": float(self.gae_lambda),
            f"{self.metric_prefix}/value_scope_global": float(self.value_scope == "global"),
            f"{self.metric_prefix}/agent_update_order_random": float(self.agent_update_order == "random"),
            f"{self.metric_prefix}/exploration_step": float(self.exploration_step),
            f"{self.metric_prefix}/warm_start_policy_enabled": float(self._warm_start_policy is not None),
            f"{self.metric_prefix}/warm_start_policy_phaseout_steps": float(self.warm_start_policy_phaseout_steps),
            f"{self.metric_prefix}/warm_start_policy_phaseout_probability": float(
                self._last_warm_start_phaseout_probability
            ),
            f"{self.metric_prefix}/warm_start_policy_phaseout_used": float(self._last_warm_start_phaseout_used),
            f"{self.metric_prefix}/behavior_cloning_weight": float(self.actor_behavior_cloning_weight),
            f"{self.metric_prefix}/behavior_cloning_min_weight": float(self.actor_behavior_cloning_min_weight),
            f"{self.metric_prefix}/behavior_cloning_replay_size": float(
                len(self.behavior_cloning_replay)
            ),
            f"{self.metric_prefix}/behavior_cloning_replay_capacity": float(
                self.actor_behavior_cloning_replay_capacity
            ),
            f"{self.metric_prefix}/behavior_cloning_ev_multiplier": float(
                self.actor_ev_behavior_cloning_multiplier
            ),
            f"{self.metric_prefix}/behavior_cloning_ev_positive_target_weight": float(
                self.actor_ev_behavior_cloning_positive_target_weight
            ),
            f"{self.metric_prefix}/behavior_cloning_ev_zero_target_weight": float(
                self.actor_ev_behavior_cloning_zero_target_weight
            ),
            f"{self.metric_prefix}/behavior_cloning_ev_zero_target_threshold": float(
                self.actor_ev_behavior_cloning_zero_target_threshold
            ),
            f"{self.metric_prefix}/behavior_cloning_storage_multiplier": float(
                self.actor_storage_behavior_cloning_multiplier
            ),
            f"{self.metric_prefix}/behavior_cloning_deferrable_multiplier": float(
                self.actor_deferrable_behavior_cloning_multiplier
            ),
            f"{self.metric_prefix}/behavior_cloning_deferrable_positive_target_weight": float(
                self.actor_deferrable_behavior_cloning_positive_target_weight
            ),
            f"{self.metric_prefix}/initial_exploration_done": float(
                self.exploration_step >= self.end_initial_exploration_time_step
            ),
            f"{self.metric_prefix}/local_action_safety_enabled": float(
                self.local_action_safety_enabled
            ),
            f"{self.metric_prefix}/local_action_safety_service_teacher_enabled": float(
                self.local_action_safety_service_teacher_enabled
            ),
            f"{self.metric_prefix}/local_action_safety_service_teacher_eval_enabled": float(
                self.local_action_safety_service_teacher_eval_enabled
            ),
            f"{self.metric_prefix}/local_action_safety_service_teacher_applied": float(
                self._last_service_teacher_applied
            ),
            f"{self.metric_prefix}/local_price_conditioning_enabled": float(
                self.local_price_conditioning_enabled
            ),
            f"{self.metric_prefix}/local_price_context_non_neutral": float(
                self._last_local_price_context_non_neutral
            ),
            f"{self.metric_prefix}/local_price_clipping_count": float(
                sum(
                    diagnostics.clipping_count
                    for diagnostics in self._last_local_price_diagnostics
                )
            ),
            f"{self.metric_prefix}/local_action_safety_interventions": float(
                sum(
                    len(result.interventions)
                    for result in self._last_local_action_projections
                )
            ),
            f"{self.metric_prefix}/local_action_safety_infeasible": float(
                sum(
                    len(result.infeasible_reasons)
                    for result in self._last_local_action_projections
                )
            ),
        }

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = dict(self._latest_training_metrics)
        self._latest_training_metrics = {}
        return metrics

    def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._latest_training_metrics = dict(metrics)
        if mlflow.active_run() and not bool(getattr(self, "managed_by_ensemble", False)):
            mlflow.log_metrics(metrics, step=step)

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        checkpoint: Dict[str, Any] = {
            "step": int(step),
            "rollout": self.rollout,
            "behavior_cloning_replay": list(self.behavior_cloning_replay),
            "exploration_step": int(self.exploration_step),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        for agent_idx in range(int(self.num_agents)):
            checkpoint[f"actor_state_dict_{agent_idx}"] = self.actors[agent_idx].state_dict()
            checkpoint[f"value_state_dict_{agent_idx}"] = self.value_nets[agent_idx].state_dict()
            checkpoint[f"actor_optimizer_state_dict_{agent_idx}"] = self.actor_optimizers[agent_idx].state_dict()
            checkpoint[f"value_optimizer_state_dict_{agent_idx}"] = self.value_optimizers[agent_idx].state_dict()

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        latest_path = output_dir_path / (self.checkpoint_artifact or "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        return str(latest_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        # Rollout and demonstration tensors are intentionally accumulated on
        # CPU and moved to the active device only when a minibatch is built.
        # Loading the entire checkpoint directly onto CUDA makes restored
        # rows incompatible with newly appended CPU rows.
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        for agent_idx in range(int(self.num_agents)):
            self.actors[agent_idx].load_state_dict(checkpoint[f"actor_state_dict_{agent_idx}"])
            self.value_nets[agent_idx].load_state_dict(checkpoint[f"value_state_dict_{agent_idx}"])
            if bool(getattr(self, "restore_optimizers", not self.fine_tune)):
                self.actor_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"actor_optimizer_state_dict_{agent_idx}"]
                )
                self.value_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"value_optimizer_state_dict_{agent_idx}"]
                )
        if bool(getattr(self, "restore_replay_buffer", not self.reset_replay_buffer)):
            self.rollout = list(checkpoint.get("rollout", []))
            restored_demonstrations = checkpoint.get("behavior_cloning_replay", [])
            self.behavior_cloning_replay.clear()
            self.behavior_cloning_replay.extend(restored_demonstrations)
        if bool(getattr(self, "restore_exploration_state", True)):
            self.exploration_step = int(
                checkpoint.get("exploration_step", self.exploration_step)
            )
        rng_state = checkpoint.get("rng_state")
        if bool(getattr(self, "restore_exploration_state", True)) and isinstance(
            rng_state, dict
        ):
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"].cpu())
            if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["torch_cuda"]])

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if (
            self.local_action_safety_enabled
            and not self.local_action_safety_runtime_only_export
        ):
            raise RuntimeError(
                f"{self.metric_prefix} local action safety is not embedded in the "
                "ONNX actor. Set local_action_safety_runtime_only_export=true "
                "only for non-deployable experiment evidence, or implement a "
                "composite bundle."
            )
        context = context or {}
        bundle_cfg = ((context.get("config") or {}).get("bundle") or {})
        global_artifact_config = dict(bundle_cfg.get("artifact_config") or {})
        raw_per_agent_config = bundle_cfg.get("per_agent_artifact_config") or {}
        per_agent_artifact_config = raw_per_agent_config if isinstance(raw_per_agent_config, dict) else {}
        require_observations_envelope = bool(bundle_cfg.get("require_observations_envelope", False))
        agent_index_offset = int(context.get("agent_index_offset", 0) or 0)

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        metadata: Dict[str, Any] = {"format": "onnx", "artifacts": []}

        for agent_idx, actor in enumerate(self.actors):
            global_agent_idx = agent_index_offset + agent_idx
            export_path = onnx_dir / f"agent_{agent_idx}.onnx"
            dummy_input = torch.randn(1, self.observation_dimension[agent_idx], device=self.device)
            export_model = ActionScaledActor(
                actor,
                low=self._action_low_for_agent(agent_idx),
                high=self._action_high_for_agent(agent_idx),
            ).to(self.device)
            export_model.eval()
            torch.onnx.export(
                export_model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=DEFAULT_ONNX_OPSET,
                do_constant_folding=True,
                input_names=[f"observation_agent_{agent_idx}"],
                output_names=[f"action_agent_{agent_idx}"],
                dynamic_axes={
                    f"observation_agent_{agent_idx}": {0: "batch_size"},
                    f"action_agent_{agent_idx}": {0: "batch_size"},
                },
            )

            raw_agent_override = (
                per_agent_artifact_config.get(str(global_agent_idx))
                if str(global_agent_idx) in per_agent_artifact_config
                else per_agent_artifact_config.get(global_agent_idx)
            )
            agent_override = raw_agent_override if isinstance(raw_agent_override, dict) else {}
            artifact_config: Dict[str, Any] = {}
            artifact_config.update(build_auto_artifact_config(context=context, agent_index=global_agent_idx))
            artifact_config.update(global_artifact_config)
            artifact_config.update(agent_override)
            if require_observations_envelope:
                artifact_config["require_observations_envelope"] = True

            metadata["artifacts"].append(
                {
                    "agent_index": global_agent_idx,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                    "observation_dimension": self.observation_dimension[agent_idx],
                    "action_dimension": self.action_dimension[agent_idx],
                    "config": artifact_config,
                }
            )
            if self.local_action_safety_enabled:
                metadata["artifacts"][-1].setdefault("config", {}).update(
                    {
                        "deployable": False,
                        "runtime_only_reason": "external_local_action_safety_projector",
                        "requires_runtime_local_action_safety": True,
                        "requires_runtime_service_teacher": bool(
                            self.local_action_safety_service_teacher_enabled
                        ),
                    }
                )
            if self.local_price_conditioning_enabled:
                metadata["artifacts"][-1].setdefault("config", {}).update(
                    {
                        "requires_runtime_local_price_adapter": True,
                        "local_price_forecast_mode": self.local_price_forecast_mode.value,
                        "local_price_context_scope": "effective_local_price_only",
                        "community_observations_used_by_leaf": False,
                    }
                )
            if mlflow.active_run():
                mlflow.log_artifact(str(export_path), artifact_path="onnx")

        return metadata


class PPO(_PPOBase):
    """Strict single-agent PPO for exactly one local environment slot.

    Multiple independent building controllers are composed with
    :class:`algorithms.pipeline.Ensemble`; this class deliberately never owns
    more than one actor, value function, optimiser pair, or local reward stream.
    """

    single_agent_only = True
    value_scope = "local"
    metric_prefix = "PPO"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        if int(self.num_agents) != 1:
            raise ValueError(
                "PPO controls exactly one environment slot. For a distributed "
                "multi-building run configure one PPO stage with count equal to "
                "the number of buildings."
            )


class IPPO(_PPOBase):
    """Independent PPO: local actor and local value function per agent."""

    value_scope = "local"
    metric_prefix = "IPPO"


class MAPPO(_PPOBase):
    """Multi-Agent PPO with decentralized actors and centralized value inputs."""

    value_scope = "global"
    metric_prefix = "MAPPO"


class HAPPO(_PPOBase):
    """HAPPO-style sequential multi-agent PPO with centralized value inputs."""

    value_scope = "global"
    metric_prefix = "HAPPO"
