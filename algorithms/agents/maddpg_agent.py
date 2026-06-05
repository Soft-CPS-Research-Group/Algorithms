from __future__ import annotations

import inspect
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlflow
import numpy as np
import torch
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.nn.functional import mse_loss, smooth_l1_loss
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.networks import Actor, build_critic_network
from algorithms.utils.replay_buffer import (
    MultiAgentReplayBuffer,
    PrioritizedReplayBuffer,
    RewardWeightedMultiAgentReplayBuffer,
)
from utils.artifact_config_builder import build_auto_artifact_config

REPLAY_BUFFER_REGISTRY = {
    "MultiAgentReplayBuffer": MultiAgentReplayBuffer,
    "RewardWeightedMultiAgentReplayBuffer": RewardWeightedMultiAgentReplayBuffer,
    "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
}


def _select_torch_device(*, require_cuda: bool = False) -> torch.device:
    """Select the torch device and fail early when CUDA was explicitly required."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "MADDPG was configured with require_cuda=true, but torch.cuda.is_available() is false."
        )
    return torch.device("cpu")


def _log_torch_runtime(device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    logger.info(
        "Torch runtime: torch_version={}, torch_cuda_version={}, cuda_available={}, cuda_device_count={}",
        torch.__version__,
        torch.version.cuda,
        cuda_available,
        cuda_device_count,
    )
    if cuda_available:
        logger.info("CUDA device selected: {}", torch.cuda.get_device_name(device))


class ActionScaledActor(torch.nn.Module):
    """Actor wrapper that converts tanh policy output to environment action bounds."""

    def __init__(self, actor: torch.nn.Module, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32).reshape(1, -1))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32).reshape(1, -1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw_action = self.actor(observation)
        return self.low + 0.5 * (raw_action + 1.0) * (self.high - self.low)


class MADDPG(BaseAgent):
    """Multi-Agent DDPG implementation with MLflow logging and ONNX export."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        exploration_cfg = self.config["algorithm"]["exploration"]["params"]
        buffer_cfg = self.config["algorithm"]["replay_buffer"]
        network_cfg = self.config["algorithm"]["networks"]

        hyperparams = self.config["algorithm"]["hyperparameters"]
        self.require_cuda = bool(exploration_cfg.get("require_cuda", hyperparams.get("require_cuda", False)))
        self.device = _select_torch_device(require_cuda=self.require_cuda)
        logger.info("Device selected: {}", self.device)
        _log_torch_runtime(self.device)
        torch.backends.cudnn.benchmark = self.device.type == "cuda"

        self.gamma = float(hyperparams.get("gamma", exploration_cfg.get("gamma", 0.99)))
        self.tau = float(exploration_cfg.get("tau", 0.001))
        self.sigma = float(exploration_cfg.get("sigma", 0.2))
        self.sigma_decay = float(exploration_cfg.get("decay", 1.0))
        self.min_sigma = float(exploration_cfg.get("min_sigma", 0.0))
        self.bias = float(exploration_cfg.get("bias", 0.0))
        noise_clip_raw = exploration_cfg.get("noise_clip")
        self.noise_clip = float(noise_clip_raw) if noise_clip_raw is not None else None
        if self.noise_clip is not None and self.noise_clip <= 0.0:
            self.noise_clip = None
        self.storage_exploration_noise_multiplier = max(
            0.0,
            float(exploration_cfg.get("storage_exploration_noise_multiplier", 1.0) or 0.0),
        )
        self.ev_negative_exploration_noise_multiplier = max(
            0.0,
            float(exploration_cfg.get("ev_negative_exploration_noise_multiplier", 1.0) or 0.0),
        )
        self.warm_start_policy_name = self._optional_string(exploration_cfg.get("warm_start_policy"))
        self.initial_exploration_strategy = self._resolve_initial_exploration_strategy(exploration_cfg)
        self.warm_start_policy_deterministic = bool(exploration_cfg.get("warm_start_policy_deterministic", True))
        self.warm_start_policy_noise_scale = max(0.0, float(exploration_cfg.get("warm_start_policy_noise_scale", 0.0) or 0.0))
        self.warm_start_policy_phaseout_steps = max(
            0,
            int(exploration_cfg.get("warm_start_policy_phaseout_steps", 0) or 0),
        )
        self.warm_start_policy_phaseout_mode = str(
            exploration_cfg.get("warm_start_policy_phaseout_mode", "probability") or "probability"
        ).strip().lower()
        if self.warm_start_policy_phaseout_mode not in {"probability", "blend"}:
            raise ValueError("MADDPG warm_start_policy_phaseout_mode must be 'probability' or 'blend'.")
        self.train_during_initial_exploration = bool(
            exploration_cfg.get("train_during_initial_exploration", False)
        )
        self.initial_exploration_training_start_step = max(
            0,
            int(exploration_cfg.get("initial_exploration_training_start_step", 0) or 0),
        )
        self.noop_noise_scale = max(0.0, float(exploration_cfg.get("noop_noise_scale", 0.15) or 0.0))
        self.deferrable_on_probability = float(np.clip(float(exploration_cfg.get("deferrable_on_probability", 0.2) or 0.0), 0.0, 1.0))
        self.deferrable_trigger_threshold = float(np.clip(float(exploration_cfg.get("deferrable_trigger_threshold", 0.5) or 0.5), 0.0, 1.0))
        self.noop_actor_initialization = bool(exploration_cfg.get("noop_actor_initialization", False))
        self.noop_actor_initialization_epsilon = float(
            np.clip(float(exploration_cfg.get("noop_actor_initialization_epsilon", 0.05) or 0.05), 1.0e-4, 0.49)
        )
        self.critic_update_mode = str(exploration_cfg.get("critic_update_mode", "joint_mean") or "joint_mean").strip().lower()
        if self.critic_update_mode not in {"joint_mean", "per_agent"}:
            raise ValueError("MADDPG critic_update_mode must be 'joint_mean' or 'per_agent'.")
        self.actor_update_interval = max(1, int(exploration_cfg.get("actor_update_interval", 1) or 1))
        self.actor_policy_loss_weight = max(
            0.0,
            float(exploration_cfg.get("actor_policy_loss_weight", 1.0) or 0.0),
        )
        self.actor_policy_loss_warmup_weight = max(
            0.0,
            float(
                exploration_cfg.get(
                    "actor_policy_loss_warmup_weight",
                    self.actor_policy_loss_weight,
                )
                or 0.0
            ),
        )
        self.actor_policy_loss_warmup_steps = max(
            0,
            int(exploration_cfg.get("actor_policy_loss_warmup_steps", 0) or 0),
        )
        self.actor_policy_loss_warmup_start_step = max(
            0,
            int(exploration_cfg.get("actor_policy_loss_warmup_start_step", 0) or 0),
        )
        self.target_policy_smoothing = bool(exploration_cfg.get("target_policy_smoothing", False))
        self.target_policy_noise = max(0.0, float(exploration_cfg.get("target_policy_noise", 0.05) or 0.0))
        self.target_policy_noise_clip = max(
            0.0,
            float(exploration_cfg.get("target_policy_noise_clip", 0.10) or 0.0),
        )
        self.actor_action_l2_penalty = max(0.0, float(exploration_cfg.get("actor_action_l2_penalty", 0.0) or 0.0))
        self.actor_action_saturation_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_action_saturation_penalty", 0.0) or 0.0),
        )
        self.actor_storage_action_l2_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_storage_action_l2_penalty", 0.0) or 0.0),
        )
        self.actor_ev_v2g_action_l2_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_ev_v2g_action_l2_penalty", 0.0) or 0.0),
        )
        self.actor_ev_v2g_action_mass_penalty = max(
            0.0,
            float(exploration_cfg.get("actor_ev_v2g_action_mass_penalty", 0.0) or 0.0),
        )
        self.actor_action_saturation_threshold = float(
            np.clip(float(exploration_cfg.get("actor_action_saturation_threshold", 0.85) or 0.85), 0.0, 1.0)
        )
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
        self.actor_behavior_cloning_min_weight = max(
            0.0,
            float(exploration_cfg.get("actor_behavior_cloning_min_weight", 0.0) or 0.0),
        )
        self.actor_behavior_cloning_decay_steps = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_decay_steps", 0) or 0),
        )
        self.actor_behavior_cloning_decay_start_step = max(
            0,
            int(exploration_cfg.get("actor_behavior_cloning_decay_start_step", 0) or 0),
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
        self.actor_behavior_cloning_source = str(
            exploration_cfg.get("actor_behavior_cloning_source", "replay_action") or "replay_action"
        ).strip().lower()
        if self.actor_behavior_cloning_source not in {"replay_action", "warm_start_policy"}:
            raise ValueError(
                "MADDPG actor_behavior_cloning_source must be 'replay_action' or 'warm_start_policy'."
            )
        self.residual_policy_enabled = bool(exploration_cfg.get("residual_policy_enabled", False))
        self.residual_action_scale = float(
            np.clip(float(exploration_cfg.get("residual_action_scale", 0.0) or 0.0), 0.0, 1.0)
        )
        self.residual_action_final_scale = float(
            np.clip(
                float(
                    exploration_cfg.get(
                        "residual_action_final_scale",
                        self.residual_action_scale,
                    )
                    or 0.0
                ),
                0.0,
                1.0,
            )
        )
        self.residual_action_start_step = max(
            0,
            int(exploration_cfg.get("residual_action_start_step", 0) or 0),
        )
        self.residual_action_growth_steps = max(
            0,
            int(exploration_cfg.get("residual_action_growth_steps", 0) or 0),
        )
        self.residual_storage_action_scale_multiplier = max(
            0.0,
            float(exploration_cfg.get("residual_storage_action_scale_multiplier", 1.0) or 0.0),
        )
        self.residual_ev_action_scale_multiplier = max(
            0.0,
            float(exploration_cfg.get("residual_ev_action_scale_multiplier", 1.0) or 0.0),
        )
        self.residual_deferrable_action_scale_multiplier = max(
            0.0,
            float(exploration_cfg.get("residual_deferrable_action_scale_multiplier", 1.0) or 0.0),
        )
        self._last_residual_action_scale = 0.0
        self.reward_normalization_enabled = bool(exploration_cfg.get("reward_normalization", False))
        self.reward_normalization_clip = float(exploration_cfg.get("reward_normalization_clip", 10.0) or 10.0)
        if self.reward_normalization_clip <= 0.0:
            self.reward_normalization_clip = 10.0
        self.reward_normalization_epsilon = float(exploration_cfg.get("reward_normalization_epsilon", 1.0e-8) or 1.0e-8)
        if self.reward_normalization_epsilon <= 0.0:
            self.reward_normalization_epsilon = 1.0e-8
        self.reward_norm_count = 0
        self.reward_norm_mean = 0.0
        self.reward_norm_m2 = 0.0
        replay_cfg = self.config["algorithm"].get("replay_buffer", {})
        self.replay_observation_event_priority_weight = max(
            0.0,
            float(replay_cfg.get("observation_event_priority_weight", 0.0) or 0.0),
        )
        self.replay_observation_event_priority_mode = str(
            replay_cfg.get("observation_event_priority_mode", "ev_departure_service") or "ev_departure_service"
        ).strip().lower()
        if self.replay_observation_event_priority_mode not in {
            "ev_departure_service",
            "ev_pv_price_peak",
            "combined",
        }:
            raise ValueError(
                "MADDPG replay_buffer.observation_event_priority_mode must be one of "
                "'ev_departure_service', 'ev_pv_price_peak' or 'combined'."
            )
        self._last_observation_event_priority_boost = 0.0
        self.critic_loss_function = str(exploration_cfg.get("critic_loss", "mse") or "mse").strip().lower()
        if self.critic_loss_function not in {"mse", "huber"}:
            raise ValueError("MADDPG critic_loss must be 'mse' or 'huber'.")
        self.critic_huber_beta = max(
            float(exploration_cfg.get("critic_huber_beta", 1.0) or 1.0),
            1.0e-6,
        )
        self.critic_target_clip_abs = max(
            0.0,
            float(exploration_cfg.get("critic_target_clip_abs", 0.0) or 0.0),
        )
        self.end_initial_exploration_time_step = max(
            0,
            int(exploration_cfg.get("end_initial_exploration_time_step", 0) or 0),
        )
        self.random_exploration_steps = max(
            0,
            int(exploration_cfg.get("random_exploration_steps", self.end_initial_exploration_time_step) or 0),
        )
        self.exploration_step = 0
        self.batch_size = buffer_cfg["batch_size"]
        self.lr_actor = float(network_cfg["actor"]["lr"])
        self.lr_critic = float(network_cfg["critic"]["lr"])

        training_cfg = self.config.get("training", {})
        checkpoint_cfg = self.config.get("checkpointing", {})
        tracking_cfg = self.config.get("tracking", {})

        self.seed = training_cfg.get("seed", 22)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.checkpoint_artifact = checkpoint_cfg.get("checkpoint_artifact", "latest_checkpoint.pth")
        self.reset_replay_buffer = checkpoint_cfg.get("reset_replay_buffer", False)
        self.freeze_pretrained_layers = checkpoint_cfg.get("freeze_pretrained_layers", False)
        self.fine_tune = checkpoint_cfg.get("fine_tune", False)
        try:
            self.mlflow_step_sample_interval = int(tracking_cfg.get("mlflow_step_sample_interval", 10) or 10)
        except (TypeError, ValueError):
            self.mlflow_step_sample_interval = 10
        if self.mlflow_step_sample_interval < 1:
            self.mlflow_step_sample_interval = 1
        self.runtime_profiling_enabled = bool(tracking_cfg.get("runtime_profiling_enabled", False))
        try:
            self.runtime_profiling_interval = int(tracking_cfg.get("runtime_profiling_interval", 512) or 512)
        except (TypeError, ValueError):
            self.runtime_profiling_interval = 512
        if self.runtime_profiling_interval < 1:
            self.runtime_profiling_interval = 512
        self.runtime_profiling_detail = str(
            tracking_cfg.get("runtime_profiling_detail", "summary") or "summary"
        ).strip().lower()
        if self.runtime_profiling_detail not in {"summary", "detailed"}:
            logger.warning(
                "Unknown runtime_profiling_detail '{}'; falling back to 'summary'.",
                self.runtime_profiling_detail,
            )
            self.runtime_profiling_detail = "summary"
        self.training_diagnostics_enabled = bool(tracking_cfg.get("training_diagnostics_enabled", True))
        self.training_diagnostics_detail = str(
            tracking_cfg.get("training_diagnostics_detail", "summary") or "summary"
        ).strip().lower()
        if self.training_diagnostics_detail not in {"summary", "per_agent"}:
            logger.warning(
                "Unknown training_diagnostics_detail '{}'; falling back to 'summary'.",
                self.training_diagnostics_detail,
            )
            self.training_diagnostics_detail = "summary"
        self._latest_training_metrics: Dict[str, float] = {}

        topology = self.config.get("topology", {})

        self.num_agents = topology.get("num_agents") or hyperparams.get("num_agents")
        self.observation_dimension = topology.get("observation_dimensions") or hyperparams.get("observation_dimensions")
        self.action_dimension = topology.get("action_dimensions") or hyperparams.get("action_dimensions")

        if self.num_agents is None or self.observation_dimension is None or self.action_dimension is None:
            raise ValueError("Topology information (num_agents / observation_dimensions / action_dimensions) is required for MADDPG.")

        self.action_low, self.action_high = self._default_action_bounds()
        self.action_names: List[List[str]] = [[] for _ in range(int(self.num_agents))]
        self.observation_names: List[List[str]] = [[] for _ in range(int(self.num_agents))]
        self.observation_space: List[Any] = []
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_raw_next_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_next_observations: Optional[List[np.ndarray]] = None
        self._warm_start_policy = None
        self._warned_missing_raw_context = False
        self._last_warm_start_policy_actions: Optional[List[List[float]]] = None
        self._last_warm_start_next_policy_actions: Optional[List[List[float]]] = None
        self._last_warm_start_phaseout_probability = 0.0
        self._last_warm_start_phaseout_used = False
        self._noop_actor_initialized = False
        self._replay_push_accepts_behavior_actions: Optional[bool] = None
        self._replay_push_accepts_next_behavior_actions: Optional[bool] = None
        self._replay_push_accepts_priority_boost: Optional[bool] = None
        self.replay_buffer = self._initialize_replay_buffer()
        self.actors, self.critics, self.actor_targets, self.critic_targets = self._initialize_networks()
        self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers()
        self.use_amp = bool(exploration_cfg.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        logger.info("MADDPG initialization complete.")

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "false"}:
            return None
        return text

    def _resolve_initial_exploration_strategy(self, exploration_cfg: Dict[str, Any]) -> str:
        configured = self._optional_string(exploration_cfg.get("initial_exploration_strategy"))
        strategy = configured or ("policy" if self.warm_start_policy_name else "uniform_full_range")
        strategy = strategy.strip().lower()
        aliases = {
            "uniform": "uniform_full_range",
            "random": "uniform_full_range",
            "full_range": "uniform_full_range",
            "noop": "noop_centered",
            "noop-centred": "noop_centered",
            "noop-centered": "noop_centered",
            "rbc": "policy",
            "baseline": "policy",
            "warm_start_policy": "policy",
        }
        strategy = aliases.get(strategy, strategy)
        if strategy not in {"uniform_full_range", "noop_centered", "policy"}:
            raise ValueError(
                "MADDPG initial_exploration_strategy must be one of "
                "'uniform_full_range', 'noop_centered' or 'policy'."
            )
        if strategy == "policy" and not self.warm_start_policy_name:
            raise ValueError("MADDPG initial_exploration_strategy='policy' requires warm_start_policy.")
        return strategy

    def _default_action_bounds(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        lows = [
            np.full(int(self.action_dimension[i]), -1.0, dtype=np.float32)
            for i in range(int(self.num_agents))
        ]
        highs = [
            np.full(int(self.action_dimension[i]), 1.0, dtype=np.float32)
            for i in range(int(self.num_agents))
        ]
        return lows, highs

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.observation_names = [list(names) for names in observation_names]
        self.action_names = [list(names) for names in action_names]
        self.observation_space = list(observation_space)
        lows, highs = self._default_action_bounds()
        for agent_idx, space in enumerate(action_space[: self.num_agents]):
            if not hasattr(space, "low") or not hasattr(space, "high"):
                continue
            low = np.asarray(space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(space.high, dtype=np.float32).reshape(-1)
            expected_dim = int(self.action_dimension[agent_idx])
            if low.shape[0] != expected_dim or high.shape[0] != expected_dim:
                logger.warning(
                    "Action bounds dimension mismatch for agent {}: low={}, high={}, expected={}. "
                    "Using default [-1, 1] bounds for this agent.",
                    agent_idx,
                    low.shape[0],
                    high.shape[0],
                    expected_dim,
                )
                continue
            lows[agent_idx] = low
            highs[agent_idx] = high
        self.action_low = lows
        self.action_high = highs
        self._configure_replay_behavior_action_priority()
        self._initialize_warm_start_policy(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self._apply_noop_actor_initialization()

    def _configure_replay_behavior_action_priority(self) -> None:
        replay_buffer = getattr(self, "replay_buffer", None)
        if not hasattr(replay_buffer, "set_behavior_action_priority_masks"):
            return
        replay_cfg = self.config["algorithm"].get("replay_buffer", {})
        scope = str(replay_cfg.get("behavior_action_priority_scope", "all") or "all").strip().lower()
        if scope != "ev":
            return
        masks = []
        for agent_idx in range(int(self.num_agents)):
            action_names = self._action_names_for_agent(agent_idx)
            action_dim = int(self.action_dimension[agent_idx])
            masks.append(
                [
                    bool(action_idx < len(action_names) and self._is_ev_action_name(action_names[action_idx]))
                    for action_idx in range(action_dim)
                ]
            )
        replay_buffer.set_behavior_action_priority_masks(masks)

    def _initialize_warm_start_policy(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        if getattr(self, "initial_exploration_strategy", "uniform_full_range") != "policy":
            return

        from algorithms.agents.baseline_policies import (  # Local import avoids registry cycles.
            NormalNoBatteryPolicy,
            NormalPolicy,
            RBCBasicPolicy,
            RBCCommunityPolicy,
            RBCSmartPolicy,
            RandomPolicy,
        )
        from algorithms.agents.rbc_agent import RuleBasedPolicy

        policy_registry = {
            "RuleBasedPolicy": RuleBasedPolicy,
            "RandomPolicy": RandomPolicy,
            "NormalNoBatteryPolicy": NormalNoBatteryPolicy,
            "NormalPolicy": NormalPolicy,
            "RBCBasicPolicy": RBCBasicPolicy,
            "RBCCommunityPolicy": RBCCommunityPolicy,
            "RBCSmartPolicy": RBCSmartPolicy,
        }
        policy_cls = policy_registry.get(str(self.warm_start_policy_name))
        if policy_cls is None:
            supported = ", ".join(sorted(policy_registry))
            raise ValueError(
                f"Unsupported MADDPG warm_start_policy '{self.warm_start_policy_name}'. "
                f"Supported policies: {supported}."
            )

        exploration_cfg = self.config["algorithm"]["exploration"]["params"]
        policy_hyperparams = exploration_cfg.get("warm_start_policy_hyperparameters") or {}
        if not isinstance(policy_hyperparams, dict):
            raise ValueError("MADDPG warm_start_policy_hyperparameters must be an object when provided.")

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
        logger.info("MADDPG initial exploration will use warm-start policy '{}'.", self.warm_start_policy_name)
        if getattr(self, "residual_policy_enabled", False):
            logger.info(
                "MADDPG residual policy enabled over warm-start policy '{}' with scale {} -> {}.",
                self.warm_start_policy_name,
                self.residual_action_scale,
                self.residual_action_final_scale,
            )

    def set_observation_context(
        self,
        *,
        raw_observations: Optional[List[np.ndarray]] = None,
        encoded_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Receive wrapper-side context for optional policy warm-start exploration."""
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
        self._latest_raw_next_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in raw_next_observations]
            if raw_next_observations is not None
            else None
        )
        if encoded_observations is not None:
            self._latest_encoded_observations = [
                np.asarray(obs, dtype=np.float64) for obs in encoded_observations
            ]
        self._latest_encoded_next_observations = (
            [np.asarray(obs, dtype=np.float64) for obs in encoded_next_observations]
            if encoded_next_observations is not None
            else None
        )
        self._last_warm_start_next_policy_actions = self._predict_warm_start_policy_for_observations(
            self._latest_raw_next_observations,
        )

    def _apply_noop_actor_initialization(self) -> None:
        if not getattr(self, "noop_actor_initialization", False) or getattr(self, "_noop_actor_initialized", False):
            return
        if not hasattr(self, "actors") or not hasattr(self, "actor_targets"):
            return

        for agent_idx, (actor, actor_target) in enumerate(zip(self.actors, self.actor_targets)):
            target_bias = self._noop_raw_actor_bias(agent_idx)
            for model in (actor, actor_target):
                output_layer = getattr(model, "fc_layers", [])[-1]
                with torch.no_grad():
                    output_layer.weight.zero_()
                    output_layer.bias.copy_(torch.as_tensor(target_bias, dtype=output_layer.bias.dtype, device=output_layer.bias.device))

        self._noop_actor_initialized = True
        logger.info("Applied MADDPG no-op-aware actor initialization.")

    def _noop_raw_actor_bias(self, agent_idx: int) -> np.ndarray:
        low = self._action_low_for_agent(agent_idx)
        high = self._action_high_for_agent(agent_idx)
        span = np.maximum(high - low, 1.0e-6)
        noop = self._noop_action_for_agent(agent_idx)
        desired = noop.copy()

        at_low = np.isclose(noop, low)
        at_high = np.isclose(noop, high)
        desired[at_low] = low[at_low] + self.noop_actor_initialization_epsilon * span[at_low]
        desired[at_high] = high[at_high] - self.noop_actor_initialization_epsilon * span[at_high]

        raw = 2.0 * (desired - low) / span - 1.0
        raw = np.clip(raw, -0.999, 0.999)
        return np.arctanh(raw).astype(np.float32)

    def _initialize_replay_buffer(self):
        logger.debug("Initializing replay buffer.")
        replay_buffer_name = self.config["algorithm"]["replay_buffer"]["class"]
        if replay_buffer_name == "PrioritizedReplayBuffer":
            raise ValueError(
                "PrioritizedReplayBuffer is not supported by MADDPG because it is single-agent. "
                "Use MultiAgentReplayBuffer or implement a multi-agent prioritized buffer."
            )
        try:
            replay_cls = REPLAY_BUFFER_REGISTRY[replay_buffer_name]
        except KeyError as exc:
            raise ValueError(f"Unknown replay buffer '{replay_buffer_name}'.") from exc

        params = {
            "capacity": self.config["algorithm"]["replay_buffer"]["capacity"],
            "batch_size": self.config["algorithm"]["replay_buffer"]["batch_size"],
            "num_agents": self.num_agents,
        }
        if replay_buffer_name == "RewardWeightedMultiAgentReplayBuffer":
            buffer_cfg = self.config["algorithm"]["replay_buffer"]
            params.update(
                {
                    "priority_fraction": buffer_cfg.get("priority_fraction", 0.5),
                    "priority_alpha": buffer_cfg.get("priority_alpha", 0.6),
                    "priority_epsilon": buffer_cfg.get("priority_epsilon", 1.0e-3),
                    "priority_mode": buffer_cfg.get("priority_mode", "abs_reward"),
                    "priority_max": buffer_cfg.get("priority_max"),
                    "behavior_action_priority_weight": buffer_cfg.get(
                        "behavior_action_priority_weight",
                        0.0,
                    ),
                    "behavior_action_priority_mode": buffer_cfg.get(
                        "behavior_action_priority_mode",
                        "positive",
                    ),
                    "behavior_action_priority_scope": buffer_cfg.get(
                        "behavior_action_priority_scope",
                        "all",
                    ),
                }
            )
        return replay_cls(**params)

    def _initialize_networks(self):
        logger.debug("Initializing actor and critic networks.")
        actor_fc_units = self.config["algorithm"]["networks"]["actor"]["layers"]
        critic_cfg = self.config["algorithm"]["networks"]["critic"]

        actors, critics, actor_targets, critic_targets = [], [], [], []
        for i in range(self.num_agents):
            state_size = self.observation_dimension[i]
            action_size = self.action_dimension[i]
            global_state_size = sum(self.observation_dimension)
            global_action_size = sum(self.action_dimension)

            actors.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critics.append(build_critic_network(global_state_size, global_action_size, self.seed, critic_cfg).to(self.device))
            actor_targets.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critic_targets.append(build_critic_network(global_state_size, global_action_size, self.seed, critic_cfg).to(self.device))

        for actor, actor_target in zip(actors, actor_targets):
            actor_target.load_state_dict(actor.state_dict())
        for critic, critic_target in zip(critics, critic_targets):
            critic_target.load_state_dict(critic.state_dict())

        return actors, critics, actor_targets, critic_targets

    def _initialize_optimizers(self):
        logger.debug("Initializing optimizers.")
        actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]
        critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics]
        return actor_optimizers, critic_optimizers

    def update(
        self,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor],
        rewards: List[float],
        next_observations: List[torch.Tensor],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        logger.debug("Starting update phase.")
        update_start_time = time.time()
        update_perf_start_time = time.perf_counter()
        should_log_runtime_profile = self._should_runtime_profile_step(global_learning_step)
        runtime_profile_metrics: Dict[str, float] = {}

        done = bool(terminated or truncated)
        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        self._update_reward_normalizer(rewards)
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_reward_normalizer_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        behavior_actions = self._transition_behavior_actions(actions)
        next_behavior_actions = self._transition_next_behavior_actions(behavior_actions)
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_behavior_action_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        priority_boost = self._transition_observation_event_priority_boost()
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_priority_boost_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        self._push_replay_transition(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            behavior_actions=behavior_actions,
            next_behavior_actions=next_behavior_actions,
            priority_boost=priority_boost,
        )
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_replay_push_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        if len(self.replay_buffer) < self.batch_size:
            logger.debug("Not enough samples in the replay buffer. Skipping update.")
            self._record_update_runtime_skip_metrics(
                runtime_profile_metrics,
                global_learning_step,
                update_perf_start_time,
                reason="replay_warmup",
            )
            return

        if not self._should_train_on_step(initial_exploration_done, global_learning_step):
            logger.debug("Initial exploration phase not finished. Skipping update.")
            self._record_update_runtime_skip_metrics(
                runtime_profile_metrics,
                global_learning_step,
                update_perf_start_time,
                reason="initial_exploration",
            )
            return

        if not update_step:
            logger.debug("Update step skipped based on schedule.")
            self._record_update_runtime_skip_metrics(
                runtime_profile_metrics,
                global_learning_step,
                update_perf_start_time,
                reason="schedule",
            )
            return

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        if hasattr(self.replay_buffer, "sample_with_policy_context_actions"):
            states, actions_all, rewards_all, next_states, dones_all, behavior_actions_all, next_behavior_actions_all = (
                self.replay_buffer.sample_with_policy_context_actions()
            )
        elif hasattr(self.replay_buffer, "sample_with_behavior_actions"):
            states, actions_all, rewards_all, next_states, dones_all, behavior_actions_all = (
                self.replay_buffer.sample_with_behavior_actions()
            )
            next_behavior_actions_all = behavior_actions_all
        else:
            states, actions_all, rewards_all, next_states, dones_all = self.replay_buffer.sample()
            behavior_actions_all = actions_all
            next_behavior_actions_all = actions_all
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_replay_sample_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        raw_rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)
        rewards_all = self._normalize_reward_tensor(raw_rewards_all)
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)
        states = [s.to(self.device, non_blocking=True) for s in states]
        actions_all = [a.to(self.device, non_blocking=True) for a in actions_all]
        behavior_actions_all = [a.to(self.device, non_blocking=True) for a in behavior_actions_all]
        next_behavior_actions_all = [a.to(self.device, non_blocking=True) for a in next_behavior_actions_all]
        next_states = [ns.to(self.device, non_blocking=True) for ns in next_states]
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_tensor_transfer_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_tensor_prepare_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        with torch.no_grad():
            next_policy_actions = []
            for agent_idx in range(self.num_agents):
                next_action = self._policy_action_from_actor_output(
                    agent_idx,
                    self.actor_targets[agent_idx](next_states[agent_idx]),
                    base_action=next_behavior_actions_all[agent_idx],
                    global_learning_step=global_learning_step,
                )
                if self.target_policy_smoothing:
                    next_action = self._add_target_policy_smoothing(agent_idx, next_action)
                next_policy_actions.append(next_action)
            global_next_actions = torch.cat(next_policy_actions, dim=1)
            q_targets_next = torch.stack(
                [critic(global_next_state, global_next_actions) for critic in self.critic_targets]
            )
            q_targets = rewards_all + self.gamma * q_targets_next * (1 - dones_all)
            if self.critic_target_clip_abs > 0.0:
                q_targets = torch.clamp(
                    q_targets,
                    -self.critic_target_clip_abs,
                    self.critic_target_clip_abs,
                )
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_target_compute_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        if self.critic_update_mode == "per_agent":
            critic_loss_values: List[float] = []
            critic_td_abs_values: List[float] = []
            critic_grad_norm_values: List[float] = []
            q_expected_stat_tensors: List[torch.Tensor] = []
            for agent_num, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    q_expected_agent = critic(global_state, global_actions)
                    critic_loss_agent = self._critic_loss(q_expected_agent, q_targets[agent_num])

                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(critic_loss_agent).backward()
                self.scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(critic.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                critic_loss_values.append(float(critic_loss_agent.item()))
                critic_td_abs_values.append(
                    float((q_expected_agent.detach() - q_targets[agent_num].detach()).abs().mean().item())
                )
                critic_grad_norm_values.append(float(grad_norm))
                q_expected_stat_tensors.append(q_expected_agent.detach())
            critic_loss_scalar = float(np.mean(critic_loss_values)) if critic_loss_values else 0.0
        else:
            q_expected = torch.stack([critic(global_state, global_actions) for critic in self.critics])
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                critic_loss = self._critic_loss(q_expected, q_targets).mean()

            for optimizer in self.critic_optimizers:
                optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(critic_loss).backward()
            for optimizer in self.critic_optimizers:
                self.scaler.unscale_(optimizer)
            critic_grad_norm = clip_grad_norm_(
                [param for critic in self.critics for param in critic.parameters()],
                max_norm=1.0,
            )
            for optimizer in self.critic_optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()

            critic_loss_scalar = float(critic_loss.item())
            critic_loss_values = [
                float(self._critic_loss(q_expected[agent_num], q_targets[agent_num]).item())
                for agent_num in range(self.num_agents)
            ]
            critic_td_abs_values = [
                float((q_expected[agent_num].detach() - q_targets[agent_num].detach()).abs().mean().item())
                for agent_num in range(self.num_agents)
            ]
            critic_grad_norm_values = [float(critic_grad_norm)]
            q_expected_stat_tensors = [q_expected.detach()]
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_critic_update_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        logger.debug("Critics updated. Loss: {:.4f}.", critic_loss_scalar)

        should_log_step_metrics = (
            self._should_log_training_step(global_learning_step) or should_log_runtime_profile
        )

        # Compute detached policy actions once and reuse them during actor updates.
        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        with torch.no_grad():
            detached_policy_actions = [
                self._policy_action_from_actor_output(
                    agent_idx,
                    actor(state),
                    base_action=behavior_actions_all[agent_idx],
                    global_learning_step=global_learning_step,
                ).detach()
                for agent_idx, (actor, state) in enumerate(zip(self.actors, states))
            ]
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_detached_policy_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        actor_update_due = (
            self.actor_update_interval <= 1
            or global_learning_step % self.actor_update_interval == 0
        )
        total_actor_loss = 0.0
        actor_loss_values: List[float] = []
        actor_policy_loss_values: List[float] = []
        actor_policy_loss_weighted_values: List[float] = []
        actor_regularization_values: List[float] = []
        actor_action_l2_values: List[float] = []
        actor_action_saturation_values: List[float] = []
        actor_storage_action_l2_values: List[float] = []
        actor_ev_v2g_action_l2_values: List[float] = []
        actor_ev_v2g_action_mass_values: List[float] = []
        actor_behavior_cloning_loss_values: List[float] = []
        actor_behavior_cloning_ev_loss_values: List[float] = []
        actor_behavior_cloning_storage_loss_values: List[float] = []
        actor_behavior_cloning_regularization_values: List[float] = []
        actor_behavior_cloning_extra_loss_values: List[float] = []
        actor_behavior_cloning_extra_grad_norm_values: List[float] = []
        actor_grad_norm_values: List[float] = []
        actor_behavior_cloning_effective_weight = self._actor_behavior_cloning_effective_weight(
            global_learning_step
        )
        actor_policy_loss_effective_weight = self._actor_policy_loss_effective_weight(
            global_learning_step
        )
        actor_behavior_cloning_extra_updates = self._actor_behavior_cloning_extra_updates_for_step(
            global_learning_step,
            actor_behavior_cloning_effective_weight,
        )
        phase_start_time = time.perf_counter() if should_log_runtime_profile else 0.0
        if actor_update_due:
            for agent_num, (actor, critic, actor_optimizer) in enumerate(
                zip(self.actors, self.critics, self.actor_optimizers)
            ):
                obs = states[agent_num]
                extra_losses, extra_grad_norms = self._run_actor_behavior_cloning_extra_updates(
                    agent_num=agent_num,
                    actor=actor,
                    actor_optimizer=actor_optimizer,
                    observations=obs,
                    behavior_actions=behavior_actions_all[agent_num],
                    behavior_cloning_weight=actor_behavior_cloning_effective_weight,
                    extra_updates=actor_behavior_cloning_extra_updates,
                )
                actor_behavior_cloning_extra_loss_values.extend(extra_losses)
                actor_behavior_cloning_extra_grad_norm_values.extend(extra_grad_norms)

                predicted_action = self._policy_action_from_actor_output(
                    agent_num,
                    actor(obs),
                    base_action=behavior_actions_all[agent_num],
                    global_learning_step=global_learning_step,
                )
                joint_policy_actions = list(detached_policy_actions)
                joint_policy_actions[agent_num] = predicted_action
                global_predicted_actions = torch.cat(joint_policy_actions, dim=1)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    actor_policy_loss = -critic(global_state, global_predicted_actions).mean()
                    weighted_actor_policy_loss = actor_policy_loss_effective_weight * actor_policy_loss
                    (
                        action_l2,
                        action_saturation,
                        storage_action_l2,
                        ev_v2g_action_l2,
                        ev_v2g_action_mass,
                        actor_regularization,
                    ) = self._actor_action_regularization_terms(
                        agent_num,
                        predicted_action,
                    )
                    behavior_cloning_loss = self._actor_behavior_cloning_loss(
                        agent_num,
                        predicted_action,
                        behavior_actions_all[agent_num],
                    )
                    behavior_cloning_ev_loss, behavior_cloning_storage_loss = (
                        self._actor_behavior_cloning_type_losses(
                            agent_num,
                            predicted_action,
                            behavior_actions_all[agent_num],
                        )
                    )
                    behavior_cloning_regularization = (
                        actor_behavior_cloning_effective_weight * behavior_cloning_loss
                    )
                    total_regularization = actor_regularization + behavior_cloning_regularization
                    actor_loss = weighted_actor_policy_loss + total_regularization

                actor_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(actor_optimizer)
                actor_grad_norm = clip_grad_norm_(actor.parameters(), max_norm=1.0)
                self.scaler.step(actor_optimizer)
                self.scaler.update()

                # Keep cached detached actions in sync with sequential actor updates.
                with torch.no_grad():
                    detached_policy_actions[agent_num] = self._policy_action_from_actor_output(
                        agent_num,
                        actor(obs),
                        base_action=behavior_actions_all[agent_num],
                        global_learning_step=global_learning_step,
                    ).detach()

                total_actor_loss += actor_loss.item()
                actor_loss_values.append(float(actor_loss.item()))
                actor_policy_loss_values.append(float(actor_policy_loss.detach().item()))
                actor_policy_loss_weighted_values.append(float(weighted_actor_policy_loss.detach().item()))
                actor_regularization_values.append(float(total_regularization.detach().item()))
                actor_action_l2_values.append(float(action_l2.detach().item()))
                actor_action_saturation_values.append(float(action_saturation.detach().item()))
                actor_storage_action_l2_values.append(float(storage_action_l2.detach().item()))
                actor_ev_v2g_action_l2_values.append(float(ev_v2g_action_l2.detach().item()))
                actor_ev_v2g_action_mass_values.append(float(ev_v2g_action_mass.detach().item()))
                actor_behavior_cloning_loss_values.append(float(behavior_cloning_loss.detach().item()))
                actor_behavior_cloning_ev_loss_values.append(float(behavior_cloning_ev_loss.detach().item()))
                actor_behavior_cloning_storage_loss_values.append(
                    float(behavior_cloning_storage_loss.detach().item())
                )
                actor_behavior_cloning_regularization_values.append(
                    float(behavior_cloning_regularization.detach().item())
                )
                actor_grad_norm_values.append(float(actor_grad_norm))
                logger.debug("Actor {} updated. Loss: {:.4f}.", agent_num, actor_loss.item())

                if update_target_step:
                    logger.debug("Updating target networks for agent {}.", agent_num)
                    self._soft_update(critic, self.critic_targets[agent_num], self.tau)
                    self._soft_update(actor, self.actor_targets[agent_num], self.tau)
        else:
            actor_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_weighted_values = [0.0 for _ in range(self.num_agents)]
            actor_regularization_values = [0.0 for _ in range(self.num_agents)]
            actor_action_l2_values = [0.0 for _ in range(self.num_agents)]
            actor_action_saturation_values = [0.0 for _ in range(self.num_agents)]
            actor_storage_action_l2_values = [0.0 for _ in range(self.num_agents)]
            actor_ev_v2g_action_l2_values = [0.0 for _ in range(self.num_agents)]
            actor_ev_v2g_action_mass_values = [0.0 for _ in range(self.num_agents)]
            actor_behavior_cloning_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_behavior_cloning_ev_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_behavior_cloning_storage_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_behavior_cloning_regularization_values = [0.0 for _ in range(self.num_agents)]
            actor_behavior_cloning_extra_loss_values = [0.0]
            actor_behavior_cloning_extra_grad_norm_values = [0.0]
            actor_grad_norm_values = [0.0 for _ in range(self.num_agents)]
            logger.debug(
                "Actor update delayed at step {} by actor_update_interval={}.",
                global_learning_step,
                self.actor_update_interval,
            )
        if should_log_runtime_profile:
            runtime_profile_metrics["MADDPG/runtime_actor_update_seconds"] = (
                time.perf_counter() - phase_start_time
            )

        if should_log_step_metrics and self.training_diagnostics_enabled:
            metrics_start_time = time.perf_counter()
            q_expected_flat = torch.cat([tensor.reshape(-1) for tensor in q_expected_stat_tensors])
            q_targets_flat = q_targets.detach().reshape(-1)
            training_metrics: Dict[str, float] = {
                "MADDPG/average_critic_loss": critic_loss_scalar,
                "MADDPG/average_actor_loss": total_actor_loss / self.num_agents,
                "MADDPG/actor_update_performed": float(actor_update_due),
                "MADDPG/actor_policy_loss_mean": float(np.mean(actor_policy_loss_values)),
                "MADDPG/actor_policy_loss_weighted_mean": float(np.mean(actor_policy_loss_weighted_values)),
                "MADDPG/actor_policy_loss_effective_weight": float(actor_policy_loss_effective_weight),
                "MADDPG/actor_regularization_loss_mean": float(np.mean(actor_regularization_values)),
                "MADDPG/actor_action_l2_mean": float(np.mean(actor_action_l2_values)),
                "MADDPG/actor_action_saturation_excess_mean": float(np.mean(actor_action_saturation_values)),
                "MADDPG/actor_storage_action_l2_mean": float(np.mean(actor_storage_action_l2_values)),
                "MADDPG/actor_ev_v2g_action_l2_mean": float(np.mean(actor_ev_v2g_action_l2_values)),
                "MADDPG/actor_ev_v2g_action_mass_mean": float(np.mean(actor_ev_v2g_action_mass_values)),
                "MADDPG/actor_behavior_cloning_loss_mean": float(np.mean(actor_behavior_cloning_loss_values)),
                "MADDPG/actor_behavior_cloning_ev_loss_mean": float(
                    np.mean(actor_behavior_cloning_ev_loss_values)
                ),
                "MADDPG/actor_behavior_cloning_storage_loss_mean": float(
                    np.mean(actor_behavior_cloning_storage_loss_values)
                ),
                "MADDPG/actor_behavior_cloning_regularization_mean": float(
                    np.mean(actor_behavior_cloning_regularization_values)
                ),
                "MADDPG/actor_behavior_cloning_extra_updates": float(
                    actor_behavior_cloning_extra_updates
                ),
                "MADDPG/actor_behavior_cloning_extra_loss_mean": float(
                    np.mean(actor_behavior_cloning_extra_loss_values)
                    if actor_behavior_cloning_extra_loss_values
                    else 0.0
                ),
                "MADDPG/actor_behavior_cloning_extra_grad_norm_mean": float(
                    np.mean(actor_behavior_cloning_extra_grad_norm_values)
                    if actor_behavior_cloning_extra_grad_norm_values
                    else 0.0
                ),
                "MADDPG/actor_behavior_cloning_effective_weight": float(
                    actor_behavior_cloning_effective_weight
                ),
                "MADDPG/actor_behavior_cloning_source_warm_start_policy": float(
                    getattr(self, "actor_behavior_cloning_source", "replay_action") == "warm_start_policy"
                ),
                "MADDPG/residual_policy_enabled": float(
                    getattr(self, "residual_policy_enabled", False)
                ),
                "MADDPG/residual_action_scale_effective": float(
                    getattr(self, "_last_residual_action_scale", 0.0)
                ),
                "MADDPG/reward_raw_mean": float(raw_rewards_all.mean().item()),
                "MADDPG/reward_raw_std": float(raw_rewards_all.std(unbiased=False).item()),
                "MADDPG/reward_train_mean": float(rewards_all.mean().item()),
                "MADDPG/reward_train_std": float(rewards_all.std(unbiased=False).item()),
                "MADDPG/reward_norm_count": float(getattr(self, "reward_norm_count", 0)),
                "MADDPG/reward_norm_mean": float(getattr(self, "reward_norm_mean", 0.0)),
                "MADDPG/reward_norm_std": float(self._reward_normalization_std()),
                "MADDPG/critic_loss_huber": float(getattr(self, "critic_loss_function", "mse") == "huber"),
                "MADDPG/critic_target_clip_abs": float(getattr(self, "critic_target_clip_abs", 0.0)),
                "MADDPG/critic_td_abs_mean": float(np.mean(critic_td_abs_values)),
                "MADDPG/critic_td_abs_max": float(np.max(critic_td_abs_values)),
                "MADDPG/critic_grad_norm_mean": float(np.mean(critic_grad_norm_values)),
                "MADDPG/critic_grad_norm_max": float(np.max(critic_grad_norm_values)),
                "MADDPG/actor_grad_norm_mean": float(np.mean(actor_grad_norm_values)),
                "MADDPG/actor_grad_norm_max": float(np.max(actor_grad_norm_values)),
                "MADDPG/q_expected_mean": float(q_expected_flat.mean().item()),
                "MADDPG/q_expected_std": float(q_expected_flat.std(unbiased=False).item()),
                "MADDPG/q_expected_min": float(q_expected_flat.min().item()),
                "MADDPG/q_expected_max": float(q_expected_flat.max().item()),
                "MADDPG/q_target_mean": float(q_targets_flat.mean().item()),
                "MADDPG/q_target_std": float(q_targets_flat.std(unbiased=False).item()),
                "MADDPG/q_target_min": float(q_targets_flat.min().item()),
                "MADDPG/q_target_max": float(q_targets_flat.max().item()),
                "MADDPG/replay_buffer_size": float(len(self.replay_buffer)),
                "MADDPG/replay_observation_event_priority_last": float(
                    getattr(self, "_last_observation_event_priority_boost", 0.0)
                ),
                "MADDPG/exploration_sigma": float(getattr(self, "sigma", 0.0)),
                "MADDPG/exploration_step": float(getattr(self, "exploration_step", 0)),
                "MADDPG/training_step_time": time.time() - update_start_time,
                "MADDPG/training_step_perf_seconds": time.perf_counter() - update_perf_start_time,
            }
            if self.training_diagnostics_detail == "per_agent":
                for agent_num in range(self.num_agents):
                    training_metrics[f"MADDPG/critic_loss_agent_{agent_num}"] = critic_loss_values[agent_num]
                    training_metrics[f"MADDPG/critic_td_abs_agent_{agent_num}"] = critic_td_abs_values[agent_num]
                    training_metrics[f"MADDPG/actor_loss_agent_{agent_num}"] = actor_loss_values[agent_num]
                    training_metrics[f"MADDPG/actor_policy_loss_agent_{agent_num}"] = actor_policy_loss_values[agent_num]
                    training_metrics[f"MADDPG/actor_policy_loss_weighted_agent_{agent_num}"] = (
                        actor_policy_loss_weighted_values[agent_num]
                    )
                    training_metrics[f"MADDPG/actor_regularization_loss_agent_{agent_num}"] = actor_regularization_values[agent_num]
                    training_metrics[f"MADDPG/actor_action_l2_agent_{agent_num}"] = actor_action_l2_values[agent_num]
                    training_metrics[f"MADDPG/actor_action_saturation_excess_agent_{agent_num}"] = actor_action_saturation_values[agent_num]
                    training_metrics[f"MADDPG/actor_storage_action_l2_agent_{agent_num}"] = actor_storage_action_l2_values[agent_num]
                    training_metrics[f"MADDPG/actor_ev_v2g_action_l2_agent_{agent_num}"] = actor_ev_v2g_action_l2_values[agent_num]
                    training_metrics[f"MADDPG/actor_ev_v2g_action_mass_agent_{agent_num}"] = actor_ev_v2g_action_mass_values[agent_num]
                    training_metrics[f"MADDPG/actor_behavior_cloning_loss_agent_{agent_num}"] = (
                        actor_behavior_cloning_loss_values[agent_num]
                    )
                    training_metrics[f"MADDPG/actor_behavior_cloning_ev_loss_agent_{agent_num}"] = (
                        actor_behavior_cloning_ev_loss_values[agent_num]
                    )
                    training_metrics[f"MADDPG/actor_behavior_cloning_storage_loss_agent_{agent_num}"] = (
                        actor_behavior_cloning_storage_loss_values[agent_num]
                    )
                    training_metrics[f"MADDPG/actor_behavior_cloning_regularization_agent_{agent_num}"] = (
                        actor_behavior_cloning_regularization_values[agent_num]
                    )
                    training_metrics[f"MADDPG/actor_grad_norm_agent_{agent_num}"] = actor_grad_norm_values[agent_num]
            if should_log_runtime_profile:
                runtime_profile_metrics["MADDPG/runtime_metrics_build_seconds"] = (
                    time.perf_counter() - metrics_start_time
                )
                training_metrics.update(runtime_profile_metrics)
            self._record_training_metrics(training_metrics, global_learning_step)

        log_message = "Update complete. Avg Critic Loss: {:.4f}, Avg Actor Loss: {:.4f}."
        if should_log_step_metrics:
            logger.info(log_message, critic_loss_scalar, total_actor_loss / self.num_agents)
        else:
            logger.debug(log_message, critic_loss_scalar, total_actor_loss / self.num_agents)

    def _push_replay_transition(
        self,
        *,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor],
        rewards: List[float],
        next_observations: List[torch.Tensor],
        done: bool,
        behavior_actions: List[Any],
        priority_boost: float,
        next_behavior_actions: Optional[List[Any]] = None,
    ) -> None:
        """Push a transition while preserving compatibility with simple buffers."""
        push = self.replay_buffer.push
        kwargs: Dict[str, Any] = {}

        accepts_behavior_actions = getattr(self, "_replay_push_accepts_behavior_actions", None)
        accepts_next_behavior_actions = getattr(self, "_replay_push_accepts_next_behavior_actions", None)
        accepts_priority_boost = getattr(self, "_replay_push_accepts_priority_boost", None)
        if (
            accepts_behavior_actions is None
            or accepts_next_behavior_actions is None
            or accepts_priority_boost is None
        ):
            try:
                parameters = inspect.signature(push).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            accepts_behavior_actions = accepts_kwargs or "behavior_actions" in parameters
            accepts_next_behavior_actions = accepts_kwargs or "next_behavior_actions" in parameters
            accepts_priority_boost = accepts_kwargs or "priority_boost" in parameters
            self._replay_push_accepts_behavior_actions = accepts_behavior_actions
            self._replay_push_accepts_next_behavior_actions = accepts_next_behavior_actions
            self._replay_push_accepts_priority_boost = accepts_priority_boost

        if accepts_behavior_actions:
            kwargs["behavior_actions"] = behavior_actions
        if accepts_next_behavior_actions and next_behavior_actions is not None:
            kwargs["next_behavior_actions"] = next_behavior_actions
        if accepts_priority_boost:
            kwargs["priority_boost"] = priority_boost

        push(observations, actions, rewards, next_observations, done, **kwargs)

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.end_initial_exploration_time_step

    def _should_train_on_step(self, initial_exploration_done: bool, global_learning_step: int) -> bool:
        if initial_exploration_done:
            return True
        return bool(getattr(self, "train_during_initial_exploration", False)) and (
            global_learning_step >= getattr(self, "initial_exploration_training_start_step", 0)
        )

    def _should_log_training_step(self, global_learning_step: int) -> bool:
        return global_learning_step % self.mlflow_step_sample_interval == 0

    def _should_runtime_profile_step(self, global_learning_step: int) -> bool:
        return bool(getattr(self, "runtime_profiling_enabled", False)) and (
            global_learning_step % getattr(self, "runtime_profiling_interval", 512) == 0
        )

    def _record_update_runtime_skip_metrics(
        self,
        metrics: Dict[str, float],
        global_learning_step: int,
        update_perf_start_time: float,
        *,
        reason: str,
    ) -> None:
        if not self._should_runtime_profile_step(global_learning_step):
            return
        metrics = dict(metrics)
        metrics["MADDPG/training_step_perf_seconds"] = time.perf_counter() - update_perf_start_time
        metrics["MADDPG/update_skip_replay_warmup"] = float(reason == "replay_warmup")
        metrics["MADDPG/update_skip_initial_exploration"] = float(reason == "initial_exploration")
        metrics["MADDPG/update_skip_schedule"] = float(reason == "schedule")
        self._record_training_metrics(metrics, global_learning_step)

    def _update_reward_normalizer(self, rewards: List[float]) -> None:
        if not getattr(self, "reward_normalization_enabled", False):
            return
        values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        for value in values:
            if not np.isfinite(value):
                continue
            self.reward_norm_count += 1
            delta = float(value) - self.reward_norm_mean
            self.reward_norm_mean += delta / self.reward_norm_count
            delta2 = float(value) - self.reward_norm_mean
            self.reward_norm_m2 += delta * delta2

    def _reward_normalization_std(self) -> float:
        if getattr(self, "reward_norm_count", 0) < 2:
            return 1.0
        variance = max(self.reward_norm_m2 / (self.reward_norm_count - 1), 0.0)
        epsilon = float(getattr(self, "reward_normalization_epsilon", 1.0e-8))
        return max(float(np.sqrt(variance)), epsilon)

    def _normalize_reward_tensor(self, rewards: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "reward_normalization_enabled", False) or getattr(self, "reward_norm_count", 0) < 2:
            return rewards
        mean = torch.as_tensor(self.reward_norm_mean, dtype=rewards.dtype, device=rewards.device)
        std = torch.as_tensor(self._reward_normalization_std(), dtype=rewards.dtype, device=rewards.device)
        normalized = (rewards - mean) / std
        clip_value = float(getattr(self, "reward_normalization_clip", 10.0))
        return torch.clamp(normalized, -clip_value, clip_value)

    def _critic_loss(self, expected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if getattr(self, "critic_loss_function", "mse") == "huber":
            return smooth_l1_loss(
                expected,
                target,
                beta=float(getattr(self, "critic_huber_beta", 1.0)),
            )
        return mse_loss(expected, target)

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        metrics = {
            "MADDPG/replay_buffer_size": float(len(self.replay_buffer)),
            "MADDPG/exploration_step": float(getattr(self, "exploration_step", 0)),
            "MADDPG/exploration_sigma": float(getattr(self, "sigma", 0.0)),
            "MADDPG/storage_exploration_noise_multiplier": float(
                getattr(self, "storage_exploration_noise_multiplier", 1.0)
            ),
            "MADDPG/ev_negative_exploration_noise_multiplier": float(
                getattr(self, "ev_negative_exploration_noise_multiplier", 1.0)
            ),
            "MADDPG/random_exploration_steps": float(getattr(self, "random_exploration_steps", 0)),
            "MADDPG/warm_start_policy_phaseout_steps": float(
                getattr(self, "warm_start_policy_phaseout_steps", 0)
            ),
            "MADDPG/warm_start_policy_phaseout_probability": float(
                getattr(self, "_last_warm_start_phaseout_probability", 0.0)
            ),
            "MADDPG/warm_start_policy_phaseout_used": float(
                getattr(self, "_last_warm_start_phaseout_used", False)
            ),
            "MADDPG/warm_start_policy_phaseout_mode_blend": float(
                getattr(self, "warm_start_policy_phaseout_mode", "probability") == "blend"
            ),
            "MADDPG/train_during_initial_exploration": float(
                getattr(self, "train_during_initial_exploration", False)
            ),
            "MADDPG/initial_exploration_training_start_step": float(
                getattr(self, "initial_exploration_training_start_step", 0)
            ),
            "MADDPG/action_warmup_done": float(
                getattr(self, "exploration_step", 0) >= getattr(self, "random_exploration_steps", 0)
            ),
            "MADDPG/initial_exploration_done": float(
                getattr(self, "exploration_step", 0) >= getattr(self, "end_initial_exploration_time_step", 0)
            ),
            "MADDPG/reward_normalization_enabled": float(getattr(self, "reward_normalization_enabled", False)),
            "MADDPG/reward_norm_count": float(getattr(self, "reward_norm_count", 0)),
            "MADDPG/reward_norm_mean": float(getattr(self, "reward_norm_mean", 0.0)),
            "MADDPG/reward_norm_std": float(self._reward_normalization_std()),
            "MADDPG/critic_loss_huber": float(getattr(self, "critic_loss_function", "mse") == "huber"),
            "MADDPG/critic_target_clip_abs": float(getattr(self, "critic_target_clip_abs", 0.0)),
            "MADDPG/actor_update_interval": float(getattr(self, "actor_update_interval", 1)),
            "MADDPG/actor_policy_loss_weight": float(getattr(self, "actor_policy_loss_weight", 1.0)),
            "MADDPG/actor_policy_loss_warmup_weight": float(
                getattr(self, "actor_policy_loss_warmup_weight", getattr(self, "actor_policy_loss_weight", 1.0))
            ),
            "MADDPG/actor_policy_loss_warmup_steps": float(
                getattr(self, "actor_policy_loss_warmup_steps", 0)
            ),
            "MADDPG/target_policy_smoothing": float(getattr(self, "target_policy_smoothing", False)),
            "MADDPG/actor_action_l2_penalty": float(getattr(self, "actor_action_l2_penalty", 0.0)),
            "MADDPG/actor_action_saturation_penalty": float(
                getattr(self, "actor_action_saturation_penalty", 0.0)
            ),
            "MADDPG/actor_storage_action_l2_penalty": float(
                getattr(self, "actor_storage_action_l2_penalty", 0.0)
            ),
            "MADDPG/actor_ev_v2g_action_l2_penalty": float(
                getattr(self, "actor_ev_v2g_action_l2_penalty", 0.0)
            ),
            "MADDPG/actor_ev_v2g_action_mass_penalty": float(
                getattr(self, "actor_ev_v2g_action_mass_penalty", 0.0)
            ),
            "MADDPG/actor_behavior_cloning_weight": float(
                getattr(self, "actor_behavior_cloning_weight", 0.0)
            ),
            "MADDPG/actor_ev_behavior_cloning_multiplier": float(
                getattr(self, "actor_ev_behavior_cloning_multiplier", 1.0)
            ),
            "MADDPG/actor_ev_behavior_cloning_positive_target_weight": float(
                getattr(self, "actor_ev_behavior_cloning_positive_target_weight", 0.0)
            ),
            "MADDPG/actor_ev_behavior_cloning_positive_target_power": float(
                getattr(self, "actor_ev_behavior_cloning_positive_target_power", 1.0)
            ),
            "MADDPG/actor_ev_behavior_cloning_zero_target_weight": float(
                getattr(self, "actor_ev_behavior_cloning_zero_target_weight", 0.0)
            ),
            "MADDPG/actor_ev_behavior_cloning_zero_target_threshold": float(
                getattr(self, "actor_ev_behavior_cloning_zero_target_threshold", 0.05)
            ),
            "MADDPG/actor_storage_behavior_cloning_multiplier": float(
                getattr(self, "actor_storage_behavior_cloning_multiplier", 1.0)
            ),
            "MADDPG/actor_behavior_cloning_min_weight": float(
                getattr(self, "actor_behavior_cloning_min_weight", 0.0)
            ),
            "MADDPG/actor_behavior_cloning_decay_steps": float(
                getattr(self, "actor_behavior_cloning_decay_steps", 0)
            ),
            "MADDPG/actor_behavior_cloning_extra_updates": float(
                getattr(self, "actor_behavior_cloning_extra_updates", 0)
            ),
            "MADDPG/actor_behavior_cloning_extra_update_start_step": float(
                getattr(self, "actor_behavior_cloning_extra_update_start_step", 0)
            ),
            "MADDPG/actor_behavior_cloning_extra_update_end_step": float(
                getattr(self, "actor_behavior_cloning_extra_update_end_step", 0)
            ),
            "MADDPG/actor_behavior_cloning_source_warm_start_policy": float(
                getattr(self, "actor_behavior_cloning_source", "replay_action") == "warm_start_policy"
            ),
            "MADDPG/residual_policy_enabled": float(getattr(self, "residual_policy_enabled", False)),
            "MADDPG/residual_action_scale": float(getattr(self, "residual_action_scale", 0.0)),
            "MADDPG/residual_action_final_scale": float(
                getattr(self, "residual_action_final_scale", 0.0)
            ),
            "MADDPG/residual_action_scale_effective": float(
                getattr(self, "_last_residual_action_scale", 0.0)
            ),
            "MADDPG/residual_action_start_step": float(
                getattr(self, "residual_action_start_step", 0)
            ),
            "MADDPG/residual_action_growth_steps": float(
                getattr(self, "residual_action_growth_steps", 0)
            ),
            "MADDPG/replay_observation_event_priority_weight": float(
                getattr(self, "replay_observation_event_priority_weight", 0.0)
            ),
            "MADDPG/replay_observation_event_priority_last": float(
                getattr(self, "_last_observation_event_priority_boost", 0.0)
            ),
        }
        replay_buffer = getattr(self, "replay_buffer", None)
        if replay_buffer is not None and hasattr(replay_buffer, "priority_fraction"):
            metrics.update(
                {
                    "MADDPG/replay_priority_fraction": float(
                        getattr(replay_buffer, "priority_fraction", 0.0)
                    ),
                    "MADDPG/replay_priority_alpha": float(
                        getattr(replay_buffer, "priority_alpha", 0.0)
                    ),
                    "MADDPG/replay_priority_max": float(
                        getattr(replay_buffer, "priority_max", 0.0) or 0.0
                    ),
                    "MADDPG/replay_behavior_action_priority_weight": float(
                        getattr(replay_buffer, "behavior_action_priority_weight", 0.0)
                    ),
                    "MADDPG/replay_behavior_action_priority_scope_ev": float(
                        getattr(replay_buffer, "behavior_action_priority_scope", "all") == "ev"
                    ),
                }
            )
        return metrics

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = dict(getattr(self, "_latest_training_metrics", {}))
        self._latest_training_metrics = {}
        return metrics

    def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not metrics:
            return
        self._latest_training_metrics = dict(metrics)
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def _soft_update(self, local_model, target_model, tau):
        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.lerp_(local_param.data, tau)

    def predict(self, observations, deterministic: bool = False) -> List[List[float]]:
        logger.debug("Predicting actions with deterministic={}.", deterministic)
        if deterministic:
            return self._predict_deterministic(observations)
        return self._predict_with_exploration(observations)

    def _predict_deterministic(self, observations):
        base_actions = self._current_residual_base_actions()
        actions = []
        with torch.inference_mode():
            for agent_idx, (actor, obs) in enumerate(zip(self.actors, observations)):
                raw_action = actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                base_action = None
                if base_actions is not None and agent_idx < len(base_actions):
                    base_action = torch.as_tensor(
                        base_actions[agent_idx],
                        dtype=raw_action.dtype,
                        device=raw_action.device,
                    )
                action = self._policy_action_from_actor_output(
                    agent_idx,
                    raw_action,
                    base_action=base_action,
                ).cpu().numpy()
                actions.append(action)
        logger.debug("Deterministic actions predicted: {}", actions)
        return actions

    def _predict_with_exploration(self, observations):
        self.exploration_step += 1
        if self.exploration_step <= self.random_exploration_steps:
            initial_strategy = getattr(self, "initial_exploration_strategy", "uniform_full_range")
            if initial_strategy == "policy":
                return self._predict_warm_start_policy()
            if initial_strategy == "noop_centered":
                return self._predict_noop_centered()
            return self._predict_random()

        phaseout_probability = self._warm_start_phaseout_probability()
        self._last_warm_start_phaseout_probability = float(phaseout_probability)
        self._last_warm_start_phaseout_used = False
        phaseout_mode = getattr(self, "warm_start_policy_phaseout_mode", "probability")
        if (
            phaseout_probability > 0.0
            and phaseout_mode == "probability"
            and np.random.random() < phaseout_probability
        ):
            self._last_warm_start_phaseout_used = True
            return self._predict_warm_start_policy()

        deterministic_actions = self._predict_deterministic(observations)

        noisy_actions = []
        for agent_idx, action in enumerate(deterministic_actions):
            noise = np.random.normal(loc=self.bias, scale=self.sigma, size=action.shape)
            noise = self._scale_exploration_noise(agent_idx, noise)
            if self.noise_clip is not None and self.noise_clip > 0.0:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            noisy_actions.append(self._clip_action_array(agent_idx, action + noise))

        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)

        if phaseout_probability > 0.0 and phaseout_mode == "blend":
            warm_start_actions = self._predict_warm_start_policy()
            blended_actions = self._blend_phaseout_actions(
                warm_start_actions=warm_start_actions,
                actor_actions=noisy_actions,
                phaseout_probability=phaseout_probability,
            )
            self._last_warm_start_phaseout_used = True
            logger.debug("Blended phase-out exploration actions predicted: {}", blended_actions)
            return blended_actions

        logger.debug("Actions with exploration applied: {}", noisy_actions)
        return [action.tolist() for action in noisy_actions]

    def _warm_start_phaseout_probability(self) -> float:
        if getattr(self, "initial_exploration_strategy", None) != "policy":
            return 0.0
        if self._warm_start_policy is None:
            return 0.0
        phaseout_steps = int(getattr(self, "warm_start_policy_phaseout_steps", 0) or 0)
        if phaseout_steps <= 0:
            return 0.0
        warmup_steps = int(getattr(self, "random_exploration_steps", 0) or 0)
        phaseout_elapsed = int(getattr(self, "exploration_step", 0)) - warmup_steps
        if phaseout_elapsed <= 0 or phaseout_elapsed > phaseout_steps:
            return 0.0
        return float(max(0.0, 1.0 - (phaseout_elapsed - 1) / phaseout_steps))

    def _blend_phaseout_actions(
        self,
        *,
        warm_start_actions: List[List[float]],
        actor_actions: List[np.ndarray],
        phaseout_probability: float,
    ) -> List[List[float]]:
        blended_actions: List[List[float]] = []
        teacher_weight = float(np.clip(phaseout_probability, 0.0, 1.0))
        actor_weight = 1.0 - teacher_weight
        for agent_idx, actor_action in enumerate(actor_actions):
            teacher_action = (
                np.asarray(warm_start_actions[agent_idx], dtype=np.float64).reshape(-1)
                if agent_idx < len(warm_start_actions)
                else np.zeros_like(actor_action, dtype=np.float64)
            )
            actor_array = np.asarray(actor_action, dtype=np.float64).reshape(-1)
            action_dim = min(actor_array.shape[0], teacher_action.shape[0])
            blended = actor_array.copy()
            if action_dim > 0:
                blended[:action_dim] = (
                    teacher_weight * teacher_action[:action_dim]
                    + actor_weight * actor_array[:action_dim]
                )
            blended_actions.append(self._clip_action_array(agent_idx, blended).tolist())
        return blended_actions

    def _scale_exploration_noise(self, agent_idx: int, noise: np.ndarray) -> np.ndarray:
        scaled_noise = np.asarray(noise, dtype=np.float64).copy()
        action_names = self._action_names_for_agent(agent_idx)
        if not action_names:
            return scaled_noise

        low = self._action_low_for_agent(agent_idx)
        storage_multiplier = float(getattr(self, "storage_exploration_noise_multiplier", 1.0))
        ev_negative_multiplier = float(getattr(self, "ev_negative_exploration_noise_multiplier", 1.0))
        for action_idx, action_name in enumerate(action_names[: scaled_noise.shape[0]]):
            if self._is_storage_action_name(action_name):
                scaled_noise[action_idx] *= storage_multiplier
            if (
                self._is_ev_action_name(action_name)
                and action_idx < low.shape[0]
                and low[action_idx] < 0.0
                and scaled_noise[action_idx] < 0.0
            ):
                scaled_noise[action_idx] *= ev_negative_multiplier
        return scaled_noise

    def _predict_random(self) -> List[List[float]]:
        random_actions = [
            np.random.uniform(
                low=self._action_low_for_agent(i),
                high=self._action_high_for_agent(i),
                size=(self.action_dimension[i],),
            ).tolist()
            for i in range(self.num_agents)
        ]
        logger.debug("Random exploration actions predicted: {}", random_actions)
        return random_actions

    def _transition_behavior_actions(self, actions: List[Any]) -> List[Any]:
        if (
            getattr(self, "actor_behavior_cloning_source", "replay_action") != "warm_start_policy"
            or self._warm_start_policy is None
            or self._latest_raw_observations is None
        ):
            return actions
        if self.warm_start_policy_deterministic and self._last_warm_start_policy_actions is not None:
            return [list(action) for action in self._last_warm_start_policy_actions]
        return self._predict_warm_start_policy(apply_noise=False, deterministic=True)

    def _transition_next_behavior_actions(self, fallback_actions: List[Any]) -> List[Any]:
        if (
            getattr(self, "_warm_start_policy", None) is None
            or getattr(self, "_latest_raw_next_observations", None) is None
            or (
                not getattr(self, "residual_policy_enabled", False)
                and getattr(self, "actor_behavior_cloning_source", "replay_action") != "warm_start_policy"
            )
        ):
            return fallback_actions
        if self._last_warm_start_next_policy_actions is not None:
            return [list(action) for action in self._last_warm_start_next_policy_actions]
        predicted = self._predict_warm_start_policy_for_observations(
            getattr(self, "_latest_raw_next_observations", None)
        )
        return predicted if predicted is not None else fallback_actions

    def _transition_observation_event_priority_boost(self) -> float:
        weight = float(getattr(self, "replay_observation_event_priority_weight", 0.0) or 0.0)
        if weight <= 0.0:
            self._last_observation_event_priority_boost = 0.0
            return 0.0

        observations = getattr(self, "_latest_raw_observations", None)
        if observations is None:
            observations = getattr(self, "_latest_encoded_observations", None)
        if observations is None:
            self._last_observation_event_priority_boost = 0.0
            return 0.0

        mode = getattr(self, "replay_observation_event_priority_mode", "ev_departure_service")
        scores = []
        for agent_idx, observation in enumerate(observations):
            ev_score = self._ev_departure_service_priority_score(agent_idx, observation)
            energy_score = self._pv_price_peak_priority_score(agent_idx, observation)
            if mode == "ev_departure_service":
                score = ev_score
            elif mode == "ev_pv_price_peak":
                score = energy_score
            else:
                score = max(ev_score, energy_score)
            scores.append(score)
        score = float(max(scores, default=0.0))
        boost = weight * score
        self._last_observation_event_priority_boost = boost
        return boost

    def _ev_departure_service_priority_score(self, agent_idx: int, observation: Any) -> float:
        names = self.observation_names[agent_idx] if agent_idx < len(self.observation_names) else []
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        if not names or values.size == 0:
            return 0.0

        departure_available = self._max_observation_value(
            names,
            values,
            include=("departure_available",),
            finite_only=True,
        )
        hours_until_departure = self._min_observation_value(
            names,
            values,
            include=("hours_until_departure", "time_until_departure_hours"),
            finite_only=True,
            lower_bound=0.0,
        )
        if departure_available <= 0.0 and not np.isfinite(hours_until_departure):
            return 0.0

        urgency_from_feature = self._max_observation_value(
            names,
            values,
            include=("departure_urgency",),
            finite_only=True,
        )
        urgency_from_hours = 0.0
        if np.isfinite(hours_until_departure):
            urgency_from_hours = float(np.clip(1.0 - hours_until_departure / 24.0, 0.0, 1.0))
        urgency = float(np.clip(max(urgency_from_feature, urgency_from_hours), 0.0, 1.0))

        soc_deficit = self._max_observation_value(
            names,
            values,
            include=("soc_deficit",),
            exclude=("surplus",),
            finite_only=True,
        )
        energy_deficit = self._max_observation_value(
            names,
            values,
            include=("energy_to_required_soc",),
            finite_only=True,
        )
        required_power = self._max_observation_value(
            names,
            values,
            include=("required_average_power", "avg_power_to_departure"),
            finite_only=True,
        )

        # These observations may be raw physical values or already normalized
        # entity features depending on the encoding profile. Clamp each signal
        # conservatively so priority remains a sampler hint, not a reward.
        deficit_signal = max(
            float(np.clip(soc_deficit, 0.0, 1.0)),
            float(np.clip(energy_deficit / 20.0, 0.0, 1.0)),
            float(np.clip(required_power / 7.4, 0.0, 1.0)),
        )
        if deficit_signal <= 1.0e-6:
            return 0.0

        return float(np.clip(max(deficit_signal, urgency * (0.25 + deficit_signal)), 0.0, 1.0))

    def _pv_price_peak_priority_score(self, agent_idx: int, observation: Any) -> float:
        names = self.observation_names[agent_idx] if agent_idx < len(self.observation_names) else []
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        if not names or values.size == 0:
            return 0.0

        pv = self._max_observation_value(
            names,
            values,
            include=("pv", "solar", "photovoltaic", "production"),
            exclude=("soc", "required", "arrival", "departure"),
            finite_only=True,
        )
        price = self._max_observation_value(
            names,
            values,
            include=("price", "pricing", "tariff", "electricity_rate"),
            finite_only=True,
        )
        import_load = self._max_observation_value(
            names,
            values,
            include=("net_electricity_consumption", "community_import", "grid_import", "net_import"),
            finite_only=True,
        )
        min_net_exchange = self._min_observation_value(
            names,
            values,
            include=("net_electricity_consumption", "community_export", "grid_export", "net_export"),
            finite_only=True,
        )
        export_load = -min_net_exchange if np.isfinite(min_net_exchange) else 0.0

        pv_score = float(np.clip(pv / 10.0, 0.0, 1.0))
        price_score = float(np.clip(price, 0.0, 1.0))
        if price > 1.0:
            price_score = float(np.clip(price / 0.5, 0.0, 1.0))
        import_score = float(np.clip(import_load / 25.0, 0.0, 1.0))
        export_score = float(np.clip(export_load / 10.0, 0.0, 1.0))
        return float(np.clip(max(0.65 * pv_score, 0.75 * price_score, import_score, export_score), 0.0, 1.0))

    @staticmethod
    def _max_observation_value(
        names: List[str],
        values: np.ndarray,
        *,
        include: tuple[str, ...],
        exclude: tuple[str, ...] = (),
        finite_only: bool = True,
    ) -> float:
        candidates: List[float] = []
        for index, raw_name in enumerate(names[: values.shape[0]]):
            name = str(raw_name).lower()
            if not any(token in name for token in include):
                continue
            if any(token in name for token in exclude):
                continue
            value = float(values[index])
            if finite_only and not np.isfinite(value):
                continue
            candidates.append(value)
        if not candidates:
            return 0.0
        return max(candidates)

    @staticmethod
    def _min_observation_value(
        names: List[str],
        values: np.ndarray,
        *,
        include: tuple[str, ...],
        exclude: tuple[str, ...] = (),
        finite_only: bool = True,
        lower_bound: Optional[float] = None,
    ) -> float:
        candidates: List[float] = []
        for index, raw_name in enumerate(names[: values.shape[0]]):
            name = str(raw_name).lower()
            if not any(token in name for token in include):
                continue
            if any(token in name for token in exclude):
                continue
            value = float(values[index])
            if finite_only and not np.isfinite(value):
                continue
            if lower_bound is not None and value < lower_bound:
                continue
            candidates.append(value)
        if not candidates:
            return float("inf")
        return min(candidates)

    def _current_residual_base_actions(self) -> Optional[List[List[float]]]:
        if not getattr(self, "residual_policy_enabled", False):
            return None
        predicted = self._predict_warm_start_policy_for_observations(self._latest_raw_observations)
        if predicted is not None:
            self._last_warm_start_policy_actions = [list(action) for action in predicted]
            return predicted
        return [
            self._noop_action_for_agent(agent_idx).astype(np.float64).tolist()
            for agent_idx in range(int(self.num_agents))
        ]

    def _predict_warm_start_policy_for_observations(
        self,
        observations: Optional[List[np.ndarray]],
    ) -> Optional[List[List[float]]]:
        if self._warm_start_policy is None or observations is None:
            return None
        actions = self._warm_start_policy.predict(
            observations,
            deterministic=bool(getattr(self, "warm_start_policy_deterministic", True)),
        )
        clipped_actions: List[List[float]] = []
        for agent_idx, action in enumerate(actions):
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            clipped_actions.append(self._clip_action_array(agent_idx, action_array).tolist())
        return clipped_actions

    def _predict_warm_start_policy(
        self,
        *,
        apply_noise: bool = True,
        deterministic: Optional[bool] = None,
    ) -> List[List[float]]:
        if self._warm_start_policy is None or self._latest_raw_observations is None:
            if not self._warned_missing_raw_context:
                logger.warning(
                    "MADDPG warm-start policy exploration requested, but raw observation context is unavailable. "
                    "Falling back to no-op-centered exploration."
                )
                self._warned_missing_raw_context = True
            return self._predict_noop_centered()

        deterministic_effective = (
            self.warm_start_policy_deterministic if deterministic is None else bool(deterministic)
        )
        actions = self._warm_start_policy.predict(
            self._latest_raw_observations,
            deterministic=deterministic_effective,
        )
        base_clipped_actions: List[List[float]] = []
        clipped_actions: List[List[float]] = []
        for agent_idx, action in enumerate(actions):
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            action_array = self._clip_action_array(agent_idx, action_array)
            base_clipped_actions.append(action_array.tolist())
            if apply_noise and self.warm_start_policy_noise_scale > 0.0:
                span = self._action_span_for_agent(agent_idx)
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.warm_start_policy_noise_scale * span,
                    size=action_array.shape,
                )
                action_array = action_array + noise
            clipped_actions.append(self._clip_action_array(agent_idx, action_array).tolist())
        if deterministic_effective:
            self._last_warm_start_policy_actions = [list(action) for action in base_clipped_actions]
        logger.debug("Warm-start policy exploration actions predicted: {}", clipped_actions)
        return clipped_actions

    def _predict_noop_centered(self) -> List[List[float]]:
        actions: List[List[float]] = []
        for agent_idx in range(self.num_agents):
            low = self._action_low_for_agent(agent_idx)
            high = self._action_high_for_agent(agent_idx)
            span = self._action_span_for_agent(agent_idx)
            action = self._noop_action_for_agent(agent_idx)
            noise = np.random.normal(
                loc=self.bias,
                scale=getattr(self, "noop_noise_scale", 0.15) * span,
                size=action.shape,
            )
            action = action + noise

            for action_idx, action_name in enumerate(self._action_names_for_agent(agent_idx)):
                if action_idx >= action.shape[0] or not self._is_deferrable_action_name(action_name):
                    continue
                threshold = low[action_idx] + getattr(self, "deferrable_trigger_threshold", 0.5) * span[action_idx]
                if np.random.random() < getattr(self, "deferrable_on_probability", 0.2):
                    action[action_idx] = np.random.uniform(
                        min(high[action_idx], threshold + 1.0e-6),
                        high[action_idx],
                    )
                else:
                    action[action_idx] = np.random.uniform(low[action_idx], min(high[action_idx], threshold))

            actions.append(np.clip(action, low, high).tolist())

        logger.debug("No-op-centered exploration actions predicted: {}", actions)
        return actions

    def _action_low_for_agent(self, agent_idx: int) -> np.ndarray:
        if hasattr(self, "action_low") and agent_idx < len(self.action_low):
            return np.asarray(self.action_low[agent_idx], dtype=np.float32)
        return np.full(int(self.action_dimension[agent_idx]), -1.0, dtype=np.float32)

    def _action_high_for_agent(self, agent_idx: int) -> np.ndarray:
        if hasattr(self, "action_high") and agent_idx < len(self.action_high):
            return np.asarray(self.action_high[agent_idx], dtype=np.float32)
        return np.full(int(self.action_dimension[agent_idx]), 1.0, dtype=np.float32)

    def _action_span_for_agent(self, agent_idx: int) -> np.ndarray:
        low = self._action_low_for_agent(agent_idx)
        high = self._action_high_for_agent(agent_idx)
        return np.maximum(high - low, 1.0e-6)

    def _noop_action_for_agent(self, agent_idx: int) -> np.ndarray:
        return np.clip(
            np.zeros(int(self.action_dimension[agent_idx]), dtype=np.float32),
            self._action_low_for_agent(agent_idx),
            self._action_high_for_agent(agent_idx),
        )

    def _action_names_for_agent(self, agent_idx: int) -> List[str]:
        if hasattr(self, "action_names") and agent_idx < len(self.action_names):
            return [str(name) for name in self.action_names[agent_idx]]
        return []

    @staticmethod
    def _is_deferrable_action_name(action_name: str) -> bool:
        raw = str(action_name or "")
        return raw.startswith("deferrable_appliance") or raw.endswith("::start") or raw == "start"

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "electrical_storage" in raw or raw in {"battery", "storage"}

    @staticmethod
    def _is_ev_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return (
            "electric_vehicle" in raw
            or "charger" in raw
            or raw.startswith("ev_")
            or raw in {"ev", "v2g"}
        )

    def _scale_action_tensor(self, agent_idx: int, raw_action: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(
            self._action_low_for_agent(agent_idx),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        high = torch.as_tensor(
            self._action_high_for_agent(agent_idx),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        scaled = low + 0.5 * (raw_action + 1.0) * (high - low)
        return torch.max(torch.min(scaled, high), low)

    def _policy_action_from_actor_output(
        self,
        agent_idx: int,
        raw_action: torch.Tensor,
        *,
        base_action: Optional[torch.Tensor] = None,
        global_learning_step: Optional[int] = None,
    ) -> torch.Tensor:
        if (
            not getattr(self, "residual_policy_enabled", False)
            or base_action is None
            or float(getattr(self, "residual_action_final_scale", 0.0) or 0.0) <= 0.0
        ):
            return self._scale_action_tensor(agent_idx, raw_action)

        base = base_action.to(dtype=raw_action.dtype, device=raw_action.device)
        if base.dim() == 1 and raw_action.dim() == 2:
            base = base.view(1, -1).expand(raw_action.shape[0], -1)
        elif base.dim() == 1:
            base = base.reshape(raw_action.shape)
        elif raw_action.dim() == 1 and base.dim() == 2:
            base = base.reshape(raw_action.shape)

        span = torch.as_tensor(
            self._action_span_for_agent(agent_idx),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        scale = self._residual_action_effective_scale(global_learning_step)
        scale_mask = self._residual_action_scale_mask(
            agent_idx,
            action_dim=int(raw_action.shape[-1]),
            dtype=raw_action.dtype,
            device=raw_action.device,
        )
        residual = 0.5 * span * scale * scale_mask * raw_action
        return self._clip_action_tensor(agent_idx, base + residual)

    def _residual_action_effective_scale(self, global_learning_step: Optional[int] = None) -> float:
        if not getattr(self, "residual_policy_enabled", False):
            self._last_residual_action_scale = 0.0
            return 0.0

        step = int(
            self.exploration_step if global_learning_step is None else global_learning_step
        )
        start_step = int(getattr(self, "residual_action_start_step", 0) or 0)
        if step < start_step:
            self._last_residual_action_scale = 0.0
            return 0.0

        initial = float(getattr(self, "residual_action_scale", 0.0) or 0.0)
        final = float(getattr(self, "residual_action_final_scale", initial) or 0.0)
        growth_steps = int(getattr(self, "residual_action_growth_steps", 0) or 0)
        if growth_steps <= 0:
            value = final
        else:
            progress = min(max((step - start_step) / growth_steps, 0.0), 1.0)
            value = initial + (final - initial) * progress
        self._last_residual_action_scale = float(np.clip(value, 0.0, 1.0))
        return self._last_residual_action_scale

    def _residual_action_scale_mask(
        self,
        agent_idx: int,
        *,
        action_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        values = np.ones(int(action_dim), dtype=np.float32)
        names = self._action_names_for_agent(agent_idx)
        for action_idx, action_name in enumerate(names[: int(action_dim)]):
            if self._is_storage_action_name(action_name):
                values[action_idx] *= float(
                    getattr(self, "residual_storage_action_scale_multiplier", 1.0)
                )
            if self._is_ev_action_name(action_name):
                values[action_idx] *= float(
                    getattr(self, "residual_ev_action_scale_multiplier", 1.0)
                )
            if self._is_deferrable_action_name(action_name):
                values[action_idx] *= float(
                    getattr(self, "residual_deferrable_action_scale_multiplier", 1.0)
                )
        return torch.as_tensor(values, dtype=dtype, device=device)

    def _clip_action_tensor(self, agent_idx: int, action: torch.Tensor) -> torch.Tensor:
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
        return torch.max(torch.min(action, high), low)

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
        normalized = 2.0 * (action - low) / span - 1.0
        return torch.clamp(normalized, -1.0, 1.0)

    def _add_target_policy_smoothing(self, agent_idx: int, action: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "target_policy_smoothing", False) or getattr(self, "target_policy_noise", 0.0) <= 0.0:
            return self._clip_action_tensor(agent_idx, action)

        span = torch.as_tensor(
            self._action_span_for_agent(agent_idx),
            dtype=action.dtype,
            device=action.device,
        )
        noise = torch.randn_like(action) * (float(self.target_policy_noise) * span)
        noise_clip = float(getattr(self, "target_policy_noise_clip", 0.0) or 0.0)
        if noise_clip > 0.0:
            noise = torch.max(torch.min(noise, noise_clip * span), -noise_clip * span)
        return self._clip_action_tensor(agent_idx, action + noise)

    def _actor_action_regularization_terms(
        self,
        agent_idx: int,
        scaled_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_action = self._normalize_scaled_action_tensor(agent_idx, scaled_action)
        action_l2 = normalized_action.pow(2).mean()
        threshold = float(getattr(self, "actor_action_saturation_threshold", 0.85))
        saturation_excess = torch.relu(normalized_action.abs() - threshold).pow(2).mean()
        storage_action_l2 = self._masked_action_l2(
            agent_idx,
            normalized_action,
            predicate=self._is_storage_action_name,
        )
        ev_v2g_action_l2 = self._negative_masked_action_l2(
            agent_idx,
            scaled_action,
            normalized_action,
            predicate=self._is_ev_action_name,
        )
        ev_v2g_action_mass = self._negative_masked_action_mass(
            agent_idx,
            scaled_action,
            predicate=self._is_ev_action_name,
        )
        regularization = (
            float(getattr(self, "actor_action_l2_penalty", 0.0)) * action_l2
            + float(getattr(self, "actor_action_saturation_penalty", 0.0)) * saturation_excess
            + float(getattr(self, "actor_storage_action_l2_penalty", 0.0)) * storage_action_l2
            + float(getattr(self, "actor_ev_v2g_action_l2_penalty", 0.0)) * ev_v2g_action_l2
            + float(getattr(self, "actor_ev_v2g_action_mass_penalty", 0.0)) * ev_v2g_action_mass
        )
        return (
            action_l2,
            saturation_excess,
            storage_action_l2,
            ev_v2g_action_l2,
            ev_v2g_action_mass,
            regularization,
        )

    def _masked_action_l2(
        self,
        agent_idx: int,
        normalized_action: torch.Tensor,
        *,
        predicate,
    ) -> torch.Tensor:
        names = self._action_names_for_agent(agent_idx)
        mask_values = [
            1.0 if action_idx < len(names) and predicate(names[action_idx]) else 0.0
            for action_idx in range(normalized_action.shape[1])
        ]
        if not any(mask_values):
            return normalized_action.new_tensor(0.0)
        mask = torch.as_tensor(mask_values, dtype=normalized_action.dtype, device=normalized_action.device).view(1, -1)
        denominator = torch.clamp(mask.sum() * normalized_action.shape[0], min=1.0)
        return (normalized_action.pow(2) * mask).sum() / denominator

    def _negative_masked_action_l2(
        self,
        agent_idx: int,
        scaled_action: torch.Tensor,
        normalized_action: torch.Tensor,
        *,
        predicate,
    ) -> torch.Tensor:
        names = self._action_names_for_agent(agent_idx)
        mask_values = [
            1.0 if action_idx < len(names) and predicate(names[action_idx]) else 0.0
            for action_idx in range(normalized_action.shape[1])
        ]
        if not any(mask_values):
            return normalized_action.new_tensor(0.0)
        mask = torch.as_tensor(mask_values, dtype=normalized_action.dtype, device=normalized_action.device).view(1, -1)
        negative_mask = (scaled_action < 0.0).to(dtype=normalized_action.dtype)
        effective_mask = mask * negative_mask
        denominator = torch.clamp(effective_mask.sum(), min=1.0)
        return (normalized_action.pow(2) * effective_mask).sum() / denominator

    def _negative_masked_action_mass(
        self,
        agent_idx: int,
        scaled_action: torch.Tensor,
        *,
        predicate,
    ) -> torch.Tensor:
        names = self._action_names_for_agent(agent_idx)
        mask_values = [
            1.0 if action_idx < len(names) and predicate(names[action_idx]) else 0.0
            for action_idx in range(scaled_action.shape[1])
        ]
        if not any(mask_values):
            return scaled_action.new_tensor(0.0)

        mask = torch.as_tensor(mask_values, dtype=scaled_action.dtype, device=scaled_action.device).view(1, -1)
        low = torch.as_tensor(
            np.abs(self._action_low_for_agent(agent_idx)[: scaled_action.shape[1]]),
            dtype=scaled_action.dtype,
            device=scaled_action.device,
        ).view(1, -1)
        negative_fraction = torch.relu(-scaled_action) / torch.clamp(low, min=1.0e-6)
        denominator = torch.clamp(mask.sum() * scaled_action.shape[0], min=1.0)
        return (negative_fraction * mask).sum() / denominator

    def _actor_behavior_cloning_extra_updates_for_step(
        self,
        global_learning_step: int,
        behavior_cloning_weight: float,
    ) -> int:
        if behavior_cloning_weight <= 0.0:
            return 0
        extra_updates = max(0, int(getattr(self, "actor_behavior_cloning_extra_updates", 0) or 0))
        if extra_updates <= 0:
            return 0

        start_step = max(
            0,
            int(getattr(self, "actor_behavior_cloning_extra_update_start_step", 0) or 0),
        )
        if global_learning_step < start_step:
            return 0

        end_step = max(
            0,
            int(getattr(self, "actor_behavior_cloning_extra_update_end_step", 0) or 0),
        )
        if end_step > 0 and global_learning_step > end_step:
            return 0

        return extra_updates

    def _run_actor_behavior_cloning_extra_updates(
        self,
        *,
        agent_num: int,
        actor: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        observations: torch.Tensor,
        behavior_actions: torch.Tensor,
        behavior_cloning_weight: float,
        extra_updates: int,
    ) -> tuple[List[float], List[float]]:
        if extra_updates <= 0:
            return [], []

        loss_values: List[float] = []
        grad_norm_values: List[float] = []
        for _ in range(extra_updates):
            predicted_action = self._policy_action_from_actor_output(
                agent_num,
                actor(observations),
                base_action=behavior_actions,
            )
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                behavior_cloning_loss = self._actor_behavior_cloning_loss(
                    agent_num,
                    predicted_action,
                    behavior_actions,
                )
                weighted_loss = behavior_cloning_weight * behavior_cloning_loss

            actor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(weighted_loss).backward()
            self.scaler.unscale_(actor_optimizer)
            grad_norm = clip_grad_norm_(actor.parameters(), max_norm=1.0)
            self.scaler.step(actor_optimizer)
            self.scaler.update()

            loss_values.append(float(weighted_loss.detach().item()))
            grad_norm_values.append(float(grad_norm))

        return loss_values, grad_norm_values

    def _actor_behavior_cloning_loss(
        self,
        agent_idx: int,
        predicted_action: torch.Tensor,
        replay_action: torch.Tensor,
    ) -> torch.Tensor:
        if float(getattr(self, "actor_behavior_cloning_weight", 0.0)) <= 0.0:
            return predicted_action.new_tensor(0.0)

        predicted_normalized = self._normalize_scaled_action_tensor(agent_idx, predicted_action)
        replay_normalized = self._normalize_scaled_action_tensor(agent_idx, replay_action.detach())
        weights = self._actor_behavior_cloning_action_weights(
            agent_idx,
            action_dim=predicted_normalized.shape[1],
            dtype=predicted_normalized.dtype,
            device=predicted_normalized.device,
        )
        weights = self._actor_behavior_cloning_sample_weights(
            agent_idx,
            base_weights=weights,
            replay_action=replay_action.detach(),
        )
        denominator = torch.clamp(weights.sum(), min=1.0)
        return ((predicted_normalized - replay_normalized).pow(2) * weights).sum() / denominator

    def _actor_behavior_cloning_type_losses(
        self,
        agent_idx: int,
        predicted_action: torch.Tensor,
        replay_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_normalized = self._normalize_scaled_action_tensor(agent_idx, predicted_action)
        replay_normalized = self._normalize_scaled_action_tensor(agent_idx, replay_action.detach())
        squared_error = (predicted_normalized - replay_normalized).pow(2)
        ev_mask = self._action_type_mask_tensor(
            agent_idx,
            action_dim=squared_error.shape[1],
            predicate=self._is_ev_action_name,
            dtype=squared_error.dtype,
            device=squared_error.device,
        )
        storage_mask = self._action_type_mask_tensor(
            agent_idx,
            action_dim=squared_error.shape[1],
            predicate=self._is_storage_action_name,
            dtype=squared_error.dtype,
            device=squared_error.device,
        )
        return (
            self._masked_mean(squared_error, ev_mask),
            self._masked_mean(squared_error, storage_mask),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        effective_mask = mask.view(1, -1).expand_as(values)
        denominator = torch.clamp(effective_mask.sum(), min=1.0)
        return (values * effective_mask).sum() / denominator

    def _action_type_mask_tensor(
        self,
        agent_idx: int,
        *,
        action_dim: int,
        predicate,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        names = self._action_names_for_agent(agent_idx)
        values = [
            1.0 if action_idx < len(names) and predicate(names[action_idx]) else 0.0
            for action_idx in range(int(action_dim))
        ]
        return torch.as_tensor(values, dtype=dtype, device=device)

    def _actor_behavior_cloning_action_weights(
        self,
        agent_idx: int,
        *,
        action_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weights = np.ones(int(action_dim), dtype=np.float32)
        names = self._action_names_for_agent(agent_idx)
        ev_multiplier = float(getattr(self, "actor_ev_behavior_cloning_multiplier", 1.0))
        storage_multiplier = float(getattr(self, "actor_storage_behavior_cloning_multiplier", 1.0))
        for action_idx, action_name in enumerate(names[: int(action_dim)]):
            if self._is_ev_action_name(action_name):
                weights[action_idx] *= ev_multiplier
            if self._is_storage_action_name(action_name):
                weights[action_idx] *= storage_multiplier
        return torch.as_tensor(weights, dtype=dtype, device=device)

    def _actor_behavior_cloning_sample_weights(
        self,
        agent_idx: int,
        *,
        base_weights: torch.Tensor,
        replay_action: torch.Tensor,
    ) -> torch.Tensor:
        weights = base_weights.view(1, -1).expand(replay_action.shape[0], -1)
        positive_target_weight = float(
            getattr(self, "actor_ev_behavior_cloning_positive_target_weight", 0.0) or 0.0
        )
        zero_target_weight = float(
            getattr(self, "actor_ev_behavior_cloning_zero_target_weight", 0.0) or 0.0
        )
        if positive_target_weight <= 0.0 and zero_target_weight <= 0.0:
            return weights

        action_dim = min(int(replay_action.shape[1]), int(base_weights.shape[0]))
        names = self._action_names_for_agent(agent_idx)
        ev_mask_values = [
            1.0 if action_idx < len(names) and self._is_ev_action_name(names[action_idx]) else 0.0
            for action_idx in range(action_dim)
        ]
        if not any(ev_mask_values):
            return weights

        ev_mask = torch.as_tensor(
            ev_mask_values,
            dtype=replay_action.dtype,
            device=replay_action.device,
        ).view(1, -1)
        positive_high = torch.as_tensor(
            np.maximum(self._action_high_for_agent(agent_idx)[:action_dim], 1.0e-6),
            dtype=replay_action.dtype,
            device=replay_action.device,
        ).view(1, -1)
        positive_target = torch.clamp(replay_action[:, :action_dim], min=0.0) / positive_high
        positive_target = torch.clamp(positive_target, 0.0, 1.0)
        multiplier = torch.ones_like(positive_target)
        if positive_target_weight > 0.0:
            power = float(getattr(self, "actor_ev_behavior_cloning_positive_target_power", 1.0) or 1.0)
            positive_term = positive_target.pow(power) if power != 1.0 else positive_target
            multiplier = multiplier + positive_target_weight * ev_mask * positive_term
        if zero_target_weight > 0.0:
            zero_target_threshold = float(
                getattr(self, "actor_ev_behavior_cloning_zero_target_threshold", 0.05) or 0.05
            )
            zero_target = (torch.abs(replay_action[:, :action_dim]) <= zero_target_threshold * positive_high).to(
                dtype=replay_action.dtype
            )
            multiplier = multiplier + zero_target_weight * ev_mask * zero_target
        return weights[:, :action_dim] * multiplier

    def _actor_behavior_cloning_effective_weight(self, global_learning_step: int) -> float:
        base_weight = float(getattr(self, "actor_behavior_cloning_weight", 0.0) or 0.0)
        if base_weight <= 0.0:
            return 0.0

        min_weight = min(
            base_weight,
            max(0.0, float(getattr(self, "actor_behavior_cloning_min_weight", 0.0) or 0.0)),
        )
        decay_steps = max(0, int(getattr(self, "actor_behavior_cloning_decay_steps", 0) or 0))
        if decay_steps <= 0:
            return base_weight

        decay_start = max(0, int(getattr(self, "actor_behavior_cloning_decay_start_step", 0) or 0))
        if global_learning_step <= decay_start:
            return base_weight

        progress = min(max((global_learning_step - decay_start) / decay_steps, 0.0), 1.0)
        return base_weight + (min_weight - base_weight) * progress

    def _actor_policy_loss_effective_weight(self, global_learning_step: int) -> float:
        base_weight = max(0.0, float(getattr(self, "actor_policy_loss_weight", 1.0) or 0.0))
        warmup_steps = max(0, int(getattr(self, "actor_policy_loss_warmup_steps", 0) or 0))
        if warmup_steps <= 0:
            return base_weight

        warmup_weight = max(
            0.0,
            float(getattr(self, "actor_policy_loss_warmup_weight", base_weight) or 0.0),
        )
        warmup_start = max(0, int(getattr(self, "actor_policy_loss_warmup_start_step", 0) or 0))
        if global_learning_step <= warmup_start:
            return warmup_weight

        progress = min(max((global_learning_step - warmup_start) / warmup_steps, 0.0), 1.0)
        return warmup_weight + (base_weight - warmup_weight) * progress

    def _clip_action_array(self, agent_idx: int, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float64),
            self._action_low_for_agent(agent_idx),
            self._action_high_for_agent(agent_idx),
        )

    def freeze_layers(self, freeze_actor: bool = True, freeze_critic: bool = False) -> None:
        for actor in self.actors:
            for param in actor.parameters():
                param.requires_grad = not freeze_actor
        for critic in self.critics:
            for param in critic.parameters():
                param.requires_grad = not freeze_critic
        logger.info("Freezing actors={}, Freezing critics={}", freeze_actor, freeze_critic)

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        checkpoint: Dict[str, Any] = {}
        for i in range(self.num_agents):
            checkpoint[f"actor_state_dict_{i}"] = self.actors[i].state_dict()
            checkpoint[f"critic_state_dict_{i}"] = self.critics[i].state_dict()
            checkpoint[f"actor_target_state_dict_{i}"] = self.actor_targets[i].state_dict()
            checkpoint[f"critic_target_state_dict_{i}"] = self.critic_targets[i].state_dict()
            checkpoint[f"actor_optimizer_state_dict_{i}"] = self.actor_optimizers[i].state_dict()
            checkpoint[f"critic_optimizer_state_dict_{i}"] = self.critic_optimizers[i].state_dict()

        if hasattr(self.replay_buffer, "get_state"):
            checkpoint["replay_buffer"] = self.replay_buffer.get_state()

        checkpoint["exploration_state"] = {
            "sigma": float(getattr(self, "sigma", 0.0)),
            "exploration_step": int(getattr(self, "exploration_step", 0)),
        }
        checkpoint["reward_normalization_state"] = {
            "enabled": bool(getattr(self, "reward_normalization_enabled", False)),
            "count": int(getattr(self, "reward_norm_count", 0)),
            "mean": float(getattr(self, "reward_norm_mean", 0.0)),
            "m2": float(getattr(self, "reward_norm_m2", 0.0)),
        }
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        latest_name = self.checkpoint_artifact or "latest_checkpoint.pth"
        latest_path = output_dir_path / latest_name
        torch.save(checkpoint, latest_path)

        logger.info("Checkpoint saved at step {} -> {}", step, latest_path)
        return str(latest_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint[f"actor_state_dict_{i}"])
            self.critics[i].load_state_dict(checkpoint[f"critic_state_dict_{i}"])
            if hasattr(self, "actor_targets"):
                self.actor_targets[i].load_state_dict(
                    checkpoint.get(f"actor_target_state_dict_{i}", checkpoint[f"actor_state_dict_{i}"])
                )
            if hasattr(self, "critic_targets"):
                self.critic_targets[i].load_state_dict(
                    checkpoint.get(f"critic_target_state_dict_{i}", checkpoint[f"critic_state_dict_{i}"])
                )
            if not self.fine_tune:
                self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_state_dict_{i}"])
                self.critic_optimizers[i].load_state_dict(checkpoint[f"critic_optimizer_state_dict_{i}"])

        if "replay_buffer" in checkpoint and not self.reset_replay_buffer:
            self.replay_buffer.set_state(checkpoint["replay_buffer"])

        exploration_state = checkpoint.get("exploration_state")
        if isinstance(exploration_state, dict):
            if "sigma" in exploration_state:
                self.sigma = float(exploration_state["sigma"])
            if "exploration_step" in exploration_state:
                self.exploration_step = int(exploration_state["exploration_step"])

        reward_norm_state = checkpoint.get("reward_normalization_state")
        if isinstance(reward_norm_state, dict):
            self.reward_norm_count = int(reward_norm_state.get("count", self.reward_norm_count))
            self.reward_norm_mean = float(reward_norm_state.get("mean", self.reward_norm_mean))
            self.reward_norm_m2 = float(reward_norm_state.get("m2", self.reward_norm_m2))

        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            python_state = rng_state.get("python")
            numpy_state = rng_state.get("numpy")
            torch_state = rng_state.get("torch")
            torch_cuda_state = rng_state.get("torch_cuda")
            if python_state is not None:
                random.setstate(python_state)
            if numpy_state is not None:
                np.random.set_state(numpy_state)
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            if torch_cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_state)

        if self.freeze_pretrained_layers:
            self.freeze_layers(freeze_actor=True, freeze_critic=False)

        logger.info("Checkpoint loaded from {}", checkpoint_file)

    @staticmethod
    def get_best_checkpoint(experiment_name: str) -> str:
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found in MLflow.")

        runs = client.search_runs(
            experiment.experiment_id,
            order_by=["metrics.validation_loss ASC"],
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
        raise ValueError(f"No runs found for experiment {experiment_name}.")

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        bundle_cfg = ((context.get("config") or {}).get("bundle") or {})
        global_artifact_config = dict(bundle_cfg.get("artifact_config") or {})
        raw_per_agent_config = bundle_cfg.get("per_agent_artifact_config") or {}
        per_agent_artifact_config = (
            raw_per_agent_config if isinstance(raw_per_agent_config, dict) else {}
        )
        require_observations_envelope = bool(bundle_cfg.get("require_observations_envelope", False))

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting MADDPG actors to ONNX under {}", onnx_dir)

        metadata: Dict[str, Any] = {"format": "onnx", "artifacts": []}

        for i, actor in enumerate(self.actors):
            export_path = onnx_dir / f"agent_{i}.onnx"
            dummy_input = torch.randn(1, self.observation_dimension[i], device=self.device)
            export_model = ActionScaledActor(
                actor,
                low=self._action_low_for_agent(i),
                high=self._action_high_for_agent(i),
            ).to(self.device)
            export_model.eval()
            torch.onnx.export(
                export_model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=DEFAULT_ONNX_OPSET,
                do_constant_folding=True,
                input_names=[f"observation_agent_{i}"],
                output_names=[f"action_agent_{i}"],
                dynamic_axes={
                    f"observation_agent_{i}": {0: "batch_size"},
                    f"action_agent_{i}": {0: "batch_size"},
                },
            )

            logger.info("ONNX model exported for agent {}: {}", i, export_path)

            relative_path = export_path.relative_to(export_root)
            raw_agent_override = (
                per_agent_artifact_config.get(str(i))
                if str(i) in per_agent_artifact_config
                else per_agent_artifact_config.get(i)
            )
            agent_override = raw_agent_override if isinstance(raw_agent_override, dict) else {}
            auto_artifact_config = build_auto_artifact_config(context=context, agent_index=i)
            artifact_config: Dict[str, Any] = {}
            artifact_config.update(auto_artifact_config)
            artifact_config.update(global_artifact_config)
            artifact_config.update(agent_override)
            if require_observations_envelope:
                artifact_config["require_observations_envelope"] = True
            metadata["artifacts"].append(
                {
                    "agent_index": i,
                    "path": str(relative_path),
                    "format": "onnx",
                    "observation_dimension": self.observation_dimension[i],
                    "action_dimension": self.action_dimension[i],
                    "config": artifact_config,
                }
            )

            if mlflow.active_run():
                mlflow.log_artifact(str(export_path), artifact_path="onnx")

        return metadata
