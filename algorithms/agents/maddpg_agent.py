from __future__ import annotations

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
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.networks import Actor, Critic
from algorithms.utils.replay_buffer import (
    MultiAgentReplayBuffer,
    PrioritizedReplayBuffer,
)
from utils.artifact_config_builder import build_auto_artifact_config

REPLAY_BUFFER_REGISTRY = {
    "MultiAgentReplayBuffer": MultiAgentReplayBuffer,
    "PrioritizedReplayBuffer": PrioritizedReplayBuffer,
}


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selected: {}", self.device)
        torch.backends.cudnn.benchmark = True

        exploration_cfg = self.config["algorithm"]["exploration"]["params"]
        buffer_cfg = self.config["algorithm"]["replay_buffer"]
        network_cfg = self.config["algorithm"]["networks"]

        hyperparams = self.config["algorithm"]["hyperparameters"]

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
        self.warm_start_policy_name = self._optional_string(exploration_cfg.get("warm_start_policy"))
        self.initial_exploration_strategy = self._resolve_initial_exploration_strategy(exploration_cfg)
        self.warm_start_policy_deterministic = bool(exploration_cfg.get("warm_start_policy_deterministic", True))
        self.warm_start_policy_noise_scale = max(0.0, float(exploration_cfg.get("warm_start_policy_noise_scale", 0.0) or 0.0))
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
        self.actor_action_saturation_threshold = float(
            np.clip(float(exploration_cfg.get("actor_action_saturation_threshold", 0.85) or 0.85), 0.0, 1.0)
        )
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
        self._latest_encoded_observations: Optional[List[np.ndarray]] = None
        self._warm_start_policy = None
        self._warned_missing_raw_context = False
        self._noop_actor_initialized = False
        self.replay_buffer = self._initialize_replay_buffer()
        self.actors, self.critics, self.actor_targets, self.critic_targets = self._initialize_networks()
        self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers()
        self.use_amp = self.device.type == "cuda"
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
        self._initialize_warm_start_policy(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self._apply_noop_actor_initialization()

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
        return replay_cls(**params)

    def _initialize_networks(self):
        logger.debug("Initializing actor and critic networks.")
        actor_fc_units = self.config["algorithm"]["networks"]["actor"]["layers"]
        critic_fc_units = self.config["algorithm"]["networks"]["critic"]["layers"]

        actors, critics, actor_targets, critic_targets = [], [], [], []
        for i in range(self.num_agents):
            state_size = self.observation_dimension[i]
            action_size = self.action_dimension[i]
            global_state_size = sum(self.observation_dimension)
            global_action_size = sum(self.action_dimension)

            actors.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critics.append(Critic(global_state_size, global_action_size, self.seed, critic_fc_units).to(self.device))
            actor_targets.append(Actor(state_size, action_size, self.seed, actor_fc_units).to(self.device))
            critic_targets.append(Critic(global_state_size, global_action_size, self.seed, critic_fc_units).to(self.device))

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

        done = bool(terminated or truncated)
        self._update_reward_normalizer(rewards)
        self.replay_buffer.push(observations, actions, rewards, next_observations, done)

        if len(self.replay_buffer) < self.batch_size:
            logger.debug("Not enough samples in the replay buffer. Skipping update.")
            return

        if not initial_exploration_done:
            logger.debug("Initial exploration phase not finished. Skipping update.")
            return

        if not update_step:
            logger.debug("Update step skipped based on schedule.")
            return

        states, actions_all, rewards_all, next_states, dones_all = self.replay_buffer.sample()

        raw_rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)
        rewards_all = self._normalize_reward_tensor(raw_rewards_all)
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)
        states = [s.to(self.device, non_blocking=True) for s in states]
        actions_all = [a.to(self.device, non_blocking=True) for a in actions_all]
        next_states = [ns.to(self.device, non_blocking=True) for ns in next_states]

        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)

        with torch.no_grad():
            next_policy_actions = []
            for agent_idx in range(self.num_agents):
                next_action = self._scale_action_tensor(agent_idx, self.actor_targets[agent_idx](next_states[agent_idx]))
                if self.target_policy_smoothing:
                    next_action = self._add_target_policy_smoothing(agent_idx, next_action)
                next_policy_actions.append(next_action)
            global_next_actions = torch.cat(next_policy_actions, dim=1)
            q_targets_next = torch.stack(
                [critic(global_next_state, global_next_actions) for critic in self.critic_targets]
            )
            q_targets = rewards_all + self.gamma * q_targets_next * (1 - dones_all)

        if self.critic_update_mode == "per_agent":
            critic_loss_values: List[float] = []
            critic_td_abs_values: List[float] = []
            critic_grad_norm_values: List[float] = []
            q_expected_stat_tensors: List[torch.Tensor] = []
            for agent_num, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    q_expected_agent = critic(global_state, global_actions)
                    critic_loss_agent = mse_loss(q_expected_agent, q_targets[agent_num])

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
                critic_loss = mse_loss(q_expected, q_targets).mean()

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
                float(mse_loss(q_expected[agent_num], q_targets[agent_num]).item())
                for agent_num in range(self.num_agents)
            ]
            critic_td_abs_values = [
                float((q_expected[agent_num].detach() - q_targets[agent_num].detach()).abs().mean().item())
                for agent_num in range(self.num_agents)
            ]
            critic_grad_norm_values = [float(critic_grad_norm)]
            q_expected_stat_tensors = [q_expected.detach()]

        logger.debug("Critics updated. Loss: {:.4f}.", critic_loss_scalar)

        should_log_step_metrics = self._should_log_training_step(global_learning_step)

        # Compute detached policy actions once and reuse them during actor updates.
        with torch.no_grad():
            detached_policy_actions = [
                self._scale_action_tensor(agent_idx, actor(state)).detach()
                for agent_idx, (actor, state) in enumerate(zip(self.actors, states))
            ]

        actor_update_due = (
            self.actor_update_interval <= 1
            or global_learning_step % self.actor_update_interval == 0
        )
        total_actor_loss = 0.0
        actor_loss_values: List[float] = []
        actor_policy_loss_values: List[float] = []
        actor_regularization_values: List[float] = []
        actor_action_l2_values: List[float] = []
        actor_action_saturation_values: List[float] = []
        actor_grad_norm_values: List[float] = []
        if actor_update_due:
            for agent_num, (actor, critic, actor_optimizer) in enumerate(
                zip(self.actors, self.critics, self.actor_optimizers)
            ):
                obs = states[agent_num]

                predicted_action = self._scale_action_tensor(agent_num, actor(obs))
                joint_policy_actions = list(detached_policy_actions)
                joint_policy_actions[agent_num] = predicted_action
                global_predicted_actions = torch.cat(joint_policy_actions, dim=1)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    actor_policy_loss = -critic(global_state, global_predicted_actions).mean()
                    action_l2, action_saturation, actor_regularization = self._actor_action_regularization_terms(
                        agent_num,
                        predicted_action,
                    )
                    actor_loss = actor_policy_loss + actor_regularization

                actor_optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(actor_optimizer)
                actor_grad_norm = clip_grad_norm_(actor.parameters(), max_norm=1.0)
                self.scaler.step(actor_optimizer)
                self.scaler.update()

                # Keep cached detached actions in sync with sequential actor updates.
                with torch.no_grad():
                    detached_policy_actions[agent_num] = self._scale_action_tensor(agent_num, actor(obs)).detach()

                total_actor_loss += actor_loss.item()
                actor_loss_values.append(float(actor_loss.item()))
                actor_policy_loss_values.append(float(actor_policy_loss.detach().item()))
                actor_regularization_values.append(float(actor_regularization.detach().item()))
                actor_action_l2_values.append(float(action_l2.detach().item()))
                actor_action_saturation_values.append(float(action_saturation.detach().item()))
                actor_grad_norm_values.append(float(actor_grad_norm))
                logger.debug("Actor {} updated. Loss: {:.4f}.", agent_num, actor_loss.item())

                if update_target_step:
                    logger.debug("Updating target networks for agent {}.", agent_num)
                    self._soft_update(critic, self.critic_targets[agent_num], self.tau)
                    self._soft_update(actor, self.actor_targets[agent_num], self.tau)
        else:
            actor_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_policy_loss_values = [0.0 for _ in range(self.num_agents)]
            actor_regularization_values = [0.0 for _ in range(self.num_agents)]
            actor_action_l2_values = [0.0 for _ in range(self.num_agents)]
            actor_action_saturation_values = [0.0 for _ in range(self.num_agents)]
            actor_grad_norm_values = [0.0 for _ in range(self.num_agents)]
            logger.debug(
                "Actor update delayed at step {} by actor_update_interval={}.",
                global_learning_step,
                self.actor_update_interval,
            )

        if should_log_step_metrics and self.training_diagnostics_enabled:
            q_expected_flat = torch.cat([tensor.reshape(-1) for tensor in q_expected_stat_tensors])
            q_targets_flat = q_targets.detach().reshape(-1)
            training_metrics: Dict[str, float] = {
                "MADDPG/average_critic_loss": critic_loss_scalar,
                "MADDPG/average_actor_loss": total_actor_loss / self.num_agents,
                "MADDPG/actor_update_performed": float(actor_update_due),
                "MADDPG/actor_policy_loss_mean": float(np.mean(actor_policy_loss_values)),
                "MADDPG/actor_regularization_loss_mean": float(np.mean(actor_regularization_values)),
                "MADDPG/actor_action_l2_mean": float(np.mean(actor_action_l2_values)),
                "MADDPG/actor_action_saturation_excess_mean": float(np.mean(actor_action_saturation_values)),
                "MADDPG/reward_raw_mean": float(raw_rewards_all.mean().item()),
                "MADDPG/reward_raw_std": float(raw_rewards_all.std(unbiased=False).item()),
                "MADDPG/reward_train_mean": float(rewards_all.mean().item()),
                "MADDPG/reward_train_std": float(rewards_all.std(unbiased=False).item()),
                "MADDPG/reward_norm_count": float(getattr(self, "reward_norm_count", 0)),
                "MADDPG/reward_norm_mean": float(getattr(self, "reward_norm_mean", 0.0)),
                "MADDPG/reward_norm_std": float(self._reward_normalization_std()),
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
                "MADDPG/exploration_sigma": float(getattr(self, "sigma", 0.0)),
                "MADDPG/exploration_step": float(getattr(self, "exploration_step", 0)),
                "MADDPG/training_step_time": time.time() - update_start_time,
            }
            if self.training_diagnostics_detail == "per_agent":
                for agent_num in range(self.num_agents):
                    training_metrics[f"MADDPG/critic_loss_agent_{agent_num}"] = critic_loss_values[agent_num]
                    training_metrics[f"MADDPG/critic_td_abs_agent_{agent_num}"] = critic_td_abs_values[agent_num]
                    training_metrics[f"MADDPG/actor_loss_agent_{agent_num}"] = actor_loss_values[agent_num]
                    training_metrics[f"MADDPG/actor_policy_loss_agent_{agent_num}"] = actor_policy_loss_values[agent_num]
                    training_metrics[f"MADDPG/actor_regularization_loss_agent_{agent_num}"] = actor_regularization_values[agent_num]
                    training_metrics[f"MADDPG/actor_action_l2_agent_{agent_num}"] = actor_action_l2_values[agent_num]
                    training_metrics[f"MADDPG/actor_action_saturation_excess_agent_{agent_num}"] = actor_action_saturation_values[agent_num]
                    training_metrics[f"MADDPG/actor_grad_norm_agent_{agent_num}"] = actor_grad_norm_values[agent_num]
            self._record_training_metrics(training_metrics, global_learning_step)

        log_message = "Update complete. Avg Critic Loss: {:.4f}, Avg Actor Loss: {:.4f}."
        if should_log_step_metrics:
            logger.info(log_message, critic_loss_scalar, total_actor_loss / self.num_agents)
        else:
            logger.debug(log_message, critic_loss_scalar, total_actor_loss / self.num_agents)

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.end_initial_exploration_time_step

    def _should_log_training_step(self, global_learning_step: int) -> bool:
        return global_learning_step % self.mlflow_step_sample_interval == 0

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

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        metrics = {
            "MADDPG/replay_buffer_size": float(len(self.replay_buffer)),
            "MADDPG/exploration_step": float(getattr(self, "exploration_step", 0)),
            "MADDPG/exploration_sigma": float(getattr(self, "sigma", 0.0)),
            "MADDPG/random_exploration_steps": float(getattr(self, "random_exploration_steps", 0)),
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
            "MADDPG/actor_update_interval": float(getattr(self, "actor_update_interval", 1)),
            "MADDPG/target_policy_smoothing": float(getattr(self, "target_policy_smoothing", False)),
            "MADDPG/actor_action_l2_penalty": float(getattr(self, "actor_action_l2_penalty", 0.0)),
            "MADDPG/actor_action_saturation_penalty": float(
                getattr(self, "actor_action_saturation_penalty", 0.0)
            ),
        }
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
        actions = []
        with torch.inference_mode():
            for agent_idx, (actor, obs) in enumerate(zip(self.actors, observations)):
                raw_action = actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                action = self._scale_action_tensor(agent_idx, raw_action).cpu().numpy()
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

        deterministic_actions = self._predict_deterministic(observations)

        noisy_actions = []
        for agent_idx, action in enumerate(deterministic_actions):
            noise = np.random.normal(loc=self.bias, scale=self.sigma, size=action.shape)
            if self.noise_clip is not None and self.noise_clip > 0.0:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            noisy_actions.append(self._clip_action_array(agent_idx, action + noise))

        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)

        logger.debug("Actions with exploration applied: {}", noisy_actions)
        return [action.tolist() for action in noisy_actions]

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

    def _predict_warm_start_policy(self) -> List[List[float]]:
        if self._warm_start_policy is None or self._latest_raw_observations is None:
            if not self._warned_missing_raw_context:
                logger.warning(
                    "MADDPG warm-start policy exploration requested, but raw observation context is unavailable. "
                    "Falling back to no-op-centered exploration."
                )
                self._warned_missing_raw_context = True
            return self._predict_noop_centered()

        actions = self._warm_start_policy.predict(
            self._latest_raw_observations,
            deterministic=self.warm_start_policy_deterministic,
        )
        clipped_actions: List[List[float]] = []
        for agent_idx, action in enumerate(actions):
            action_array = np.asarray(action, dtype=np.float64).reshape(-1)
            if self.warm_start_policy_noise_scale > 0.0:
                span = self._action_span_for_agent(agent_idx)
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.warm_start_policy_noise_scale * span,
                    size=action_array.shape,
                )
                action_array = action_array + noise
            clipped_actions.append(self._clip_action_array(agent_idx, action_array).tolist())
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_action = self._normalize_scaled_action_tensor(agent_idx, scaled_action)
        action_l2 = normalized_action.pow(2).mean()
        threshold = float(getattr(self, "actor_action_saturation_threshold", 0.85))
        saturation_excess = torch.relu(normalized_action.abs() - threshold).pow(2).mean()
        regularization = (
            float(getattr(self, "actor_action_l2_penalty", 0.0)) * action_l2
            + float(getattr(self, "actor_action_saturation_penalty", 0.0)) * saturation_excess
        )
        return action_l2, saturation_excess, regularization

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
