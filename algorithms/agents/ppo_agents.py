from __future__ import annotations

import random
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
from algorithms.utils.networks import GaussianActor, ValueNetwork
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
        self._warm_start_policy = None
        self._latest_raw_observations: Optional[List[np.ndarray]] = None
        self._latest_encoded_observations: Optional[List[np.ndarray]] = None
        self._last_warm_start_policy_actions: Optional[List[List[float]]] = None
        self._last_warm_start_phaseout_probability = 0.0
        self._last_warm_start_phaseout_used = False

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
        _ = context
        deterministic = bool(deterministic)
        self.exploration_step += 1
        self._last_warm_start_policy_actions = None
        if not deterministic and self.exploration_step <= self.random_exploration_steps:
            if self.initial_exploration_strategy == "policy":
                return self._predict_warm_start_policy()
            return self._predict_random()

        actions = self._predict_actor(observations, deterministic=deterministic)
        if not deterministic:
            actions = self._apply_warm_start_phaseout(actions)
        return actions

    def _predict_actor(self, observations, *, deterministic: bool) -> List[List[float]]:
        actions: List[List[float]] = []
        with torch.inference_mode():
            for agent_idx, obs in enumerate(observations):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
                if deterministic:
                    normalized = self.actors[agent_idx](obs_tensor)
                else:
                    distribution = self.actors[agent_idx].distribution(obs_tensor)
                    normalized = torch.clamp(distribution.rsample(), -1.0, 1.0)
                scaled = self._scale_action_tensor(agent_idx, normalized)
                actions.append(scaled.squeeze(0).cpu().numpy().tolist())
        return actions

    def _predict_warm_start_policy(self) -> List[List[float]]:
        if self._warm_start_policy is None:
            return self._predict_random()
        observations = self._latest_raw_observations or self._latest_encoded_observations
        if observations is None:
            return self._predict_random()
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
            return actor_actions

        teacher_actions = self._predict_warm_start_policy()
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
        teacher_actions = []
        old_log_probs = []
        values = []
        with torch.no_grad():
            for agent_idx in range(int(self.num_agents)):
                action_tensor = torch.as_tensor(actions[agent_idx], dtype=torch.float32, device=self.device).view(1, -1)
                normalized = self._normalize_scaled_action_tensor(agent_idx, action_tensor)
                teacher_action = None
                if self._last_warm_start_policy_actions is not None and agent_idx < len(self._last_warm_start_policy_actions):
                    teacher_tensor = torch.as_tensor(
                        self._last_warm_start_policy_actions[agent_idx],
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, -1)
                    if teacher_tensor.shape[-1] == normalized.shape[-1]:
                        teacher_action = self._normalize_scaled_action_tensor(agent_idx, teacher_tensor)
                obs_batch = obs_tensors[agent_idx].to(self.device).view(1, -1)
                distribution = self.actors[agent_idx].distribution(obs_batch)
                log_prob = distribution.log_prob(normalized).sum(dim=-1)
                value_input = self._value_input_for_agent(agent_idx, obs_tensors).to(self.device).view(1, -1)
                value = self.value_nets[agent_idx](value_input).squeeze(-1)
                normalized_actions.append(normalized.squeeze(0).cpu())
                if teacher_action is None:
                    teacher_actions.append(torch.full_like(normalized.squeeze(0).cpu(), float("nan")))
                else:
                    teacher_actions.append(teacher_action.squeeze(0).cpu())
                old_log_probs.append(log_prob.squeeze(0).cpu())
                values.append(value.squeeze(0).cpu())

        self.rollout.append(
            {
                "observations": obs_tensors,
                "next_observations": next_obs_tensors,
                "actions": normalized_actions,
                "teacher_actions": teacher_actions,
                "rewards": torch.as_tensor(rewards, dtype=torch.float32).view(-1),
                "done": bool(done),
                "old_log_probs": torch.stack(old_log_probs),
                "values": torch.stack(values),
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

        flat_advantages = advantages.reshape(-1)
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
        behavior_cloning_extra_losses, behavior_cloning_extra_grad_norms = (
            self._run_actor_behavior_cloning_extra_updates(
                indices,
                behavior_cloning_weight=behavior_cloning_weight,
                extra_updates=behavior_cloning_extra_updates,
            )
        )

        for _epoch in range(self.ppo_epochs):
            shuffled = indices[torch.randperm(rollout_size, device=self.device)]
            stop_epoch = False
            for start in range(0, rollout_size, self.minibatch_size):
                batch_idx = shuffled[start : start + self.minibatch_size]
                for agent_idx in self._ppo_agent_update_order():
                    obs_batch = self._stack_agent_observations(agent_idx, batch_idx)
                    action_batch = self._stack_agent_actions(agent_idx, batch_idx)
                    value_input_batch = self._stack_value_inputs(agent_idx, batch_idx)

                    distribution = self.actors[agent_idx].distribution(obs_batch)
                    log_prob = distribution.log_prob(action_batch).sum(dim=-1)
                    entropy = distribution.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(log_prob - old_log_probs[batch_idx, agent_idx])
                    advantage_batch = advantages[batch_idx, agent_idx]
                    unclipped_loss = ratio * advantage_batch
                    clipped_loss = torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    ) * advantage_batch
                    policy_loss = -torch.minimum(unclipped_loss, clipped_loss).mean()

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

                    with torch.no_grad():
                        approx_kl = (old_log_probs[batch_idx, agent_idx] - log_prob).mean().abs()

                    policy_losses.append(float(policy_loss.detach().item()))
                    value_losses.append(float(value_loss.detach().item()))
                    behavior_cloning_losses.append(float(behavior_cloning_loss.detach().item()))
                    actor_regularization_losses.append(float(actor_regularization_loss.detach().item()))
                    entropy_values.append(float(entropy.detach().item()))
                    approx_kl_values.append(float(approx_kl.detach().item()))
                    grad_norm_values.append(float(grad_norm))

                    if self.target_kl is not None and approx_kl.item() > self.target_kl:
                        stop_epoch = True
                        break
                if stop_epoch:
                    break
            if stop_epoch:
                break

        if self.training_diagnostics_enabled and global_learning_step % self.mlflow_step_sample_interval == 0:
            metrics = {
                f"{self.metric_prefix}/rollout_size": float(rollout_size),
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

    def _stack_agent_observations(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [self.rollout[int(idx.item())]["observations"][agent_idx] for idx in indices]
        return torch.stack(selected).to(self.device)

    def _stack_agent_actions(self, agent_idx: int, indices: torch.Tensor) -> torch.Tensor:
        selected = [self.rollout[int(idx.item())]["actions"][agent_idx] for idx in indices]
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
        rollout_size = int(indices.numel())
        for _update in range(extra_updates):
            shuffled = indices[torch.randperm(rollout_size, device=self.device)]
            for start in range(0, rollout_size, self.minibatch_size):
                batch_idx = shuffled[start : start + self.minibatch_size]
                for agent_idx in self._ppo_agent_update_order():
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

    def _actor_behavior_cloning_loss(
        self,
        agent_idx: int,
        obs_batch: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if float(getattr(self, "actor_behavior_cloning_weight", 0.0)) <= 0.0:
            return torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        teacher_actions = self._stack_agent_teacher_actions(agent_idx, indices)
        valid_mask = torch.isfinite(teacher_actions).all(dim=1)
        if not torch.any(valid_mask):
            return torch.as_tensor(0.0, dtype=obs_batch.dtype, device=obs_batch.device)

        predicted = self.actors[agent_idx](obs_batch[valid_mask])
        target = teacher_actions[valid_mask]
        return mse_loss(predicted, target)

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
            f"{self.metric_prefix}/initial_exploration_done": float(
                self.exploration_step >= self.end_initial_exploration_time_step
            ),
        }

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        metrics = dict(self._latest_training_metrics)
        self._latest_training_metrics = {}
        return metrics

    def _record_training_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._latest_training_metrics = dict(metrics)
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)

    def save_checkpoint(self, output_dir: str, step: int) -> str:
        checkpoint: Dict[str, Any] = {
            "step": int(step),
            "rollout": self.rollout,
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
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        for agent_idx in range(int(self.num_agents)):
            self.actors[agent_idx].load_state_dict(checkpoint[f"actor_state_dict_{agent_idx}"])
            self.value_nets[agent_idx].load_state_dict(checkpoint[f"value_state_dict_{agent_idx}"])
            if not self.fine_tune:
                self.actor_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"actor_optimizer_state_dict_{agent_idx}"]
                )
                self.value_optimizers[agent_idx].load_state_dict(
                    checkpoint[f"value_optimizer_state_dict_{agent_idx}"]
                )
        if not self.reset_replay_buffer:
            self.rollout = list(checkpoint.get("rollout", []))
        self.exploration_step = int(checkpoint.get("exploration_step", self.exploration_step))
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"])
            if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        bundle_cfg = ((context.get("config") or {}).get("bundle") or {})
        global_artifact_config = dict(bundle_cfg.get("artifact_config") or {})
        raw_per_agent_config = bundle_cfg.get("per_agent_artifact_config") or {}
        per_agent_artifact_config = raw_per_agent_config if isinstance(raw_per_agent_config, dict) else {}
        require_observations_envelope = bool(bundle_cfg.get("require_observations_envelope", False))

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        metadata: Dict[str, Any] = {"format": "onnx", "artifacts": []}

        for agent_idx, actor in enumerate(self.actors):
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
                per_agent_artifact_config.get(str(agent_idx))
                if str(agent_idx) in per_agent_artifact_config
                else per_agent_artifact_config.get(agent_idx)
            )
            agent_override = raw_agent_override if isinstance(raw_agent_override, dict) else {}
            artifact_config: Dict[str, Any] = {}
            artifact_config.update(build_auto_artifact_config(context=context, agent_index=agent_idx))
            artifact_config.update(global_artifact_config)
            artifact_config.update(agent_override)
            if require_observations_envelope:
                artifact_config["require_observations_envelope"] = True

            metadata["artifacts"].append(
                {
                    "agent_index": agent_idx,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                    "observation_dimension": self.observation_dimension[agent_idx],
                    "action_dimension": self.action_dimension[agent_idx],
                    "config": artifact_config,
                }
            )
            if mlflow.active_run():
                mlflow.log_artifact(str(export_path), artifact_path="onnx")

        return metadata


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
