from __future__ import annotations

import random
import time
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

        topology = self.config.get("topology", {})

        self.num_agents = topology.get("num_agents") or hyperparams.get("num_agents")
        self.observation_dimension = topology.get("observation_dimensions") or hyperparams.get("observation_dimensions")
        self.action_dimension = topology.get("action_dimensions") or hyperparams.get("action_dimensions")

        if self.num_agents is None or self.observation_dimension is None or self.action_dimension is None:
            raise ValueError("Topology information (num_agents / observation_dimensions / action_dimensions) is required for MADDPG.")

        self.replay_buffer = self._initialize_replay_buffer()
        self.actors, self.critics, self.actor_targets, self.critic_targets = self._initialize_networks()
        self.actor_optimizers, self.critic_optimizers = self._initialize_optimizers()
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        logger.info("MADDPG initialization complete.")

    def _initialize_replay_buffer(self):
        logger.debug("Initializing replay buffer.")
        replay_buffer_name = self.config["algorithm"]["replay_buffer"]["class"]
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

        rewards_all = torch.stack(rewards_all).to(self.device, dtype=torch.float32, non_blocking=True)
        dones_all = dones_all.to(self.device, dtype=torch.float32, non_blocking=True)
        states = [s.to(self.device, non_blocking=True) for s in states]
        actions_all = [a.to(self.device, non_blocking=True) for a in actions_all]
        next_states = [ns.to(self.device, non_blocking=True) for ns in next_states]

        global_state = torch.cat(states, dim=1)
        global_next_state = torch.cat(next_states, dim=1)
        global_actions = torch.cat(actions_all, dim=1)

        with torch.no_grad():
            global_next_actions = torch.cat([self.actor_targets[i](next_states[i]) for i in range(self.num_agents)], dim=1)
            q_targets_next = torch.stack(
                [critic(global_next_state, global_next_actions) for critic in self.critic_targets]
            )
            q_targets = rewards_all + self.gamma * q_targets_next * (1 - dones_all)

        q_expected = torch.stack([critic(global_state, global_actions) for critic in self.critics])
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            critic_loss = mse_loss(q_expected, q_targets).mean()

        for optimizer in self.critic_optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        for optimizer in self.critic_optimizers:
            self.scaler.unscale_(optimizer)
        clip_grad_norm_([param for critic in self.critics for param in critic.parameters()], max_norm=1.0)
        for optimizer in self.critic_optimizers:
            self.scaler.step(optimizer)
        self.scaler.update()

        logger.debug("Critics updated. Loss: {:.4f}.", critic_loss.item())

        should_log_step_metrics = self._should_log_training_step(global_learning_step)

        for agent_num in range(self.num_agents):
            critic_loss_value = (q_expected[agent_num] - q_targets[agent_num]).abs().mean()
            if should_log_step_metrics and mlflow.active_run():
                mlflow.log_metric(
                    f"critic_loss_agent_{agent_num}",
                    critic_loss_value.item(),
                    step=global_learning_step,
                )

        # Compute detached policy actions once and reuse them during actor updates.
        with torch.no_grad():
            detached_policy_actions = [
                actor(state).detach()
                for actor, state in zip(self.actors, states)
            ]

        total_actor_loss = 0.0
        for agent_num, (actor, critic, actor_optimizer) in enumerate(
            zip(self.actors, self.critics, self.actor_optimizers)
        ):
            obs = states[agent_num]

            predicted_action = actor(obs)
            joint_policy_actions = list(detached_policy_actions)
            joint_policy_actions[agent_num] = predicted_action
            global_predicted_actions = torch.cat(joint_policy_actions, dim=1)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                actor_loss = -critic(global_state, global_predicted_actions).mean()

            actor_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(actor_optimizer)
            clip_grad_norm_(actor.parameters(), max_norm=1.0)
            self.scaler.step(actor_optimizer)
            self.scaler.update()

            # Keep cached detached actions in sync with sequential actor updates.
            with torch.no_grad():
                detached_policy_actions[agent_num] = actor(obs).detach()

            total_actor_loss += actor_loss.item()
            logger.debug("Actor {} updated. Loss: {:.4f}.", agent_num, actor_loss.item())

            if update_target_step:
                logger.debug("Updating target networks for agent {}.", agent_num)
                self._soft_update(critic, self.critic_targets[agent_num], self.tau)
                self._soft_update(actor, self.actor_targets[agent_num], self.tau)

            if should_log_step_metrics and mlflow.active_run():
                mlflow.log_metric(
                    f"actor_loss_agent_{agent_num}",
                    actor_loss.item(),
                    step=global_learning_step,
                )

        if should_log_step_metrics and mlflow.active_run():
            mlflow.log_metrics(
                {
                    "average_critic_loss": critic_loss.item(),
                    "average_actor_loss": total_actor_loss / self.num_agents,
                    "training_step_time": time.time() - update_start_time,
                },
                step=global_learning_step,
            )

        log_message = "Update complete. Avg Critic Loss: {:.4f}, Avg Actor Loss: {:.4f}."
        if should_log_step_metrics:
            logger.info(log_message, critic_loss.item(), total_actor_loss / self.num_agents)
        else:
            logger.debug(log_message, critic_loss.item(), total_actor_loss / self.num_agents)

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return global_learning_step >= self.end_initial_exploration_time_step

    def _should_log_training_step(self, global_learning_step: int) -> bool:
        return global_learning_step % self.mlflow_step_sample_interval == 0

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
            for actor, obs in zip(self.actors, observations):
                action = actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).cpu().numpy()
                actions.append(action)
        logger.debug("Deterministic actions predicted: {}", actions)
        return actions

    def _predict_with_exploration(self, observations):
        self.exploration_step += 1
        if self.exploration_step <= self.random_exploration_steps:
            return self._predict_random()

        deterministic_actions = self._predict_deterministic(observations)

        noisy_actions = []
        for action in deterministic_actions:
            noise = np.random.normal(loc=self.bias, scale=self.sigma, size=action.shape)
            if self.noise_clip is not None and self.noise_clip > 0.0:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            noisy_actions.append(np.clip(action + noise, -1, 1))

        self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)

        logger.debug("Actions with exploration applied: {}", noisy_actions)
        return [action.tolist() for action in noisy_actions]

    def _predict_random(self) -> List[List[float]]:
        random_actions = [
            np.random.uniform(low=-1.0, high=1.0, size=(self.action_dimension[i],)).tolist()
            for i in range(self.num_agents)
        ]
        logger.debug("Random exploration actions predicted: {}", random_actions)
        return random_actions

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
            checkpoint[f"actor_optimizer_state_dict_{i}"] = self.actor_optimizers[i].state_dict()
            checkpoint[f"critic_optimizer_state_dict_{i}"] = self.critic_optimizers[i].state_dict()

        if hasattr(self.replay_buffer, "get_state"):
            checkpoint["replay_buffer"] = self.replay_buffer.get_state()

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
            if not self.fine_tune:
                self.actor_optimizers[i].load_state_dict(checkpoint[f"actor_optimizer_state_dict_{i}"])
                self.critic_optimizers[i].load_state_dict(checkpoint[f"critic_optimizer_state_dict_{i}"])

        if "replay_buffer" in checkpoint and not self.reset_replay_buffer:
            self.replay_buffer.set_state(checkpoint["replay_buffer"])

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
            torch.onnx.export(
                actor,
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
