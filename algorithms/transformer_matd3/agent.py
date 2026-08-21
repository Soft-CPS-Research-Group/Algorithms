from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import _log_torch_runtime, _select_torch_device
from algorithms.transformer_matd3.components import (
    CentralizedCritic,
    DeterministicActorHead,
)
from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer
from algorithms.transformer_matd3.types import LayoutSignature, ReplayBatch
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
    layout: BuildingTokenLayout
    action_names: Tuple[str, ...]
    action_low: torch.Tensor
    action_high: torch.Tensor


class AgentTransformerMATD3(BaseAgent):
    """Static-layout Transformer MATD3 learner.

    Dynamic topology, persistence, residual control, safety, behavior cloning,
    price conditioning, and export belong to later implementation stages.
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
        use_deterministic = bool(deterministic)
        result: List[List[float]] = []
        for state, observation in zip(self._per_building, observations):
            observation_tensor = self._tensor(observation).unsqueeze(0)
            actor_modules = self._actor_modules(state)
            prior_modes = [module.training for module in actor_modules]
            actor_modules.eval()
            try:
                with torch.no_grad():
                    action = self._actor_action(
                        state, observation_tensor, target=False
                    )
                    if not use_deterministic:
                        action = self._explore(action, state)
            finally:
                for module, training in zip(actor_modules, prior_modes):
                    module.train(training)
            result.append(action.squeeze(0).cpu().tolist())
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
        self._update_reward_normalizer(rewards)
        transition = {
            "observations": self._copied_vectors(observations),
            "actions": self._copied_vectors(actions),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1).copy(),
            "next_observations": self._copied_vectors(next_observations),
            "terminated": self._done_vector(terminated),
            "truncated": self._done_vector(truncated),
        }
        self._validate_transition_vectors(transition)
        self._store_transition(transition)

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
        return {
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
        }

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
        raw_rewards = self._tensor(batch.rewards)
        individual_rewards = self._normalize_reward_tensor(raw_rewards)
        train_rewards = self._team_rewards(individual_rewards)
        self._last_train_rewards = train_rewards.detach().cpu()
        done = self._tensor(batch.done.astype(np.float32))

        with torch.no_grad():
            next_actions = [
                self._target_action(state, observation)
                for state, observation in zip(self._per_building, next_observations)
            ]
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
            gap_values.append(float((expected_1.detach() - expected_2.detach()).abs().mean()))
            critic_grad_norms.extend((float(grad_1), float(grad_2)))

        actor_update_due = global_learning_step % self.actor_update_interval == 0
        actor_losses: List[float] = []
        actor_q_abs: List[float] = []
        actor_grad_norms: List[float] = []
        if actor_update_due:
            with torch.no_grad():
                detached_actions = [
                    self._actor_action(state, observation, target=False).detach()
                    for state, observation in zip(self._per_building, observations)
                ]
            for index, state in enumerate(self._per_building):
                joint_actions = list(detached_actions)
                joint_actions[index] = self._actor_action(
                    state, observations[index], target=False
                )
                self._set_requires_grad(state.critic_1, False)
                try:
                    q_policy = state.critic_1(
                        observations, self._layouts, joint_actions
                    )
                    actor_loss = -q_policy.mean()
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
                actor_q_abs.append(float(q_policy.detach().abs().mean()))
                actor_grad_norms.append(float(actor_grad))
            if update_target_step:
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
                actor_losses
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

    @property
    def _layouts(self) -> List[BuildingTokenLayout]:
        return [state.layout for state in self._per_building]

    def _actor_action(
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
        unit_action = torch.tanh(actor(ca_embeddings)).squeeze(-1)
        return state.action_low + (unit_action + 1.0) * (
            state.action_high - state.action_low
        ) / 2.0

    def _target_action(
        self,
        state: _PerBuildingState,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_action(state, observations, target=True)
        if not self.target_policy_smoothing or self.target_policy_noise <= 0.0:
            return action
        span = state.action_high - state.action_low
        noise = torch.randn_like(action) * (self.target_policy_noise * span)
        limit = self.target_policy_noise_clip * span
        noise = torch.maximum(torch.minimum(noise, limit), -limit)
        return torch.maximum(
            torch.minimum(action + noise, state.action_high), state.action_low
        )

    def _explore(
        self,
        action: torch.Tensor,
        state: _PerBuildingState,
    ) -> torch.Tensor:
        if self.exploration_step < self.random_exploration_steps:
            return state.action_low + torch.rand_like(action) * (
                state.action_high - state.action_low
            )
        span = state.action_high - state.action_low
        noise = torch.randn_like(action) * (self.exploration_sigma * span) + (
            self.bias * span
        )
        if self.noise_clip is not None:
            limit = self.noise_clip * span
            noise = torch.maximum(torch.minimum(noise, limit), -limit)
        return torch.maximum(
            torch.minimum(action + noise, state.action_high), state.action_low
        )

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
        )

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
            actor_optimizer=torch.optim.Adam(
                actor_modules.parameters(), lr=self.learning_rate
            ),
            critic_1_optimizer=torch.optim.Adam(
                critic_1.parameters(), lr=self.learning_rate
            ),
            critic_2_optimizer=torch.optim.Adam(
                critic_2.parameters(), lr=self.learning_rate
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
