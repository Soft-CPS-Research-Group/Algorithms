import random
from collections import deque

import numpy as np
import torch


def _maybe_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    """Pin memory only when CUDA is available in the current runtime."""
    if torch.cuda.is_available():
        return tensor.pin_memory()
    return tensor


class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, batch_size):
        """
        Multi-Agent Replay Buffer optimized for CPU storage with fast GPU transfers.

        Args:
            capacity (int): Maximum number of experiences per agent.
            num_agents (int): Number of agents.
            batch_size (int): Number of experiences to sample per agent.
        """

        self.capacity = capacity
        self.num_agents = num_agents
        self.batch_size = batch_size
        # Store joint transitions so sampling preserves alignment across agents.
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        states,
        actions,
        rewards,
        next_states,
        done,
        behavior_actions=None,
        priority_boost=None,
    ):
        """
        Store experiences for each agent in CPU memory.

        Args:
            states (list): List of states per agent.
            actions (list): List of actions per agent.
            rewards (list): List of rewards per agent.
            next_states (list): List of next states per agent.
            done (bool): Single done flag shared across all agents.
            behavior_actions (list, optional): Optional per-agent action targets
                for supervised actor regularization. Defaults to the executed
                actions for backward-compatible behavior cloning.
            priority_boost (float, optional): Ignored by the uniform buffer.
                Weighted replay implementations can use it as an external
                event-priority signal.
        """
        # Keep transitions on pageable CPU memory and pin only sampled batches.
        # This avoids long-lived pinned allocations that can hurt host memory performance.
        if behavior_actions is None:
            behavior_actions = actions
        state_tensors = [
            torch.tensor(states[agent_idx], dtype=torch.float32)
            for agent_idx in range(self.num_agents)
        ]
        action_tensors = [
            torch.tensor(actions[agent_idx], dtype=torch.float32)
            for agent_idx in range(self.num_agents)
        ]
        behavior_action_tensors = [
            torch.tensor(behavior_actions[agent_idx], dtype=torch.float32)
            for agent_idx in range(self.num_agents)
        ]
        reward_tensors = [
            torch.tensor(rewards[agent_idx], dtype=torch.float32).unsqueeze(0)
            for agent_idx in range(self.num_agents)
        ]
        next_state_tensors = [
            torch.tensor(next_states[agent_idx], dtype=torch.float32)
            for agent_idx in range(self.num_agents)
        ]
        done_tensor = torch.tensor(float(done), dtype=torch.float32).unsqueeze(0)

        self.buffer.append(
            (
                state_tensors,
                action_tensors,
                reward_tensors,
                next_state_tensors,
                done_tensor,
                behavior_action_tensors,
            )
        )

    def sample(self):
        """
        Sample a batch of experiences for each agent.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated) as CPU tensors.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch = random.sample(self.buffer, self.batch_size)
        return self._build_sample(batch, include_behavior_actions=False)

    def sample_with_behavior_actions(self):
        """
        Sample a batch and include per-agent behavior-cloning action targets.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated,
            behavior_actions) as CPU tensors.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch = random.sample(self.buffer, self.batch_size)
        return self._build_sample(batch, include_behavior_actions=True)

    def _build_sample(self, batch, *, include_behavior_actions: bool):
        unpacked = [self._unpack_transition(experience) for experience in batch]
        (
            states_batch,
            actions_batch,
            rewards_batch,
            next_states_batch,
            dones_batch,
            behavior_actions_batch,
        ) = zip(*unpacked)

        states = [
            _maybe_pin_memory(torch.stack([transition[agent_idx] for transition in states_batch]))
            for agent_idx in range(self.num_agents)
        ]
        actions = [
            _maybe_pin_memory(torch.stack([transition[agent_idx] for transition in actions_batch]))
            for agent_idx in range(self.num_agents)
        ]
        rewards = [
            _maybe_pin_memory(torch.stack([transition[agent_idx] for transition in rewards_batch]))
            for agent_idx in range(self.num_agents)
        ]
        next_states = [
            _maybe_pin_memory(torch.stack([transition[agent_idx] for transition in next_states_batch]))
            for agent_idx in range(self.num_agents)
        ]
        behavior_actions = [
            _maybe_pin_memory(torch.stack([transition[agent_idx] for transition in behavior_actions_batch]))
            for agent_idx in range(self.num_agents)
        ]

        # Keep historical shape: [num_agents, batch_size, 1].
        done_tensor = _maybe_pin_memory(torch.stack(dones_batch))
        done_tensor = done_tensor.unsqueeze(0).expand(self.num_agents, -1, -1)

        if include_behavior_actions:
            return states, actions, rewards, next_states, done_tensor, behavior_actions
        return states, actions, rewards, next_states, done_tensor

    @staticmethod
    def _unpack_transition(experience):
        if len(experience) == 6:
            return experience
        if len(experience) == 5:
            states, actions, rewards, next_states, done = experience
            return states, actions, rewards, next_states, done, actions
        raise ValueError("Unsupported replay transition format.")

    def get_state(self):
        """Return a serialisable snapshot of the replay buffer."""
        return {
            "format": "joint_transitions_v3",
            "buffer": list(self.buffer),
        }

    def set_state(self, state):
        """Restore buffer contents from :meth:`get_state`."""
        if state is None:
            return
        self.buffer = deque(maxlen=self.capacity)

        if isinstance(state, dict) and "buffer" in state:
            experiences = state["buffer"]
            for experience in experiences:
                self.buffer.append(experience)
            return

        # Backward compatibility for older checkpoints where each agent had its own deque.
        if isinstance(state, list) and len(state) == self.num_agents:
            agent_buffers = state
            min_len = min(len(agent_buffer) for agent_buffer in agent_buffers) if agent_buffers else 0
            for transition_idx in range(min_len):
                per_agent = [agent_buffers[agent_idx][transition_idx] for agent_idx in range(self.num_agents)]
                states = [entry[0] for entry in per_agent]
                actions = [entry[1] for entry in per_agent]
                rewards = [entry[2] for entry in per_agent]
                next_states = [entry[3] for entry in per_agent]
                done = per_agent[0][4]
                self.buffer.append((states, actions, rewards, next_states, done, actions))
            return

        if isinstance(state, list):
            for experience in state:
                self.buffer.append(experience)
            return

        raise ValueError("Unsupported replay buffer state format.")

    def __len__(self):
        """Get the current replay size."""
        return len(self.buffer)


class RewardWeightedMultiAgentReplayBuffer(MultiAgentReplayBuffer):
    """Joint multi-agent replay with optional reward-weighted sampling.

    This keeps the same output contract as :class:`MultiAgentReplayBuffer`, but
    samples a configurable fraction of the batch from transitions with large
    reward signal according to ``priority_mode``. In this project those
    transitions usually correspond to EV service windows, grid violations,
    deferrable deadlines or other rare constraint signals that uniform replay
    can under-sample.
    """

    def __init__(
        self,
        capacity,
        num_agents,
        batch_size,
        priority_fraction: float = 0.5,
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1.0e-3,
        priority_mode: str = "abs_reward",
        priority_max: float | None = None,
        behavior_action_priority_weight: float = 0.0,
        behavior_action_priority_mode: str = "positive",
        behavior_action_priority_scope: str = "all",
    ):
        super().__init__(capacity=capacity, num_agents=num_agents, batch_size=batch_size)
        self.priority_fraction = float(np.clip(float(priority_fraction), 0.0, 1.0))
        self.priority_alpha = max(float(priority_alpha), 0.0)
        self.priority_epsilon = max(float(priority_epsilon), 1.0e-12)
        self.priority_mode = str(priority_mode or "abs_reward").strip().lower()
        if self.priority_mode not in {"abs_reward", "negative_reward", "positive_reward"}:
            raise ValueError(
                "RewardWeightedMultiAgentReplayBuffer priority_mode must be "
                "'abs_reward', 'negative_reward' or 'positive_reward'."
            )
        self.priority_max = None if priority_max is None else max(float(priority_max), self.priority_epsilon)
        self.behavior_action_priority_weight = max(float(behavior_action_priority_weight), 0.0)
        self.behavior_action_priority_mode = str(behavior_action_priority_mode or "positive").strip().lower()
        if self.behavior_action_priority_mode not in {"positive", "abs"}:
            raise ValueError(
                "RewardWeightedMultiAgentReplayBuffer behavior_action_priority_mode must be "
                "'positive' or 'abs'."
            )
        self.behavior_action_priority_scope = str(behavior_action_priority_scope or "all").strip().lower()
        if self.behavior_action_priority_scope not in {"all", "ev"}:
            raise ValueError(
                "RewardWeightedMultiAgentReplayBuffer behavior_action_priority_scope must be "
                "'all' or 'ev'."
            )
        self.behavior_action_priority_masks = None
        self.priorities = deque(maxlen=capacity)

    def set_behavior_action_priority_masks(self, masks) -> None:
        """Restrict behavior-action priority to selected per-agent action dimensions."""
        parsed_masks = []
        for mask in masks or []:
            parsed_masks.append(np.asarray(mask, dtype=bool).reshape(-1))
        self.behavior_action_priority_masks = parsed_masks or None

    def push(
        self,
        states,
        actions,
        rewards,
        next_states,
        done,
        behavior_actions=None,
        priority_boost=None,
    ):
        super().push(
            states,
            actions,
            rewards,
            next_states,
            done,
            behavior_actions=behavior_actions,
            priority_boost=priority_boost,
        )
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        finite_rewards = reward_values[np.isfinite(reward_values)]
        priority = self._priority_from_rewards(finite_rewards)
        priority += self.behavior_action_priority_weight * self._priority_from_behavior_actions(
            behavior_actions if behavior_actions is not None else actions
        )
        priority += self._priority_from_external_boost(priority_boost)
        if self.priority_max is not None:
            priority = min(priority, self.priority_max)
        self.priorities.append(priority + self.priority_epsilon)

    def _priority_from_rewards(self, finite_rewards: np.ndarray) -> float:
        if finite_rewards.size == 0:
            return 0.0
        if self.priority_mode == "negative_reward":
            return float(np.max(np.maximum(-finite_rewards, 0.0)))
        if self.priority_mode == "positive_reward":
            return float(np.max(np.maximum(finite_rewards, 0.0)))
        return float(np.max(np.abs(finite_rewards)))

    def _priority_from_behavior_actions(self, behavior_actions) -> float:
        if self.behavior_action_priority_weight <= 0.0:
            return 0.0
        values = []
        for agent_idx, action in enumerate(behavior_actions or []):
            array = np.asarray(action, dtype=np.float64).reshape(-1)
            if self.behavior_action_priority_scope != "all":
                masks = self.behavior_action_priority_masks or []
                if agent_idx >= len(masks):
                    continue
                mask = np.asarray(masks[agent_idx], dtype=bool).reshape(-1)
                if mask.shape[0] != array.shape[0] or not np.any(mask):
                    continue
                array = array[mask]
            finite = array[np.isfinite(array)]
            if finite.size > 0:
                values.append(finite)
        if not values:
            return 0.0
        action_values = np.concatenate(values)
        if self.behavior_action_priority_mode == "abs":
            return float(np.max(np.abs(action_values)))
        return float(np.max(np.maximum(action_values, 0.0)))

    @staticmethod
    def _priority_from_external_boost(priority_boost) -> float:
        if priority_boost is None:
            return 0.0
        try:
            value = float(priority_boost)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(value, 0.0)

    def sample(self):
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch_indices = self._sample_indices()
        batch = [self.buffer[index] for index in batch_indices]
        return self._build_sample(batch, include_behavior_actions=False)

    def sample_with_behavior_actions(self):
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch_indices = self._sample_indices()
        batch = [self.buffer[index] for index in batch_indices]
        return self._build_sample(batch, include_behavior_actions=True)

    def _sample_indices(self) -> list[int]:
        replay_size = len(self.buffer)
        priority_count = int(round(self.batch_size * self.priority_fraction))
        uniform_count = self.batch_size - priority_count

        indices: list[int] = []
        if uniform_count > 0:
            indices.extend(random.choices(range(replay_size), k=uniform_count))

        if priority_count > 0:
            priorities = np.asarray(list(self.priorities), dtype=np.float64)
            if priorities.shape[0] != replay_size:
                priorities = np.ones(replay_size, dtype=np.float64)
            weights = np.power(np.maximum(priorities, self.priority_epsilon), self.priority_alpha)
            total = float(np.sum(weights))
            if total <= 0.0 or not np.isfinite(total):
                probabilities = np.full(replay_size, 1.0 / replay_size, dtype=np.float64)
            else:
                probabilities = weights / total
            indices.extend(np.random.choice(replay_size, size=priority_count, replace=True, p=probabilities).tolist())

        random.shuffle(indices)
        return indices

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "format": "joint_reward_weighted_transitions_v2",
                "priorities": list(self.priorities),
                "priority_fraction": self.priority_fraction,
                "priority_alpha": self.priority_alpha,
                "priority_epsilon": self.priority_epsilon,
                "priority_mode": self.priority_mode,
                "priority_max": self.priority_max,
                "behavior_action_priority_weight": self.behavior_action_priority_weight,
                "behavior_action_priority_mode": self.behavior_action_priority_mode,
                "behavior_action_priority_scope": self.behavior_action_priority_scope,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        if isinstance(state, dict):
            if "priority_mode" in state:
                mode = str(state.get("priority_mode") or self.priority_mode).strip().lower()
                if mode in {"abs_reward", "negative_reward", "positive_reward"}:
                    self.priority_mode = mode
            if "priority_max" in state:
                raw_max = state.get("priority_max")
                self.priority_max = None if raw_max is None else max(float(raw_max), self.priority_epsilon)
            if "behavior_action_priority_weight" in state:
                self.behavior_action_priority_weight = max(
                    float(state.get("behavior_action_priority_weight") or 0.0),
                    0.0,
                )
            if "behavior_action_priority_mode" in state:
                mode = str(state.get("behavior_action_priority_mode") or self.behavior_action_priority_mode)
                mode = mode.strip().lower()
                if mode in {"positive", "abs"}:
                    self.behavior_action_priority_mode = mode
            if "behavior_action_priority_scope" in state:
                scope = str(state.get("behavior_action_priority_scope") or self.behavior_action_priority_scope)
                scope = scope.strip().lower()
                if scope in {"all", "ev"}:
                    self.behavior_action_priority_scope = scope
        self.priorities = deque(maxlen=self.capacity)
        if isinstance(state, dict) and isinstance(state.get("priorities"), list):
            for value in state["priorities"][-self.capacity :]:
                self.priorities.append(float(value))
        while len(self.priorities) < len(self.buffer):
            self.priorities.append(self.priority_epsilon)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, device="cpu"):
        """
        Prioritized Replay Buffer that stores experiences as PyTorch tensors.

        Args:
            capacity (int): Maximum buffer size.
            alpha (float): Prioritization exponent.
            device (str): Device to store tensors on ('cpu' or 'cuda').
        """
        self.capacity = capacity
        self.alpha = alpha
        self.device = torch.device(device)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        """
        Store an experience tuple with priority.

        Args:
            state (array-like): Current state.
            action (array-like): Action taken.
            reward (float): Reward received.
            next_state (array-like): Next state.
            done (bool): Whether the episode is done.
            td_error (float): TD error used for prioritization.
        """
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        # Convert to PyTorch tensors before storing
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences using prioritized sampling.

        Args:
            batch_size (int): Number of samples.
            beta (float): Importance sampling correction factor.

        Returns:
            Tensors of (states, actions, rewards, next_states, dones), indices, and importance sampling weights.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.stack(states).to(self.device),
            torch.stack(actions).to(self.device),
            torch.stack(rewards).to(self.device),
            torch.stack(next_states).to(self.device),
            torch.stack(dones).to(self.device),
            torch.tensor(indices, dtype=torch.int64, device=self.device),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled experiences.

        Args:
            indices (array-like): Indices of sampled experiences.
            priorities (array-like): New priority values.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def get_state(self):
        """Return a serialisable snapshot of the buffer."""
        return {
            "buffer": list(self.buffer),
            "priorities": self.priorities.copy(),
            "position": self.position,
            "capacity": self.capacity,
            "alpha": self.alpha,
            "device": str(self.device),
        }

    def set_state(self, state):
        """Restore buffer contents from :meth:`get_state`."""
        if state is None:
            return

        self.buffer = state["buffer"]
        self.priorities = state["priorities"]
        self.position = state["position"]
        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
        self.device = torch.device(state["device"])
