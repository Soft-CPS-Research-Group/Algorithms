import random

import numpy as np
import torch


_PIN_MEMORY_AVAILABLE = torch.cuda.is_available()


def _maybe_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    """Pin memory only when CUDA is available in the current runtime."""
    if _PIN_MEMORY_AVAILABLE:
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
        # Store joint transitions in compact preallocated arrays. This avoids
        # keeping millions of tiny Tensor/list objects alive in long 15s runs.
        self.position = 0
        self.size = 0
        self._last_insert_index = -1
        self._state_dims: list[int] = []
        self._action_dims: list[int] = []
        self._behavior_action_dims: list[int] = []
        self._next_behavior_action_dims: list[int] = []
        self._states: list[np.ndarray] | None = None
        self._actions: list[np.ndarray] | None = None
        self._behavior_actions: list[np.ndarray] | None = None
        self._next_behavior_actions: list[np.ndarray] | None = None
        self._next_states: list[np.ndarray] | None = None
        self._rewards: np.ndarray | None = None
        self._dones: np.ndarray | None = None

    def push(
        self,
        states,
        actions,
        rewards,
        next_states,
        done,
        behavior_actions=None,
        next_behavior_actions=None,
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
            next_behavior_actions (list, optional): Optional per-agent action
                targets for the next state. Residual policies use these as the
                base policy in critic targets. Defaults to ``behavior_actions``.
            priority_boost (float, optional): Ignored by the uniform buffer.
                Weighted replay implementations can use it as an external
                event-priority signal.
        """
        if behavior_actions is None:
            behavior_actions = actions
        if next_behavior_actions is None:
            next_behavior_actions = behavior_actions

        self._ensure_storage(states, actions, next_states, behavior_actions, next_behavior_actions)
        insert_index = self.position

        assert self._states is not None
        assert self._actions is not None
        assert self._behavior_actions is not None
        assert self._next_behavior_actions is not None
        assert self._next_states is not None
        assert self._rewards is not None
        assert self._dones is not None

        for agent_idx in range(self.num_agents):
            self._states[agent_idx][insert_index] = self._coerce_vector(
                states[agent_idx],
                self._state_dims[agent_idx],
                label=f"state[{agent_idx}]",
            )
            self._actions[agent_idx][insert_index] = self._coerce_vector(
                actions[agent_idx],
                self._action_dims[agent_idx],
                label=f"action[{agent_idx}]",
            )
            self._behavior_actions[agent_idx][insert_index] = self._coerce_vector(
                behavior_actions[agent_idx],
                self._behavior_action_dims[agent_idx],
                label=f"behavior_action[{agent_idx}]",
            )
            self._next_behavior_actions[agent_idx][insert_index] = self._coerce_vector(
                next_behavior_actions[agent_idx],
                self._next_behavior_action_dims[agent_idx],
                label=f"next_behavior_action[{agent_idx}]",
            )
            self._next_states[agent_idx][insert_index] = self._coerce_vector(
                next_states[agent_idx],
                self._state_dims[agent_idx],
                label=f"next_state[{agent_idx}]",
            )
            reward_value = rewards[agent_idx] if agent_idx < len(rewards) else 0.0
            self._rewards[insert_index, agent_idx, 0] = self._safe_float32(reward_value)
        self._dones[insert_index, 0] = 1.0 if bool(done) else 0.0

        self.position = (insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._last_insert_index = insert_index

    def _ensure_storage(self, states, actions, next_states, behavior_actions, next_behavior_actions) -> None:
        if self._states is not None:
            return

        self._state_dims = [
            int(np.asarray(states[agent_idx], dtype=np.float32).reshape(-1).shape[0])
            for agent_idx in range(self.num_agents)
        ]
        next_state_dims = [
            int(np.asarray(next_states[agent_idx], dtype=np.float32).reshape(-1).shape[0])
            for agent_idx in range(self.num_agents)
        ]
        self._action_dims = [
            int(np.asarray(actions[agent_idx], dtype=np.float32).reshape(-1).shape[0])
            for agent_idx in range(self.num_agents)
        ]
        self._behavior_action_dims = [
            int(np.asarray(behavior_actions[agent_idx], dtype=np.float32).reshape(-1).shape[0])
            for agent_idx in range(self.num_agents)
        ]
        self._next_behavior_action_dims = [
            int(np.asarray(next_behavior_actions[agent_idx], dtype=np.float32).reshape(-1).shape[0])
            for agent_idx in range(self.num_agents)
        ]
        if next_state_dims != self._state_dims:
            raise ValueError("Replay buffer next_state dimensions must match state dimensions.")
        if self._next_behavior_action_dims != self._behavior_action_dims:
            raise ValueError("Replay buffer next_behavior_action dimensions must match behavior_action dimensions.")

        self._states = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._state_dims
        ]
        self._actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._action_dims
        ]
        self._behavior_actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._behavior_action_dims
        ]
        self._next_behavior_actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._next_behavior_action_dims
        ]
        self._next_states = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._state_dims
        ]
        self._rewards = np.zeros((self.capacity, self.num_agents, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)

    @staticmethod
    def _safe_float32(value) -> np.float32:
        try:
            parsed = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            parsed = 0.0
        if not np.isfinite(parsed):
            parsed = 0.0
        return np.float32(parsed)

    @staticmethod
    def _coerce_vector(value, expected_dim: int, *, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] != expected_dim:
            raise ValueError(
                f"Replay buffer {label} dimension changed from {expected_dim} to {array.shape[0]}."
            )
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def sample(self):
        """
        Sample a batch of experiences for each agent.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated) as CPU tensors.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        indices = np.asarray(random.sample(range(self.size), self.batch_size), dtype=np.int64)
        return self._build_sample(indices, include_behavior_actions=False)

    def sample_with_behavior_actions(self):
        """
        Sample a batch and include per-agent behavior-cloning action targets.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated,
            behavior_actions) as CPU tensors.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        indices = np.asarray(random.sample(range(self.size), self.batch_size), dtype=np.int64)
        return self._build_sample(indices, include_behavior_actions=True)

    def sample_with_policy_context_actions(self):
        """
        Sample a batch and include current/next behavior action targets.

        Residual policies learn deltas around a teacher policy, so the actor
        loss needs current base actions and critic targets need next-state base
        actions. Existing callers can keep using ``sample_with_behavior_actions``.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        indices = np.asarray(random.sample(range(self.size), self.batch_size), dtype=np.int64)
        return self._build_sample(indices, include_behavior_actions=True, include_next_behavior_actions=True)

    def _build_sample(
        self,
        indices: np.ndarray,
        *,
        include_behavior_actions: bool,
        include_next_behavior_actions: bool = False,
    ):
        if self._states is None or self._actions is None or self._next_states is None:
            raise ValueError("Replay buffer storage is not initialized.")
        if self._rewards is None or self._dones is None or self._behavior_actions is None:
            raise ValueError("Replay buffer storage is not initialized.")
        if include_next_behavior_actions and self._next_behavior_actions is None:
            raise ValueError("Replay buffer next behavior-action storage is not initialized.")

        states = [
            self._tensor_from_array(storage[indices])
            for storage in self._states
        ]
        actions = [
            self._tensor_from_array(storage[indices])
            for storage in self._actions
        ]
        rewards = [
            self._tensor_from_array(self._rewards[indices, agent_idx, :])
            for agent_idx in range(self.num_agents)
        ]
        next_states = [
            self._tensor_from_array(storage[indices])
            for storage in self._next_states
        ]
        behavior_actions = [
            self._tensor_from_array(storage[indices])
            for storage in self._behavior_actions
        ]
        next_behavior_actions = [
            self._tensor_from_array(storage[indices])
            for storage in (self._next_behavior_actions or self._behavior_actions)
        ]

        # Keep historical shape: [num_agents, batch_size, 1].
        done_tensor = self._tensor_from_array(self._dones[indices])
        done_tensor = done_tensor.unsqueeze(0).expand(self.num_agents, -1, -1)

        if include_behavior_actions and include_next_behavior_actions:
            return states, actions, rewards, next_states, done_tensor, behavior_actions, next_behavior_actions
        if include_behavior_actions:
            return states, actions, rewards, next_states, done_tensor, behavior_actions
        return states, actions, rewards, next_states, done_tensor

    @staticmethod
    def _tensor_from_array(array: np.ndarray) -> torch.Tensor:
        return _maybe_pin_memory(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)))

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
        if self._states is None:
            return {
                "format": "joint_transitions_compact_v1",
                "position": self.position,
                "size": self.size,
                "state_dims": self._state_dims,
                "action_dims": self._action_dims,
                "behavior_action_dims": self._behavior_action_dims,
                "next_behavior_action_dims": self._next_behavior_action_dims,
                "states": [],
                "actions": [],
                "behavior_actions": [],
                "next_behavior_actions": [],
                "next_states": [],
                "rewards": np.zeros((0, self.num_agents, 1), dtype=np.float32),
                "dones": np.zeros((0, 1), dtype=np.float32),
            }

        active = slice(0, self.size)
        return {
            "format": "joint_transitions_compact_v1",
            "position": self.position,
            "size": self.size,
            "state_dims": list(self._state_dims),
            "action_dims": list(self._action_dims),
            "behavior_action_dims": list(self._behavior_action_dims),
            "next_behavior_action_dims": list(self._next_behavior_action_dims),
            "states": [array[active].copy() for array in self._states],
            "actions": [array[active].copy() for array in self._actions],
            "behavior_actions": [array[active].copy() for array in self._behavior_actions],
            "next_behavior_actions": [array[active].copy() for array in self._next_behavior_actions],
            "next_states": [array[active].copy() for array in self._next_states],
            "rewards": self._rewards[active].copy(),
            "dones": self._dones[active].copy(),
        }

    def set_state(self, state):
        """Restore buffer contents from :meth:`get_state`."""
        if state is None:
            return
        self._reset_storage()

        if isinstance(state, dict) and state.get("format") == "joint_transitions_compact_v1":
            self._set_compact_state(state)
            return

        # Backward compatibility for older compact snapshots that may carry a
        # newer format name but the same arrays.
        if isinstance(state, dict) and {"states", "actions", "next_states", "rewards", "dones"} <= set(state):
            self._set_compact_state(state)
            return

        # Backward compatibility for older checkpoints where transitions were a
        # list of tensors.
        self.position = 0

        if isinstance(state, dict) and "buffer" in state:
            experiences = list(state["buffer"])[-self.capacity :]
            for experience in experiences:
                self._push_legacy_transition(experience)
            self.position = int(state.get("position", len(self) % self.capacity)) % self.capacity
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
                self.push(states, actions, rewards, next_states, done, behavior_actions=actions)
            return

        if isinstance(state, list):
            for experience in state[-self.capacity :]:
                self._push_legacy_transition(experience)
            return

        raise ValueError("Unsupported replay buffer state format.")

    def _reset_storage(self) -> None:
        self.position = 0
        self.size = 0
        self._last_insert_index = -1
        self._state_dims = []
        self._action_dims = []
        self._behavior_action_dims = []
        self._next_behavior_action_dims = []
        self._states = None
        self._actions = None
        self._behavior_actions = None
        self._next_behavior_actions = None
        self._next_states = None
        self._rewards = None
        self._dones = None

    def _set_compact_state(self, state: dict) -> None:
        states = [np.asarray(array, dtype=np.float32) for array in state.get("states", [])]
        actions = [np.asarray(array, dtype=np.float32) for array in state.get("actions", [])]
        behavior_actions = [
            np.asarray(array, dtype=np.float32)
            for array in state.get("behavior_actions", actions)
        ]
        next_behavior_actions = [
            np.asarray(array, dtype=np.float32)
            for array in state.get("next_behavior_actions", behavior_actions)
        ]
        next_states = [np.asarray(array, dtype=np.float32) for array in state.get("next_states", [])]
        if not states:
            self._reset_storage()
            return

        size = int(state.get("size", min(array.shape[0] for array in states)))
        size = max(0, min(size, self.capacity))
        self._state_dims = [int(array.shape[1]) for array in states]
        self._action_dims = [int(array.shape[1]) for array in actions]
        self._behavior_action_dims = [int(array.shape[1]) for array in behavior_actions]
        self._next_behavior_action_dims = [int(array.shape[1]) for array in next_behavior_actions]
        self._states = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._state_dims
        ]
        self._actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._action_dims
        ]
        self._behavior_actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._behavior_action_dims
        ]
        self._next_behavior_actions = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._next_behavior_action_dims
        ]
        self._next_states = [
            np.zeros((self.capacity, dim), dtype=np.float32)
            for dim in self._state_dims
        ]
        self._rewards = np.zeros((self.capacity, self.num_agents, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)

        for agent_idx in range(self.num_agents):
            self._states[agent_idx][:size] = states[agent_idx][:size]
            self._actions[agent_idx][:size] = actions[agent_idx][:size]
            self._behavior_actions[agent_idx][:size] = behavior_actions[agent_idx][:size]
            self._next_behavior_actions[agent_idx][:size] = next_behavior_actions[agent_idx][:size]
            self._next_states[agent_idx][:size] = next_states[agent_idx][:size]
        rewards = np.asarray(state.get("rewards", []), dtype=np.float32)
        dones = np.asarray(state.get("dones", []), dtype=np.float32)
        if rewards.size:
            self._rewards[:size] = rewards[:size].reshape(size, self.num_agents, 1)
        if dones.size:
            self._dones[:size] = dones[:size].reshape(size, 1)
        self.size = size
        self.position = int(state.get("position", size % self.capacity)) % self.capacity

    def _push_legacy_transition(self, experience) -> None:
        states, actions, rewards, next_states, done, behavior_actions = self._unpack_transition(experience)
        scalar_rewards = [
            self._safe_float32(reward)
            for reward in rewards
        ]
        done_value = bool(float(np.asarray(done, dtype=np.float32).reshape(-1)[0]) > 0.5)
        self.push(
            states,
            actions,
            scalar_rewards,
            next_states,
            done_value,
            behavior_actions=behavior_actions,
        )

    def __len__(self):
        """Get the current replay size."""
        return int(self.size)


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
        self._priorities = np.zeros((self.capacity,), dtype=np.float32)

    @property
    def priorities(self) -> np.ndarray:
        return self._priorities[: len(self)]

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
        next_behavior_actions=None,
        priority_boost=None,
    ):
        super().push(
            states,
            actions,
            rewards,
            next_states,
            done,
            behavior_actions=behavior_actions,
            next_behavior_actions=next_behavior_actions,
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
        priority += self.priority_epsilon
        insert_index = getattr(self, "_last_insert_index", len(self.priorities))
        self._priorities[insert_index] = np.float32(priority)

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

        batch_indices = np.asarray(self._sample_indices(), dtype=np.int64)
        return self._build_sample(batch_indices, include_behavior_actions=False)

    def sample_with_behavior_actions(self):
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch_indices = np.asarray(self._sample_indices(), dtype=np.int64)
        return self._build_sample(batch_indices, include_behavior_actions=True)

    def sample_with_policy_context_actions(self):
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch_indices = np.asarray(self._sample_indices(), dtype=np.int64)
        return self._build_sample(
            batch_indices,
            include_behavior_actions=True,
            include_next_behavior_actions=True,
        )

    def _sample_indices(self) -> list[int]:
        replay_size = len(self)
        priority_count = int(round(self.batch_size * self.priority_fraction))
        uniform_count = self.batch_size - priority_count

        indices: list[int] = []
        if uniform_count > 0:
            indices.extend(np.random.randint(0, replay_size, size=uniform_count).tolist())

        if priority_count > 0:
            priorities = self._priorities[:replay_size].astype(np.float64, copy=False)
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
                "priorities": self._priorities[: len(self)].copy(),
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
        self._priorities = np.zeros((self.capacity,), dtype=np.float32)
        if isinstance(state, dict) and "priorities" in state:
            values = np.asarray(state.get("priorities"), dtype=np.float32).reshape(-1)[-self.capacity :]
            count = min(values.shape[0], len(self))
            if count > 0:
                self._priorities[:count] = values[:count]
        if len(self) > 0:
            active_priorities = self._priorities[: len(self)]
            active_priorities[active_priorities <= 0.0] = np.float32(self.priority_epsilon)


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
