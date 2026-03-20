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

    def push(self, states, actions, rewards, next_states, done):
        """
        Store experiences for each agent in CPU memory.

        Args:
            states (list): List of states per agent.
            actions (list): List of actions per agent.
            rewards (list): List of rewards per agent.
            next_states (list): List of next states per agent.
            done (bool): Single done flag shared across all agents.
        """
        state_tensors = [
            _maybe_pin_memory(torch.tensor(states[agent_idx], dtype=torch.float32))
            for agent_idx in range(self.num_agents)
        ]
        action_tensors = [
            _maybe_pin_memory(torch.tensor(actions[agent_idx], dtype=torch.float32))
            for agent_idx in range(self.num_agents)
        ]
        reward_tensors = [
            _maybe_pin_memory(torch.tensor(rewards[agent_idx], dtype=torch.float32).unsqueeze(0))
            for agent_idx in range(self.num_agents)
        ]
        next_state_tensors = [
            _maybe_pin_memory(torch.tensor(next_states[agent_idx], dtype=torch.float32))
            for agent_idx in range(self.num_agents)
        ]
        done_tensor = _maybe_pin_memory(torch.tensor(float(done), dtype=torch.float32).unsqueeze(0))

        self.buffer.append((state_tensors, action_tensors, reward_tensors, next_state_tensors, done_tensor))

    def sample(self):
        """
        Sample a batch of experiences for each agent.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated) as CPU tensors.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        batch = random.sample(self.buffer, self.batch_size)
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)

        states = [
            torch.stack([transition[agent_idx] for transition in states_batch])
            for agent_idx in range(self.num_agents)
        ]
        actions = [
            torch.stack([transition[agent_idx] for transition in actions_batch])
            for agent_idx in range(self.num_agents)
        ]
        rewards = [
            torch.stack([transition[agent_idx] for transition in rewards_batch])
            for agent_idx in range(self.num_agents)
        ]
        next_states = [
            torch.stack([transition[agent_idx] for transition in next_states_batch])
            for agent_idx in range(self.num_agents)
        ]

        # Keep historical shape: [num_agents, batch_size, 1].
        done_tensor = torch.stack(dones_batch)
        done_tensor = done_tensor.unsqueeze(0).expand(self.num_agents, -1, -1)

        return states, actions, rewards, next_states, done_tensor

    def get_state(self):
        """Return a serialisable snapshot of the replay buffer."""
        return {
            "format": "joint_transitions_v2",
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
                self.buffer.append((states, actions, rewards, next_states, done))
            return

        if isinstance(state, list):
            for experience in state:
                self.buffer.append(experience)
            return

        raise ValueError("Unsupported replay buffer state format.")

    def __len__(self):
        """Get the current replay size."""
        return len(self.buffer)


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
