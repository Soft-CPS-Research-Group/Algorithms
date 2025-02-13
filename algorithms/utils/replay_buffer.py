import random
import numpy as np
import torch
from collections import deque

class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, batch_size, device="cpu"):
        """
        Multi-Agent Replay Buffer that stores experiences as PyTorch tensors.

        Args:
            capacity (int): Maximum number of experiences per agent.
            num_agents (int): Number of agents.
            batch_size (int): Number of experiences to sample per agent.
            device (str): Device to store tensors on ('cpu' or 'cuda').
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.buffers = [deque(maxlen=capacity) for _ in range(num_agents)]

    def push(self, states, actions, rewards, next_states, terminated):
        """
        Store experiences for each agent as PyTorch tensors.

        Args:
            states (list): List of states, one for each agent.
            actions (list): List of actions, one for each agent.
            rewards (list): List of rewards, one for each agent.
            next_states (list): List of next states, one for each agent.
            terminated (bool or list): Done flag(s) (shared across agents).
        """
        for agent_idx in range(self.num_agents):
            # Convert to tensors before storing
            state_tensor = torch.tensor(states[agent_idx], dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(actions[agent_idx], dtype=torch.float32, device=self.device)
            reward_tensor = torch.tensor(rewards[agent_idx], dtype=torch.float32, device=self.device).unsqueeze(0)
            next_state_tensor = torch.tensor(next_states[agent_idx], dtype=torch.float32, device=self.device)
            terminated_tensor = torch.tensor(terminated, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.buffers[agent_idx].append((state_tensor, action_tensor, reward_tensor, next_state_tensor, terminated_tensor))

    def sample(self):
        """
        Sample a batch of experiences for each agent.

        Returns:
            Tuple of (states, actions, rewards, next_states, terminated) as tensors for all agents.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        states, actions, rewards, next_states, terminated = [], [], [], [], []
        for agent_idx in range(self.num_agents):
            batch = random.sample(self.buffers[agent_idx], self.batch_size)
            states_agent, actions_agent, rewards_agent, next_states_agent, terminated_agent = zip(*batch)

            states.append(torch.stack(states_agent).to(self.device))
            actions.append(torch.stack(actions_agent).to(self.device))
            rewards.append(torch.stack(rewards_agent).to(self.device))
            next_states.append(torch.stack(next_states_agent).to(self.device))
            terminated.append(torch.stack(terminated_agent).to(self.device))

        return states, actions, rewards, next_states, terminated

    def __len__(self):
        """Get the current size of the smallest buffer."""
        return min(len(buffer) for buffer in self.buffers)


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
