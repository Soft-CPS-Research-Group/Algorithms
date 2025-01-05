import random
import numpy as np
from collections import deque

class MultiAgentReplayBuffer:
    def __init__(self, capacity, num_agents, batch_size):
        """
        Multi-Agent Replay Buffer

        Args:
            capacity (int): Maximum number of experiences per agent.
            num_agents (int): Number of agents.
            batch_size (int): Number of experiences to sample per agent.
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.buffers = [deque(maxlen=capacity) for _ in range(num_agents)]

    def push(self, states, actions, rewards, next_states, dones):
        """
        Store experiences for each agent.

        Args:
            states (list): List of states, one for each agent.
            actions (list): List of actions, one for each agent.
            rewards (list): List of rewards, one for each agent.
            next_states (list): List of next states, one for each agent.
            dones (bool): Done flag (shared across agents).
        """
        for agent_idx in range(self.num_agents):
            self.buffers[agent_idx].append(
                (states[agent_idx], actions[agent_idx], rewards[agent_idx], next_states[agent_idx], dones)
            )

    def sample(self):
        """
        Sample a batch of experiences for each agent.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) for all agents.
        """
        if len(self) < self.batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for agent_idx in range(self.num_agents):
            batch = random.sample(self.buffers[agent_idx], self.batch_size)
            states_agent, actions_agent, rewards_agent, next_states_agent, dones_agent = zip(*batch)
            states.append(np.stack(states_agent))
            actions.append(np.stack(actions_agent))
            rewards.append(np.stack(rewards_agent))
            next_states.append(np.stack(next_states_agent))
            dones.append(np.stack(dones_agent))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Get the current size of the smallest buffer.
        Ensures sampling fairness for all agents.
        """
        return min(len(buffer) for buffer in self.buffers)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
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
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
