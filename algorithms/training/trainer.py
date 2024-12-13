import torch

class MADDPGTrainer:
    def __init__(self, agents, environment, replay_buffer, batch_size=64):
        self.agents = agents
        self.environment = environment
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            states = self.environment.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = {agent_id: agent.act(states[agent_id]) for agent_id, agent in self.agents.items()}
                next_states, rewards, done, _ = self.environment.step(actions)

                # Add experience to the replay buffer
                self.replay_buffer.add(states, actions, rewards, next_states, done)

                # Update agents
                if len(self.replay_buffer) >= self.batch_size:
                    for agent_id, agent in self.agents.items():
                        agent.update(self.replay_buffer, self.batch_size)

                states = next_states
                episode_reward += sum(rewards.values())

            print(f"Episode {episode + 1}: Reward {episode_reward}")