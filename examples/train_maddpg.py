from marl.agents.maddpg_agent import MADDPGAgent
from marl.training.trainer import MADDPGTrainer
from marl.utils.replay_buffer import ReplayBuffer

# Initialize agents, environment, and replay buffer
agents = {
    "agent_1": MADDPGAgent(actor_model_1, critic_model_1),
    "agent_2": MADDPGAgent(actor_model_2, critic_model_2),
}
environment = CustomEnvironment()
replay_buffer = ReplayBuffer(buffer_size=100000)

trainer = MADDPGTrainer(agents, environment, replay_buffer)
trainer.train(num_episodes=500)


maddpg = MADDPG(config_path='config.yaml')