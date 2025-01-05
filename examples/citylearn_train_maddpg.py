import yaml
from citylearn import CityLearnEnv
from algorithms.agents.maddpg_agent import MADDPG
from wrapper_citylearn import Wrapper
from citylearn.reward_function import V2GPenaltyReward

# Load config file
with open("citylearn_config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Map reward functions
reward_function_map = {
    "V2GPenaltyReward": V2GPenaltyReward,
    # Add more reward functions here as needed
}

# Get parameters from config
dataset_name = config["dataset_name"]
central_agent = config["central_agent"]
reward_function_name = config["reward_function"]
reward_function = reward_function_map[reward_function_name]  # Map reward function

# Initialize CityLearn environment with parameters from config
env = CityLearnEnv(dataset_name=dataset_name, central_agent=central_agent, reward_function=reward_function)

# Initialize MADDPG agent
agent = MADDPG(env, config_path="maddpg_config.yaml")

# Wrap the agent
wrapper = Wrapper(env, agent)

# Run learning
wrapper.learn()
