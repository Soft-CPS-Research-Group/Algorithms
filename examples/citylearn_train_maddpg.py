import os
import yaml
from loguru import logger
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.maddpg_agent import MADDPG
from wrapper_citylearn import Wrapper_CityLearn as Wrapper
from citylearn.reward_function import V2GPenaltyReward, RewardFunction
from utils.mlflow_helper import start_mlflow_run, end_mlflow_run

# 1. Setup
config_file_path = "../configs/config.yaml"

# 1.1. Reward Function Map
reward_function_map = {
    "V2GPenaltyReward": V2GPenaltyReward,
    "RewardFunction": RewardFunction
    # Add more reward functions here as needed
}

# 1.2 Load config file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)

# 1.3 Get parameters from config
experiment_name = config["experiment"]["name"]
log_dir = config["experiment"]["logging"].get("log_dir", "./logs")  # Directory path for logs
dataset_name = config["dataset_name"]
dataset_path = config["dataset_path"]
central_agent = config["central_agent"]
reward_function = reward_function_map[config["reward_function"]]

# 1.4 Setup Loguru
os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
log_file = os.path.join(log_dir, f"{experiment_name}.log")  # Log file matches experiment name
logger.add(
    log_file,
    level=config["experiment"]["logging"].get("log_level", "INFO").upper(),
    format="{time} - {level} - {message}",
    rotation="1 week",  # Rotate logs weekly
    retention="1 month",  # Retain logs for a month
    compression="zip"  # Compress old logs
)
logger.info(f"Logger initialized for experiment: {experiment_name}")

# 1.5 Setup MLFlow
start_mlflow_run(config=config, logger=logger)


# 2. Configure Experiment
# 2.1. Initialize CityLearn environment with parameters from config
logger.info(f"Initializing CityLearn environment with dataset: {dataset_path}")
env = CityLearnEnv(schema=dataset_path, central_agent=central_agent, reward_function=reward_function)

# 2.2. Initialize the algorithm (e.x.: MADDPG) agent
logger.info(f"Initializing MADDPG agent with config: {config_file_path}")
agent = MADDPG(env, config_path=config_file_path)

# 2.3 Wrap the algorithm, the environment, and high-level logging/MLFlow logic
logger.info("Wrapping environment and agent into the experiment wrapper.")
wrapper = Wrapper(env, agent)


# 3. Run learning
logger.info("Starting learning process.")
wrapper.learn()


# 4. Test (KPIs)
logger.info("Evaluating environment KPIs.")
kpis = wrapper.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
logger.info(f"KPI Evaluation complete: {kpis}")
print(kpis)


# 5. Cleaning up 
end_mlflow_run()
