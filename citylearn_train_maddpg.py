import os
import yaml
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.maddpg_agent import MADDPG
from utils.wrapper_citylearn import Wrapper_CityLearn as Wrapper
from citylearn.reward_function import RewardFunction
from reward_function import V2G_Reward
from utils.mlflow_helper import start_mlflow_run, end_mlflow_run
from loguru import logger
from typing import List

# 1. Setup
config_file_path = "./configs/config.yaml"


# 1.1. Reward Function Map
reward_function_map = {
    "V2GPenaltyReward": V2G_Reward,
    "RewardFunction": RewardFunction
}

# 1.2 Load config file
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)

# 1.3 Get experiment-specific paths
experiment_name = config["experiment"]["name"]
base_experiment_dir = os.path.join("./experiments", experiment_name)  # Root for this experiment
log_dir = os.path.join(base_experiment_dir, "logs")  # Logs directory
mlflow_uri = os.path.join(base_experiment_dir, "mlruns")  # MLflow URI

# Ensure all directories exist
os.makedirs(log_dir, exist_ok=True)

# 1.4 Update config paths dynamically
config["experiment"]["logging"]["log_dir"] = log_dir

# 1.5 Setup Loguru
log_file = os.path.join(log_dir, f"{experiment_name}.log")
logger.add(
    log_file,
    level=config["experiment"]["logging"].get("log_level", "INFO").upper(),
    format="{time} - {level} - {message}",
    rotation="1 week",  # Rotate logs weekly
    retention="1 month",  # Retain logs for a month
    compression="zip"  # Compress old logs
)
logger.info(f"Logger initialized for experiment: {experiment_name}")

# 1.6 Setup MLFlow
start_mlflow_run(config=config)

# 2. Configure Experiment
# 2.1 Initialize CityLearn environment with parameters from config
logger.info(f"Initializing CityLearn environment with dataset: {config['simulator']['dataset_path']}")
env = CityLearnEnv(
    schema=config["simulator"]["dataset_path"],
    central_agent=config["simulator"]["central_agent"],
    reward_function=reward_function_map[config["simulator"]["reward_function"]]
)

# 2.2 Initialize the algorithm (e.x.: MADDPG) agent
logger.info(f"Initializing MADDPG agent for experiment: {experiment_name}")
agent = MADDPG(config=config)

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

# 5. Cleaning up
end_mlflow_run()
