import os
import yaml
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.maddpg_agent import MADDPG
from utils.wrapper_citylearn import Wrapper_CityLearn as Wrapper
from citylearn.reward_function import RewardFunction
from reward_function import V2G_Reward
from utils.mlflow_helper import start_mlflow_run, end_mlflow_run
from utils.helpers import set_default_config
from loguru import logger
import mlflow
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

# 1.4 Setup MLFlow and retrieve run ID
start_mlflow_run(config=config)
active_run = mlflow.active_run()
if active_run is None:
    logger.error("1.4: No active MLflow run found.")
    raise RuntimeError("1.4: No active MLflow run found.")
run_id = active_run.info.run_id
run_name = active_run.info.run_name

# 1.5 Dynamically update log_dir to include run_id
log_dir = os.path.join(base_experiment_dir, "logs", run_name)  # Logs directory with run ID
os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
config["experiment"]["logging"]["log_dir"] = log_dir

# 1.6 Setup Loguru
log_file = os.path.join(log_dir, f"{run_id}.log")
logger.add(
    log_file,
    level=config["experiment"]["logging"].get("log_level", "INFO").upper(),
    format="{time} - {level} - {message}",
    rotation="1 week",  # Rotate logs weekly
    retention="1 month",  # Retain logs for a month
    compression="zip"  # Compress old logs
)
logger.info(f"1.6: Logger initialized for experiment: {experiment_name} with MLflow run ID: {run_id}")

# 2. Configure Experiment
# 2.1 Initialize CityLearn environment with parameters from config
logger.info(f"2.1: Initializing CityLearn environment with dataset: {config['simulator']['dataset_path']}")
env = CityLearnEnv(
    schema=config["simulator"]["dataset_path"],
    central_agent=config["simulator"]["central_agent"],
    reward_function=reward_function_map[config["simulator"]["reward_function"]]
)

# 2.2 Inititalize the wrapper with the environment
logger.info("Wrapping environment into the experiment wrapper.")
wrapper = Wrapper(env=env, config=config)

# 2.3 Get relevant information from the env to configure the algorithm
logger.debug("Setting/Getting necessary config parameters for the algorithm from the Env.")
set_default_config(config, ["algorithm","hyperparameters","observation_dimensions"], wrapper.observation_dimension)
set_default_config(config, ["algorithm","hyperparameters","action_dimensions"], wrapper.action_dimension)
set_default_config(config, ["algorithm","hyperparameters","action_space"], wrapper.action_space)
set_default_config(config, ["algorithm","hyperparameters","num_agents"], len(wrapper.action_space))
logger.debug(f'Observation Dimension: {config["algorithm"]["hyperparameters"]["observation_dimensions"]}')
logger.debug(f'Action Dimension: {config["algorithm"]["hyperparameters"]["action_dimensions"]}')
logger.debug(f'Action Space: {config["algorithm"]["hyperparameters"]["action_space"]}')
logger.debug(f'Num Agents: {config["algorithm"]["hyperparameters"]["num_agents"]}')

# 2.4 Initialize the algorithm (e.x.: MADDPG) agent
logger.info(f"Initializing MADDPG agent for experiment: {experiment_name}")
agent = MADDPG(config=config)

# 2.5 Wrap the algorithm with the environment
logger.info("Wrapping agent into the experiment wrapper.")
wrapper.set_model(agent)

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
