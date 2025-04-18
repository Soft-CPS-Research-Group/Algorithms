# citylearn_train.py

import os
import yaml
import argparse
import json
from citylearn.citylearn import CityLearnEnv
from algorithms.agents.maddpg_agent import MADDPG
from utils.wrapper_citylearn import Wrapper_CityLearn as Wrapper
from citylearn.reward_function import RewardFunction
from reward_function import V2G_Reward
from utils.mlflow_helper import start_mlflow_run, end_mlflow_run
from utils.helpers import set_default_config
from loguru import logger
import mlflow

# -------------------------------
# 1. Parse CLI Arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to the config file (mounted from shared volume)')
parser.add_argument('--job_id', required=True, help='Job ID to structure output folders')
args = parser.parse_args()

config_file_path = args.config
print(config_file_path)
job_id = args.job_id
print(job_id)

# -------------------------------
# 2. Setup Directories
# -------------------------------
BASE_DIR = "/data"
job_dir = os.path.join(BASE_DIR, "jobs", job_id)
log_dir = os.path.join(job_dir, "logs")
result_path = os.path.join(job_dir, "results", "result.json")
progress_path = os.path.join(job_dir, "progress", "progress.json")
mlflow_uri = os.path.join(BASE_DIR, "mlflow", "mlruns")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(result_path), exist_ok=True)
os.makedirs(os.path.dirname(progress_path), exist_ok=True)
os.makedirs(mlflow_uri, exist_ok=True)

print("DIRS OK")
print(log_dir)
print(result_path)
print(progress_path)
print(mlflow_uri)


# -------------------------------
# 3. Load Config
# -------------------------------
with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

# Inject logging paths into config
config["experiment"]["logging"]["log_dir"] = log_dir
config["experiment"]["logging"]["mlflow_uri"] = f"file:{mlflow_uri}"

# -------------------------------
# 4. Set up MLflow & Logging
# -------------------------------
print("Starting MLFlow")
start_mlflow_run(config=config)
run = mlflow.active_run()
if run is None:
    raise RuntimeError("MLflow run could not be started.")
run_id = run.info.run_id
run_name = run.info.run_name

# -------------------------------
# 4.1 Update job_info.json with MLflow run ID
# -------------------------------
job_info_path = os.path.join(job_dir, "job_info.json")

try:
    if os.path.exists(job_info_path):
        with open(job_info_path, "r") as f:
            job_info = json.load(f)
    else:
        job_info = {}

    job_info["mlflow_run_id"] = run_id
    job_info["mlflow_uri"] = f"file:{mlflow_uri}"  # optional, nice to track

    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)

    logger.info(f"MLflow run ID saved to job_info.json: {run_id}")

except Exception as e:
    logger.warning(f"Could not update job_info.json with MLflow run ID: {e}")


print("Starting logs")
log_file = os.path.join(log_dir, f"{run_id}.log")
level = config["experiment"]["logging"].get("log_level", "INFO").upper()
logger.add(log_file, level=level)

logger.info(f"Started run {run_name} with ID {run_id} for job {job_id}")

# -------------------------------
# 5. Initialize Environment
# -------------------------------
print("Initializing ENV")
reward_function_map = {
    "V2GPenaltyReward": V2G_Reward,
    "RewardFunction": RewardFunction
}

env = CityLearnEnv(
    schema=config["simulator"]["dataset_path"],
    central_agent=config["simulator"]["central_agent"],
    reward_function=reward_function_map[config["simulator"]["reward_function"]]
)

# -------------------------------
# 6. Initialize Agent + Wrapper
# -------------------------------
wrapper = Wrapper(env=env, config=config, job_id=job_id, progress_path=progress_path)
set_default_config(config, ["algorithm","hyperparameters","observation_dimensions"], wrapper.observation_dimension)
set_default_config(config, ["algorithm","hyperparameters","action_dimensions"], wrapper.action_dimension)
set_default_config(config, ["algorithm","hyperparameters","action_space"], wrapper.action_space)
set_default_config(config, ["algorithm","hyperparameters","num_agents"], len(wrapper.action_space))

agent = MADDPG(config=config)
wrapper.set_model(agent)

# -------------------------------
# 7. Run Training
# -------------------------------
wrapper.learn()

# -------------------------------
# 8. Evaluate & Save Results
# -------------------------------
kpis = wrapper.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3).dropna(how='all')
logger.info(f"KPI Evaluation:\n{kpis}")

for kpi_name, kpi_value in kpis.items():
    mlflow.log_metric(f"kpi_{kpi_name}", kpi_value)

with open(result_path, "w") as f:
    json.dump(kpis.to_dict(), f, indent=2)

# -------------------------------
# 9. End MLflow Run
# -------------------------------
end_mlflow_run()
