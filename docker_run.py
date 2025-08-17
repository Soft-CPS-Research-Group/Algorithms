"""Entry point script executed inside the Docker container.

The script configures directories and logging, trains a selected
reinforcement-learning agent on CityLearn and records evaluation metrics
with MLflow. It is intended to be invoked by the Docker image's entry
point.
"""

import argparse
import json
import os
from importlib import import_module
from typing import Dict, Tuple, Type

import mlflow
import yaml
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from loguru import logger

from reward_function import V2G_Reward
from utils.helpers import set_default_config
from utils.mlflow_helper import end_mlflow_run, start_mlflow_run
from utils.wrapper_citylearn import Wrapper_CityLearn as Wrapper


# Map configuration keys to reward function classes
REWARD_FUNCTIONS: Dict[str, Type[RewardFunction]] = {
    "V2GPenaltyReward": V2G_Reward,
    "RewardFunction": RewardFunction,
}

# Map algorithm names to their import paths; extend as more agents are added
ALGORITHMS: Dict[str, str] = {
    "MADDPG": "algorithms.agents.maddpg_agent.MADDPG",
    "RBC": "algorithms.agents.rbc_agent.RBCAgent",
    "GNN": "algorithms.agents.gnn_agent.GNNAgent",
    "Transformer": "algorithms.agents.transformer_agent.TransformerAgent",
}

BASE_DIR = "/data"


def load_agent_class(name: str):
    """Dynamically import an agent class based on its registered name."""

    try:
        module_path, class_name = ALGORITHMS[name].rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown algorithm: {name}") from exc


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config file (mounted from shared volume)",
    )
    parser.add_argument(
        "--job_id",
        required=True,
        help="Job ID used to structure output folders",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override log level from config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a previous MLflow run",
    )
    parser.add_argument(
        "--checkpoint-run-id",
        help="MLflow run ID from which to load a checkpoint",
    )
    return parser.parse_args()


def setup_directories(job_id: str) -> Tuple[str, str, str, str, str]:
    """Create required directories and return their paths."""

    # 1) Build base paths
    logger.debug(f"Creating directory structure for job_id={job_id}")
    job_dir = os.path.join(BASE_DIR, "jobs", job_id)
    log_dir = os.path.join(job_dir, "logs")
    result_path = os.path.join(job_dir, "results", "result.json")
    progress_path = os.path.join(job_dir, "progress", "progress.json")
    mlflow_uri = os.path.join(BASE_DIR, "mlflow", "mlruns")

    # 2) Create directories if they do not exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    os.makedirs(mlflow_uri, exist_ok=True)

    # 3) Return constructed paths
    return job_dir, log_dir, result_path, progress_path, mlflow_uri


def load_config(config_file_path: str, log_dir: str, mlflow_uri: str) -> dict:
    """Load the YAML configuration and inject runtime paths."""

    # 1) Parse the YAML configuration
    logger.debug(f"Loading configuration file from {config_file_path}")
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # 2) Inject runtime paths for logging
    config["experiment"]["logging"]["log_dir"] = log_dir
    config["experiment"]["logging"]["mlflow_uri"] = f"file:{mlflow_uri}"
    return config


def initialise_mlflow(config: dict, job_dir: str, mlflow_uri: str) -> Tuple[str, str]:
    """Start an MLflow run and persist run information for later reference."""

    # 1) Start a new MLflow run
    logger.debug("Starting MLflow run")
    start_mlflow_run(config=config)
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("MLflow run could not be started.")

    run_id = run.info.run_id
    run_name = run.info.run_name

    job_info_path = os.path.join(job_dir, "job_info.json")
    job_info = {}
    if os.path.exists(job_info_path):
        with open(job_info_path, "r") as f:
            job_info = json.load(f)
    job_info["mlflow_run_id"] = run_id
    job_info["mlflow_uri"] = f"file:{mlflow_uri}"

    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)
    logger.info(f"MLflow run ID saved to job_info.json: {run_id}")
    return run_id, run_name


def configure_logging(config: dict, log_dir: str, run_id: str) -> None:
    """Configure loguru to write logs to a run specific file."""
    # 1) Resolve logging level and file path
    level = config["experiment"]["logging"].get("log_level", "INFO").upper()
    log_file = os.path.join(log_dir, f"{run_id}.log")

    # 2) Configure handlers: file (rotating) and stdout
    logger.remove()  # ensure no duplicate handlers
    logger.add(log_file, level=level, rotation="100 MB")
    logger.add(lambda msg: print(msg, end=""), level=level)

    # 3) Report configuration
    logger.debug(f"Logging configured with level={level} at {log_file}")


def initialise_environment(config: dict) -> CityLearnEnv:
    """Construct the CityLearn environment from the configuration."""
    # 1) Resolve reward function class
    reward_name = config["simulator"]["reward_function"]
    reward_cls = REWARD_FUNCTIONS[reward_name]
    logger.info(f"Initialising CityLearn environment with reward {reward_name}")

    # 2) Instantiate the environment
    return CityLearnEnv(
        schema=config["simulator"]["dataset_path"],
        central_agent=config["simulator"]["central_agent"],
        reward_function=reward_cls,
    )


def prepare_agent_and_wrapper(
    env: CityLearnEnv, config: dict, job_id: str, progress_path: str
) -> Wrapper:
    """Initialise the agent specified in the config and attach to wrapper."""

    # 1) Create the experiment wrapper
    wrapper = Wrapper(env=env, config=config, job_id=job_id, progress_path=progress_path)

    # 2) Populate config with environment-dependent defaults
    set_default_config(
        config,
        ["algorithm", "hyperparameters", "observation_dimensions"],
        wrapper.observation_dimension,
    )
    set_default_config(
        config,
        ["algorithm", "hyperparameters", "action_dimensions"],
        wrapper.action_dimension,
    )
    set_default_config(
        config,
        ["algorithm", "hyperparameters", "action_space"],
        wrapper.action_space,
    )
    set_default_config(
        config,
        ["algorithm", "hyperparameters", "num_agents"],
        len(wrapper.action_space),
    )

    # 3) Instantiate and attach the agent
    agent_name = config["algorithm"].get("class", "MADDPG")
    agent_cls = load_agent_class(agent_name)
    logger.info(f"Initialising agent class {agent_name}")
    agent = agent_cls(config=config)
    wrapper.set_model(agent)
    return wrapper


def evaluate_and_save(wrapper: Wrapper, result_path: str) -> None:
    """Evaluate KPIs, log them to MLflow and persist results to disk."""
    # 1) Evaluate KPIs from the environment
    kpis = wrapper.env.evaluate()
    kpis = (
        kpis.pivot(index="cost_function", columns="name", values="value")
        .round(3)
        .dropna(how="all")
    )
    logger.info(f"KPI Evaluation:\n{kpis}")

    # 2) Log KPIs to MLflow
    for kpi_name, kpi_value in kpis.items():
        mlflow.log_metric(f"kpi_{kpi_name}", kpi_value)

    # 3) Persist KPIs to disk
    with open(result_path, "w") as f:
        json.dump(kpis.to_dict(), f, indent=2)


def main() -> None:
    # 1) Parse CLI arguments
    args = parse_args()

    # 2) Prepare directory structure
    job_dir, log_dir, result_path, progress_path, mlflow_uri = setup_directories(args.job_id)

    # 3) Load configuration file
    config = load_config(args.config, log_dir, mlflow_uri)

    # 4) Override config values from CLI
    if args.log_level:
        config["experiment"]["logging"]["log_level"] = args.log_level
    if args.resume:
        config["experiment"]["resume_training"] = True
    if args.checkpoint_run_id:
        config["experiment"]["checkpoint_run_id"] = args.checkpoint_run_id

    # 5) Start MLflow tracking and persist run details
    run_id, run_name = initialise_mlflow(config, job_dir, mlflow_uri)

    # 6) Configure logging
    configure_logging(config, log_dir, run_id)
    logger.info(f"Started run {run_name} with ID {run_id} for job {args.job_id}")

    # 7) Build the environment
    env = initialise_environment(config)

    # 8) Prepare the agent and wrapper
    wrapper = prepare_agent_and_wrapper(env, config, args.job_id, progress_path)

    # 9) Begin learning
    wrapper.learn()

    # 10) Evaluate results and persist to disk
    evaluate_and_save(wrapper, result_path)

    # 11) Close MLflow run
    end_mlflow_run()


if __name__ == "__main__":
    main()

