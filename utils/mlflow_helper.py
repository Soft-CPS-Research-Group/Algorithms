import mlflow
import json
import tempfile
from loguru import logger
from utils.helpers import flatten_dict  

def log_to_mlflow(metric_name, value, step=None):
    """
    Logs a metric to MLflow if mlflow_enabled is True and an active MLflow run exists.

    Parameters:
    - metric_name: Name of the metric to log.
    - value: Value of the metric.
    - step: Step associated with the metric (optional).
    """
    if mlflow.active_run():
        mlflow.log_metric(metric_name, value, step)

def log_param_to_mlflow(param_name, value):
    if mlflow.active_run():
        mlflow.log_param(param_name, value)

def log_artifact_to_mlflow(file_path, artifact_path=None):
    """
    Logs a file or directory as an artifact to the active MLflow run. 
    Artifacts can include any files (e.g., model weights, configurations, logs, plots) that you want to store in MLflow for later use.

    Parameters:
    - file_path: Path to the file or directory to log.
    - artifact_path: Optional path within the artifact storage.
    """
    if mlflow.active_run():
        mlflow.log_artifact(file_path, artifact_path)

def log_dict_to_mlflow(data, artifact_path="config.json"):
    """
    Logs a dictionary as a JSON artifact in MLflow.
    Sometimes you might want to log a dictionary (e.g., model configuration, results) as a JSON artifact.

    Parameters:
    - data: Dictionary to log.
    - artifact_path: Path to save the JSON file within MLflow.
    """
    if mlflow.active_run():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            json.dump(data, temp_file)
            temp_file.flush()  # Ensure data is written
            mlflow.log_artifact(temp_file.name, artifact_path)

def log_model_to_mlflow(model, model_dir="models", model_name=None):
    """
    Logs a trained model as an artifact in MLflow.
    If you’re saving models (e.g., PyTorch, TensorFlow, or custom), this function can help.
    For frameworks like PyTorch or TensorFlow, you can directly use mlflow.pytorch.log_model() or mlflow.tensorflow.log_model().

    Parameters:
    - model: The model object to save.
    - model_dir: Directory to save the model before logging.
    - model_name: Optional name for the saved model file.
    """
    import os
    import tempfile

    if mlflow.active_run():
        # Save model to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, model_name or "model.pth")
            model.save(save_path)  # Replace with model-specific save logic
            mlflow.log_artifact(save_path, artifact_path=model_dir)

def log_text_to_mlflow(text, file_name="info.txt"):
    """
    Logs a string as a text artifact in MLflow.
    This can be useful for logging textual artifacts, such as evaluation summaries, debug logs, or experiment descriptions.

    Parameters:
    - text: The string content to log.
    - file_name: The name of the text file to save in MLflow.
    """
    if mlflow.active_run():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text.encode("utf-8"))
            temp_file.flush()
            mlflow.log_artifact(temp_file.name, artifact_path=file_name)

def end_mlflow_run():
    """
    Ends the current MLflow run if one is active.
    """
    if mlflow.active_run():
        mlflow.end_run()

def log_params_to_mlflow(params):
    """
    Logs multiple parameters to MLflow.

    Parameters:
    - params: Dictionary of param_name: value pairs.
    """
    if mlflow.active_run():
        mlflow.log_params(params)

import mlflow
from loguru import logger


def start_mlflow_run(config):
    """
    Starts an MLflow run based on the provided configuration and logs setup information.

    Parameters:
    - config: Dictionary loaded from the YAML configuration file.
    """
    # Use the provided logger instance or default to the global logger

    try:
        # Check if MLflow is enabled
        mlflow_enabled = config.get("experiment", {}).get("logging", {}).get("mlflow", False)
        if not mlflow_enabled:
            logger.warning("MLflow is disabled in the configuration.")
            return

        # Get MLflow tracking URI
        mlflow_uri = config.get("experiment", {}).get("logging", {}).get("mlflow_uri", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_uri)

        # Get the experiment name
        experiment_name = config.get("experiment", {}).get("name", "default_experiment")

        # Create or get the experiment in MLflow
        experiment_id = mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set: {experiment_name} (ID: {experiment_id})")

        # Start the MLflow run
        with mlflow.start_run(run_name=experiment_name):
            logger.info(f"MLflow run started: {experiment_name}")

            # Log configuration parameters
            logger.info("Logging setup config parameters to MLflow.")

            # Assuming flatten_dict is a utility function to flatten nested dictionaries
            flattened_params = flatten_dict(config)
            mlflow.log_params(flattened_params)

    except Exception as e:
        logger.error(f"Failed to start MLflow run: {e}")
