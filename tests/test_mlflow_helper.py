import pytest

from utils.mlflow_helper import start_mlflow_run


@pytest.mark.parametrize("function_name", ["set_tracking_uri", "set_experiment", "start_run", "log_params"])
def test_start_mlflow_run_skips_all_mlflow_calls_when_disabled(monkeypatch, function_name):
    def _should_not_be_called(*_args, **_kwargs):
        raise AssertionError(f"mlflow.{function_name} should not be called when mlflow_enabled=false")

    monkeypatch.setattr(f"utils.mlflow_helper.mlflow.{function_name}", _should_not_be_called)

    config = {
        "tracking": {"mlflow_enabled": False},
        "runtime": {"mlflow_uri": "file:./mlruns"},
        "metadata": {"experiment_name": "exp", "run_name": "run"},
    }

    start_mlflow_run(config)
