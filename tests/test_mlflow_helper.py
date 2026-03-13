import pytest
from types import SimpleNamespace

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


def test_start_mlflow_run_prefers_tracking_uri_field(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "utils.mlflow_helper.mlflow.set_tracking_uri",
        lambda uri: captured.setdefault("tracking_uri", uri),
    )
    monkeypatch.setattr(
        "utils.mlflow_helper.mlflow.set_experiment",
        lambda name: SimpleNamespace(experiment_id="exp-1", name=name),
    )
    monkeypatch.setattr(
        "utils.mlflow_helper.mlflow.start_run",
        lambda run_name: SimpleNamespace(info=SimpleNamespace(run_id="run-1", run_name=run_name)),
    )
    monkeypatch.setattr("utils.mlflow_helper.mlflow.log_params", lambda params: captured.setdefault("params", params))

    config = {
        "tracking": {"mlflow_enabled": True},
        "runtime": {"tracking_uri": "http://tracking-from-runtime:5000", "mlflow_uri": "file:./mlruns"},
        "metadata": {"experiment_name": "exp", "run_name": "run"},
    }

    start_mlflow_run(config)

    assert captured["tracking_uri"] == "http://tracking-from-runtime:5000"
    assert captured["params"]["metadata.experiment_name"] == "exp"
