from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import run_experiment as runner
from utils.local_metrics import LocalMetricsLogger


class _DummyConfigModel:
    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


class _DummyKpiTable:
    def pivot(self, **_kwargs):
        return self

    def round(self, *_args, **_kwargs):
        return self

    def dropna(self, **_kwargs):
        return self

    def stack(self, **_kwargs):
        return {("cost", "building"): 1.23}

    def to_dict(self):
        return {"cost": {"building": 1.23}}


class _DummyEnv:
    def evaluate(self):
        return _DummyKpiTable()


class _DummyWrapper:
    last_instance = None

    def __init__(self, env, config, job_id, progress_path):
        self.env = env
        self.config = config
        self.job_id = job_id
        self.progress_path = progress_path
        self.observation_dimension = [2]
        self.action_dimension = [1]
        self.action_space = [object()]
        self.local_metrics_logger = None
        self.learn_call_kwargs = None
        _DummyWrapper.last_instance = self

    def set_model(self, _agent):
        return None

    def learn(self, **kwargs):
        self.learn_call_kwargs = kwargs
        return None

    def describe_environment(self):
        return {
            "observation_names": [["feat_1", "feat_2"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}, {"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0.0], "high": [1.0]}]],
            "action_names": ["action_0"],
            "action_names_by_agent": {"0": ["action_0"]},
            "reward_function": {"name": "RewardFunction", "params": {}},
        }


class _DummyWrapperWithLocalMetrics(_DummyWrapper):
    def __init__(self, env, config, job_id, progress_path):
        super().__init__(env, config, job_id, progress_path)
        self.local_metrics_logger = LocalMetricsLogger(config.get("runtime", {}).get("log_dir"))

    def learn(self, **kwargs):
        self.learn_call_kwargs = kwargs
        if self.local_metrics_logger:
            self.local_metrics_logger.log({"sample_metric": 1.0}, 0)


class _DummyAgent:
    def export_artifacts(self, output_dir: str, context: dict | None = None):
        _ = context
        output_path = Path(output_dir) / "onnx_models" / "agent_0.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy")
        return {
            "format": "onnx",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "onnx_models/agent_0.onnx",
                    "format": "onnx",
                    "observation_dimension": 2,
                    "action_dimension": 1,
                    "config": {},
                }
            ],
        }


class _DummyResumeAgent(_DummyAgent):
    def __init__(self):
        self.loaded_checkpoint_paths = []

    def load_checkpoint(self, checkpoint_path: str):
        self.loaded_checkpoint_paths.append(checkpoint_path)

    @staticmethod
    def get_best_checkpoint(experiment_name: str) -> str:
        _ = experiment_name
        return "best-run-id"


class _DummyRunInfo:
    def __init__(self, run_id: str, run_name: str, experiment_id: str):
        self.run_id = run_id
        self.run_name = run_name
        self.experiment_id = experiment_id


class _DummyRun:
    def __init__(self, run_id: str, run_name: str, experiment_id: str):
        self.info = _DummyRunInfo(run_id=run_id, run_name=run_name, experiment_id=experiment_id)


def _build_enabled_config(*, artifact_profile: str) -> dict:
    return {
        "metadata": {"experiment_name": "exp", "run_name": "run"},
        "runtime": {"log_dir": None, "mlflow_uri": "http://from-config:5000"},
        "tracking": {
            "mlflow_enabled": True,
            "log_level": "INFO",
            "log_frequency": 1,
            "mlflow_step_sample_interval": 10,
            "mlflow_artifacts_profile": artifact_profile,
        },
        "checkpointing": {"checkpoint_interval": None},
        "bundle": {"require_observations_envelope": False, "artifact_config": {}},
        "simulator": {
            "dataset_name": "dummy_dataset",
            "dataset_path": "dummy.json",
            "central_agent": False,
            "reward_function": "RewardFunction",
            "episodes": 3,
            "simulation_start_time_step": None,
            "simulation_end_time_step": None,
            "episode_time_steps": None,
            "export": {
                "mode": "none",
                "export_kpis_on_episode_end": False,
                "session_name": None,
            },
        },
        "training": {
            "seed": 1,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "topology": {
            "num_agents": None,
            "observation_dimensions": None,
            "action_dimensions": None,
            "action_space": None,
        },
        "algorithm": {
            "name": "MADDPG",
            "hyperparameters": {"gamma": 0.99},
            "networks": {
                "actor": {"class": "Actor", "layers": [8], "lr": 1e-4},
                "critic": {"class": "Critic", "layers": [8], "lr": 1e-3},
            },
            "replay_buffer": {
                "class": "MultiAgentReplayBuffer",
                "capacity": 8,
                "batch_size": 2,
            },
            "exploration": {
                "strategy": "GaussianNoise",
                "params": {
                    "bias": 0.0,
                    "sigma": 0.1,
                    "decay": 0.99,
                    "gamma": 0.99,
                    "tau": 0.001,
                },
            },
        },
    }


def test_run_experiment_mlflow_disabled_writes_stable_outputs(monkeypatch, tmp_path):
    config = {
        "metadata": {"experiment_name": "exp", "run_name": "run"},
        "runtime": {"log_dir": None, "mlflow_uri": None},
        "tracking": {"mlflow_enabled": False, "log_level": "INFO", "log_frequency": 1},
        "checkpointing": {"checkpoint_interval": None},
        "bundle": {"require_observations_envelope": False, "artifact_config": {}},
        "simulator": {
            "dataset_name": "dummy",
            "dataset_path": "dummy.json",
            "central_agent": False,
            "reward_function": "RewardFunction",
            "episodes": 2,
            "simulation_start_time_step": 12,
            "simulation_end_time_step": 48,
            "episode_time_steps": 24,
            "export": {
                "mode": "end",
                "export_kpis_on_episode_end": True,
                "session_name": "job-session",
            },
        },
        "training": {
            "seed": 1,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "topology": {"num_agents": None, "observation_dimensions": None, "action_dimensions": None, "action_space": None},
        "algorithm": {
            "name": "MADDPG",
            "hyperparameters": {"gamma": 0.99},
            "networks": {
                "actor": {"class": "Actor", "layers": [8], "lr": 1e-4},
                "critic": {"class": "Critic", "layers": [8], "lr": 1e-3},
            },
            "replay_buffer": {"class": "MultiAgentReplayBuffer", "capacity": 8, "batch_size": 2},
            "exploration": {"strategy": "GaussianNoise", "params": {"bias": 0.0, "sigma": 0.1, "decay": 0.99, "gamma": 0.99, "tau": 0.001}},
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: None)
    captured_env_kwargs = {}

    def _dummy_citylearn_env(**kwargs):
        captured_env_kwargs.update(kwargs)
        return _DummyEnv()

    monkeypatch.setattr(runner, "CityLearnEnv", _dummy_citylearn_env)
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    runner.run_experiment(str(config_path), "job-mlflow-off", tmp_path)
    assert _DummyWrapper.last_instance is not None
    assert _DummyWrapper.last_instance.learn_call_kwargs == {"episodes": 2}

    job_root = tmp_path / "jobs" / "job-mlflow-off"
    assert (job_root / "logs").exists()
    log_path = job_root / "logs" / "local-job-mlflow-off.log"
    assert log_path.exists()
    assert "Starting experiment" in log_path.read_text(encoding="utf-8")
    assert (job_root / "progress" / "progress.json").exists()
    assert (job_root / "results" / "result.json").exists()
    assert (job_root / "onnx_models" / "agent_0.onnx").exists()
    assert (job_root / "artifact_manifest.json").exists()
    assert (job_root / "job_info.json").exists()
    assert (job_root / "config.resolved.yaml").exists()
    assert (job_root / "results" / "summary.json").exists()

    job_info = json.loads((job_root / "job_info.json").read_text(encoding="utf-8"))
    assert job_info["mlflow_enabled"] is False
    assert job_info["run_id"] == "local-job-mlflow-off"
    assert job_info["mlflow_run_id"] is None
    assert job_info["mlflow_experiment_id"] is None

    resolved_config = yaml.safe_load((job_root / "config.resolved.yaml").read_text(encoding="utf-8"))
    runtime_cfg = resolved_config["runtime"]
    assert runtime_cfg["job_id"] == "job-mlflow-off"
    assert runtime_cfg["run_id"] == "local-job-mlflow-off"
    assert runtime_cfg["run_name"] == "run"
    assert runtime_cfg["job_dir"] == str(job_root)
    assert runtime_cfg["log_dir"] == str(job_root / "logs")
    assert runtime_cfg["mlflow_uri"] == f"file:{tmp_path / 'mlflow' / 'mlruns'}"
    assert runtime_cfg["tracking_uri"] == f"file:{tmp_path / 'mlflow' / 'mlruns'}"

    topology_cfg = resolved_config["topology"]
    assert topology_cfg["num_agents"] == 1
    assert topology_cfg["observation_dimensions"] == [2]
    assert topology_cfg["action_dimensions"] == [1]
    simulator_cfg = resolved_config["simulator"]
    assert simulator_cfg["export"]["mode"] == "end"
    assert simulator_cfg["export"]["export_kpis_on_episode_end"] is True
    assert simulator_cfg["export"]["session_name"] == "job-session"
    assert simulator_cfg["simulation_start_time_step"] == 12
    assert simulator_cfg["simulation_end_time_step"] == 48
    assert simulator_cfg["episode_time_steps"] == 24
    assert captured_env_kwargs["schema"] == "dummy.json"
    assert captured_env_kwargs["render_mode"] == "end"
    assert captured_env_kwargs["export_kpis_on_episode_end"] is True
    assert captured_env_kwargs["render_session_name"] == "job-session"
    assert captured_env_kwargs["simulation_start_time_step"] == 12
    assert captured_env_kwargs["simulation_end_time_step"] == 48
    assert captured_env_kwargs["episode_time_steps"] == 24
    assert captured_env_kwargs["render_directory"] == str(job_root / "results" / "simulation_data")
    result_payload = json.loads((job_root / "results" / "result.json").read_text(encoding="utf-8"))
    assert result_payload["status"] == "completed"
    assert result_payload["kpi_source"] == "simulator_export"
    assert result_payload["export_kpis_on_episode_end"] is True

    # Input config file must remain unchanged; resolved values are written separately.
    unchanged_input = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert unchanged_input == config


def test_run_experiment_mlflow_disabled_keeps_local_metrics_fallback(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config["tracking"]["mlflow_enabled"] = False

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapperWithLocalMetrics)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    runner.run_experiment(str(config_path), "job-local-metrics", tmp_path)

    metrics_path = tmp_path / "jobs" / "job-local-metrics" / "logs" / "metrics.jsonl"
    assert metrics_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("sample_metric" in record.get("metrics", {}) for record in records)


def test_run_experiment_writes_disabled_kpi_result_when_export_toggle_off(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config["tracking"]["mlflow_enabled"] = False
    config["simulator"]["export"]["export_kpis_on_episode_end"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    runner.run_experiment(str(config_path), "job-kpi-disabled", tmp_path)
    assert _DummyWrapper.last_instance is not None
    assert _DummyWrapper.last_instance.learn_call_kwargs == {"episodes": 3}

    result_payload = json.loads(
        (tmp_path / "jobs" / "job-kpi-disabled" / "results" / "result.json").read_text(encoding="utf-8")
    )
    assert result_payload["status"] == "completed"
    assert result_payload["kpi_source"] == "disabled"


def test_run_experiment_uses_env_tracking_uri_adds_mlflow_identity_and_curated_artifacts(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="curated")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    dummy_run = _DummyRun(run_id="run-123", run_name="mlflow-run", experiment_id="42")
    logged_artifacts = []
    logged_tags = {}

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-mlflow:5000")
    monkeypatch.setenv("MLFLOW_UI_BASE_URL", "https://mlflow-ui.local")
    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: dummy_run)
    monkeypatch.setattr(runner.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner.mlflow,
        "set_tags",
        lambda tags: logged_tags.update(tags),
    )
    monkeypatch.setattr(
        runner.mlflow,
        "log_artifact",
        lambda file_path, artifact_path=None: logged_artifacts.append((Path(file_path).name, artifact_path)),
    )
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    runner.run_experiment(str(config_path), "job-mlflow-on", tmp_path)

    job_root = tmp_path / "jobs" / "job-mlflow-on"
    job_info = json.loads((job_root / "job_info.json").read_text(encoding="utf-8"))
    resolved_config = yaml.safe_load((job_root / "config.resolved.yaml").read_text(encoding="utf-8"))

    assert job_info["tracking_uri"] == "http://env-mlflow:5000"
    assert job_info["mlflow_run_id"] == "run-123"
    assert job_info["mlflow_experiment_id"] == "42"
    assert job_info["mlflow_run_url"] == "https://mlflow-ui.local/#/experiments/42/runs/run-123"

    runtime_cfg = resolved_config["runtime"]
    assert runtime_cfg["mlflow_uri"] == "http://env-mlflow:5000"
    assert runtime_cfg["tracking_uri"] == "http://env-mlflow:5000"
    assert runtime_cfg["experiment_id"] == "42"
    assert runtime_cfg["mlflow_run_url"] == "https://mlflow-ui.local/#/experiments/42/runs/run-123"

    assert logged_tags["opeva.job_id"] == "job-mlflow-on"
    assert logged_tags["opeva.algorithm"] == "MADDPG"
    assert logged_tags["opeva.dataset"] == "dummy_dataset"
    assert logged_tags["opeva.run_name"] == "mlflow-run"
    assert len(logged_tags["opeva.config_hash"]) == 64

    artifact_names = [name for name, _ in logged_artifacts]
    assert "config.resolved.yaml" in artifact_names
    assert "artifact_manifest.json" in artifact_names
    assert "result.json" in artifact_names
    assert "summary.json" in artifact_names


def test_run_experiment_minimal_profile_skips_curated_artifacts(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    dummy_run = _DummyRun(run_id="run-001", run_name="mlflow-run", experiment_id="7")
    logged_artifacts = []

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: dummy_run)
    monkeypatch.setattr(runner.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.mlflow, "set_tags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner.mlflow,
        "log_artifact",
        lambda file_path, artifact_path=None: logged_artifacts.append((Path(file_path).name, artifact_path)),
    )
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    runner.run_experiment(str(config_path), "job-mlflow-minimal", tmp_path)

    artifact_names = {name for name, _ in logged_artifacts}
    # minimal profile keeps model/checkpoint artifacts only (from agent/checkpoint manager)
    assert "config.resolved.yaml" not in artifact_names
    assert "artifact_manifest.json" not in artifact_names
    assert "result.json" not in artifact_names
    assert "summary.json" not in artifact_names


def test_build_checkpoint_artifact_candidates_has_fallback_order():
    candidates = runner._build_checkpoint_artifact_candidates("latest_checkpoint.pth")
    assert candidates == ["checkpoints/latest_checkpoint.pth", "latest_checkpoint.pth"]


def test_download_checkpoint_from_mlflow_uses_legacy_fallback_when_needed(monkeypatch, tmp_path):
    downloaded_from = []
    checkpoint_file = tmp_path / "downloaded" / "latest_checkpoint.pth"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text("checkpoint", encoding="utf-8")

    def _download_artifacts(**kwargs):
        downloaded_from.append(kwargs["artifact_path"])
        if kwargs["artifact_path"] == "latest_checkpoint.pth":
            return str(checkpoint_file)
        raise FileNotFoundError("new path missing")

    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.mlflow.artifacts, "download_artifacts", _download_artifacts)

    resolved_path = runner._download_checkpoint_from_mlflow(
        checkpoint_run_id="run-id",
        checkpoint_artifact="latest_checkpoint.pth",
        tracking_uri="http://mlflow.local",
        download_dir=tmp_path / "downloads",
    )

    assert downloaded_from == ["checkpoints/latest_checkpoint.pth", "latest_checkpoint.pth"]
    assert resolved_path == checkpoint_file


def test_run_experiment_resume_uses_mlflow_download_artifacts_and_load(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config["checkpointing"].update(
        {
            "resume_training": True,
            "checkpoint_run_id": "resume-run-id",
            "checkpoint_artifact": "latest_checkpoint.pth",
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    dummy_run = _DummyRun(run_id="run-002", run_name="mlflow-run", experiment_id="9")
    resume_agent = _DummyResumeAgent()
    downloaded_from = []
    checkpoint_file = tmp_path / "downloaded" / "latest_checkpoint.pth"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text("checkpoint", encoding="utf-8")

    def _download_artifacts(**kwargs):
        downloaded_from.append(kwargs["artifact_path"])
        if kwargs["artifact_path"].startswith("checkpoints/"):
            return str(checkpoint_file)
        raise FileNotFoundError("legacy path not needed")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: dummy_run)
    monkeypatch.setattr(runner.mlflow, "set_tags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.mlflow, "log_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.mlflow, "set_tracking_uri", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.mlflow.artifacts, "download_artifacts", _download_artifacts)
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: resume_agent)

    runner.run_experiment(str(config_path), "job-resume", tmp_path)

    assert downloaded_from == ["checkpoints/latest_checkpoint.pth"]
    assert resume_agent.loaded_checkpoint_paths == [str(checkpoint_file)]


def test_run_experiment_resume_fails_if_agent_does_not_implement_load(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config["checkpointing"]["resume_training"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))
    monkeypatch.setattr(runner, "start_mlflow_run", lambda config: None)
    monkeypatch.setattr(runner, "end_mlflow_run", lambda: None)
    monkeypatch.setattr(runner.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(runner, "CityLearnEnv", lambda **_kwargs: _DummyEnv())
    monkeypatch.setattr(runner, "Wrapper", _DummyWrapper)
    monkeypatch.setattr(runner, "create_agent", lambda config: _DummyAgent())

    with pytest.raises(RuntimeError, match="does not implement load_checkpoint"):
        runner.run_experiment(str(config_path), "job-resume-fail", tmp_path)


def test_run_experiment_fails_fast_for_placeholder_algorithm(monkeypatch, tmp_path):
    config = _build_enabled_config(artifact_profile="minimal")
    config["algorithm"]["name"] = "SingleAgentRL"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setattr(runner, "validate_config", lambda raw: _DummyConfigModel(raw))

    with pytest.raises(ValueError, match="placeholder"):
        runner.run_experiment(str(config_path), "job-placeholder", tmp_path)
