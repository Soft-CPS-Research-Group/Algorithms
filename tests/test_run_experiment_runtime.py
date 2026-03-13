from __future__ import annotations

import json
from pathlib import Path

import yaml

import run_experiment as runner


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
    def __init__(self, env, config, job_id, progress_path):
        self.env = env
        self.config = config
        self.job_id = job_id
        self.progress_path = progress_path
        self.observation_dimension = [2]
        self.action_dimension = [1]
        self.action_space = [object()]
        self.local_metrics_logger = None

    def set_model(self, _agent):
        return None

    def learn(self):
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
            "end_initial_exploration_time_step": 0,
            "end_exploration_time_step": 0,
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

    job_root = tmp_path / "jobs" / "job-mlflow-off"
    assert (job_root / "logs").exists()
    assert (job_root / "progress" / "progress.json").exists()
    assert (job_root / "results" / "result.json").exists()
    assert (job_root / "onnx_models" / "agent_0.onnx").exists()
    assert (job_root / "artifact_manifest.json").exists()
    assert (job_root / "job_info.json").exists()
    assert (job_root / "config.resolved.yaml").exists()

    job_info = json.loads((job_root / "job_info.json").read_text(encoding="utf-8"))
    assert job_info["mlflow_enabled"] is False
    assert job_info["run_id"] == "local-job-mlflow-off"
    assert job_info["mlflow_run_id"] is None

    resolved_config = yaml.safe_load((job_root / "config.resolved.yaml").read_text(encoding="utf-8"))
    runtime_cfg = resolved_config["runtime"]
    assert runtime_cfg["job_id"] == "job-mlflow-off"
    assert runtime_cfg["run_id"] == "local-job-mlflow-off"
    assert runtime_cfg["run_name"] == "run"
    assert runtime_cfg["job_dir"] == str(job_root)
    assert runtime_cfg["log_dir"] == str(job_root / "logs")
    assert runtime_cfg["mlflow_uri"] == f"file:{tmp_path / 'mlflow' / 'mlruns'}"

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

    # Input config file must remain unchanged; resolved values are written separately.
    unchanged_input = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert unchanged_input == config
