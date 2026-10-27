"""Failure, preservation and stochastic-boundary regressions for the shared runtime."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from algorithms.configuration.project import validate_config
from algorithms.runtime.atomic import atomic_json
from algorithms.runtime.checkpoint_index import discover_legacy, resolve_index
from algorithms.runtime.controller import SimulationController
from algorithms.runtime.inputs import unique_object, validate_table
from algorithms.runtime.paths import _prepare_paths
from algorithms.runtime.randomness import RandomStreams, derived_seed
from algorithms.runtime.tracking import mlflow, tracking_session
from utils.checkpoint_manager import CheckpointManager


def base_config():
    return {
        "metadata": {"experiment_name": "test", "run_name": "test"},
        "pipeline": [{"algorithm": "NormalPolicy"}],
        "simulator": {
            "dataset_name": "test",
            "dataset_path": "unused",
            "reward_function": "RewardFunction",
        },
    }


def test_explicit_train_count_adds_evaluation_without_removing_training():
    config = base_config()
    config["training"] = {"episodes": 4, "seed": 0}
    resolved = validate_config(config).to_dict()
    assert resolved["simulator"]["episodes"] == 5
    assert resolved["simulator"]["deterministic_finish"] is True
    assert resolved["training"]["seed"] == 0
    assert resolved["tracking"]["mlflow_enabled"] is False
    config["evaluation"] = {"episodes": 0}
    assert validate_config(config).simulator.episodes == 4


def test_legacy_episode_total_retains_effective_training_count():
    config = base_config()
    config["simulator"].update(episodes=4, deterministic_finish=True)
    resolved = validate_config(config)
    assert resolved.simulator.episodes == 4


def test_named_streams_do_not_advance_replay_when_exploration_changes():
    first, second = RandomStreams(0), RandomStreams(0)
    with first.scope("exploration"):
        random.random()
        np.random.normal(size=100)
        torch.rand(14)
    with first.scope("learning"):
        a = (random.random(), np.random.rand(), torch.rand(2))
    with second.scope("learning"):
        b = (random.random(), np.random.rand(), torch.rand(2))
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])
    assert derived_seed(0, "stage", 0) != derived_seed(0, "stage", 1)


@pytest.mark.parametrize("actions", [[[float("nan")]], [[float("inf")]], [[0, 1]], [], [[0], [0]]])
def test_invalid_actions_rejected_before_simulator_step(actions):
    wrapper = SimpleNamespace(
        config={"simulator": {}},
        _get_action_bounds_cache=lambda: [(np.array([-1.0]), np.array([1.0]))],
    )
    with pytest.raises(ValueError):
        SimulationController._clip_actions(wrapper, actions)


def test_clipping_is_explicit_and_reject_policy_is_supported():
    wrapper = SimpleNamespace(
        config={"simulator": {"action_policy": "reject"}},
        _get_action_bounds_cache=lambda: [(np.array([-1.0]), np.array([1.0]))],
    )
    with pytest.raises(ValueError, match="outside bounds"):
        SimulationController._clip_actions(wrapper, [[2.0]])
    wrapper.config["simulator"]["action_policy"] = "clip"
    assert SimulationController._clip_actions(wrapper, [[2.0]]) == [[1.0]]


def test_atomic_json_preserves_previous_record_on_nonfinite_failure(tmp_path):
    target = tmp_path / "record.json"
    atomic_json(target, {"status": "running"})
    with pytest.raises(ValueError):
        atomic_json(target, {"metric": float("nan")})
    assert json.loads(target.read_text()) == {"status": "running"}


def test_repeated_job_id_retains_old_attempt(tmp_path):
    first = _prepare_paths(tmp_path, "same")
    first["result_path"].write_text('{"status":"completed"}')
    second = _prepare_paths(tmp_path, "same")
    assert first["job_dir"] != second["job_dir"]
    assert first["result_path"].read_text() == '{"status":"completed"}'
    assert second["job_dir"].parent == first["job_dir"] / "attempts"


class Writer:
    def save_checkpoint(self, directory, step):
        target = Path(directory) / "weights.bin"
        target.write_bytes(b"weights")
        return str(target)


def test_checkpoint_index_detects_tampering_and_preserves_same_step_versions(tmp_path):
    manager = CheckpointManager(str(tmp_path), None)
    first = manager.save_final(Writer(), 1)
    second = manager.save_final(Writer(), 1)
    assert first != second and first.read_bytes() == second.read_bytes()
    entry, artifact, _ = resolve_index(tmp_path / "checkpoints")
    assert artifact == second
    assert discover_legacy(tmp_path / "checkpoints/index.json") == second
    assert entry["resume_capability"] == "weights"
    second.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        resolve_index(tmp_path / "checkpoints")


def test_partial_checkpoint_and_disk_full_never_replace_latest(tmp_path, monkeypatch):
    manager = CheckpointManager(str(tmp_path), None)
    valid = manager.save_final(Writer(), 1)

    class Broken:
        def save_checkpoint(self, directory, step):
            (Path(directory) / "partial.bin").write_bytes(b"partial")
            raise OSError("disk full")

    with pytest.raises(OSError):
        manager.save_final(Broken(), 2)
    assert resolve_index(tmp_path / "checkpoints")[1] == valid
    assert list((tmp_path / "checkpoints").glob(".*.partial"))
    monkeypatch.setattr(
        "utils.checkpoint_manager.shutil.disk_usage", lambda _: SimpleNamespace(free=0)
    )
    with pytest.raises(OSError, match="free disk"):
        manager.save_final(Writer(), 3)
    assert resolve_index(tmp_path / "checkpoints")[1] == valid


def test_data_contract_checks_dates_resolution_order_bounds_and_missing_values(tmp_path):
    path = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {
            "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z"],
            "non_shiftable_load": [1.0, 2.0],
        }
    )
    frame.to_csv(path, index=False)
    contract = {
        "columns": list(frame.columns),
        "timezone": "UTC",
        "bounds": {"non_shiftable_load": [0, 3]},
    }
    assert validate_table(path, required_end=1, contract=contract)["calendar"]["timezone"] == "UTC"
    with pytest.raises(ValueError, match="resolution"):
        validate_table(path, seconds=900)
    with pytest.raises(ValueError, match="columns/order"):
        validate_table(path, contract={"columns": list(reversed(frame.columns))})
    with pytest.raises(ValueError, match="short horizon"):
        validate_table(path, required_end=2)
    with pytest.raises(ValueError, match="Timezone"):
        validate_table(path, contract={"timezone": "Europe/Lisbon"})
    frame.loc[1, "non_shiftable_load"] = float("nan")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        validate_table(path)
    with pytest.raises(ValueError, match="Duplicate"):
        json.loads('{"building":1,"building":2}', object_pairs_hook=unique_object)


def test_tracking_policy_samples_all_producers_and_retains_local_evidence(tmp_path, monkeypatch):
    import sys

    calls = []
    backend = SimpleNamespace(
        log_metrics=lambda *a, **k: calls.append((a, k)),
        log_artifact=lambda *a, **k: calls.append((a, k)),
        active_run=lambda: object(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", backend)
    paths = _prepare_paths(tmp_path, "tracking")
    config = {
        "tracking": {
            "mlflow_enabled": True,
            "mlflow_step_sample_interval": 10,
            "mlflow_artifacts_profile": "minimal",
        }
    }
    with tracking_session(config, paths):
        mlflow.log_metrics({"loss": 2}, step=1)
        mlflow.log_metrics({"loss": 1}, step=10)
        mlflow.log_artifact("unused", artifact_path="model")
        assert len(calls) == 1
    assert len((paths["log_dir"] / "metrics.jsonl").read_text().splitlines()) == 2
    summary = json.loads((paths["results_dir"] / "tracking.json").read_text())
    assert summary["remote_calls"] == 1


def test_tracking_failure_falls_back_locally_or_aborts(tmp_path, monkeypatch):
    import sys

    def fail(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(log_metrics=fail, active_run=lambda: object())
    )
    config = {"tracking": {"mlflow_enabled": True, "mlflow_step_sample_interval": 1}}
    paths = _prepare_paths(tmp_path, "fallback")
    with tracking_session(config, paths) as state:
        mlflow.log_metric("loss", 1, 1)
        mlflow.log_metric("loss", 2, 2)
        assert state["failed"] is True
    assert len((paths["log_dir"] / "metrics.jsonl").read_text().splitlines()) == 2
    config["tracking"]["mlflow_failure_policy"] = "abort"
    with pytest.raises(ConnectionError), tracking_session(config, paths):
        mlflow.log_metric("loss", 1, 1)


def test_evaluation_freezes_dropout_and_normalizer_and_hash_tracks_actual_weights():
    from algorithms.runtime.evaluation import evaluation_mode, policy_digest

    unit = SimpleNamespace(
        config={},
        networks=[
            torch.nn.Sequential(
                torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.8)
            )
        ],
    )
    model = unit.networks[0]
    before = policy_digest(unit)
    running = model[1].running_mean.clone()
    with evaluation_mode(unit):
        x = torch.ones(4, 3)
        a, b = model(x), model(x)
        assert torch.equal(a, b) and not a.requires_grad
        assert not model.training
    assert model.training and torch.equal(running, model[1].running_mean)
    assert before == policy_digest(unit)
    with torch.no_grad():
        model[0].weight.add_(1)
    assert before != policy_digest(unit)


def test_auxiliary_files_are_required_and_identified(tmp_path):
    from algorithms.runtime.inputs import validate_inputs

    config = {
        "simulator": {},
        "pipeline": [{"hyperparameters": {"schedule_path": str(tmp_path / "missing.json")}}],
    }
    schema = {"root_directory": str(tmp_path), "buildings": {}}
    with pytest.raises(FileNotFoundError, match="Missing agent input"):
        validate_inputs(config, schema)
    path = tmp_path / "missing.json"
    path.write_text("{}")
    a = validate_inputs(config, schema)
    path.write_text('{"changed":true}')
    b = validate_inputs(config, schema)
    assert a["identity_sha256"] != b["identity_sha256"]


def test_episode_resume_rejects_weights_and_middle_of_episode():
    from algorithms.runtime.resume import restore_runtime

    with pytest.raises(ValueError, match="episode-boundary"):
        restore_runtime(
            None,
            {"version": "algorithms.resume.v1", "resume_capability": "weights", "boundary": False},
        )


def test_building_agent_evaluation_is_deterministic_and_keeps_normalizer():
    from algorithms.agents.building_agent import BuildingAgent

    agent = BuildingAgent(
        {
            "algorithm": {
                "hyperparameters": {
                    "obs_dim": 2,
                    "action_dim": 1,
                    "hidden_dims": [4],
                    "num_steps": 4,
                }
            }
        }
    )
    observation = [np.array([1.0, 2.0], dtype=np.float32)]
    agent.predict(observation, deterministic=False)
    before = agent._obs_rms._n
    rng = torch.get_rng_state().clone()
    a = agent.predict(observation, deterministic=True)
    b = agent.predict(observation, deterministic=True)
    assert a == b and agent._obs_rms._n == before
    assert torch.equal(rng, torch.get_rng_state())
    assert agent.rollout_buffer.ptr == 0


def test_community_coordinator_evaluation_keeps_normalizers_and_temporal_interval(monkeypatch):
    from algorithms.agents.community_coordinator_agent import CommunityCoordinatorAgent

    agent = CommunityCoordinatorAgent(
        {
            "algorithm": {
                "hyperparameters": {
                    "c_dim": 2,
                    "b_dim": 2,
                    "num_buildings": 1,
                    "hidden_dims": [4],
                    "num_steps": 4,
                    "cc_action_interval": 2,
                }
            }
        }
    )
    agent._output_mode = "signal"
    monkeypatch.setattr(agent, "_build_community_context", lambda _: np.ones(2, dtype=np.float32))
    monkeypatch.setattr(
        agent, "_build_building_features", lambda _: np.ones((1, 2), dtype=np.float32)
    )
    monkeypatch.setattr(agent, "_log_decision", lambda *args: None)
    agent.on_episode_start(episode=0, training=False)
    rng = torch.get_rng_state().clone()
    a = agent.predict([], deterministic=True).copy()
    assert agent._step_in_interval == 1
    agent.predict([], deterministic=True)
    assert agent._step_in_interval == 0
    b = agent.predict([], deterministic=True)
    np.testing.assert_array_equal(a, b)
    assert agent._community_rms._n == agent._building_rms._n == 0
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("name", ["IPPO", "MAPPO", "HAPPO", "MASAC"])
def test_evaluation_does_not_advance_exploration_counter(name):
    from algorithms.registry import build_execution_unit
    from tests.test_single_agent_rl import _attach, _distributed_config

    config = _distributed_config("TD3" if name == "MASAC" else name)
    config["pipeline"][0].update(algorithm=name, count=1)
    agent = build_execution_unit(config)
    _attach(agent)
    before = agent.exploration_step
    agent.predict([np.zeros(3), np.zeros(2)], deterministic=True)
    assert agent.exploration_step == before


def test_dynamic_inventory_includes_future_members_and_asset_sources(tmp_path):
    from algorithms.runtime.inputs import validate_inputs

    schema = {
        "root_directory": str(tmp_path),
        "buildings": {
            "future": {"include": False, "filename": "future.bin"},
            "source": {"include": False, "filename": "source.bin"},
            "unused": {"include": False, "filename": "not-consumed.bin"},
        },
        "topology_events": [
            {"operation": "add_member", "target_member_id": "future", "time_step": 2},
            {
                "operation": "add_asset",
                "source_member_id": "source",
                "target_member_id": "future",
                "time_step": 3,
            },
        ],
    }
    config = {"simulator": {"topology_mode": "dynamic"}}
    with pytest.raises(FileNotFoundError, match="future.bin"):
        validate_inputs(config, schema)
    (tmp_path / "future.bin").write_bytes(b"future")
    (tmp_path / "source.bin").write_bytes(b"source")
    first = validate_inputs(config, schema)
    assert first["future_building_ids"] == ["future", "source"]
    assert {Path(item["path"]).name for item in first["files"]} == {"future.bin", "source.bin"}
    assert not schema["buildings"]["future"]["include"]
    (tmp_path / "source.bin").write_bytes(b"changed source")
    assert first["identity_sha256"] != validate_inputs(config, schema)["identity_sha256"]
