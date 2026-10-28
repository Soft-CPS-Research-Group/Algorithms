"""Behavioural boundaries shared by jobs, simulators and composed agents."""

from __future__ import annotations

import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import Field

from algorithms.catalogue import AGENT_SPECS
from algorithms.configuration.migration import apply_overrides, migrate_config
from algorithms.configuration.model import StrictModel
from algorithms.configuration.project import validate_config
from algorithms.contracts import (
    AgentCapabilities,
    AgentSpec,
    StageContext,
    Transition,
    UpdateSchedule,
)
from algorithms.pipeline import Ensemble, Pipeline
from algorithms.registry import ALGORITHM_REGISTRY, build_execution_unit, register_agent
from algorithms.runtime.simulator import SimulatorAdapter
from tests.test_pipeline import RecordingUnit


def base_config():
    return yaml.safe_load(
        Path("configs/templates/experimental/transformer_maddpg_smoke.yaml").read_text()
    )


def test_validation_and_catalogue_do_not_import_numerical_or_tracking_libraries():
    code = """
import sys
from algorithms.configuration.project import validate_config
from algorithms.catalogue import AGENT_SPECS
assert len(AGENT_SPECS) == 37
assert not ({"torch", "citylearn", "mlflow", "numpy", "onnx"} & sys.modules.keys())
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "section", ["tracking", "simulator", "job", "outputs", "training", "runtime", "checkpointing"]
)
def test_unknown_common_keys_fail_before_execution(section):
    config = base_config()
    config.setdefault(section, {})["misspelled_parameter"] = 1
    with pytest.raises(ValueError, match="Extra inputs"):
        validate_config(config)


def test_actor_and_offline_extensions_reject_unknown_keys():
    config = base_config()
    config["pipeline"][0]["exploration"] = {"strategy": "normal", "params": {"sigmma": 0.1}}
    with pytest.raises(ValueError, match="sigmma"):
        validate_config(config)
    config["pipeline"] = [
        {
            "algorithm": "IQLAgent",
            "hyperparameters": {
                "offline_training": {
                    "dataset_path": "transitions.parquet",
                    "parameters": {"learning_rae": 0.01},
                }
            },
        }
    ]
    with pytest.raises(ValueError, match="learning_rae"):
        validate_config(config)
    config["pipeline"][0]["hyperparameters"]["offline_training"]["parameters"] = {
        "gradient_steps": -1
    }
    with pytest.raises(ValueError, match="greater than 0"):
        validate_config(config)


def test_version_migration_and_overrides_are_explicit_and_non_mutating():
    original = {
        "algorithm": {
            "name": "RBCBasicPolicy",
            "hyperparameters": {"ev_service_soc_tolerance": 0.2},
        }
    }
    copy = deepcopy(original)
    migrated, notes = migrate_config(original, legacy_algorithm=True)
    assert original == copy
    assert "ev_service_soc_tolerance" not in migrated["pipeline"][0]["hyperparameters"]
    assert len(notes) == 2
    overridden = apply_overrides(
        migrated, ["pipeline.0.hyperparameters.ev_service_soc_tolerance=0.2"]
    )
    assert overridden["pipeline"][0]["hyperparameters"]["ev_service_soc_tolerance"] == 0.2
    assert "ev_service_soc_tolerance" not in migrated["pipeline"][0]["hyperparameters"]
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        migrate_config({"schema_version": "unknown"})


def test_profiles_resolve_to_bounded_training_or_read_only_evaluation():
    config = base_config()
    config["job"] = {"profile": "smoke"}
    smoke = validate_config(config).to_dict()
    assert smoke["simulator"]["episodes"] == 2
    assert smoke["simulator"]["episode_time_steps"] <= 5
    assert smoke["simulator"]["deterministic_finish"] is True
    assert smoke["tracking"]["mlflow_enabled"] is False
    config["job"]["profile"] = "evaluate"
    config["checkpointing"] = {"resume_training": True, "checkpoint_local_path": "trained.pt"}
    evaluation = validate_config(config).to_dict()
    assert evaluation["simulator"]["episodes"] == 1
    assert evaluation["pipeline"][0]["frozen"] is True
    assert evaluation["checkpointing"]["checkpoint_interval"] is None


def test_unsupported_checkpoint_request_fails_before_training():
    config = base_config()
    config["pipeline"] = [{"algorithm": "NormalPolicy"}]
    config["outputs"] = {"model_artifacts": "checkpoint"}
    with pytest.raises(ValueError, match="no stage supports checkpoint"):
        validate_config(config)


@pytest.mark.parametrize("legacy", [False, True])
def test_simulator_adapter_preserves_truncation_and_actual_duration(legacy):
    class Environment:
        seconds_per_time_step = 900

        def reset(self):
            return [[1]], {"reset": True}

        def step(self, actions):
            info = {"TimeLimit.truncated": True, "duration_seconds": 15}
            return ([[2]], [3], True, info) if legacy else ([[2]], [3], False, True, info)

    adapter = SimulatorAdapter(Environment())
    assert adapter.reset()[1] == {"reset": True}
    result = adapter.step([[0]])
    assert result.truncated is True and result.terminated is False
    assert result.duration_seconds == 15
    assert result.rewards == [3]


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_simulator_rejects_invalid_duration(duration):
    env = types.SimpleNamespace(
        step=lambda actions: ([], [], False, False, {"duration_seconds": duration})
    )
    with pytest.raises(ValueError, match="finite and positive"):
        SimulatorAdapter(env).step([])


def test_typed_transition_preserves_member_identity_proposals_and_simulator_actions():
    class TypedUnit(RecordingUnit):
        def learn_transition(self, transition, schedule):
            self.transition, self.schedule = transition, schedule

        def decide(self, observations, deterministic=None, *, context=None):
            self.context = context
            return [[0.1]]

    workers = [TypedUnit("a"), TypedUnit("b")]
    ensemble = Ensemble(workers)
    manager = RecordingUnit("manager", predict_output=[0.8, 1.1])
    manager.capabilities = AgentCapabilities("none", "raw", output_signal="price")
    pipeline = Pipeline([manager, ensemble])
    pipeline.member_ids = ("Building_A", "Building_B")
    pipeline.predict([[1], [2]])
    assert workers[1].context == StageContext(
        1.1, source_stage=0, kind="price", member_ids=("Building_B",)
    )
    transition = Transition(
        [[1], [2]],
        [[3], [4]],
        [1, 2],
        [[0.2], [0.3]],
        False,
        True,
        proposed_actions=[[0.4], [0.5]],
        member_ids=pipeline.member_ids,
        duration_seconds=15,
        info={"sample": "real"},
    )
    schedule = UpdateSchedule(9, True, False, True)
    pipeline.learn_transition(transition, schedule)
    second = workers[1].transition
    assert second.member_ids == ("Building_B",)
    assert second.proposed_actions == [[0.5]] and second.submitted_actions == [[0.3]]
    assert second.executed_actions is None
    assert second.truncated and second.duration_seconds == 15
    assert workers[1].schedule is schedule


def test_new_agent_registers_without_changing_job_or_project_schema(monkeypatch):
    class NewStage(StrictModel):
        algorithm: Literal["ContractTestAgent"]
        count: Literal[1] = 1
        frozen: bool = False
        hyperparameters: dict = Field(default_factory=dict)

    class NewAgent(RecordingUnit):
        def __init__(self, config):
            super().__init__("extension")

    module = types.ModuleType("contract_test_extension")
    module.Stage, module.Agent = NewStage, NewAgent
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = AgentSpec(
        "ContractTestAgent",
        f"{module.__name__}:Agent",
        f"{module.__name__}:Stage",
        AgentCapabilities("none", "raw"),
    )
    register_agent(spec)
    try:
        config = base_config()
        config["pipeline"] = [{"algorithm": spec.name}]
        unit = build_execution_unit(validate_config(config).to_dict())
        assert unit.predict([[1]]) == [[0.0]]
        assert unit.capabilities == spec.capabilities
    finally:
        del ALGORITHM_REGISTRY[spec.name]


@pytest.mark.parametrize("name", sorted(AGENT_SPECS))
def test_declared_capabilities_match_the_existing_implementation(name):
    from algorithms.execution_unit import ExecutionUnit

    spec = AGENT_SPECS[name]
    implementation = ALGORITHM_REGISTRY[name]
    assert issubclass(implementation, ExecutionUnit)
    assert spec.capabilities.dynamic_topology == bool(
        getattr(implementation, "supports_dynamic_topology", False)
    )
    assert spec.capabilities.final_stage_only == bool(
        getattr(implementation, "requires_final_pipeline_stage", False)
    )


@pytest.mark.parametrize(
    "expression", ["pipeline.9.count=2", "pipeline.-1.count=2", "pipeline.0.count.x=2"]
)
def test_invalid_override_targets_are_configuration_errors(expression):
    with pytest.raises(ValueError):
        apply_overrides({"pipeline": [{"count": 1}]}, [expression])


@pytest.mark.parametrize("job_id", ["../escape", "/absolute", "..", "nested/path"])
def test_job_id_cannot_escape_the_job_output_root(job_id):
    from algorithms.runtime.paths import _derive_job_id

    with pytest.raises(ValueError, match="directory name"):
        _derive_job_id(job_id)


def test_evaluation_requires_a_model_source():
    config = base_config()
    config["job"] = {"profile": "evaluate"}
    with pytest.raises(ValueError, match="trained model source"):
        validate_config(config)
