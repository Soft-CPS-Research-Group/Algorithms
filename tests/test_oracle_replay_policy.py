from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorithms.agents.baseline_policies import RBCSmartLocalPolicy
from algorithms.agents.oracle_replay_policy import FixedServiceOracleReplayPolicy
from algorithms.oracles import SemanticActionSeries, SemanticSchedule


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def test_oracle_replay_uses_strict_local_service_teacher() -> None:
    assert issubclass(FixedServiceOracleReplayPolicy, RBCSmartLocalPolicy)


def test_oracle_replay_converts_kw_to_normalized_action_and_wraps_episode(
    tmp_path: Path,
) -> None:
    schedule = SemanticSchedule(
        problem_id="fixture",
        horizon=2,
        timestep_hours=0.25,
        series=(
            SemanticActionSeries(
                building_id="Building_1",
                action_name="electrical_storage",
                values=(1.0, -0.5),
            ),
        ),
    )
    path = tmp_path / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    agent = FixedServiceOracleReplayPolicy(
        {
            "algorithm": {
                "name": "FixedServiceOracleReplayPolicy",
                "hyperparameters": {
                    "schedule_path": str(path),
                    "local_action_safety_enabled": False,
                },
            }
        }
    )
    names = [
        "district__community_import_power_kw",
        "storage::Building_1/electrical_storage::nominal_power_kw",
    ]
    agent.attach_environment(
        observation_names=[names],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )
    observation = [np.asarray([999.0, 5.0])]

    assert agent.predict(observation)[0] == pytest.approx([0.2])
    assert agent.predict(observation)[0] == pytest.approx([-0.1])
    assert agent.predict(observation)[0] == pytest.approx([0.2])

    metadata = agent.export_artifacts(str(tmp_path / "artifacts"))
    assert metadata["parameters"]["service_teacher"] == {
        "algorithm": "RBCSmartLocalPolicy",
        "observation_scope": "building_plus_public_exogenous",
        "blocked_observation_token": "community",
    }


def test_oracle_replay_applies_schedule_step_offset(tmp_path: Path) -> None:
    schedule = SemanticSchedule(
        problem_id="fixture-offset",
        horizon=2,
        timestep_hours=0.25,
        series=(
            SemanticActionSeries(
                building_id="Building_1",
                action_name="electrical_storage",
                values=(1.0, -0.5),
            ),
        ),
    )
    path = tmp_path / "schedule-offset.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    agent = FixedServiceOracleReplayPolicy(
        {
            "algorithm": {
                "name": "FixedServiceOracleReplayPolicy",
                "hyperparameters": {
                    "schedule_path": str(path),
                    "schedule_step_offset": 1,
                    "local_action_safety_enabled": False,
                },
            }
        }
    )
    names = [
        "district__community_import_power_kw",
        "storage::Building_1/electrical_storage::nominal_power_kw",
    ]
    agent.attach_environment(
        observation_names=[names],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    action = agent.predict_at_step(
        [np.asarray([999.0, 5.0])],
        schedule_step=0,
    )

    assert action[0] == pytest.approx([-0.1])
    metadata = agent.export_artifacts(str(tmp_path / "artifacts-offset"))
    assert metadata["parameters"]["oracle_schedule"]["schedule_step_offset"] == 1
