from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.total_oracle_replay_policy import TotalOracleReplayPolicy
from algorithms.oracles import SemanticActionSeries, SemanticSchedule


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


def _write_schedule(
    tmp_path: Path,
    series: tuple[SemanticActionSeries, ...],
    *,
    horizon: int = 2,
    metadata=None,
) -> Path:
    schedule = SemanticSchedule(
        problem_id="total-oracle-fixture",
        horizon=horizon,
        timestep_hours=0.25,
        series=series,
        metadata=metadata or {"formulation": "total_energy_milp"},
    )
    path = tmp_path / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    return path


def _agent(path: Path) -> TotalOracleReplayPolicy:
    return TotalOracleReplayPolicy(
        {
            "algorithm": {
                "name": "TotalOracleReplayPolicy",
                "hyperparameters": {"schedule_path": str(path)},
            }
        }
    )


def _series(
    building: str,
    action: str,
    values,
    *,
    unit: str = "kW",
    direction: str = "charge",
) -> SemanticActionSeries:
    return SemanticActionSeries(
        building_id=building,
        action_name=action,
        values=tuple(values),
        unit=unit,
        positive_direction=direction,
    )


def test_total_replay_is_standalone_raw_policy_without_rbc_teacher(tmp_path: Path) -> None:
    path = _write_schedule(
        tmp_path,
        (_series("Building_1", "electrical_storage", (0.0, 0.0)),),
    )
    agent = _agent(path)

    assert isinstance(agent, BaseAgent)
    assert agent.use_raw_observations is True
    agent.attach_environment(
        observation_names=[["storage::Building_1/electrical_storage::nominal_power_kw"]],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_1"],
            "seconds_per_time_step": 900,
        },
    )
    exported = agent.export_artifacts(str(tmp_path / "artifacts"))
    assert exported["format"] == "rule_based"
    assert exported["parameters"]["semantic_policy_format"] == "semantic_schedule_replay"
    assert exported["parameters"]["service_teacher"] is None
    assert exported["parameters"]["oracle_schedule"]["wraps"] is False
    assert exported["artifacts"] == [
        {
            "agent_index": 0,
            "path": "policy_agent_0.json",
            "format": "rule_based",
            "config": {"use_preprocessor": False},
        }
    ]
    assert (tmp_path / "artifacts" / "policy_agent_0.json").is_file()


def test_total_replay_preserves_building_groups_and_converts_all_semantic_units(
    tmp_path: Path,
) -> None:
    ev_1 = "electric_vehicle_storage_charger_1_1"
    ev_2 = "charger::Building_2/charger_2_1::electric_vehicle_storage"
    deferrable = "deferrable_appliance_deferrable_appliance_1"
    path = _write_schedule(
        tmp_path,
        (
            _series("Building_1", "electrical_storage", (2.5, 0.0)),
            _series("Building_1", ev_1, (5.5, 0.0)),
            _series(
                "Building_1",
                deferrable,
                (1.0, 0.0),
                unit="normalized_action",
                direction="start",
            ),
            _series("Building_2", "electrical_storage", (-2.0, 0.0)),
            _series("Building_2", ev_2, (-3.6, 0.0)),
        ),
    )
    agent = _agent(path)
    observation_names = [
        [
            "storage::Building_1/electrical_storage::nominal_power_kw",
            "charger::Building_1/charger_1_1::max_charging_power_kw",
        ],
        [
            "storage::Building_2/electrical_storage::nominal_power_kw",
            "charger::Building_2/charger_2_1::max_discharging_power_kw",
        ],
    ]
    action_names = [
        [deferrable, "electrical_storage", ev_1],
        [ev_2, "electrical_storage"],
    ]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=[_Box([0.0, -1.0, -1.0], [1.0, 1.0, 1.0]), _Box([-1.0, -1.0], [1.0, 1.0])],
        observation_space=[None, None],
        metadata={
            "building_names": ["Building_1", "Building_2"],
            "seconds_per_time_step": 900,
        },
    )

    actions = agent.predict(
        [np.asarray([5.0, 11.0]), np.asarray([4.0, 7.2])]
    )

    assert actions[0] == pytest.approx([1.0, 0.5, 0.5])
    assert actions[1] == pytest.approx([-0.5, -0.5])
    assert agent.predict([np.asarray([5.0, 11.0]), np.asarray([4.0, 7.2])]) == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
    ]


def test_total_replay_uses_explicit_metadata_when_raw_power_is_unavailable(
    tmp_path: Path,
) -> None:
    action = "electric_vehicle_storage_charger_1_1"
    path = _write_schedule(
        tmp_path,
        (_series("Building_1", action, (5.5, -3.6)),),
    )
    agent = _agent(path)
    agent.attach_environment(
        observation_names=[[]],
        action_names=[[action]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_1"],
            "seconds_per_time_step": 900,
            "total_oracle_replay": {
                "action_power_limits_kw": {
                    "Building_1": {
                        action: {
                            "max_charging_power_kw": 11.0,
                            "max_discharging_power_kw": 7.2,
                        }
                    }
                }
            },
        },
    )

    assert agent.predict([np.asarray([])])[0] == pytest.approx([0.5])
    assert agent.predict([np.asarray([])])[0] == pytest.approx([-0.5])


def test_total_replay_uses_embedded_schedule_ev_limits_and_legacy_binary_trigger(
    tmp_path: Path,
) -> None:
    ev_action = "electric_vehicle_storage_charger_1_1"
    deferrable = "deferrable_appliance_washer"
    path = _write_schedule(
        tmp_path,
        (
            _series("Building_1", ev_action, (5.5, -3.6)),
            _series(
                "Building_1",
                deferrable,
                (1.0, 0.0),
                unit="binary_start",
                direction="start_cycle",
            ),
        ),
        metadata={
            "scope": "individual_total_home_linear_milp",
            "action_power_limits_kw": {
                ev_action: {"max_charge_kw": 11.0, "max_discharge_kw": 7.2}
            },
        },
    )
    agent = _agent(path)
    agent.attach_environment(
        observation_names=[[]],
        action_names=[[ev_action, deferrable]],
        action_space=[_Box([-1.0, 0.0], [1.0, 1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    assert agent.predict([np.asarray([])])[0] == pytest.approx([0.5, 1.0])
    assert agent.predict([np.asarray([])])[0] == pytest.approx([-0.5, 0.0])


def test_total_replay_fails_explicitly_when_charger_power_contract_is_missing(
    tmp_path: Path,
) -> None:
    action = "electric_vehicle_storage_charger_1_1"
    path = _write_schedule(
        tmp_path,
        (_series("Building_1", action, (5.5, 0.0)),),
    )

    with pytest.raises(ValueError, match="action_power_limits_kw"):
        _agent(path).attach_environment(
            observation_names=[["hour"]],
            action_names=[[action]],
            action_space=[_Box([-1.0], [1.0])],
            observation_space=[None],
            metadata={
                "building_names": ["Building_1"],
                "seconds_per_time_step": 900,
            },
        )


def test_total_replay_is_non_wrapping_and_rejects_out_of_range_explicit_step(
    tmp_path: Path,
) -> None:
    path = _write_schedule(
        tmp_path,
        (_series("Building_1", "electrical_storage", (0.0, 0.0)),),
    )
    agent = _agent(path)
    agent.attach_environment(
        observation_names=[[]],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    assert agent.predict([np.asarray([])]) == [[0.0]]
    assert agent.predict([np.asarray([])]) == [[0.0]]
    with pytest.raises(RuntimeError, match="does not wrap"):
        agent.predict([np.asarray([])])
    with pytest.raises(IndexError, match="outside"):
        agent.predict_at_step([np.asarray([])], schedule_step=2)


@pytest.mark.parametrize(
    ("attached_actions", "schedule_series", "message"),
    [
        (
            ["electrical_storage", "electric_vehicle_storage_charger_1_1"],
            (_series("Building_1", "electrical_storage", (0.0, 0.0)),),
            "no series",
        ),
        (
            ["electrical_storage"],
            (
                _series("Building_1", "electrical_storage", (0.0, 0.0)),
                _series("Building_2", "electrical_storage", (0.0, 0.0)),
            ),
            "outside the attached",
        ),
    ],
)
def test_total_replay_requires_exact_local_action_coverage(
    tmp_path: Path,
    attached_actions: list[str],
    schedule_series: tuple[SemanticActionSeries, ...],
    message: str,
) -> None:
    path = _write_schedule(tmp_path, schedule_series)
    with pytest.raises(ValueError, match=message):
        _agent(path).attach_environment(
            observation_names=[[]],
            action_names=[attached_actions],
            action_space=[_Box([-1.0] * len(attached_actions), [1.0] * len(attached_actions))],
            observation_space=[None],
            metadata={
                "building_names": ["Building_1"],
                "seconds_per_time_step": 900,
            },
        )


def test_total_replay_rejects_wrong_semantic_unit(tmp_path: Path) -> None:
    path = _write_schedule(
        tmp_path,
        (
            _series(
                "Building_1",
                "deferrable_appliance_deferrable_appliance_1",
                (1.0, 0.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="normalized_action"):
        _agent(path).attach_environment(
            observation_names=[[]],
            action_names=[["deferrable_appliance_deferrable_appliance_1"]],
            action_space=[_Box([0.0], [1.0])],
            observation_space=[None],
            metadata={
                "building_names": ["Building_1"],
                "seconds_per_time_step": 900,
            },
        )


def test_total_replay_clips_only_numerical_action_bound_residue() -> None:
    key = ("Building_1", "electrical_storage")

    assert TotalOracleReplayPolicy._within_bounds(
        1.0 + 5.0e-8, -1.0, 1.0, key
    ) == pytest.approx(1.0)
    assert TotalOracleReplayPolicy._within_bounds(
        -1.0 - 5.0e-8, -1.0, 1.0, key
    ) == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="exceeds attached bounds"):
        TotalOracleReplayPolicy._within_bounds(1.0 + 2.0e-6, -1.0, 1.0, key)


def test_total_replay_combined_schedule_can_bind_one_strict_local_member(
    tmp_path: Path,
) -> None:
    path = _write_schedule(
        tmp_path,
        (
            _series("Building_1", "electrical_storage", (0.0, 0.0)),
            _series("Building_2", "electrical_storage", (0.0, 0.0)),
        ),
    )
    agent = TotalOracleReplayPolicy(
        {
            "algorithm": {
                "name": "TotalOracleReplayPolicy",
                "hyperparameters": {
                    "schedule_path": str(path),
                    "allow_attached_action_subset": True,
                    "repeat_schedule_for_training": True,
                },
            }
        }
    )
    agent.attach_environment(
        observation_names=[[]],
        action_names=[["electrical_storage"]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_2"],
            "seconds_per_time_step": 900,
        },
    )

    assert agent.predict([np.asarray([])]) == [[0.0]]
    assert agent.predict([np.asarray([])]) == [[0.0]]
    assert agent.predict([np.asarray([])]) == [[0.0]]
    schedule_metadata = agent.export_artifacts(
        str(tmp_path / "subset-artifacts")
    )["parameters"]["oracle_schedule"]
    assert schedule_metadata["allow_attached_action_subset"] is True
    assert schedule_metadata["wraps"] is True


def test_total_replay_nudges_exact_ev_deadband_above_float_residue(
    tmp_path: Path,
) -> None:
    action = "electric_vehicle_storage_charger_1_1"
    path = _write_schedule(
        tmp_path,
        (_series("Building_1", action, (1.4, 0.0)),),
        metadata={
            "action_power_limits_kw": {
                "Building_1": {
                    action: {
                        "max_charging_power_kw": 11.0,
                        "max_discharging_power_kw": 7.2,
                        "min_charging_power_kw": 1.4,
                        "min_discharging_power_kw": 1.4,
                    }
                }
            }
        },
    )
    agent = _agent(path)
    agent.attach_environment(
        observation_names=[[]],
        action_names=[[action]],
        action_space=[_Box([-1.0], [1.0])],
        observation_space=[None],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    normalized = agent.predict([np.asarray([])])[0][0]
    assert normalized * 11.0 > 1.4
    assert normalized * 11.0 == pytest.approx(1.400001)
