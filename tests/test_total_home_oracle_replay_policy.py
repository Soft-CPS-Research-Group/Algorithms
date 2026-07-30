import json

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from algorithms.agents.total_home_oracle_replay_policy import TotalHomeOracleReplayPolicy
from algorithms.oracles import SemanticActionSeries, SemanticSchedule
from scripts.prepare_total_home_replay_configs import _config
from utils.config_schema import validate_config


def test_total_home_replay_normalizes_all_scheduled_action_types(tmp_path):
    schedule = SemanticSchedule(
        problem_id="total",
        horizon=2,
        timestep_hours=0.25,
        series=(
            SemanticActionSeries("Building_1", "electrical_storage", (2.5, -1.0)),
            SemanticActionSeries(
                "Building_1", "electric_vehicle_storage_charger_1", (5.5, -3.6)
            ),
            SemanticActionSeries(
                "Building_1",
                "deferrable_appliance_washer",
                (1.0, 0.0),
                unit="binary_start",
            ),
        ),
        metadata={
            "action_power_limits_kw": {
                "electrical_storage": {"max_charge_kw": 5.0, "max_discharge_kw": 5.0},
                "electric_vehicle_storage_charger_1": {
                    "max_charge_kw": 11.0,
                    "max_discharge_kw": 7.2,
                },
            }
        },
    )
    path = tmp_path / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    policy = TotalHomeOracleReplayPolicy(
        {
            "algorithm": {
                "hyperparameters": {
                    "schedule_path": str(path),
                    "local_action_safety_enabled": False,
                }
            }
        }
    )
    names = [[
        "electrical_storage",
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_washer",
    ]]
    policy.attach_environment(
        observation_names=[[]],
        action_names=names,
        action_space=[spaces.Box(low=-1.0, high=1.0, shape=(3,))],
        observation_space=[spaces.Box(low=-1.0, high=1.0, shape=(0,))],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    assert policy.predict_at_step([np.zeros(0)], schedule_step=0) == [[0.5, 0.5, 1.0]]
    assert policy.predict_at_step([np.zeros(0)], schedule_step=1) == [[-0.2, -0.5, 0.0]]


def test_total_home_replay_rejects_incomplete_schedule(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(
        SemanticSchedule(
            problem_id="incomplete",
            horizon=1,
            timestep_hours=1.0,
            series=(SemanticActionSeries("Building_1", "electrical_storage", (0.0,)),),
            metadata={
                "action_power_limits_kw": {
                    "electrical_storage": {"max_charge_kw": 1.0, "max_discharge_kw": 1.0}
                }
            },
        ).to_json(),
        encoding="utf-8",
    )
    policy = TotalHomeOracleReplayPolicy(
        {"algorithm": {"hyperparameters": {"schedule_path": str(path)}}}
    )

    try:
        policy.attach_environment(
            observation_names=[[]],
            action_names=[["electrical_storage", "electric_vehicle_storage_charger_1"]],
            action_space=[spaces.Box(low=-1.0, high=1.0, shape=(2,))],
            observation_space=[spaces.Box(low=-1.0, high=1.0, shape=(0,))],
            metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
        )
    except ValueError as error:
        assert "action mismatch" in str(error)
    else:
        raise AssertionError("Incomplete total-home schedule was accepted.")


def test_total_home_replay_repeats_only_when_explicitly_enabled_for_training(tmp_path):
    schedule = SemanticSchedule(
        problem_id="repeatable-training-teacher",
        horizon=2,
        timestep_hours=0.25,
        series=(
            SemanticActionSeries("Building_1", "electrical_storage", (2.5, -1.0)),
        ),
        metadata={
            "action_power_limits_kw": {
                "electrical_storage": {"max_charge_kw": 5.0, "max_discharge_kw": 5.0}
            }
        },
    )
    path = tmp_path / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    policy = TotalHomeOracleReplayPolicy(
        {
            "algorithm": {
                "hyperparameters": {
                    "schedule_path": str(path),
                    "local_action_safety_enabled": False,
                    "repeat_schedule_for_training": True,
                }
            }
        }
    )
    policy.attach_environment(
        observation_names=[[]],
        action_names=[["electrical_storage"]],
        action_space=[spaces.Box(low=-1.0, high=1.0, shape=(1,))],
        observation_space=[spaces.Box(low=-1.0, high=1.0, shape=(0,))],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 900},
    )

    assert policy.predict_at_step([np.zeros(0)], schedule_step=2) == [[0.5]]


def test_total_home_replay_applies_deadline_ev_floor_and_service_target_cap(tmp_path):
    building = "Building_4"
    charger = "charger_4_1"
    action_name = "electric_vehicle_storage_charger_4_1"
    schedule = SemanticSchedule(
        problem_id="ev-safety-controls",
        horizon=2,
        timestep_hours=0.25,
        series=(SemanticActionSeries(building, action_name, (0.0, 22.0)),),
        metadata={
            "action_power_limits_kw": {
                action_name: {"max_charge_kw": 22.0, "max_discharge_kw": 22.0}
            }
        },
    )
    path = tmp_path / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    policy = TotalHomeOracleReplayPolicy(
        {
            "algorithm": {
                "hyperparameters": {
                    "schedule_path": str(path),
                    "local_action_safety_ev_minimum_mode": "deadline_feasible",
                    "local_action_safety_protect_ev_service_target": True,
                }
            }
        }
    )
    prefix = f"charger::{building}/{charger}::"
    observation_names = [[
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}min_charging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
        f"{prefix}departure_energy_margin_kwh",
        f"{prefix}available_charge_power_kw",
        f"{prefix}charger_efficiency_ratio",
        f"{prefix}energy_to_required_soc_kwh",
    ]]
    policy.attach_environment(
        observation_names=observation_names,
        action_names=[[action_name]],
        action_space=[spaces.Box(low=-1.0, high=1.0, shape=(1,))],
        observation_space=[spaces.Box(low=-np.inf, high=np.inf, shape=(11,))],
        metadata={"building_names": [building], "seconds_per_time_step": 900},
    )

    # An average floor would force 0.4, but enough future capacity remains.
    deferred = [1.0, 22.0, 22.0, 3.7, 1.0, 1.0, 0.4, 6.0, 22.0, 0.9, 20.0]
    assert policy.predict_at_step([deferred], schedule_step=0)[0] == pytest.approx([0.0])

    # The 22 kW oracle request is capped at the 1.2375 kWh still required.
    capped = [1.0, 22.0, 22.0, 3.7, 1.0, 1.0, 0.0, 6.0, 22.0, 0.9, 1.2375]
    assert policy.predict_at_step([capped], schedule_step=1)[0] == pytest.approx([0.25])


def test_total_home_replay_exports_safety_configuration_with_compatible_defaults(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(
        SemanticSchedule(
            problem_id="defaults",
            horizon=1,
            timestep_hours=1.0,
            series=(SemanticActionSeries("Building_1", "electrical_storage", (0.0,)),),
            metadata={
                "action_power_limits_kw": {
                    "electrical_storage": {
                        "max_charge_kw": 1.0,
                        "max_discharge_kw": 1.0,
                    }
                }
            },
        ).to_json(),
        encoding="utf-8",
    )
    policy = TotalHomeOracleReplayPolicy(
        {"algorithm": {"hyperparameters": {"schedule_path": str(path)}}}
    )
    policy.attach_environment(
        observation_names=[[]],
        action_names=[["electrical_storage"]],
        action_space=[spaces.Box(low=-1.0, high=1.0, shape=(1,))],
        observation_space=[spaces.Box(low=-1.0, high=1.0, shape=(0,))],
        metadata={"building_names": ["Building_1"], "seconds_per_time_step": 3600},
    )

    parameters = policy.export_artifacts(str(tmp_path / "artifacts"))["parameters"]

    assert parameters["local_action_safety_fail_on_infeasible"] is False
    assert parameters["local_action_safety_protect_ev_minimum"] is True
    assert parameters["local_action_safety_ev_minimum_mode"] == "average"
    assert parameters["local_action_safety_protect_ev_service_target"] is False
    assert parameters["local_action_safety_protect_deferrable_must_start"] is True
    assert parameters["local_action_safety_allow_discretionary_deferrable_start"] is True
    assert parameters["local_action_safety_headroom_reserve_kw"] == 0.0


def test_prepare_total_home_replay_uses_deadline_floor_and_target_cap(tmp_path):
    schedule_path = tmp_path / "schedule.json"
    replay = _config(
        dataset_path=tmp_path / "schema.json",
        building="Building_1",
        start=0,
        end=4,
        session_name="replay",
        algorithm="TotalHomeOracleReplayPolicy",
        schedule_path=schedule_path,
    )

    hyperparameters = replay["pipeline"][0]["hyperparameters"]
    assert hyperparameters["local_action_safety_ev_minimum_mode"] == "deadline_feasible"
    assert hyperparameters["local_action_safety_protect_ev_service_target"] is True


def test_validate_dump_and_policy_preserve_total_home_safety_controls(tmp_path):
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        SemanticSchedule(
            problem_id="resolved-config",
            horizon=4,
            timestep_hours=0.25,
            series=(SemanticActionSeries("Building_1", "electrical_storage", (0.0,) * 4),),
            metadata={
                "action_power_limits_kw": {
                    "electrical_storage": {
                        "max_charge_kw": 1.0,
                        "max_discharge_kw": 1.0,
                    }
                }
            },
        ).to_json(),
        encoding="utf-8",
    )
    raw_config = _config(
        dataset_path=tmp_path / "schema.json",
        building="Building_1",
        start=0,
        end=4,
        session_name="resolved-config",
        algorithm="TotalHomeOracleReplayPolicy",
        schedule_path=schedule_path,
    )
    expected = {
        "repeat_schedule_for_training": True,
        "local_action_safety_enabled": True,
        "local_action_safety_fail_on_infeasible": True,
        "local_action_safety_protect_ev_minimum": False,
        "local_action_safety_ev_minimum_mode": "deadline_feasible",
        "local_action_safety_protect_ev_service_target": True,
        "local_action_safety_protect_deferrable_must_start": False,
        "local_action_safety_allow_discretionary_deferrable_start": False,
        "local_action_safety_headroom_reserve_kw": 0.25,
    }
    raw_config["pipeline"][0]["hyperparameters"].update(expected)

    resolved = validate_config(raw_config).to_dict()
    # Mirror run_experiment's config.resolved.yaml serialization boundary.
    dumped = yaml.safe_load(yaml.safe_dump(resolved, sort_keys=False))
    resolved_hyperparameters = dumped["pipeline"][0]["hyperparameters"]

    assert {key: resolved_hyperparameters[key] for key in expected} == expected

    policy = TotalHomeOracleReplayPolicy(
        {"algorithm": {"hyperparameters": resolved_hyperparameters}}
    )
    artifact_parameters = policy.export_artifacts(str(tmp_path / "artifacts"))["parameters"]
    assert {key: artifact_parameters[key] for key in expected} == expected


def test_validate_config_rejects_unknown_total_home_ev_minimum_mode(tmp_path):
    raw_config = _config(
        dataset_path=tmp_path / "schema.json",
        building="Building_1",
        start=0,
        end=4,
        session_name="invalid-mode",
        algorithm="TotalHomeOracleReplayPolicy",
        schedule_path=tmp_path / "schedule.json",
    )
    raw_config["pipeline"][0]["hyperparameters"][
        "local_action_safety_ev_minimum_mode"
    ] = "optimistic"

    with pytest.raises(Exception, match="deadline_feasible"):
        validate_config(raw_config)
