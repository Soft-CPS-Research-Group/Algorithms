from __future__ import annotations

import json
from pathlib import Path


ENTITY_OBSERVATION_BUNDLES = (
    "entity_core_electrical",
    "entity_community_operational",
    "entity_forecasts_existing",
    "entity_temporal_derived",
)


def _assert_entity_schema_uses_all_observation_bundles(schema: dict) -> None:
    assert schema["interface"] == "entity"
    assert schema["ev_departure_within_tolerance"] == 0.05
    assert schema["ev_departure_service_tolerance"] == 0.05

    bundles = schema["observation_bundles"]
    for bundle_name in ENTITY_OBSERVATION_BUNDLES:
        assert bundles[bundle_name]["active"] is True


def test_15s_parquet_dataset_uses_deferrable_appliance_names():
    dataset_dir = Path("datasets/citylearn_three_phase_electrical_service_demo_15s_parquet")
    schema_path = dataset_dir / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_text = json.dumps(schema)

    _assert_entity_schema_uses_all_observation_bundles(schema)
    assert schema["root_directory"] == f"./{dataset_dir.as_posix()}"
    assert "washing_machine" not in schema_text
    assert "washer" not in schema_text
    assert "dryer" not in schema_text

    appliances = schema["buildings"]["Building_1"]["deferrable_appliances"]
    appliance = appliances["deferrable_appliance_1"]
    cycle_profiles = dataset_dir / appliance["cycle_profiles_file"]
    flexibility_schedule = dataset_dir / appliance["flexibility_schedule_file"]

    assert cycle_profiles.name == "deferrable_appliance_1_cycle_profiles.parquet"
    assert flexibility_schedule.name == "deferrable_appliance_1_flexibility_schedule.parquet"
    assert cycle_profiles.is_file()
    assert flexibility_schedule.is_file()


def test_2022_all_plus_evs_dataset_uses_entity_interface_and_all_bundles():
    schema_path = Path("datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    _assert_entity_schema_uses_all_observation_bundles(schema)
    assert schema["seconds_per_time_step"] == 3600
