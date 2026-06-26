from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ENTITY_OBSERVATION_BUNDLES = (
    "entity_core_electrical",
    "entity_community_operational",
    "entity_forecasts_existing",
    "entity_forecasts_derived",
    "entity_temporal_derived",
    "entity_action_feedback",
)


def _assert_ev_countdowns_match_15min_source(source_dir: Path, dataset_dir: Path) -> None:
    for source_charger in source_dir.glob("charger_*.parquet"):
        resampled_charger = dataset_dir / source_charger.name
        source = pd.read_parquet(source_charger)
        resampled = pd.read_parquet(resampled_charger)

        change_columns = [
            column
            for column in ("electric_vehicle_charger_state", "electric_vehicle_id")
            if column in source
        ]
        if change_columns:
            changes = source[change_columns].ne(source[change_columns].shift()).any(axis=1)
            change_indices = source.index[changes & (source.index != 0)]
            assert all(int(index) % 60 == 0 for index in change_indices)

        sampled_source = source.iloc[::60].reset_index(drop=True)
        for column in ("electric_vehicle_departure_time", "electric_vehicle_estimated_arrival_time"):
            if column not in source:
                continue

            expected = sampled_source[column].map(
                lambda value: value if pd.isna(value) or float(value) < 0 else float(math.ceil(float(value) / 60.0))
            )
            pd.testing.assert_series_equal(
                resampled[column].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
            )


def _assert_entity_schema_uses_all_observation_bundles(schema: dict) -> None:
    if "interface" in schema:
        assert schema["interface"] == "entity"
    if "ev_departure_within_tolerance" in schema:
        assert schema["ev_departure_within_tolerance"] == 0.05
    if "ev_departure_service_tolerance" in schema:
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
    assert schema["root_directory"] == "./datasets/citylearn_challenge_2022_phase_all_plus_evs"


def test_dynamic_15s_parquet_datasets_use_all_entity_bundles():
    for dataset_name in (
        "citylearn_three_phase_dynamic_asset_changes_demo_15s_parquet",
        "citylearn_three_phase_dynamic_assets_only_demo_15s_parquet",
    ):
        dataset_dir = Path("datasets") / dataset_name
        schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))

        _assert_entity_schema_uses_all_observation_bundles(schema)
        assert schema["seconds_per_time_step"] == 15
        assert schema["root_directory"] == f"./{dataset_dir.as_posix()}"


def test_15min_parquet_datasets_use_all_entity_bundles_and_full_year_resolution():
    for dataset_name in (
        "citylearn_three_phase_electrical_service_demo_15min_parquet",
        "citylearn_three_phase_dynamic_assets_only_demo_15min_parquet",
    ):
        dataset_dir = Path("datasets") / dataset_name
        schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))

        _assert_entity_schema_uses_all_observation_bundles(schema)
        assert schema["seconds_per_time_step"] == 900
        assert schema["simulation_start_time_step"] == 0
        assert schema["simulation_end_time_step"] == 35039
        assert schema["root_directory"] == f"./{dataset_dir.as_posix()}"

        building = pd.read_parquet(dataset_dir / "Building_1.parquet")
        charger = pd.read_parquet(dataset_dir / "charger_1_1.parquet")
        assert len(building) == 35040
        assert len(charger) == 35040
        assert building["seconds"].isin([0]).all()
        assert set(building["minutes"].unique()).issubset({0, 15, 30, 45})

    _assert_ev_countdowns_match_15min_source(
        Path("datasets/citylearn_three_phase_electrical_service_demo_15s_parquet"),
        Path("datasets/citylearn_three_phase_electrical_service_demo_15min_parquet"),
    )
    _assert_ev_countdowns_match_15min_source(
        Path("datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet"),
        Path("datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet"),
    )

    static_dir = Path("datasets/citylearn_three_phase_electrical_service_demo_15min_parquet")
    profiles = pd.read_parquet(static_dir / "deferrable_appliance_1_cycle_profiles.parquet")
    schedule = pd.read_parquet(static_dir / "deferrable_appliance_1_flexibility_schedule.parquet")
    assert set(profiles["duration_steps"].unique()).issubset({4, 8, 12, 16})
    assert (schedule["latest_start_time_step"] >= schedule["earliest_start_time_step"]).all()


def test_algorithm_dataset_variants_use_all_entity_bundles():
    for dataset_name in (
        "citylearn_three_phase_electrical_service_demo_15s_parquet_no_v2g_data_2026_05_20",
        "citylearn_three_phase_electrical_service_demo_15s_parquet_multi_charger_data_2026_05_20",
        "citylearn_challenge_2022_phase_all_plus_evs_no_v2g_data_2026_05_20",
        "citylearn_challenge_2022_phase_all_plus_evs_multi_charger_data_2026_05_20",
    ):
        schema = json.loads((Path("datasets") / dataset_name / "schema.json").read_text(encoding="utf-8"))
        _assert_entity_schema_uses_all_observation_bundles(schema)
