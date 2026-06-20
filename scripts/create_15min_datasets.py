from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_SECONDS_PER_STEP = 15
TARGET_SECONDS_PER_STEP = 900
RESAMPLE_FACTOR = TARGET_SECONDS_PER_STEP // SOURCE_SECONDS_PER_STEP

DATASETS = {
    "static": (
        "citylearn_three_phase_electrical_service_demo_15s_parquet",
        "citylearn_three_phase_electrical_service_demo_15min_parquet",
    ),
    "dynamic_assets_only": (
        "citylearn_three_phase_dynamic_assets_only_demo_15s_parquet",
        "citylearn_three_phase_dynamic_assets_only_demo_15min_parquet",
    ),
}

ENERGY_COLUMNS = {
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
}

TIME_INDEX_COLUMNS = {
    "earliest_start_time_step",
    "latest_start_time_step",
    "deadline_time_step",
    "time_step",
    "start_time_step",
    "end_time_step",
}

COUNTDOWN_COLUMNS = {
    "electric_vehicle_departure_time",
    "electric_vehicle_estimated_arrival_time",
}


def _read_schema(dataset_dir: Path) -> dict[str, Any]:
    return json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))


def _write_schema(schema: dict[str, Any], dataset_dir: Path) -> None:
    (dataset_dir / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _group_ids(length: int) -> pd.Series:
    if length % RESAMPLE_FACTOR != 0:
        raise ValueError(f"Expected row count to divide by {RESAMPLE_FACTOR}, got {length}")

    return pd.Series(range(length), dtype="int64") // RESAMPLE_FACTOR


def _aggregate_general_timeseries(dataframe: pd.DataFrame) -> pd.DataFrame:
    grouped = dataframe.groupby(_group_ids(len(dataframe)), sort=False)
    columns: dict[str, pd.Series] = {}

    for column in dataframe.columns:
        if column in ENERGY_COLUMNS:
            columns[column] = grouped[column].sum()
        else:
            columns[column] = grouped[column].first()

    return pd.DataFrame(columns).reset_index(drop=True)


def _ceil_divide_countdown(value: Any) -> Any:
    if pd.isna(value):
        return value

    value = float(value)
    if value < 0:
        return value

    return float(math.ceil(value / RESAMPLE_FACTOR))


def _aggregate_charger_timeseries(dataframe: pd.DataFrame, file_name: str) -> pd.DataFrame:
    change_columns = [c for c in ("electric_vehicle_charger_state", "electric_vehicle_id") if c in dataframe]
    if change_columns:
        changes = dataframe[change_columns].ne(dataframe[change_columns].shift()).any(axis=1)
        change_indices = dataframe.index[changes & (dataframe.index != 0)]
        non_aligned = [int(i) for i in change_indices if int(i) % RESAMPLE_FACTOR != 0]
        if non_aligned:
            sample = ", ".join(map(str, non_aligned[:10]))
            raise ValueError(f"{file_name}: EV schedule has changes inside 15min windows: {sample}")

    grouped = dataframe.groupby(_group_ids(len(dataframe)), sort=False)
    columns: dict[str, pd.Series] = {}

    for column in dataframe.columns:
        series = grouped[column].first()
        if column in COUNTDOWN_COLUMNS:
            series = series.map(_ceil_divide_countdown)
        columns[column] = series

    return pd.DataFrame(columns).reset_index(drop=True)


def _parse_profile(value: Any) -> list[float]:
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]

    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]

    if hasattr(value, "tolist"):
        return [float(v) for v in value.tolist()]

    raise TypeError(f"Unsupported load_profile value type: {type(value)!r}")


def _aggregate_profile_values(values: list[float]) -> list[float]:
    if len(values) % RESAMPLE_FACTOR != 0:
        raise ValueError(f"Deferrable profile length must divide by {RESAMPLE_FACTOR}, got {len(values)}")

    return [
        float(sum(values[start : start + RESAMPLE_FACTOR]))
        for start in range(0, len(values), RESAMPLE_FACTOR)
    ]


def _resample_deferrable_profiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    new_profiles: list[str] = []
    new_durations: list[int] = []
    new_totals: list[float] = []

    for _, row in output.iterrows():
        profile = _aggregate_profile_values(_parse_profile(row["load_profile"]))
        new_profiles.append(json.dumps(profile, separators=(",", ":")))
        new_durations.append(len(profile))
        new_totals.append(float(sum(profile)))

    output["duration_steps"] = new_durations
    output["total_energy_kwh"] = new_totals
    output["load_profile"] = new_profiles
    return output


def _convert_start_index(value: Any) -> int:
    return int(math.ceil(float(value) / RESAMPLE_FACTOR))


def _convert_latest_start_index(value: Any) -> int:
    return int(math.floor(float(value) / RESAMPLE_FACTOR))


def _convert_inclusive_end_index(value: Any) -> int:
    return int(math.floor((float(value) + 1.0) / RESAMPLE_FACTOR) - 1)


def _resample_deferrable_schedule(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output["earliest_start_time_step"] = output["earliest_start_time_step"].map(_convert_start_index)
    output["latest_start_time_step"] = output["latest_start_time_step"].map(_convert_latest_start_index)
    output["deadline_time_step"] = output["deadline_time_step"].map(_convert_inclusive_end_index)

    invalid = output["latest_start_time_step"] < output["earliest_start_time_step"]
    if invalid.any():
        bad_cycles = ", ".join(output.loc[invalid, "cycle_id"].astype(str).head(10))
        raise ValueError(f"Invalid 15min deferrable windows after resampling: {bad_cycles}")

    return output


def _resample_parquet(source: Path, destination: Path) -> tuple[int | None, int | None]:
    dataframe = pd.read_parquet(source)

    if source.name.endswith("_cycle_profiles.parquet") and "load_profile" in dataframe:
        output = _resample_deferrable_profiles(dataframe)
    elif source.name.endswith("_flexibility_schedule.parquet"):
        output = _resample_deferrable_schedule(dataframe)
    elif source.name.startswith("charger_"):
        output = _aggregate_charger_timeseries(dataframe, source.name)
    else:
        output = _aggregate_general_timeseries(dataframe)

    output.to_parquet(destination, index=False, compression="zstd")
    return len(dataframe), len(output)


def _convert_time_step(value: Any, *, inclusive_end: bool = False) -> Any:
    if value is None:
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return _convert_inclusive_end_index(value) if inclusive_end else _convert_start_index(value)

    if isinstance(value, float) and value.is_integer():
        return _convert_inclusive_end_index(value) if inclusive_end else _convert_start_index(value)

    return value


def _update_nested_time_steps(value: Any) -> Any:
    if isinstance(value, list):
        return [_update_nested_time_steps(item) for item in value]

    if isinstance(value, dict):
        updated: dict[str, Any] = {}
        for key, item in value.items():
            if key == "time_step" or key == "start_time_step":
                updated[key] = _convert_time_step(item)
            elif key == "end_time_step":
                updated[key] = _convert_time_step(item, inclusive_end=True)
            elif key in TIME_INDEX_COLUMNS:
                updated[key] = _convert_time_step(item)
            else:
                updated[key] = _update_nested_time_steps(item)
        return updated

    return value


def _update_schema(
    schema: dict[str, Any],
    *,
    source_name: str,
    destination_dir: Path,
    output_rows: int,
) -> dict[str, Any]:
    updated = _update_nested_time_steps(schema)
    updated["root_directory"] = f"./{destination_dir.as_posix()}"
    updated["seconds_per_time_step"] = TARGET_SECONDS_PER_STEP
    updated["simulation_start_time_step"] = 0
    updated["simulation_end_time_step"] = output_rows - 1
    updated["_derived_from"] = source_name
    updated["_derivation"] = {
        "type": "15s_to_15min_resample",
        "source_dataset": f"./datasets/{source_name}",
        "source_seconds_per_time_step": SOURCE_SECONDS_PER_STEP,
        "target_seconds_per_time_step": TARGET_SECONDS_PER_STEP,
        "aggregation_factor": RESAMPLE_FACTOR,
        "energy_columns": sorted(ENERGY_COLUMNS),
        "countdown_columns": sorted(COUNTDOWN_COLUMNS),
    }
    return updated


def create_dataset(source_dir: Path, destination_dir: Path, *, overwrite: bool) -> None:
    if destination_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{destination_dir} already exists; pass --overwrite to regenerate it.")
        shutil.rmtree(destination_dir)

    destination_dir.mkdir(parents=True)

    schema = _read_schema(source_dir)
    if int(schema["seconds_per_time_step"]) != SOURCE_SECONDS_PER_STEP:
        raise ValueError(f"{source_dir} is not a {SOURCE_SECONDS_PER_STEP}s dataset.")

    output_rows: int | None = None
    for source_file in sorted(source_dir.iterdir()):
        if source_file.name == "schema.json":
            continue

        destination_file = destination_dir / source_file.name
        if source_file.suffix == ".parquet":
            input_count, file_output_rows = _resample_parquet(source_file, destination_file)
            if input_count and input_count % RESAMPLE_FACTOR == 0:
                if output_rows is None or file_output_rows > output_rows:
                    output_rows = file_output_rows
        else:
            shutil.copy2(source_file, destination_file)

    if output_rows is None:
        raise RuntimeError(f"No timeseries parquet files were converted for {source_dir}.")

    updated_schema = _update_schema(
        schema,
        source_name=source_dir.name,
        destination_dir=destination_dir,
        output_rows=output_rows,
    )
    _write_schema(updated_schema, destination_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 15-minute parquet datasets from tracked 15-second datasets.")
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        type=Path,
        help="Directory containing source datasets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated 15-minute dataset directories.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=f"Dataset variants to generate: {', '.join(sorted(DATASETS))}.",
    )
    args = parser.parse_args()

    dataset_keys = args.datasets or sorted(DATASETS)
    unknown = sorted(set(dataset_keys) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset variant(s): {', '.join(unknown)}")

    for dataset_key in dataset_keys:
        source_name, destination_name = DATASETS[dataset_key]
        source_dir = args.dataset_root / source_name
        destination_dir = args.dataset_root / destination_name
        print(f"{source_name} -> {destination_name}")
        create_dataset(source_dir, destination_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
