"""Audit stationary battery behavior from exported simulation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _schema_rows(schema_path: Path) -> list[dict[str, Any]]:
    schema = _read_json(schema_path)
    rows: list[dict[str, Any]] = []
    for building, data in sorted((schema.get("buildings") or {}).items()):
        storage = ((data or {}).get("electrical_storage") or {}).get("attributes") or {}
        if not storage:
            continue
        rows.append(
            {
                "dataset": schema_path.parent.name,
                "schema_path": str(schema_path),
                "building": building,
                "capacity_kwh": storage.get("capacity"),
                "efficiency": storage.get("efficiency"),
                "capacity_loss_coefficient": storage.get("capacity_loss_coefficient"),
                "loss_coefficient": storage.get("loss_coefficient"),
                "initial_soc": storage.get("initial_soc"),
                "nominal_power_kw": storage.get("nominal_power", storage.get("power")),
                "phase_connection": storage.get("phase_connection"),
            }
        )
    return rows


def _storage_rows(run_name: str, data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("exported_data_building_*_battery_ep*.csv")):
        df = pd.read_csv(path)
        if "Battery (Dis)Charge-kWh" not in df.columns:
            continue
        building = path.name.split("_battery_")[0].replace("exported_data_", "")
        episode = path.stem.split("_ep")[-1]
        energy = pd.to_numeric(df["Battery (Dis)Charge-kWh"], errors="coerce").fillna(0.0)
        soc = (
            pd.to_numeric(df["Battery Soc-%"], errors="coerce")
            if "Battery Soc-%" in df.columns
            else pd.Series(dtype=float)
        )
        charge_kwh = float(energy[energy > 1.0e-9].sum())
        discharge_kwh = float(-energy[energy < -1.0e-9].sum())
        throughput_kwh = float(energy.abs().sum())
        idle_mask = energy.abs() <= 1.0e-9
        idle_soc_delta = 0.0
        if len(soc) > 1:
            idle_soc_delta = float(soc.diff().abs().fillna(0.0)[idle_mask].sum())
        rows.append(
            {
                "run": run_name,
                "data_dir": str(data_dir),
                "building": building,
                "episode": int(episode) if episode.isdigit() else episode,
                "rows": len(df),
                "charge_kwh": charge_kwh,
                "discharge_kwh": discharge_kwh,
                "throughput_kwh": throughput_kwh,
                "idle_fraction": float(idle_mask.mean()) if len(idle_mask) else 1.0,
                "soc_min": float(soc.min()) if len(soc) else 0.0,
                "soc_max": float(soc.max()) if len(soc) else 0.0,
                "soc_final": float(soc.iloc[-1]) if len(soc) else 0.0,
                "idle_soc_abs_delta_sum": idle_soc_delta,
            }
        )
    return rows


def _sum_exported_data(data_dir: Path, pattern: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for path in sorted(data_dir.glob(pattern)):
        df = pd.read_csv(path)
        for column in df.columns:
            if column == "timestamp":
                continue
            values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            totals[column] = totals.get(column, 0.0) + float(values.sum())
    return totals


def _run_summary(run_name: str, data_dir: Path, storage: pd.DataFrame) -> dict[str, Any]:
    subset = storage[storage["run"] == run_name]
    community = _sum_exported_data(data_dir, "exported_data_community_ep*.csv")
    pricing = _sum_exported_data(data_dir, "exported_data_pricing_ep*.csv")
    row: dict[str, Any] = {
        "run": run_name,
        "data_dir": str(data_dir),
        "battery_files": int(len(subset)),
        "episodes": int(subset["episode"].nunique()) if len(subset) else 0,
        "charge_kwh": float(subset["charge_kwh"].sum()) if len(subset) else 0.0,
        "discharge_kwh": float(subset["discharge_kwh"].sum()) if len(subset) else 0.0,
        "throughput_kwh": float(subset["throughput_kwh"].sum()) if len(subset) else 0.0,
        "storage_idle_fraction_mean": float(subset["idle_fraction"].mean()) if len(subset) else 1.0,
        "soc_min": float(subset["soc_min"].min()) if len(subset) else 0.0,
        "soc_max": float(subset["soc_max"].max()) if len(subset) else 0.0,
        "soc_final_mean": float(subset["soc_final"].mean()) if len(subset) else 0.0,
        "idle_soc_abs_delta_sum": float(subset["idle_soc_abs_delta_sum"].sum()) if len(subset) else 0.0,
        "buildings_with_storage_use": int((subset["throughput_kwh"] > 1.0e-9).sum()) if len(subset) else 0,
    }
    for key in (
        "Net Electricity Consumption-kWh",
        "Self Consumption-kWh",
        "Stored energy by community- kWh",
        "Total Solar Generation-kWh",
        "Price-$",
        "CO2-kg_co2",
    ):
        row[f"community__{key}"] = community.get(key, 0.0)
    for key, value in pricing.items():
        row[f"pricing__{key}"] = value
    return row


def _benchmark_rows(paths: list[Path]) -> list[dict[str, Any]]:
    columns = [
        "dataset_key",
        "agent_key",
        "variant_label",
        "kpi__community_settled_cost_total_eur",
        "metric__Action_storage_positive_fraction__mean",
        "metric__Action_storage_negative_fraction__mean",
        "metric__Action_storage_idle_fraction__mean",
        "metric__RewardComponent_battery_safety_penalty_mean__mean",
        "metric__RewardComponent_battery_throughput_penalty_mean__mean",
        "metric__RewardComponent_local_import_cost_mean__mean",
        "metric__RewardComponent_community_settlement_cost_mean__mean",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, record in df.iterrows():
            row = {"benchmark_summary": str(path)}
            for column in columns:
                row[column] = record[column] if column in df.columns else None
            rows.append(row)
    return rows


def _parse_run_argument(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema", type=Path, action="append", default=[])
    parser.add_argument("--run", action="append", default=[], help="name=simulation_data_dir")
    parser.add_argument("--benchmark-summary", type=Path, action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema = pd.DataFrame([row for path in args.schema for row in _schema_rows(path)])
    storage = pd.DataFrame([row for raw in args.run for row in _storage_rows(*_parse_run_argument(raw))])
    summaries = pd.DataFrame(
        [
            _run_summary(name, path, storage)
            for name, path in (_parse_run_argument(raw) for raw in args.run)
        ]
    )
    benchmark = pd.DataFrame(_benchmark_rows(args.benchmark_summary))

    schema.to_csv(args.output_dir / "dataset_storage_schema.csv", index=False)
    storage.to_csv(args.output_dir / "storage_by_building_episode.csv", index=False)
    summaries.to_csv(args.output_dir / "storage_run_summary.csv", index=False)
    benchmark.to_csv(args.output_dir / "storage_reward_metric_summary.csv", index=False)

    readme = [
        "# Storage Behavior Audit",
        "",
        "Generated files:",
        "",
        "- `dataset_storage_schema.csv`",
        "- `storage_by_building_episode.csv`",
        "- `storage_run_summary.csv`",
        "- `storage_reward_metric_summary.csv`",
        "",
        "Interpretation notes:",
        "",
        "- `idle_soc_abs_delta_sum` should stay near zero when the battery action is zero; non-zero values suggest standby loss or dataset/simulator drift while idle.",
        "- `Battery (Dis)Charge-kWh` positive values are exported as stored energy; negative values are exported as self-consumption/discharge.",
        "- Reward metrics are sampled training metrics, not exported KPI totals.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
