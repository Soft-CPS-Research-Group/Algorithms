"""Measure export overhead on identical jobs, with no network tracking or model export."""

import argparse
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path

import yaml

from algorithms.runtime.atomic import atomic_json
from algorithms.runtime.job import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    base = yaml.safe_load(
        Path("configs/templates/experimental/transformer_maddpg_smoke.yaml").read_text()
    )
    base["pipeline"] = [{"algorithm": "NormalPolicy"}]
    base["outputs"] = {"model_artifacts": "none"}
    base["training"]["episodes"] = 2
    base["evaluation"] = {"episodes": 1}
    base["tracking"].update(
        mlflow_enabled=False,
        metrics_enabled=False,
        log_level="ERROR",
        log_frequency=100000,
        progress_update_interval=100000,
    )
    rows = []
    modes = {
        "final_kpis": {
            "profile": "final",
            "mode": "none",
            "export_business_as_usual_timeseries": False,
        },
        "previous_final_timeseries": {
            "profile": "final",
            "mode": "end",
            "export_business_as_usual_timeseries": False,
        },
        "diagnostic_all": {
            "profile": "diagnostic",
            "mode": "end",
            "final_episode_only": False,
            "kpis_final_episode_only": False,
            "timeseries_final_episode_only": False,
            "export_business_as_usual_timeseries": True,
        },
    }
    for label, horizon in (("small", 24), ("representative", 168), ("large", 720)):
        for mode, export in modes.items():
            for repetition in range(args.repetitions):
                config = deepcopy(base)
                config["simulator"].update(
                    episode_time_steps=horizon + 1, simulation_end_time_step=horizon
                )
                config["simulator"]["export"].update(export)
                identifier = f"{label}-{mode}-{repetition}"
                started = time.perf_counter()
                run_experiment(config, identifier, args.output)
                wall = time.perf_counter() - started
                result = json.loads(
                    (args.output / "jobs" / identifier / "results/finalization.json").read_text()
                )
                row = {
                    "size": label,
                    "steps_per_episode": horizon,
                    "profile": mode,
                    "repetition": repetition,
                    "wall_seconds": wall,
                    **result,
                }
                rows.append(row)
                atomic_json(args.output / "measurements.json", {"rows": rows})
                print(f"{identifier}: {wall:.3f}s, {result['simulation_bytes']} bytes", flush=True)
    summary = []
    for label in ("small", "representative", "large"):
        for mode in modes:
            selected = [row for row in rows if row["size"] == label and row["profile"] == mode]
            summary.append(
                {
                    "size": label,
                    "profile": mode,
                    "median_wall_seconds": statistics.median(
                        row["wall_seconds"] for row in selected
                    ),
                    "median_simulation_bytes": statistics.median(
                        row["simulation_bytes"] for row in selected
                    ),
                }
            )
    atomic_json(
        args.output / "measurements.json",
        {
            "version": 1,
            "agent": "NormalPolicy",
            "building_ids": base["simulator"]["building_ids"],
            "training_episodes": 2,
            "evaluation_episodes": 1,
            "tracking": "disabled",
            "model_export": "disabled",
            "rows": rows,
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
