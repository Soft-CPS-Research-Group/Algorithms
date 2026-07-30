#!/usr/bin/env python3
"""Prepare a single-building schema plus matched MILP/RBC replay configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.oracles import SemanticSchedule


def _config(
    *,
    dataset_path: Path,
    building: str,
    start: int,
    end: int,
    session_name: str,
    algorithm: str,
    schedule_path: Path | None,
) -> dict:
    hyperparameters = {}
    description = "Matched strict-local RBCSmart baseline for a total-home MILP replay."
    if schedule_path is not None:
        hyperparameters = {
            "schedule_path": str(schedule_path.resolve()),
            "local_action_safety_enabled": True,
            "local_action_safety_ev_minimum_mode": "deadline_feasible",
            "local_action_safety_protect_ev_service_target": True,
            "local_action_safety_headroom_reserve_kw": 0.0,
        }
        description = "Teacher-free CityLearn replay of a total-home individual MILP schedule."
    return {
        "metadata": {
            "experiment_name": "individual_total_home_milp_replay_20260730",
            "run_name": session_name,
            "community_name": f"single_{building}",
            "description": description,
        },
        "tracking": {
            "mlflow_enabled": False,
            "log_level": "INFO",
            "log_frequency": 128,
            "progress_update_interval": 128,
            "action_diagnostics_enabled": True,
            "reward_diagnostics_enabled": True,
            "tags": {
                "benchmark_track": "building_local_retail",
                "controller_scope": "one_building_local",
                "oracle_scope": "individual_total_home_linear_milp" if schedule_path else "matched_rbcsmart_local",
                "perfect_foresight": bool(schedule_path),
                "service_teacher": False,
            },
        },
        "checkpointing": {"resume_training": False},
        "simulator": {
            "dataset_name": f"single_{building}_total_home_replay",
            "dataset_path": str(dataset_path.resolve()),
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "static",
            "entity_encoding": {
                "enabled": True,
                "normalization": "minmax_space",
                "profile": "building_local_v1",
                "clip": True,
            },
            "community_market": {
                "enabled": False,
                "local_price_ratio_to_grid_import": 0.8,
                "intra_community_sell_ratio": 0.8,
                "grid_export_price": 0.0,
                "import_member_weights": {},
                "kpis": {
                    "community_local_traded_enabled": False,
                    "community_self_consumption_enabled": True,
                },
            },
            "reward_function": "LocalScorecardGuardRewardV2",
            "reward_function_kwargs": {"reward_scale": 0.01},
            "episodes": 1,
            "deterministic_finish": True,
            "simulation_start_time_step": start,
            "simulation_end_time_step": end,
            "episode_time_steps": end - start + 1,
            "export": {
                "mode": "end",
                "export_kpis_on_episode_end": True,
                "final_episode_only": True,
                "kpis_final_episode_only": True,
                "timeseries_final_episode_only": True,
                "include_business_as_usual": True,
                "export_business_as_usual_timeseries": False,
                "kpi_round_decimals": None,
                "session_name": session_name,
            },
        },
        "training": {
            "seed": 123,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "topology": {
            "num_agents": None,
            "observation_dimensions": None,
            "action_dimensions": None,
            "action_space": None,
        },
        "pipeline": [
            {
                "algorithm": algorithm,
                "count": 1,
                "hyperparameters": hyperparameters,
            }
        ],
        "execution": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-schema", type=Path, required=True)
    parser.add_argument("--building", required=True)
    parser.add_argument("--start-time-step", type=int, required=True)
    parser.add_argument("--end-time-step", type=int, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_schema.resolve()
    schema = json.loads(source_path.read_text(encoding="utf-8"))
    if args.building not in (schema.get("buildings") or {}):
        raise ValueError(f"Unknown building {args.building!r}.")
    schedule = SemanticSchedule.from_json(args.schedule.read_text(encoding="utf-8"))
    if schedule.horizon != args.end_time_step - args.start_time_step:
        raise ValueError("Schedule horizon must equal end_time_step - start_time_step.")
    schedule_buildings = {series.building_id for series in schedule.series}
    if schedule_buildings != {args.building}:
        raise ValueError("Schedule must contain exactly the selected building.")

    raw_root = Path(str(schema.get("root_directory") or source_path.parent))
    if not raw_root.is_absolute():
        raw_root = raw_root.resolve()
    schema["root_directory"] = str(raw_root)
    for building_name, building in schema["buildings"].items():
        building["include"] = building_name == args.building
        for key in ("energy_simulation", "weather", "carbon_intensity", "pricing"):
            if building.get(key):
                building[key] = str((raw_root / str(building[key])).resolve())
        for charger in (building.get("chargers") or {}).values():
            if charger.get("charger_simulation"):
                charger["charger_simulation"] = str(
                    (raw_root / str(charger["charger_simulation"])).resolve()
                )
        for appliance in (building.get("deferrable_appliances") or {}).values():
            for key in ("cycle_profiles_file", "flexibility_schedule_file"):
                if appliance.get(key):
                    appliance[key] = str((raw_root / str(appliance[key])).resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived_schema = args.output_dir / f"schema_{args.building.lower()}.json"
    derived_schema.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replay_session = f"total-home-milp-replay-{args.building.lower()}-{args.start_time_step}-{args.end_time_step}"
    baseline_session = f"rbcsmart-matched-{args.building.lower()}-{args.start_time_step}-{args.end_time_step}"
    replay = _config(
        dataset_path=derived_schema,
        building=args.building,
        start=args.start_time_step,
        end=args.end_time_step,
        session_name=replay_session,
        algorithm="TotalHomeOracleReplayPolicy",
        schedule_path=args.schedule,
    )
    baseline = _config(
        dataset_path=derived_schema,
        building=args.building,
        start=args.start_time_step,
        end=args.end_time_step,
        session_name=baseline_session,
        algorithm="RBCSmartLocalPolicy",
        schedule_path=None,
    )
    (args.output_dir / "replay.yaml").write_text(yaml.safe_dump(replay, sort_keys=False), encoding="utf-8")
    replay_without_safety = json.loads(json.dumps(replay))
    replay_without_safety["metadata"]["run_name"] = replay_session + "-no-safety"
    replay_without_safety["metadata"]["description"] = (
        "Teacher-free direct CityLearn replay of a total-home MILP schedule; runtime safety projector disabled."
    )
    replay_without_safety["pipeline"][0]["hyperparameters"]["local_action_safety_enabled"] = False
    replay_without_safety["simulator"]["export"]["session_name"] = replay_session + "-no-safety"
    (args.output_dir / "replay_no_safety.yaml").write_text(
        yaml.safe_dump(replay_without_safety, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "rbcsmart.yaml").write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8")
    print(derived_schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
