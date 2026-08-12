#!/usr/bin/env python3
"""Clone an audited fixed-service replay config for another schedule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config_schema import validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument(
        "--objective",
        choices=("cost", "scorecard", "carbon_scorecard"),
        default="scorecard",
        help="Scientific objective represented by the replayed schedule.",
    )
    parser.add_argument(
        "--service-config",
        type=Path,
        help=(
            "Optional resolved config whose RBC stage supplies the exact service "
            "policy and hyperparameters used under the replayed battery schedule."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Replay config template must contain a YAML object.")
    pipeline = config.get("pipeline") or []
    if len(pipeline) != 1 or pipeline[0].get("algorithm") != (
        "FixedServiceOracleReplayPolicy"
    ):
        raise ValueError(
            "Replay config template must contain one FixedServiceOracleReplayPolicy stage."
        )
    if not args.schedule.is_file():
        raise FileNotFoundError(args.schedule)

    objective_metadata = {
        "cost": {
            "description": (
                "CityLearn replay of a cost-minimizing fixed-service "
                "stationary-battery schedule."
            ),
            "oracle_scope": "community_fixed_service_battery_cost",
            "objective": "community_import_cost",
        },
        "scorecard": {
            "description": (
                "CityLearn replay of a cost-constrained scorecard-shaped "
                "fixed-service stationary-battery schedule."
            ),
            "oracle_scope": "community_fixed_service_battery_scorecard_shaped",
            "objective": "cost_ceiling_plus_ramp_daily_peak_all_time_peak",
        },
        "carbon_scorecard": {
            "description": (
                "CityLearn replay of a carbon-minimizing fixed-service "
                "stationary-battery schedule under cost, ramp and peak limits."
            ),
            "oracle_scope": (
                "community_fixed_service_battery_carbon_scorecard_shaped"
            ),
            "objective": (
                "carbon_under_cost_ramp_daily_peak_all_time_peak_limits"
            ),
        },
    }[args.objective]
    config["metadata"].update(
        {
            "experiment_name": args.experiment_name,
            "run_name": args.run_name,
            "description": objective_metadata["description"],
        }
    )
    for key in list(config.get("runtime") or {}):
        config["runtime"][key] = None
    config["tracking"]["tags"].update(
        {
            "oracle_scope": objective_metadata["oracle_scope"],
            "objective": objective_metadata["objective"],
            "global_optimum_claim": "False",
            "requires_exact_replay": "True",
        }
    )
    config["simulator"]["export"]["session_name"] = args.session_name
    if args.service_config is not None:
        service_config = yaml.safe_load(args.service_config.read_text(encoding="utf-8"))
        service_stages = [
            stage
            for stage in (service_config.get("pipeline") or [])
            if stage.get("algorithm")
            in {
                "RBCSmartLocalPolicy",
                "RBCSmartPolicy",
                "SignalAwareRBC",
                "SignalAwareRBCSmartLocal",
            }
        ]
        if len(service_stages) != 1:
            raise ValueError(
                "service-config must contain exactly one supported SMART service stage."
            )
        schedule_settings = {
            key: value
            for key, value in pipeline[0]["hyperparameters"].items()
            if key.startswith("schedule_") or key.startswith("local_action_safety_")
        }
        pipeline[0]["hyperparameters"] = {
            **dict(service_stages[0].get("hyperparameters") or {}),
            **schedule_settings,
            "service_policy": str(service_stages[0]["algorithm"]),
        }
        config["tracking"]["tags"]["service_policy"] = str(
            service_stages[0]["algorithm"]
        )
        config["training"]["seed"] = int(
            (service_config.get("training") or {}).get(
                "seed", config["training"]["seed"]
            )
        )
        source_market = (service_config.get("simulator") or {}).get(
            "community_market"
        )
        if isinstance(source_market, dict):
            config["simulator"]["community_market"] = dict(source_market)
    pipeline[0]["hyperparameters"]["schedule_path"] = str(args.schedule)
    validate_config(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
