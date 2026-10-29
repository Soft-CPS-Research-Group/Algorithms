"""Run the unchanged positive-price dataset through the common Algorithms job lifecycle."""

import argparse
from pathlib import Path
from algorithms.runtime.job import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument(
    "policies", nargs="+", choices=["RBCBasicPolicy", "RBCSmartLocalPolicy", "RBCCommunityPolicy"]
)
parser.add_argument("--dataset-path", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--label", default="candidate")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=8759)
parser.add_argument("--market", action="store_true")
args = parser.parse_args()
root = args.output_dir.resolve() / args.label
root.mkdir(parents=True, exist_ok=True)
for policy in args.policies:
    config = {
        "schema_version": "algorithms.config.v1",
        "metadata": {
            "experiment_name": "baseline-hierarchy-audit",
            "run_name": policy,
            "community_name": "citylearn-2022-plus-evs-17-buildings",
        },
        "job": {"profile": "evaluate"},
        "tracking": {
            "mlflow_enabled": False,
            "log_level": "WARNING",
            "log_frequency": 512,
            "progress_update_interval": 512,
            "system_metrics_enabled": False,
        },
        "simulator": {
            "dataset_name": "citylearn_challenge_2022_phase_all_plus_evs",
            "dataset_path": str(args.dataset_path.resolve()),
            "building_ids": None,
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "static",
            "reward_function": "RewardFunction",
            "simulation_start_time_step": args.start,
            "simulation_end_time_step": args.end,
            "episode_time_steps": args.end - args.start + 1,
            "random_seed": 2022,
            "export": {
                "profile": "final",
                "mode": "end",
                "include_business_as_usual": True,
                "export_business_as_usual_timeseries": False,
            },
        },
        "training": {"episodes": 0, "seed": 22},
        "evaluation": {"episodes": 1},
        "pipeline": [{"algorithm": policy, "count": 1}],
        "outputs": {"model_artifacts": "bundle"},
        "bundle": {"require_deployable": True, "verification_samples": 32},
    }
    if args.market:
        config["simulator"]["community_market"] = {
            "enabled": True,
            "local_price_ratio_to_grid_import": 0.8,
            "grid_export_price": 0.0,
        }
    print("START", policy, flush=True)
    run_experiment(config, policy, root)
    print("COMPLETED", policy, flush=True)
