#!/usr/bin/env python3
"""Generate the recoverable annual MATD3 standalone retry."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "matd3_settlement_annual_recovery_v2"
SEED = 789


def recipe() -> dict:
    source = REPO_ROOT / "configs/templates/rl/matd3_15min_residual_local.yaml"
    config = copy.deepcopy(yaml.safe_load(source.read_text(encoding="utf-8")))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"MATD3 standalone settled recoverable seed {SEED}",
            "description": (
                "Three train years plus one deterministic annual evaluation. "
                "Uses atomic progress, continuous inter-step watchdog coverage, "
                "yearly actor checkpoints and an explicit runtime-only residual export."
            ),
        }
    )
    config["tracking"]["tags"] = {
        "protocol": EXPERIMENT_NAME,
        "controller": "matd3_standalone",
        "seed": str(SEED),
        "settlement": "enabled",
        "training_years": "3",
        "deterministic_evaluation_years": "1",
        "atomic_progress": "True",
        "inter_step_watchdog": "True",
        "yearly_actor_checkpoint": "True",
        "residual_export": "runtime_only",
    }
    config["training"]["seed"] = SEED
    config["simulator"].update(
        {
            "episodes": 4,
            "deterministic_finish": True,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 35039,
            "episode_time_steps": 35040,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-seed{SEED}"
    )
    config["checkpointing"].update(
        {
            "checkpoint_mode": "inference",
            "checkpoint_interval": 35040,
            "require_update_step": True,
            "require_initial_exploration_done": True,
        }
    )
    config["pipeline"][0]["exploration"]["params"][
        "residual_policy_runtime_only_export"
    ] = True
    return config


def derive_smoke(config: dict) -> dict:
    smoke = copy.deepcopy(config)
    transitions = 1024
    smoke["metadata"]["run_name"] += " [profiling-boundary smoke]"
    smoke["tracking"]["tags"]["evidence"] = "functional_smoke"
    smoke["simulator"].update(
        {
            "episodes": 1,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": transitions - 1,
            "episode_time_steps": transitions,
        }
    )
    smoke["simulator"]["export"]["session_name"] += "-smoke"
    smoke["checkpointing"]["checkpoint_interval"] = transitions
    return smoke


def generate(output_dir: Path, *, smoke: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = recipe()
    if smoke:
        config = derive_smoke(config)
    path = output_dir / f"matd3_settled_recoverable_seed{SEED}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME,
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(generate(args.output_dir, smoke=args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
