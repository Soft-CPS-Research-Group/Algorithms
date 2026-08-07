#!/usr/bin/env python3
"""Generate settlement-on CC-L2 diagnostics and learning over SMART."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Sequence

import yaml

try:
    from scripts.generate_ppo_cc_settlement_templates import (
        CHECKPOINT_NAME,
        REPO_ROOT,
        SMART_SEED,
        _market,
        _signal_aware_stage,
        _smart_configs,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_ppo_cc_settlement_templates import (
        CHECKPOINT_NAME,
        REPO_ROOT,
        SMART_SEED,
        _market,
        _signal_aware_stage,
        _smart_configs,
    )


EXPERIMENT_NAME = "cc_level2_smart_settlement_v1"
NUM_BUILDINGS = 17
NEUTRAL_VECTOR = [1.0] * NUM_BUILDINGS

# Per-building argmin of the matched annual SMART scalar response sweep
# {0.7, 0.9, 1.0, 1.1, 1.3}, subject to local hard gates.  Buildings 1--14
# and 16--17 preferred 0.7; Building 15 required 1.3 for its best feasible
# response.  The coupled annual simulation remains the source of truth.
EMPIRICAL_VECTOR = [0.7] * NUM_BUILDINGS
EMPIRICAL_VECTOR[14] = 1.3


def _fixed_vector_recipe(
    *,
    multipliers: Sequence[float],
    recipe: str,
    run_name: str,
) -> dict[str, Any]:
    if len(multipliers) != NUM_BUILDINGS:
        raise ValueError("CC-L2 SMART vector must contain 17 multipliers")
    smart, _ = _smart_configs()
    config = copy.deepcopy(smart)
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": run_name,
            "description": (
                "Annual fixed per-building price-vector diagnostic over the "
                "frozen SignalAwareRBC SMART leaf with settlement enabled."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": "fixed_cc_level2_smart",
            "cc_level": "2",
            "recipe": recipe,
            "settlement": "enabled",
            "leaf_frozen": "True",
            "promotion_eligible": "False",
        }
    )
    config["simulator"]["reward_function"] = "CCRewardLevel1"
    config["simulator"]["reward_function_kwargs"] = {
        "cost_aggregation": "community_settled",
        "w_cost": 1.0,
        "w_peak": 0.0,
        "w_ramp": 0.0,
        "w_export": 0.0,
        "w_violation": 0.0,
    }
    config["simulator"]["entity_encoding"]["profile"] = "cc_level2"
    config["simulator"]["export"]["session_name"] = f"{EXPERIMENT_NAME}-{recipe}"
    config["pipeline"][0]["hyperparameters"] = {
        "multiplier": 1.0,
        "multipliers": [float(value) for value in multipliers],
    }
    return config


def learned_recipe() -> dict[str, Any]:
    source_path = REPO_ROOT / "configs/templates/cc_level2_local.yaml"
    config = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": "CC-L2 SMART learned from empirical vector seed 123",
            "community_name": "citylearn_static_15min",
            "description": (
                "Settlement-aligned learned CC-L2 initialized at the feasible "
                "per-building SMART response vector."
            ),
        }
    )
    config["tracking"].update(
        {
            "mlflow_enabled": False,
            "log_frequency": 512,
            "mlflow_step_sample_interval": 512,
            "progress_update_interval": 128,
            "system_metrics_enabled": False,
            "tags": {
                "protocol": EXPERIMENT_NAME,
                "controller": "learned_cc_level2_smart",
                "cc_level": "2",
                "recipe": "empirical_reference_residual",
                "settlement": "enabled",
                "cc_seed": str(SMART_SEED),
                "leaf_frozen": "True",
                "reference_source": "matched_annual_scalar_response_argmin",
            },
        }
    )
    config["checkpointing"].update(
        {
            "checkpoint_artifact": CHECKPOINT_NAME,
            "checkpoint_interval": 35040,
            "require_update_step": True,
            "require_initial_exploration_done": False,
        }
    )
    config["simulator"].update(
        {
            "reward_function": "CCRewardLevel2",
            "reward_function_kwargs": {
                "cost_aggregation": "community_settled",
                "w_cost": 1.0,
                "w_peak": 0.3,
                "w_export": 0.1,
                "w_ev": 0.5,
                "urgency_horizon": 4.0,
            },
            "episodes": 8,
            "deterministic_finish": True,
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 35039,
            "episode_time_steps": 35040,
            "community_market": _market(),
            "export": {
                "mode": "end",
                "export_kpis_on_episode_end": True,
                "final_episode_only": True,
                "kpis_final_episode_only": True,
                "timeseries_final_episode_only": True,
                "include_business_as_usual": True,
                "export_business_as_usual_timeseries": False,
                "kpi_round_decimals": None,
                "session_name": f"{EXPERIMENT_NAME}-learned-seed123",
            },
        }
    )
    config["training"] = {
        "seed": SMART_SEED,
        "steps_between_training_updates": 1,
        "target_update_interval": 2,
    }
    manager = config["pipeline"][0]
    manager["frozen"] = False
    manager["hyperparameters"].update(
        {
            "price_min": 0.5,
            "price_max": 1.5,
            "reference_multipliers": list(EMPIRICAL_VECTOR),
            "policy_residual_scale": 0.35,
            "w_factor": 0.10,
            "w_smoothness": 0.05,
            # Start at the measured feasible vector.  The historical BC
            # teacher targets neutral-centred signals and would erase it.
            "bc_pretrain_enabled": False,
        }
    )
    config["pipeline"][1] = _signal_aware_stage()
    return config


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "cc_l2_smart_neutral_vector": _fixed_vector_recipe(
            multipliers=NEUTRAL_VECTOR,
            recipe="neutral_vector",
            run_name="CC-L2 SMART neutral vector annual",
        ),
        "cc_l2_smart_empirical_vector": _fixed_vector_recipe(
            multipliers=EMPIRICAL_VECTOR,
            recipe="empirical_vector",
            run_name="CC-L2 SMART empirical response vector annual",
        ),
        "cc_l2_smart_learned_seed123": learned_recipe(),
    }
    paths: list[Path] = []
    for name, config in configs.items():
        path = output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs/experiments" / EXPERIMENT_NAME,
    )
    args = parser.parse_args()
    for path in generate(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
