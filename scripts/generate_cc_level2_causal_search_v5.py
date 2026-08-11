#!/usr/bin/env python3
"""Generate deterministic CC-L2 coordinate probes over the causal CC-L1 incumbent.

The trainable V4 policy proved that member rewards and vector routing execute,
but a few thousandths of unconstrained PPO drift were enough to lose the
global 0.90 incumbent.  V5 first treats the 17 active-event discounts as a
small deterministic black-box optimisation problem.  Exact one-coordinate
probes identify which buildings can profitably move above/below 0.90.  The
validated vector then becomes both a deployable Level-2 controller and a safe
teacher/incumbent for any later contextual policy.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_ppo_causal_online_v5p3b import (
        PPO_SEED,
        REPO_ROOT,
        recipe as build_causal_recipe,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_ppo_causal_online_v5p3b import (
        PPO_SEED,
        REPO_ROOT,
        recipe as build_causal_recipe,
    )


EXPERIMENT_NAME = "cc_level2_causal_search_v5"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
NUM_BUILDINGS = 17
INCUMBENT = 0.90
# The first real paired probes used 0.005 and produced bitwise-identical
# trajectories to the 0.90 incumbent.  The frozen residual leaf delegates the
# price-sensitive part to a threshold-based SMART policy, so infinitesimal
# finite differences sit inside a genuine action deadband.  Use a coarse
# 0.05 probe first; any winning direction is refined only after it changes the
# physical trajectory.
COORDINATE_DELTA = 0.05


def vector_for_probe(name: str) -> list[float]:
    if name == "vector_parity":
        return [INCUMBENT] * NUM_BUILDINGS
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "building" or parts[2] not in {"down", "up"}:
        raise ValueError(f"Unknown CC-L2 V5 coordinate probe: {name}")
    building_number = int(parts[1])
    if not 1 <= building_number <= NUM_BUILDINGS:
        raise ValueError(f"Building number must be within [1, {NUM_BUILDINGS}]")
    values = [INCUMBENT] * NUM_BUILDINGS
    direction = -1.0 if parts[2] == "down" else 1.0
    values[building_number - 1] = INCUMBENT + direction * COORDINATE_DELTA
    return values


def probe_names() -> tuple[str, ...]:
    names = ["vector_parity"]
    for building_number in range(1, NUM_BUILDINGS + 1):
        names.extend(
            (
                f"building_{building_number}_down",
                f"building_{building_number}_up",
            )
        )
    return tuple(names)


def build_config(
    name: str,
    *,
    pilot_steps: int | None = None,
) -> dict[str, Any]:
    discounts = vector_for_probe(name)
    config = copy.deepcopy(build_causal_recipe("hourly_cost"))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 causal coordinate {name} seed {PPO_SEED}",
            "description": (
                "Deterministic per-building causal price vector over the frozen "
                "PPO seed-789 leaf. No outcome trace or future realized data is used."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "controller_level": "CC-L2",
            "search_method": "one_coordinate_at_a_time",
            "vector_incumbent": str(INCUMBENT),
            "coordinate_delta": str(COORDINATE_DELTA),
            "leaf_price_response": "actor_current_plus_residual",
            "uses_future_realized_data": "False",
            "promotion_eligible": "False",
        }
    )
    config["tracking"].update(
        {
            "progress_phase_updates_enabled": True,
            "stall_watchdog_context_interval_steps": 64,
        }
    )
    manager = config["pipeline"][0]
    manager["frozen"] = True
    manager["hyperparameters"].update(
        {
            "discount_multipliers": discounts,
            "vector_min_multiplier": 0.5,
            "vector_max_multiplier": 1.3,
        }
    )
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    leaf_params.update(
        {
            # Matched price-path ablations selected current-only actor
            # conditioning together with the residual SMART response.  The
            # forecast coordinates remain the real forecasts; persisting the
            # current multiplier was measurably worse.
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": "real_unmodified",
            "residual_base_price_conditioning_enabled": True,
        }
    )
    residual_params = leaf_params["residual_base_policy_hyperparameters"]
    residual_params.update(
        {
            # Preserve the measured 0.90 incumbent exactly while making the
            # CC-L2 magnitude identifiable: 0.95 gives half the configured
            # charging authority and 0.85 reaches the 1.5x cap.
            "signal_price_response_mode": "linear_discount",
            "signal_price_charge_reference_multiplier": INCUMBENT,
            "signal_price_charge_gain_max": 1.5,
        }
    )
    config["simulator"]["episodes"] = 1
    config["simulator"]["deterministic_finish"] = True
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-seed{PPO_SEED}"
    )
    config["checkpointing"]["checkpoint_interval"] = None

    if pilot_steps is not None:
        if pilot_steps < 4096 or pilot_steps % 4 != 0:
            raise ValueError("pilot_steps must be a multiple of 4 and at least 4096")
        config["metadata"]["run_name"] += f" pilot {pilot_steps}"
        config["tracking"]["tags"].update(
            {
                "evidence": "matched_slice_coordinate_probe",
                "pilot_steps": str(pilot_steps),
            }
        )
        config["simulator"].update(
            {
                "simulation_start_time_step": 0,
                "simulation_end_time_step": pilot_steps - 1,
                "episode_time_steps": pilot_steps,
            }
        )
        config["simulator"]["export"]["session_name"] += f"-pilot{pilot_steps}"
    return config


def build_vector_config(
    multipliers: list[float],
    *,
    label: str,
    pilot_steps: int | None = None,
    cc_action_interval: int | None = None,
    episodes: int = 1,
) -> dict[str, Any]:
    """Build an auditable refinement/combined vector from the parity contract."""

    values = [float(value) for value in multipliers]
    if len(values) != NUM_BUILDINGS:
        raise ValueError(
            f"CC-L2 V5 custom vector must contain {NUM_BUILDINGS} multipliers"
        )
    if any(value < 0.5 or value > 1.3 for value in values):
        raise ValueError("CC-L2 V5 custom vector multipliers must lie within [0.5, 1.3]")
    slug = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    if not slug:
        raise ValueError("CC-L2 V5 custom vector label must not be empty")
    if int(episodes) < 1:
        raise ValueError("CC-L2 V5 episodes must be at least 1")

    config = build_config("vector_parity", pilot_steps=pilot_steps)
    config["pipeline"][0]["hyperparameters"]["discount_multipliers"] = values
    if cc_action_interval is not None:
        if int(cc_action_interval) < 1:
            raise ValueError("CC-L2 V5 cc_action_interval must be positive")
        config["pipeline"][0]["hyperparameters"]["cc_action_interval"] = int(
            cc_action_interval
        )
    config["metadata"]["run_name"] = f"CC-L2 causal vector {label} seed {PPO_SEED}"
    config["tracking"]["tags"].update(
        {
            "recipe": slug,
            "search_method": "vector_refinement",
            "parent_vector": "global_0.90",
            "cc_action_interval": str(
                config["pipeline"][0]["hyperparameters"]["cc_action_interval"]
            ),
            "evaluation_episode_index": str(int(episodes)),
            "episode_realization_matched": str(int(episodes) > 1),
        }
    )
    config["simulator"]["episodes"] = int(episodes)
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{slug}-seed{PPO_SEED}"
        f"-interval{config['pipeline'][0]['hyperparameters']['cc_action_interval']}"
        + ("" if int(episodes) == 1 else f"-ep{int(episodes)}")
        + ("" if pilot_steps is None else f"-pilot{pilot_steps}")
    )
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    pilot_steps: int | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if pilot_steps is None else f"_pilot{pilot_steps}"
    outputs: list[Path] = []
    for name in probe_names():
        path = output_dir / f"cc_l2_v5_{name}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(
                build_config(name, pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs


def generate_refinements(
    output_dir: Path,
    *,
    building_number: int,
    multipliers: list[float],
    pilot_steps: int | None = None,
    cc_action_interval: int | None = None,
) -> list[Path]:
    if not 1 <= int(building_number) <= NUM_BUILDINGS:
        raise ValueError(f"Building number must be within [1, {NUM_BUILDINGS}]")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if pilot_steps is None else f"_pilot{pilot_steps}"
    outputs: list[Path] = []
    for multiplier in multipliers:
        values = [INCUMBENT] * NUM_BUILDINGS
        values[int(building_number) - 1] = float(multiplier)
        value_slug = str(float(multiplier)).replace(".", "p")
        label = f"building_{building_number}_multiplier_{value_slug}"
        path = output_dir / f"cc_l2_v5_{label}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(
                build_vector_config(
                    values,
                    label=label,
                    pilot_steps=pilot_steps,
                    cc_action_interval=cc_action_interval,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs


def generate_custom_vector(
    output_dir: Path,
    *,
    multipliers: list[float],
    label: str,
    pilot_steps: int | None = None,
    cc_action_interval: int | None = None,
    episodes: int = 1,
) -> Path:
    """Write one auditable combined-vector candidate."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_vector_config(
        multipliers,
        label=label,
        pilot_steps=pilot_steps,
        cc_action_interval=cc_action_interval,
        episodes=episodes,
    )
    suffix = "" if pilot_steps is None else f"_pilot{pilot_steps}"
    slug = str(config["tracking"]["tags"]["recipe"])
    path = output_dir / f"cc_l2_v5_{slug}{suffix}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--refine-building", type=int)
    parser.add_argument("--refine-values", type=float, nargs="+")
    parser.add_argument("--vector-values", type=float, nargs="+")
    parser.add_argument("--vector-label")
    parser.add_argument("--cc-action-interval", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()
    if (args.refine_building is None) != (args.refine_values is None):
        parser.error("--refine-building and --refine-values must be supplied together")
    if (args.vector_values is None) != (args.vector_label is None):
        parser.error("--vector-values and --vector-label must be supplied together")
    if args.refine_building is not None and args.vector_values is not None:
        parser.error("refinement and custom-vector modes are mutually exclusive")
    if args.vector_values is not None:
        outputs = [
            generate_custom_vector(
                args.output_dir,
                multipliers=args.vector_values,
                label=args.vector_label,
                pilot_steps=args.pilot_steps,
                cc_action_interval=args.cc_action_interval,
                episodes=args.episodes,
            )
        ]
    elif args.refine_building is not None:
        outputs = generate_refinements(
            args.output_dir,
            building_number=args.refine_building,
            multipliers=args.refine_values,
            pilot_steps=args.pilot_steps,
            cc_action_interval=args.cc_action_interval,
        )
    else:
        outputs = generate(args.output_dir, pilot_steps=args.pilot_steps)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
