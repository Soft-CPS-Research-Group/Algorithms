#!/usr/bin/env python3
"""Generate matched fixed-price probes for the CC-L2 bidirectional leaf.

Phase A measures the global response curve at 0.70, 0.85, 1.00, 1.15 and
1.30. Phase B changes one building at a time around the exact-neutral vector.
The probes are deterministic instruments, not learned CC candidates.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_level2_ppo_distilled_v6 import (
        ANNUAL_STEPS,
        EXPERIMENT_NAME as TRAINING_EXPERIMENT,
        NUM_BUILDINGS,
        REPO_ROOT,
        build_paired_neutral_config,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_level2_ppo_distilled_v6 import (
        ANNUAL_STEPS,
        EXPERIMENT_NAME as TRAINING_EXPERIMENT,
        NUM_BUILDINGS,
        REPO_ROOT,
        build_paired_neutral_config,
    )


EXPERIMENT_NAME = "cc_level2_bidirectional_map_v6"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME
PROBE_MULTIPLIERS = (0.70, 0.85, 1.00, 1.15, 1.30)
NON_NEUTRAL_MULTIPLIERS = tuple(
    value for value in PROBE_MULTIPLIERS if value != 1.0
)


def _slug_multiplier(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def build_probe_config(
    *,
    multiplier: float,
    building_number: int | None = None,
    start_step: int = 0,
    horizon: int = 4096,
) -> dict[str, Any]:
    value = float(multiplier)
    if value not in PROBE_MULTIPLIERS:
        raise ValueError(
            f"CC-L2 causal-map multiplier must be one of {PROBE_MULTIPLIERS}"
        )
    if building_number is not None and not 1 <= int(building_number) <= NUM_BUILDINGS:
        raise ValueError(f"building_number must lie within [1, {NUM_BUILDINGS}]")

    config = copy.deepcopy(
        build_paired_neutral_config(
            episodes=1,
            start_step=start_step,
            horizon=horizon,
        )
    )
    values = [1.0] * NUM_BUILDINGS
    if building_number is None:
        values = [value] * NUM_BUILDINGS
        scope = "global"
        label = f"global_{_slug_multiplier(value)}"
    else:
        values[int(building_number) - 1] = value
        scope = "one_building"
        label = f"building_{int(building_number)}_{_slug_multiplier(value)}"

    manager = config["pipeline"][0]
    manager["hyperparameters"].update(
        {
            "multiplier": 1.0,
            "multipliers": values,
            "schedule": None,
        }
    )
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 bidirectional causal map {label}",
            "description": (
                "Deterministic matched price-response probe over the frozen "
                "PPO leaf. Exactly one vector coordinate or all coordinates "
                "are changed; no training or future trace is used."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "parent_training_protocol": TRAINING_EXPERIMENT,
            "recipe": label,
            "probe_scope": scope,
            "probe_multiplier": str(value),
            "probe_building": (
                "all" if building_number is None else str(int(building_number))
            ),
            "search_method": "matched_fixed_price_response_map",
            "uses_future_realized_data": "False",
            "promotion_eligible": "False",
        }
    )
    config["simulator"].update(
        {
            "episodes": 1,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{label}-start{start_step}-steps{horizon}"
    )
    return config


def build_pulse_probe_config(
    *,
    multiplier: float,
    building_number: int | None = None,
    pulse_start: int = 96,
    pulse_duration: int = 4,
    start_step: int = 0,
    horizon: int = 384,
) -> dict[str, Any]:
    """Build a matched neutral run with one short price intervention.

    Constant probes intentionally measure the cumulative closed-loop response,
    including battery saturation.  Pulse probes isolate the immediate causal
    response: both runs are neutral up to ``pulse_start``, the intervention is
    active for exactly ``pulse_duration`` simulator steps, then the signal
    returns to neutral.
    """
    value = float(multiplier)
    if value not in NON_NEUTRAL_MULTIPLIERS:
        raise ValueError(
            "CC-L2 pulse multiplier must be one of "
            f"{NON_NEUTRAL_MULTIPLIERS}"
        )
    if building_number is not None and not 1 <= int(building_number) <= NUM_BUILDINGS:
        raise ValueError(f"building_number must lie within [1, {NUM_BUILDINGS}]")
    if pulse_start <= 0:
        raise ValueError("pulse_start must be greater than zero")
    if pulse_duration <= 0:
        raise ValueError("pulse_duration must be positive")
    pulse_end = int(pulse_start) + int(pulse_duration)
    if pulse_end >= int(horizon):
        raise ValueError("price pulse must end before the configured horizon")

    config = copy.deepcopy(
        build_paired_neutral_config(
            episodes=1,
            start_step=start_step,
            horizon=horizon,
        )
    )
    manager = config["pipeline"][0]
    manager["hyperparameters"].update(
        {
            "multiplier": 1.0,
            "multipliers": None,
            "schedule": None,
            "vector_schedule": None,
        }
    )
    if building_number is None:
        manager["hyperparameters"]["schedule"] = [
            {"start_step": 0, "multiplier": 1.0},
            {"start_step": int(pulse_start), "multiplier": value},
            {"start_step": pulse_end, "multiplier": 1.0},
        ]
        scope = "global_pulse"
        label = f"global_pulse_{_slug_multiplier(value)}_at{pulse_start}"
    else:
        neutral = [1.0] * NUM_BUILDINGS
        intervention = list(neutral)
        intervention[int(building_number) - 1] = value
        manager["hyperparameters"]["vector_schedule"] = [
            {"start_step": 0, "multipliers": neutral},
            {
                "start_step": int(pulse_start),
                "multipliers": intervention,
            },
            {"start_step": pulse_end, "multipliers": neutral},
        ]
        scope = "one_building_pulse"
        label = (
            f"building_{int(building_number)}_pulse_"
            f"{_slug_multiplier(value)}_at{pulse_start}"
        )

    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-L2 bidirectional causal pulse {label}",
            "description": (
                "Matched deterministic intervention over the frozen PPO leaf. "
                "The price is neutral before and after a one-hour pulse, so "
                "the immediate local response is identifiable without "
                "confounding it with multi-day battery saturation."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "parent_training_protocol": TRAINING_EXPERIMENT,
            "recipe": label,
            "probe_scope": scope,
            "probe_multiplier": str(value),
            "probe_building": (
                "all" if building_number is None else str(int(building_number))
            ),
            "probe_start_step": str(int(pulse_start)),
            "probe_duration_steps": str(int(pulse_duration)),
            "search_method": "matched_price_pulse_causal_map",
            "uses_future_realized_data": "False",
            "promotion_eligible": "False",
        }
    )
    config["simulator"].update(
        {
            "episodes": 1,
            "deterministic_finish": True,
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{label}-start{start_step}-steps{horizon}"
    )
    return config


def generate(
    output_dir: Path = OUTPUT_DIR,
    *,
    start_step: int = 0,
    horizon: int = 4096,
    include_member_probes: bool = False,
    include_pulse_probes: bool = False,
    include_member_pulse_probes: bool = False,
    pulse_start: int = 96,
    pulse_duration: int = 4,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for multiplier in PROBE_MULTIPLIERS:
        label = f"global_{_slug_multiplier(multiplier)}"
        path = output_dir / f"{label}_start{start_step}_steps{horizon}.yaml"
        path.write_text(
            yaml.safe_dump(
                build_probe_config(
                    multiplier=multiplier,
                    start_step=start_step,
                    horizon=horizon,
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(path)

    if include_member_probes:
        for building_number in range(1, NUM_BUILDINGS + 1):
            for multiplier in PROBE_MULTIPLIERS:
                if multiplier == 1.0:
                    continue
                label = (
                    f"building_{building_number}_{_slug_multiplier(multiplier)}"
                )
                path = output_dir / f"{label}_start{start_step}_steps{horizon}.yaml"
                path.write_text(
                    yaml.safe_dump(
                        build_probe_config(
                            multiplier=multiplier,
                            building_number=building_number,
                            start_step=start_step,
                            horizon=horizon,
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                outputs.append(path)

    if include_pulse_probes:
        for multiplier in NON_NEUTRAL_MULTIPLIERS:
            label = (
                f"global_pulse_{_slug_multiplier(multiplier)}_at{pulse_start}"
            )
            path = output_dir / f"{label}_start{start_step}_steps{horizon}.yaml"
            path.write_text(
                yaml.safe_dump(
                    build_pulse_probe_config(
                        multiplier=multiplier,
                        pulse_start=pulse_start,
                        pulse_duration=pulse_duration,
                        start_step=start_step,
                        horizon=horizon,
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            outputs.append(path)

    if include_member_pulse_probes:
        for building_number in range(1, NUM_BUILDINGS + 1):
            for multiplier in NON_NEUTRAL_MULTIPLIERS:
                label = (
                    f"building_{building_number}_pulse_"
                    f"{_slug_multiplier(multiplier)}_at{pulse_start}"
                )
                path = output_dir / (
                    f"{label}_start{start_step}_steps{horizon}.yaml"
                )
                path.write_text(
                    yaml.safe_dump(
                        build_pulse_probe_config(
                            multiplier=multiplier,
                            building_number=building_number,
                            pulse_start=pulse_start,
                            pulse_duration=pulse_duration,
                            start_step=start_step,
                            horizon=horizon,
                        ),
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=4096)
    parser.add_argument("--include-member-probes", action="store_true")
    parser.add_argument("--include-pulse-probes", action="store_true")
    parser.add_argument("--include-member-pulse-probes", action="store_true")
    parser.add_argument("--pulse-start", type=int, default=96)
    parser.add_argument("--pulse-duration", type=int, default=4)
    args = parser.parse_args()
    if args.horizon > ANNUAL_STEPS:
        parser.error(f"--horizon must not exceed {ANNUAL_STEPS}")
    for path in generate(
        args.output_dir,
        start_step=args.start_step,
        horizon=args.horizon,
        include_member_probes=args.include_member_probes,
        include_pulse_probes=args.include_pulse_probes,
        include_member_pulse_probes=args.include_member_pulse_probes,
        pulse_start=args.pulse_start,
        pulse_duration=args.pulse_duration,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
