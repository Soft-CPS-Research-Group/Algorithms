#!/usr/bin/env python3
"""Derive short, end-to-end smokes from the frozen annual PPO/CC protocol."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_ppo_cc_settlement_templates import (
    _ppo_configs,
    _smart_configs,
)


# CityLearn exposes both interval endpoints. A 385-row window therefore
# produces 384 environment transitions: exactly 96 four-step CC decisions.
SMOKE_STEPS = 385
SMOKE_PROTOCOL = "ppo_cc_settlement_smoke_v1"


def _annual_payloads() -> dict[str, dict[str, Any]]:
    smart, cc_smart = _smart_configs()
    ppo, cc_ppo = _ppo_configs()
    return {
        "smart_settlement_smoke.yaml": smart,
        "cc_smart_settlement_smoke_seed123.yaml": cc_smart,
        "ppo_settlement_smoke_seed789.yaml": ppo,
        "cc_ppo_settlement_smoke_seed789.yaml": cc_ppo,
    }


def _derive_smoke(config: dict[str, Any], *, steps: int) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("Smoke steps must be positive")

    smoke = copy.deepcopy(config)
    metadata = smoke["metadata"]
    metadata["experiment_name"] = SMOKE_PROTOCOL
    metadata["run_name"] = f'{metadata["run_name"]} [smoke {steps}]'
    metadata["description"] = (
        f'{metadata["description"]} Local end-to-end smoke evidence only; '
        "not annual performance evidence."
    )

    tracking = smoke["tracking"]
    tracking["tags"]["parent_protocol"] = tracking["tags"]["protocol"]
    tracking["tags"]["protocol"] = SMOKE_PROTOCOL
    tracking["tags"]["evidence"] = "smoke"
    tracking["log_frequency"] = min(int(tracking["log_frequency"]), 64)
    tracking["progress_update_interval"] = min(
        int(tracking["progress_update_interval"]), 64
    )
    tracking["runtime_profiling_interval"] = min(
        int(tracking["runtime_profiling_interval"]), 64
    )

    simulator = smoke["simulator"]
    simulator["simulation_start_time_step"] = 0
    simulator["simulation_end_time_step"] = steps - 1
    simulator["episode_time_steps"] = steps
    simulator["export"]["session_name"] = (
        f'{simulator["export"]["session_name"]}-smoke-{steps}'
    )

    manager = smoke["pipeline"][0]
    if manager["algorithm"] == "CCLevel1":
        interval = int(manager["hyperparameters"]["cc_action_interval"])
        rollout_steps = int(manager["hyperparameters"]["num_steps"])
        transition_count = steps - 1
        if transition_count % interval != 0:
            raise ValueError(
                "Smoke transitions (configured time steps minus one) must be "
                f"divisible by CC interval ({interval})"
            )
        decisions_per_episode = transition_count // interval
        if decisions_per_episode != rollout_steps:
            raise ValueError(
                "Smoke window must contain exactly one canonical CC rollout: "
                f"expected {rollout_steps * interval} environment transitions"
            )

        # Episode 1 exercises the complete BC path. Episode 2 produces one
        # complete PPO rollout and therefore one real coordinator update.
        # Episode 3 is required because deterministic_finish makes the final
        # episode evaluation-only and the wrapper deliberately skips update().
        simulator["episodes"] = 3
        manager["hyperparameters"]["bc_collect_steps"] = decisions_per_episode
        smoke["checkpointing"]["checkpoint_interval"] = transition_count
    else:
        simulator["episodes"] = 1
        smoke["checkpointing"]["checkpoint_interval"] = None

    return smoke


def generate_smokes(output_dir: Path, *, steps: int = SMOKE_STEPS) -> list[Path]:
    payloads = {
        filename: _derive_smoke(config, steps=steps)
        for filename, config in _annual_payloads().items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/local_configs/ppo_cc_settlement_smoke_v1",
    )
    parser.add_argument("--steps", type=int, default=SMOKE_STEPS)
    args = parser.parse_args()
    for path in generate_smokes(args.output_dir, steps=args.steps):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
