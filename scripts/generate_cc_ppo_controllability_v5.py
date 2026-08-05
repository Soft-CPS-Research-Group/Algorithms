#!/usr/bin/env python3
"""Generate frozen-PPO price-path ablations for CC controllability V5."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_cc_causal_price_control_v4 import (
        PPO_SEED,
        REPO_ROOT,
        _derive_smoke,
        ppo_fixed_recipe,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_cc_causal_price_control_v4 import (
        PPO_SEED,
        REPO_ROOT,
        _derive_smoke,
        ppo_fixed_recipe,
    )


EXPERIMENT_NAME = "cc_ppo_controllability_v5"
PROBE_MULTIPLIER = 0.95
FORECAST_MODES = ("real_unmodified", "persist_current")


def _mode_label(forecast_mode: str) -> str:
    if forecast_mode == "real_unmodified":
        return "actor_current_only"
    if forecast_mode == "persist_current":
        return "actor_current_and_forecasts"
    raise ValueError(f"Unsupported V5 forecast mode: {forecast_mode}")


def actor_price_probe(forecast_mode: str) -> dict[str, Any]:
    """Build one frozen-checkpoint diagnostic with explicit actor conditioning.

    These probes deliberately do not claim that the existing actor is trained
    for non-neutral virtual prices.  Their purpose is to measure whether the
    current-only/forecast mismatch explains the failed first CC-PPO campaign.
    """

    label = _mode_label(forecast_mode)
    config = copy.deepcopy(ppo_fixed_recipe(PROBE_MULTIPLIER))
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": (
                f"CC-PPO frozen actor price-path ablation {label} seed {PPO_SEED}"
            ),
            "description": (
                "Annual diagnostic over the nominal-price PPO checkpoint. The "
                "0.95 multiplier controls the strict-local SMART residual base "
                "and is also injected into the frozen actor observation. This "
                "is an explicit out-of-distribution ablation, not a deployable "
                "price-responsive PPO claim."
            ),
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"cc-ppo-v5-{label}-fixed-0p95-seed{PPO_SEED}"
    )

    tags = config["tracking"]["tags"]
    tags.update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": label,
            "cc_price_scope": "strict_local_actor_and_residual_base",
            "ppo_actor_price_conditioning": "True",
            "actor_forecast_mode": forecast_mode,
            "checkpoint_price_training_support": "nominal_only",
            "inference_distribution": "explicit_ood_diagnostic",
            "promotion_eligible": "False",
        }
    )

    exploration = config["pipeline"][1]["exploration"]["params"]
    exploration.update(
        {
            "local_price_conditioning_enabled": True,
            "local_price_forecast_mode": forecast_mode,
            "residual_base_policy": "SignalAwareRBCSmartLocal",
            "residual_base_price_conditioning_enabled": True,
        }
    )
    return config


def _filename(forecast_mode: str) -> str:
    return f"cc_ppo_fixed_0p95_{_mode_label(forecast_mode)}_seed{PPO_SEED}.yaml"


def generate(output_dir: Path, *, smoke: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for forecast_mode in FORECAST_MODES:
        payload = actor_price_probe(forecast_mode)
        if smoke:
            payload = _derive_smoke(payload)
        path = output_dir / _filename(forecast_mode)
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate 384-transition functional smokes, not evidence configs.",
    )
    args = parser.parse_args()
    for path in generate(args.output_dir, smoke=args.smoke):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
