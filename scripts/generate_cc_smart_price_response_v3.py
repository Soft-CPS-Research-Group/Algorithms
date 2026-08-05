#!/usr/bin/env python3
"""Generate annual CC-SMART price-response and update-density probes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.generate_ppo_cc_settlement_templates import REPO_ROOT, _smart_configs
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_ppo_cc_settlement_templates import REPO_ROOT, _smart_configs


EXPERIMENT_NAME = "cc_smart_price_response_v3"
SEED = 123
FIXED_MULTIPLIERS = (0.7, 0.9, 1.1, 1.3)
DENSE_RECIPE = "legacy_update_dense"


def _tagged(config: dict[str, Any], *, recipe: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["metadata"]["experiment_name"] = EXPERIMENT_NAME
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "controller": "cc_smart",
            "recipe": recipe,
            "settlement": "enabled",
            "leaf_frozen": "True",
            "evidence_horizon": "full_year",
        }
    )
    return config


def fixed_recipe(multiplier: float) -> dict[str, Any]:
    smart, _ = _smart_configs()
    recipe = f"fixed_{multiplier:.1f}".replace(".", "p")
    config = _tagged(smart, recipe=recipe)
    config["metadata"].update(
        {
            "run_name": f"CC-SMART fixed multiplier {multiplier:.1f}",
            "description": (
                "Annual fixed-price probe over the frozen SignalAwareRBC leaf; "
                "identifies the response envelope of the scalar CC channel."
            ),
        }
    )
    config["tracking"]["tags"]["fixed_multiplier"] = str(multiplier)
    config["simulator"]["export"]["session_name"] = (
        f"cc-smart-price-response-v3-{recipe}"
    )
    config["pipeline"][0]["hyperparameters"]["multiplier"] = float(multiplier)
    return config


def dense_recipe() -> dict[str, Any]:
    _, cc_smart = _smart_configs()
    config = _tagged(cc_smart, recipe=DENSE_RECIPE)
    config["metadata"].update(
        {
            "run_name": f"CC-SMART V1 update-dense seed {SEED}",
            "description": (
                "Exact V1 reward, rollout and regularization with eight annual "
                "episodes; isolates genuinely increased PPO update count."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "cc_seed": str(SEED),
            "training_episodes": "8",
            "ppo_horizon_cc_decisions": "96",
            "planned_ppo_update_count_approx": "547",
        }
    )
    config["simulator"]["episodes"] = 8
    config["simulator"]["export"]["session_name"] = (
        "cc-smart-price-response-v3-legacy-update-dense-seed123"
    )
    config["checkpointing"]["checkpoint_interval"] = 35040
    config["training"]["seed"] = SEED
    return config


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configs: list[tuple[str, dict[str, Any]]] = [
        (f"cc_smart_fixed_{value:.1f}".replace(".", "p"), fixed_recipe(value))
        for value in FIXED_MULTIPLIERS
    ]
    configs.append((f"cc_smart_{DENSE_RECIPE}_seed{SEED}", dense_recipe()))

    paths: list[Path] = []
    for name, config in configs:
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
