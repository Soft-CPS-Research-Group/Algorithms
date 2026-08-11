#!/usr/bin/env python3
"""Generate causal price-response ablations for the frozen PPO leaf.

The deployed PPO is a residual controller over ``SignalAwareRBCSmartLocal``.
A coordinator signal can therefore affect the SMART residual base, the PPO
actor observation, or both.  These paired recipes establish which causal path
actually gives useful authority before a per-building CC-L2 policy is trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from scripts.generate_cc_level2_causal_search_v5 import (
        OUTPUT_DIR as SEARCH_OUTPUT_DIR,
        PPO_SEED,
        REPO_ROOT,
        build_config as build_search_config,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution
    from generate_cc_level2_causal_search_v5 import (
        OUTPUT_DIR as SEARCH_OUTPUT_DIR,
        PPO_SEED,
        REPO_ROOT,
        build_config as build_search_config,
    )


EXPERIMENT_NAME = "cc_level2_ppo_price_response_v5"
OUTPUT_DIR = REPO_ROOT / "configs" / "experiments" / EXPERIMENT_NAME

RECIPES: Mapping[str, Mapping[str, Any]] = {
    "residual_only": {
        "local_price_conditioning_enabled": False,
        "residual_base_price_conditioning_enabled": True,
        "local_price_forecast_mode": "real_unmodified",
    },
    "actor_current_only": {
        "local_price_conditioning_enabled": True,
        "residual_base_price_conditioning_enabled": False,
        "local_price_forecast_mode": "real_unmodified",
    },
    "actor_persist_only": {
        "local_price_conditioning_enabled": True,
        "residual_base_price_conditioning_enabled": False,
        "local_price_forecast_mode": "persist_current",
    },
    "actor_current_plus_residual": {
        "local_price_conditioning_enabled": True,
        "residual_base_price_conditioning_enabled": True,
        "local_price_forecast_mode": "real_unmodified",
    },
    "actor_persist_plus_residual": {
        "local_price_conditioning_enabled": True,
        "residual_base_price_conditioning_enabled": True,
        "local_price_forecast_mode": "persist_current",
    },
}


def build_config(name: str, *, pilot_steps: int | None = None) -> dict[str, Any]:
    if name not in RECIPES:
        raise ValueError(f"Unknown PPO price-response recipe: {name}")
    config = build_search_config("vector_parity", pilot_steps=pilot_steps)
    recipe = RECIPES[name]
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC price response {name} seed {PPO_SEED}",
            "description": (
                "Causal 0.90 active-event price signal over a frozen PPO leaf; "
                "paired ablation of actor and residual-SMART price paths."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": name,
            "price_response_path": name,
            "search_parent": str(SEARCH_OUTPUT_DIR.relative_to(REPO_ROOT)),
            "promotion_eligible": "False",
        }
    )
    leaf_params = config["pipeline"][1]["exploration"]["params"]
    leaf_params.update(recipe)
    config["simulator"]["export"]["session_name"] = (
        f"{EXPERIMENT_NAME}-{name}-seed{PPO_SEED}"
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
    for name in RECIPES:
        path = output_dir / f"cc_l2_ppo_price_response_{name}{suffix}.yaml"
        path.write_text(
            yaml.safe_dump(
                build_config(name, pilot_steps=pilot_steps),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pilot-steps", type=int)
    args = parser.parse_args()
    for path in generate(args.output_dir, pilot_steps=args.pilot_steps):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
