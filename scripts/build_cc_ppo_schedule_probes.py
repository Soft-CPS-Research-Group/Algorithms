#!/usr/bin/env python3
"""Build replay-required temporal CC-PPO probes from matched annual traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:
    from scripts.generate_cc_causal_price_control_v4 import (
        PPO_SEED,
        REPO_ROOT,
        ppo_fixed_recipe,
    )
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from generate_cc_causal_price_control_v4 import (
        PPO_SEED,
        REPO_ROOT,
        ppo_fixed_recipe,
    )


EXPERIMENT_NAME = "cc_ppo_controllability_v5"
DEFAULT_DISCOUNT = 0.95
DEFAULT_NEUTRAL = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows


def _normalise_column(name: str) -> str:
    return "".join(char.lower() for char in name if char.isalnum() or char == "_")


def _find_column(fieldnames: Sequence[str], prefix: str) -> str:
    target = _normalise_column(prefix)
    exact = [name for name in fieldnames if _normalise_column(name) == target]
    if len(exact) == 1:
        return exact[0]
    matches = [
        name for name in fieldnames if _normalise_column(name).startswith(target)
    ]
    if matches:
        shortest_length = min(len(_normalise_column(name)) for name in matches)
        shortest = [
            name
            for name in matches
            if len(_normalise_column(name)) == shortest_length
        ]
        if len(shortest) == 1:
            return shortest[0]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one column beginning with {prefix!r}; found {matches!r}"
        )
    return matches[0]


def _floats(rows: Sequence[Mapping[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {column!r} at row {index}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {column!r} at row {index}")
        values.append(value)
    return values


def _assert_aligned(
    reference_rows: Sequence[Mapping[str, str]],
    other_rows: Sequence[Mapping[str, str]],
    *,
    reference_name: str,
    other_name: str,
) -> None:
    if len(reference_rows) != len(other_rows):
        raise ValueError(
            f"Trace length mismatch: {reference_name}={len(reference_rows)}, "
            f"{other_name}={len(other_rows)}"
        )
    if not reference_rows:
        raise ValueError("Traces must contain at least one data row")
    if "timestamp" not in reference_rows[0] or "timestamp" not in other_rows[0]:
        return
    for index, (reference, other) in enumerate(zip(reference_rows, other_rows)):
        if reference["timestamp"] != other["timestamp"]:
            reference_timestamp = reference["timestamp"]
            other_timestamp = other["timestamp"]
            raise ValueError(
                f"Timestamp mismatch at row {index}: "
                f"{reference_timestamp!r} != {other_timestamp!r}"
            )


def _native_cheap_mask(
    current: Sequence[float],
    forecast_1: Sequence[float],
    forecast_2: Sequence[float],
    forecast_3: Sequence[float],
) -> list[bool]:
    mask: list[bool] = []
    for price, one, two, three in zip(current, forecast_1, forecast_2, forecast_3):
        forecasts = (one, two, three)
        forecast_mean = sum(forecasts) / len(forecasts)
        forecast_min = min(forecasts)
        forecast_max = max(forecasts)
        spread = max(
            forecast_max - forecast_min,
            abs(forecast_mean) * 0.05,
            1.0e-9,
        )
        mask.append(
            price <= forecast_mean - 0.20 * spread
            or price <= forecast_min + 0.10 * spread
        )
    return mask


def _block_mask(
    step_mask: Sequence[bool],
    *,
    block_steps: int,
    activation_fraction: float,
) -> list[bool]:
    if block_steps < 1:
        raise ValueError("block_steps must be >= 1")
    if not 0.0 < activation_fraction <= 1.0:
        raise ValueError("activation_fraction must be in (0, 1]")
    blocks: list[bool] = []
    for start in range(0, len(step_mask), block_steps):
        values = step_mask[start : start + block_steps]
        blocks.append(sum(values) / len(values) >= activation_fraction)
    return blocks


def _retrospective_cost_mask(
    neutral_cost: Sequence[float],
    discount_cost: Sequence[float],
    *,
    block_steps: int,
    minimum_saving: float,
) -> list[bool]:
    if len(neutral_cost) != len(discount_cost):
        raise ValueError("Matched cost traces must have the same length")
    blocks: list[bool] = []
    for start in range(0, len(neutral_cost), block_steps):
        neutral = sum(neutral_cost[start : start + block_steps])
        discount = sum(discount_cost[start : start + block_steps])
        blocks.append(neutral - discount > minimum_saving)
    return blocks


def _compress_schedule(
    block_mask: Sequence[bool],
    *,
    block_steps: int,
    discount: float,
    neutral: float,
) -> list[dict[str, float | int]]:
    if not block_mask:
        raise ValueError("Cannot build a schedule from an empty mask")
    schedule: list[dict[str, float | int]] = []
    previous: float | None = None
    for block_index, enabled in enumerate(block_mask):
        multiplier = discount if enabled else neutral
        if multiplier == previous:
            continue
        schedule.append(
            {
                "start_step": block_index * block_steps,
                "multiplier": multiplier,
            }
        )
        previous = multiplier
    return schedule


def derive_schedule_masks(
    *,
    pricing_fieldnames: Sequence[str],
    pricing_rows: Sequence[Mapping[str, str]],
    neutral_fieldnames: Sequence[str],
    neutral_rows: Sequence[Mapping[str, str]],
    discount_fieldnames: Sequence[str] | None = None,
    discount_rows: Sequence[Mapping[str, str]] | None = None,
    block_steps: int = 4,
    activation_fraction: float = 0.5,
    minimum_saving: float = 0.0,
) -> dict[str, list[bool]]:
    """Return block-level masks for causal heuristics and an optional selector."""

    _assert_aligned(
        pricing_rows,
        neutral_rows,
        reference_name="pricing",
        other_name="neutral community",
    )
    current = _floats(
        pricing_rows,
        _find_column(pricing_fieldnames, "electricity_pricing"),
    )
    forecasts = [
        _floats(
            pricing_rows,
            _find_column(
                pricing_fieldnames,
                f"electricity_pricing_predicted_{index}",
            ),
        )
        for index in (1, 2, 3)
    ]
    cheap = _native_cheap_mask(current, *forecasts)
    net = _floats(
        neutral_rows,
        _find_column(neutral_fieldnames, "Net Electricity Consumption"),
    )
    export = [value < -1.0e-9 for value in net]
    masks = {
        "native_cheap": _block_mask(
            cheap,
            block_steps=block_steps,
            activation_fraction=activation_fraction,
        ),
        "community_export": _block_mask(
            export,
            block_steps=block_steps,
            activation_fraction=activation_fraction,
        ),
        "cheap_or_export": _block_mask(
            [cheap_step or export_step for cheap_step, export_step in zip(cheap, export)],
            block_steps=block_steps,
            activation_fraction=activation_fraction,
        ),
        "cheap_and_export": _block_mask(
            [cheap_step and export_step for cheap_step, export_step in zip(cheap, export)],
            block_steps=block_steps,
            activation_fraction=activation_fraction,
        ),
    }

    if discount_rows is not None:
        if discount_fieldnames is None:
            raise ValueError("discount_fieldnames are required with discount_rows")
        _assert_aligned(
            neutral_rows,
            discount_rows,
            reference_name="neutral community",
            other_name="discount community",
        )
        neutral_cost = _floats(
            neutral_rows,
            _find_column(neutral_fieldnames, "Price"),
        )
        discount_cost = _floats(
            discount_rows,
            _find_column(discount_fieldnames, "Price"),
        )
        masks["retrospective_cost"] = _retrospective_cost_mask(
            neutral_cost,
            discount_cost,
            block_steps=block_steps,
            minimum_saving=minimum_saving,
        )
    return masks


def _probe_config(
    *,
    recipe: str,
    schedule: Sequence[Mapping[str, float | int]],
    discount: float,
    block_steps: int,
    smoke_transitions: int | None,
) -> dict[str, Any]:
    config = ppo_fixed_recipe(1.0)
    config["metadata"].update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_name": f"CC-PPO temporal probe {recipe} seed {PPO_SEED}",
            "description": (
                "Replay-required temporal diagnostic over the frozen PPO and "
                "strict-local signal-aware residual base. The schedule was "
                "derived from annual traces and is not a deployable learned CC."
            ),
        }
    )
    config["tracking"]["tags"].update(
        {
            "protocol": EXPERIMENT_NAME,
            "recipe": recipe,
            "temporal_probe": "True",
            "trace_derived": "True",
            "in_sample_diagnostic": "True",
            "promotion_eligible": "False",
            "cc_action_interval": str(block_steps),
            "discount_multiplier": str(discount),
        }
    )
    config["pipeline"][0]["hyperparameters"].update(
        {
            "multiplier": 1.0,
            "schedule": [dict(entry) for entry in schedule],
        }
    )
    config["simulator"]["export"]["session_name"] = (
        f"cc-ppo-v5-temporal-{recipe}-seed{PPO_SEED}"
    )
    if smoke_transitions is not None:
        if smoke_transitions < 1:
            raise ValueError("smoke_transitions must be >= 1")
        config["simulator"]["episodes"] = 1
        config["simulator"]["episode_time_steps"] = smoke_transitions + 1
        config["checkpointing"]["checkpoint_interval"] = None
        config["metadata"]["run_name"] += " [functional smoke]"
        config["tracking"]["tags"]["evidence"] = "functional_smoke"
        config["tracking"]["tags"]["promotion_eligible"] = "False"
        config["simulator"]["export"]["session_name"] += "-smoke"
    return config


def build_probes(
    *,
    pricing_csv: Path,
    neutral_community_csv: Path,
    output_dir: Path,
    discount_community_csv: Path | None = None,
    discount: float = DEFAULT_DISCOUNT,
    neutral: float = DEFAULT_NEUTRAL,
    block_steps: int = 4,
    activation_fraction: float = 0.5,
    minimum_saving: float = 0.0,
    smoke_transitions: int | None = None,
) -> dict[str, Any]:
    if not 0.0 < discount < neutral:
        raise ValueError("Expected 0 < discount < neutral")
    pricing_fields, pricing_rows = _read_csv(pricing_csv)
    neutral_fields, neutral_rows = _read_csv(neutral_community_csv)
    discount_fields: list[str] | None = None
    discount_rows: list[dict[str, str]] | None = None
    if discount_community_csv is not None:
        discount_fields, discount_rows = _read_csv(discount_community_csv)

    masks = derive_schedule_masks(
        pricing_fieldnames=pricing_fields,
        pricing_rows=pricing_rows,
        neutral_fieldnames=neutral_fields,
        neutral_rows=neutral_rows,
        discount_fieldnames=discount_fields,
        discount_rows=discount_rows,
        block_steps=block_steps,
        activation_fraction=activation_fraction,
        minimum_saving=minimum_saving,
    )

    retrospective_estimate: dict[str, float] | None = None
    if discount_rows is not None and discount_fields is not None:
        neutral_cost = _floats(
            neutral_rows,
            _find_column(neutral_fields, "Price"),
        )
        discount_cost = _floats(
            discount_rows,
            _find_column(discount_fields, "Price"),
        )
        neutral_total = sum(neutral_cost)
        discount_total = sum(discount_cost)
        optimistic_saving = 0.0
        optimistic_cost = 0.0
        for start in range(0, len(neutral_cost), block_steps):
            neutral_block = sum(neutral_cost[start : start + block_steps])
            discount_block = sum(discount_cost[start : start + block_steps])
            chosen = min(neutral_block, discount_block)
            optimistic_cost += chosen
            optimistic_saving += neutral_block - chosen
        retrospective_estimate = {
            "neutral_trace_cost_eur": neutral_total,
            "fixed_discount_trace_cost_eur": discount_total,
            "fixed_discount_delta_eur": discount_total - neutral_total,
            "optimistic_independent_block_cost_eur": optimistic_cost,
            "optimistic_independent_block_saving_eur": optimistic_saving,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for recipe, block_mask in masks.items():
        schedule = _compress_schedule(
            block_mask,
            block_steps=block_steps,
            discount=discount,
            neutral=neutral,
        )
        payload = _probe_config(
            recipe=recipe,
            schedule=schedule,
            discount=discount,
            block_steps=block_steps,
            smoke_transitions=smoke_transitions,
        )
        path = output_dir / f"cc_ppo_temporal_{recipe}_seed{PPO_SEED}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        files[recipe] = str(path)
        summaries[recipe] = {
            "blocks": len(block_mask),
            "discount_blocks": sum(block_mask),
            "discount_block_rate": sum(block_mask) / len(block_mask),
            "schedule_entries": len(schedule),
        }
        if recipe == "retrospective_cost" and retrospective_estimate is not None:
            summaries[recipe]["non_evidence_mixed_trace_estimate"] = (
                retrospective_estimate
            )

    sources = {
        "pricing_csv": {"path": str(pricing_csv), "sha256": _sha256(pricing_csv)},
        "neutral_community_csv": {
            "path": str(neutral_community_csv),
            "sha256": _sha256(neutral_community_csv),
        },
    }
    if discount_community_csv is not None:
        sources["discount_community_csv"] = {
            "path": str(discount_community_csv),
            "sha256": _sha256(discount_community_csv),
        }
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "status": "replay_required_in_sample_diagnostics",
        "warning": (
            "The retrospective selector combines outcomes from different state "
            "trajectories. Only a continuous simulator replay is evidence."
        ),
        "discount": discount,
        "neutral": neutral,
        "block_steps": block_steps,
        "activation_fraction": activation_fraction,
        "minimum_saving": minimum_saving,
        "smoke_transitions": smoke_transitions,
        "sources": sources,
        "recipes": summaries,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-csv", type=Path, required=True)
    parser.add_argument("--neutral-community-csv", type=Path, required=True)
    parser.add_argument("--discount-community-csv", type=Path)
    parser.add_argument("--discount", type=float, default=DEFAULT_DISCOUNT)
    parser.add_argument("--neutral", type=float, default=DEFAULT_NEUTRAL)
    parser.add_argument("--block-steps", type=int, default=4)
    parser.add_argument("--activation-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-saving", type=float, default=0.0)
    parser.add_argument("--smoke-transitions", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "remote_configs" / EXPERIMENT_NAME,
    )
    args = parser.parse_args()
    manifest = build_probes(
        pricing_csv=args.pricing_csv,
        neutral_community_csv=args.neutral_community_csv,
        discount_community_csv=args.discount_community_csv,
        output_dir=args.output_dir,
        discount=args.discount,
        neutral=args.neutral,
        block_steps=args.block_steps,
        activation_fraction=args.activation_fraction,
        minimum_saving=args.minimum_saving,
        smoke_transitions=args.smoke_transitions,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
