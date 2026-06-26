"""Rank Phase 10 training windows by opportunity to improve over an RBC baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COMMUNITY_NET_COLUMN = "Net Electricity Consumption-kWh"
COMMUNITY_SELF_CONSUMPTION_COLUMN = "Self Consumption-kWh"
COMMUNITY_SOLAR_COLUMN = "Total Solar Generation-kWh"
COMMUNITY_COST_COLUMN = "Price-$"
TIMESTAMP_COLUMN = "timestamp"
PRICE_PREFIX = "electricity_pricing-"

SCORE_WEIGHTS = {
    "import_cost": 0.30,
    "peak_import": 0.20,
    "solar_spill": 0.20,
    "export": 0.15,
    "price_spread": 0.10,
    "import": 0.05,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if parsed != parsed else parsed


def _first_existing(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _stored_energy_column(row: Mapping[str, Any]) -> str | None:
    for name in row:
        lowered = name.lower().replace(" ", "")
        if "storedenergybycommunity" in lowered:
            return name
    return None


def _solar_generation_magnitude(value: Any) -> float:
    """Return solar generation as a positive physical magnitude.

    Simulator community exports can encode production as negative energy in the
    net-balance convention. Opportunity ranking needs generated energy, not the
    accounting sign.
    """

    return abs(_to_float(value))


def _pricing_by_timestamp(pricing_path: Path | None) -> dict[str, float]:
    if pricing_path is None or not pricing_path.exists():
        return {}
    prices: dict[str, float] = {}
    for row in _read_csv(pricing_path):
        price_column = next((name for name in row if name.startswith(PRICE_PREFIX)), "")
        if not price_column:
            continue
        timestamp = str(row.get(TIMESTAMP_COLUMN) or "")
        if timestamp:
            prices[timestamp] = _to_float(row.get(price_column))
    return prices


def _load_community_steps(community_path: Path, pricing_path: Path | None = None) -> list[dict[str, float | str]]:
    prices = _pricing_by_timestamp(pricing_path)
    rows = _read_csv(community_path)
    if not rows:
        return []
    stored_column = _stored_energy_column(rows[0])
    output: list[dict[str, float | str]] = []

    for index, row in enumerate(rows):
        timestamp = str(row.get(TIMESTAMP_COLUMN) or index)
        net = _to_float(row.get(COMMUNITY_NET_COLUMN))
        solar = _solar_generation_magnitude(row.get(COMMUNITY_SOLAR_COLUMN))
        self_consumption = max(0.0, _to_float(row.get(COMMUNITY_SELF_CONSUMPTION_COLUMN)))
        stored = max(0.0, _to_float(row.get(stored_column)) if stored_column else 0.0)
        price = prices.get(timestamp, 0.0)
        step_cost = _to_float(row.get(COMMUNITY_COST_COLUMN), default=max(net, 0.0) * price)
        import_kwh = max(net, 0.0)
        export_kwh = max(-net, 0.0)
        solar_spill = max(solar - self_consumption - stored, 0.0)
        output.append(
            {
                "index": float(index),
                "timestamp": timestamp,
                "net_kwh": net,
                "import_kwh": import_kwh,
                "export_kwh": export_kwh,
                "solar_generation_kwh": solar,
                "self_consumption_kwh": self_consumption,
                "stored_kwh": stored,
                "solar_spill_kwh": solar_spill,
                "price_eur_per_kwh": price,
                "step_cost_eur": step_cost,
                "price_weighted_import": import_kwh * price,
            }
        )
    return output


def _sum_metric(rows: Sequence[Mapping[str, float | str]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def _max_metric(rows: Sequence[Mapping[str, float | str]], key: str) -> float:
    if not rows:
        return 0.0
    return max(float(row.get(key, 0.0) or 0.0) for row in rows)


def _min_metric(rows: Sequence[Mapping[str, float | str]], key: str) -> float:
    if not rows:
        return 0.0
    return min(float(row.get(key, 0.0) or 0.0) for row in rows)


def _window_rows(
    steps: Sequence[Mapping[str, float | str]],
    *,
    window_steps: int,
    stride_steps: int,
) -> Iterable[tuple[int, int, Sequence[Mapping[str, float | str]]]]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    if stride_steps <= 0:
        raise ValueError("stride_steps must be positive")
    if len(steps) < window_steps:
        return
    for start in range(0, len(steps) - window_steps + 1, stride_steps):
        end = start + window_steps
        yield start, end, steps[start:end]


def _window_metrics(
    steps: Sequence[Mapping[str, float | str]],
    *,
    start: int,
    end: int,
    prefix: str = "",
) -> dict[str, Any]:
    solar = _sum_metric(steps, "solar_generation_kwh")
    self_consumption = _sum_metric(steps, "self_consumption_kwh")
    import_kwh = _sum_metric(steps, "import_kwh")
    export_kwh = _sum_metric(steps, "export_kwh")
    spill = _sum_metric(steps, "solar_spill_kwh")
    price_max = _max_metric(steps, "price_eur_per_kwh")
    price_min = _min_metric(steps, "price_eur_per_kwh")
    first = str(steps[0].get("timestamp", start)) if steps else str(start)
    last = str(steps[-1].get("timestamp", end - 1)) if steps else str(end - 1)
    values: dict[str, Any] = {
        "start_step": start,
        "end_step": end,
        "start_timestamp": first,
        "end_timestamp": last,
        f"{prefix}community_cost_eur": _sum_metric(steps, "step_cost_eur"),
        f"{prefix}price_weighted_import": _sum_metric(steps, "price_weighted_import"),
        f"{prefix}import_kwh": import_kwh,
        f"{prefix}export_kwh": export_kwh,
        f"{prefix}peak_import_kwh": _max_metric(steps, "import_kwh"),
        f"{prefix}solar_generation_kwh": solar,
        f"{prefix}self_consumption_kwh": self_consumption,
        f"{prefix}solar_spill_kwh": spill,
        f"{prefix}self_consumption_rate": self_consumption / solar if solar > 0.0 else None,
        f"{prefix}price_min": price_min,
        f"{prefix}price_max": price_max,
        f"{prefix}price_spread": price_max - price_min,
    }
    return values


def _safe_ratio(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return max(0.0, value / maximum)


def _classify(row: Mapping[str, Any]) -> str:
    labels: list[str] = []
    if _to_float(row.get("score_peak_import_component")) >= 0.60:
        labels.append("peak_shaving")
    if _to_float(row.get("score_solar_spill_component")) >= 0.60 or _to_float(row.get("score_export_component")) >= 0.60:
        labels.append("solar_absorption")
    if _to_float(row.get("score_price_spread_component")) >= 0.60:
        labels.append("price_arbitrage")
    if _to_float(row.get("candidate_cost_delta_eur")) > 0.0:
        labels.append("candidate_regression")
    return "+".join(labels) if labels else "balanced"


def build_opportunity_windows(
    baseline_community_path: Path,
    *,
    baseline_pricing_path: Path | None = None,
    candidate_community_path: Path | None = None,
    candidate_pricing_path: Path | None = None,
    window_steps: int = 512,
    stride_steps: int = 256,
    top: int | None = None,
) -> list[dict[str, Any]]:
    """Build ranked opportunity windows from RBC community timeseries.

    The score is deliberately heuristic: it surfaces windows where an RL residual
    policy has useful work to do, not final performance claims.
    """

    baseline = _load_community_steps(baseline_community_path, baseline_pricing_path)
    candidate = (
        _load_community_steps(candidate_community_path, candidate_pricing_path)
        if candidate_community_path is not None
        else []
    )
    if candidate and len(candidate) != len(baseline):
        raise ValueError("candidate and baseline community timeseries must have the same number of rows")

    rows: list[dict[str, Any]] = []
    for start, end, window in _window_rows(
        baseline,
        window_steps=window_steps,
        stride_steps=stride_steps,
    ):
        row = _window_metrics(window, start=start, end=end)
        if candidate:
            candidate_row = _window_metrics(candidate[start:end], start=start, end=end, prefix="candidate_")
            row.update(candidate_row)
            row["candidate_cost_delta_eur"] = (
                row["candidate_community_cost_eur"] - row["community_cost_eur"]
            )
            row["candidate_import_delta_kwh"] = row["candidate_import_kwh"] - row["import_kwh"]
            row["candidate_export_delta_kwh"] = row["candidate_export_kwh"] - row["export_kwh"]
            row["candidate_peak_import_delta_kwh"] = (
                row["candidate_peak_import_kwh"] - row["peak_import_kwh"]
            )
            candidate_self_rate = row.get("candidate_self_consumption_rate")
            baseline_self_rate = row.get("self_consumption_rate")
            row["candidate_self_consumption_rate_delta"] = (
                None
                if candidate_self_rate is None or baseline_self_rate is None
                else float(candidate_self_rate) - float(baseline_self_rate)
            )
        rows.append(row)

    maxima = {
        "import_cost": max((_to_float(row.get("price_weighted_import")) for row in rows), default=0.0),
        "peak_import": max((_to_float(row.get("peak_import_kwh")) for row in rows), default=0.0),
        "solar_spill": max((_to_float(row.get("solar_spill_kwh")) for row in rows), default=0.0),
        "export": max((_to_float(row.get("export_kwh")) for row in rows), default=0.0),
        "price_spread": max((_to_float(row.get("price_spread")) for row in rows), default=0.0),
        "import": max((_to_float(row.get("import_kwh")) for row in rows), default=0.0),
    }

    for row in rows:
        components = {
            "import_cost": _safe_ratio(_to_float(row.get("price_weighted_import")), maxima["import_cost"]),
            "peak_import": _safe_ratio(_to_float(row.get("peak_import_kwh")), maxima["peak_import"]),
            "solar_spill": _safe_ratio(_to_float(row.get("solar_spill_kwh")), maxima["solar_spill"]),
            "export": _safe_ratio(_to_float(row.get("export_kwh")), maxima["export"]),
            "price_spread": _safe_ratio(_to_float(row.get("price_spread")), maxima["price_spread"]),
            "import": _safe_ratio(_to_float(row.get("import_kwh")), maxima["import"]),
        }
        for name, value in components.items():
            row[f"score_{name}_component"] = value
        row["opportunity_score"] = sum(SCORE_WEIGHTS[name] * value for name, value in components.items())
        row["opportunity_type"] = _classify(row)

    rows.sort(key=lambda item: float(item["opportunity_score"]), reverse=True)
    if top is not None:
        return rows[: max(0, top)]
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 10 Opportunity Windows",
        "",
        "These windows are heuristic training candidates, not final performance claims.",
        "",
        "| rank | steps | timestamp | score | type | cost | import | export | peak | solar spill |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {start}-{end} | {ts} | {score:.3f} | {kind} | {cost:.1f} | {imp:.1f} | {exp:.1f} | {peak:.1f} | {spill:.1f} |".format(
                rank=index,
                start=row["start_step"],
                end=row["end_step"],
                ts=row["start_timestamp"],
                score=float(row["opportunity_score"]),
                kind=row["opportunity_type"],
                cost=_to_float(row.get("community_cost_eur")),
                imp=_to_float(row.get("import_kwh")),
                exp=_to_float(row.get("export_kwh")),
                peak=_to_float(row.get("peak_import_kwh")),
                spill=_to_float(row.get("solar_spill_kwh")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-community", type=Path, required=True)
    parser.add_argument("--baseline-pricing", type=Path)
    parser.add_argument("--candidate-community", type=Path)
    parser.add_argument("--candidate-pricing", type=Path)
    parser.add_argument("--window-steps", type=int, default=512)
    parser.add_argument("--stride-steps", type=int, default=256)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    rows = build_opportunity_windows(
        args.baseline_community,
        baseline_pricing_path=args.baseline_pricing,
        candidate_community_path=args.candidate_community,
        candidate_pricing_path=args.candidate_pricing,
        window_steps=args.window_steps,
        stride_steps=args.stride_steps,
        top=args.top,
    )
    _write_csv(args.output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.output_md:
        _write_markdown(args.output_md, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
