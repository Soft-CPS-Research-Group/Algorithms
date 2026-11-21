"""Reproduce a causal 24-hour Predictor feed for short Core-30 experiments.

Each record covers 96 physical quarter-hours. Load and absolute PV use the
previous day's observed data. Price uses the dataset's already issued retail
tariff estimates at 1, 6 and 24 hours, without reading a future realised price.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd


@contextmanager
def new_archive(output: Path):
    """Publish a complete archive. Concurrent writers cannot replace an existing file."""
    if output.exists():
        raise FileExistsError(output)
    with NamedTemporaryFile("w", dir=output.parent, encoding="utf-8") as staging:
        yield staging
        staging.flush()
        os.link(staging.name, output)


def prepare(
    schema_path: Path, output: Path, first: int, last: int, calibration_end: int | None = None
):
    schema = json.loads(schema_path.read_text())
    if schema["seconds_per_time_step"] != 900:
        raise ValueError("This reference estimator requires native 15-minute data")
    total = int(schema["simulation_end_time_step"]) + 1
    if first < 0 or last < first or last >= total:
        raise ValueError("The requested issue range is outside the dataset")
    if calibration_end is not None and not 96 < calibration_end <= first:
        raise ValueError("Calibration needs prior-day residuals strictly before the first forecast")
    root = (schema_path.parent / schema.get("root_directory", ".")).resolve()
    sites, tariffs = {}, {}
    for site, building in schema["buildings"].items():
        if not building.get("include", True):
            continue
        pv = building.get("pv") or {}
        if pv and pv.get("attributes", {}).get("generation_mode") != "absolute":
            raise ValueError(f"{site} requires absolute PV generation in kWh per native step")
        source = pd.read_parquet(
            root / building["energy_simulation"], columns=["non_shiftable_load", "solar_generation"]
        )
        physical = source.iloc[: last + 1].to_numpy(dtype=float) * 4  # kWh over 15 minutes -> kW
        if not pv:
            physical[:, 1] = 0
        if not np.isfinite(physical).all() or (physical < 0).any():
            raise ValueError(f"Invalid physical profile for {site}")
        tariff_name = building["pricing"]
        if tariff_name not in tariffs:
            tariffs[tariff_name] = pd.read_parquet(root / tariff_name)
        error = None
        if calibration_end is not None:
            residuals = physical[96:calibration_end] - physical[: calibration_end - 96]
            # Keep errors separate so load and PV bounds cannot cross zero.
            error = (1.1 * np.abs(residuals).max(axis=0)).tolist()
        sites[site] = physical, tariff_name, error
    with new_archive(output) as destination:
        for issue in range(first, last + 1):
            targets = issue + np.arange(96)
            # At cold start even the prior-day reading may be unavailable.
            history = np.where(targets >= 96, targets - 96, issue)
            prices = {}
            for name, frame in tariffs.items():
                series = []
                for target in targets:
                    price = None
                    if target == issue:
                        price = float(frame["electricity_pricing"].iat[issue])
                    elif target < total:
                        for column, horizon in enumerate((4, 24, 96), start=1):
                            published_step = int(target - horizon)
                            if 0 <= published_step <= issue:
                                price = float(
                                    frame[f"electricity_pricing_predicted_{column}"].iat[
                                        published_step
                                    ]
                                )
                                break
                    series.append(price if price is not None and np.isfinite(price) else None)
                prices[name] = series
            for site, (physical, tariff_name, error) in sites.items():
                record = {
                    "site": site,
                    "issued_at_seconds": issue * 900,
                    "received_at_seconds": issue * 900,
                    "valid_from_seconds": issue * 900,
                    "interval_seconds": 900,
                    "price_eur_per_kwh": prices[tariff_name],
                    "non_shiftable_kw": physical[history, 0].tolist(),
                    "solar_kw": physical[history, 1].tolist(),
                }
                if error is not None:
                    record.update(
                        calibration_error_kw=error, calibration_end_seconds=calibration_end * 900
                    )
                destination.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--first", type=int, required=True, help="First issue time, in native steps"
    )
    parser.add_argument("--last", type=int, required=True, help="Last issue time, inclusive")
    parser.add_argument(
        "--calibration-end", type=int, help="Exclusive end of the frozen preparation window"
    )
    options = parser.parse_args()
    prepare(options.schema, options.output, options.first, options.last, options.calibration_end)
