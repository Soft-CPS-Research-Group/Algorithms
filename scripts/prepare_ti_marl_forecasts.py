"""Reproduce a causal 24-hour Predictor feed for short Core-30 experiments.

Each record covers 96 physical quarter-hours. Load and absolute PV use the
previous day's observations or a ridge model fitted to earlier meter readings.
Price uses the dataset's already issued retail
tariff estimates at 1, 6 and 24 hours, without reading a future realised price.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

if __package__:
    from .ti_marl_weather_pv import HISTORY, SOLAR_LEADS, WEATHER_COLUMNS, WeatherPV
else:  # Also support invoking this file directly.
    from ti_marl_weather_pv import HISTORY, SOLAR_LEADS, WEATHER_COLUMNS, WeatherPV


SLOTS_PER_DAY = 96
FIRST_FIT_ISSUE = 7 * SLOTS_PER_DAY + 2
# Fit separate coefficients for short and long leads, pooling nearby leads.
LEAD_GROUPS = (
    (0, 2, (0, 1, 2)),
    (3, 8, (4, 7)),
    (9, 24, (12, 20)),
    (25, 48, (28, 40)),
    (49, 95, (56, 76, 95)),
)


def _meter_features(meters, issues, leads):
    issues, leads = np.broadcast_arrays(issues, leads)
    targets = issues + leads
    days = np.stack([meters[targets - day * SLOTS_PER_DAY] for day in range(1, 8)])
    typical = days.mean(axis=0)
    last = meters[issues - 1]
    previous = meters[issues - 2]
    typical_last = np.mean(
        [meters[issues - 1 - day * SLOTS_PER_DAY] for day in range(1, 8)], axis=0
    )
    # 0.15 kW keeps the relative update bounded when the meter is near zero.
    recent_ratio = np.clip((last + 0.15) / (typical_last + 0.15), 0, 2.5)
    return np.stack(
        (
            days[0],
            np.median(days[:3], axis=0),
            typical,
            days[-1],
            np.maximum(0, typical + last - typical_last),
            typical * recent_ratio,
            np.maximum(0, last + (last - previous)),
            last,
        ),
        axis=-1,
    )


def historical_passive_export(meters):
    """Upper estimate by delivery quarter using only earlier paired meter readings.

    The preceding 28 occurrences determine the peak. Every past positive miss
    of this estimate at the same quarter is retained, with at least 28 prior errors
    observed before publication. No decision-time weather or actor inputs change.
    """
    net = np.asarray(meters[:, 1], dtype=float) - meters[:, 0]
    # Include targets just beyond the last issue. Their most recent input is
    # still at least 84 quarters old at issuance.
    future = np.full(len(net) + SOLAR_LEADS, np.nan)
    for quarter in range(SLOTS_PER_DAY):
        observed = pd.Series(net[quarter::SLOTS_PER_DAY])
        series = observed.reindex(range(len(future[quarter::SLOTS_PER_DAY])))
        peak = series.shift(1).rolling(28, min_periods=28).max()
        error = (series - peak).clip(lower=0)
        allowance = error.shift(1).expanding(min_periods=28).max()
        future[quarter::SLOTS_PER_DAY] = (peak + allowance).to_numpy()
    return future


@dataclass(frozen=True)
class MeterForecaster:
    """Per-site ridge forecasts for passive kW demand and PV, indexed by lead quarter."""

    weights: np.ndarray
    intercepts: np.ndarray

    @classmethod
    def fit(cls, meters, fit_end):
        if not FIRST_FIT_ISSUE + 95 < fit_end <= len(meters):
            raise ValueError("Meter fitting needs seven days of lags and complete 24-hour labels")
        weights = np.empty((SLOTS_PER_DAY, 2, 8))
        intercepts = np.empty((SLOTS_PER_DAY, 2))
        for lo, hi, leads in LEAD_GROUPS:
            issues = np.arange(FIRST_FIT_ISSUE, fit_end - max(leads))
            history = np.concatenate([_meter_features(meters, issues, lead) for lead in leads])
            observed = np.concatenate([meters[issues + lead] for lead in leads])
            for channel in range(2):
                features = history[:, channel, :]
                mean = features.mean(axis=0)
                scale = np.maximum(features.std(axis=0), 0.01)
                centred = (features - mean) / scale
                target = observed[:, channel]
                gram = centred.T @ centred + np.eye(centred.shape[1]) * (0.01 * len(centred))
                coef = np.linalg.solve(gram, centred.T @ (target - target.mean())) / scale
                weights[lo : hi + 1, channel] = coef
                intercepts[lo : hi + 1, channel] = target.mean() - mean @ coef
        return cls(weights, intercepts)

    def predict(self, meters, issues, leads):
        features = _meter_features(meters, issues, leads)
        estimate = (features * self.weights[leads]).sum(axis=-1) + self.intercepts[leads]
        return np.maximum(0, estimate).round(6)


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
    schema_path: Path,
    output: Path,
    first: int,
    last: int,
    calibration_end: int | None = None,
    *,
    meter_fit_end: int | None = None,
    weather_fit_end: int | None = None,
    passive_export_history: bool = False,
):
    schema = json.loads(schema_path.read_text())
    if schema["seconds_per_time_step"] != 900:
        raise ValueError("This reference estimator requires native 15-minute data")
    total = int(schema["simulation_end_time_step"]) + 1
    if first < 0 or last < first or last >= total:
        raise ValueError("The requested issue range is outside the dataset")
    if calibration_end is not None and not 96 < calibration_end <= first:
        raise ValueError("Calibration needs prior-day residuals strictly before the first forecast")
    if meter_fit_end is not None:
        if not FIRST_FIT_ISSUE + 95 < meter_fit_end <= first:
            raise ValueError("Meter fitting needs historical labels ending before the first issue")
        if calibration_end is not None and calibration_end <= meter_fit_end + 1:
            raise ValueError("Calibration needs a completed one-step forecast after fitting")
    if weather_fit_end is not None:
        if not HISTORY + 96 <= weather_fit_end <= first:
            raise ValueError("Next-quarter fitting requires 23 prior days before the first issue")
        if calibration_end is not None and calibration_end <= weather_fit_end + 1:
            raise ValueError("Next-quarter calibration needs completed observations after fitting")
    root = (schema_path.parent / schema.get("root_directory", ".")).resolve()
    first_prediction = max(weather_fit_end or 0, meter_fit_end or 0)
    sites, tariffs, weather_files = {}, {}, {}
    export_reference = {}
    for site, building in schema["buildings"].items():
        if not building.get("include", True):
            continue
        pv = building.get("pv") or {}
        if pv and pv.get("attributes", {}).get("generation_mode") != "absolute":
            raise ValueError(f"{site} requires absolute PV generation in kWh per native step")
        source = pd.read_parquet(
            root / building["energy_simulation"], columns=["non_shiftable_load", "solar_generation"]
        )
        # Fit in the precision of the native Core-30 meter readings.
        meter_dtype = float if meter_fit_end is None else np.float32
        physical = source.iloc[: last + 1].to_numpy(dtype=meter_dtype) * 4  # kWh per quarter -> kW
        if not pv:
            physical[:, 1] = 0
        if not np.isfinite(physical).all() or (physical < 0).any():
            raise ValueError(f"Invalid physical profile for {site}")
        if passive_export_history:
            export_reference[site] = historical_passive_export(physical)
        tariff_name = building["pricing"]
        if tariff_name not in tariffs:
            tariffs[tariff_name] = pd.read_parquet(root / tariff_name)
        model = MeterForecaster.fit(physical, meter_fit_end) if meter_fit_end is not None else None
        next_values = None
        if weather_fit_end is not None:
            origins = np.arange(first_prediction, last + 1)
            next_pv = np.zeros((len(origins), SOLAR_LEADS))
            if pv:
                weather_path = root / building["weather"]
                if weather_path not in weather_files:
                    weather_files[weather_path] = (
                        pd.read_parquet(weather_path, columns=list(WEATHER_COLUMNS))
                        .iloc[: last + 1]
                        .astype(float)
                    )
                weather = weather_files[weather_path]
                for lead in range(1, SOLAR_LEADS + 1):
                    estimator = WeatherPV.fit(physical[:, 1], weather, weather_fit_end, lead)
                    next_pv[:, lead - 1] = estimator.predict(physical[:, 1], weather, origins)
            # The live non-shiftable demand is an exogenous reading at the issue time.
            next_values = np.column_stack((physical[origins, 0], next_pv))
        error = None
        if calibration_end is not None:
            if next_values is not None:
                stop = calibration_end - first_prediction - 1
                residuals = (
                    physical[first_prediction + 1 : calibration_end] - next_values[:stop, :2]
                )
            elif model is None:
                residuals = physical[96:calibration_end] - physical[: calibration_end - 96]
            else:
                issues = np.arange(meter_fit_end, calibration_end - 1)
                residuals = physical[issues + 1] - model.predict(physical, issues, 1)
            # Keep errors separate so load and PV bounds cannot cross zero.
            error = (1.1 * np.abs(residuals).max(axis=0)).tolist()
        sites[site] = physical, tariff_name, error, model, next_values
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
            for site, (physical, tariff_name, error, model, next_values) in sites.items():
                forecast = (
                    physical[history]
                    if model is None
                    else model.predict(physical, issue, targets - issue)
                )
                if next_values is not None:
                    short = next_values[issue - first_prediction]
                    forecast[1] = short[:2]
                    forecast[2 : SOLAR_LEADS + 1, 1] = (
                        forecast[2 : SOLAR_LEADS + 1, 1] + short[2:]
                    ) / 2
                record = {
                    "site": site,
                    "issued_at_seconds": issue * 900,
                    "received_at_seconds": issue * 900,
                    "valid_from_seconds": issue * 900,
                    "interval_seconds": 900,
                    "price_eur_per_kwh": prices[tariff_name],
                    "non_shiftable_kw": forecast[:, 0].tolist(),
                    "solar_kw": forecast[:, 1].tolist(),
                }
                if passive_export_history:
                    upper = export_reference[site][issue + 1 : issue + SOLAR_LEADS + 1]
                    if np.isfinite(upper).all():
                        record["passive_export_upper_kw"] = upper.tolist()
                        record["passive_export_history_end_seconds"] = (
                            issue + SOLAR_LEADS - SLOTS_PER_DAY + 1
                        ) * 900
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
    parser.add_argument(
        "--meter-fit-end",
        type=int,
        help="Use a fitted passive demand and solar forecast. Excludes this native step and later labels.",
    )
    parser.add_argument(
        "--weather-fit-end",
        type=int,
        help="Use live base demand for one quarter and issued weather for three hours. "
        "Solar model labels end strictly before this native step.",
    )
    parser.add_argument(
        "--passive-export-history",
        action="store_true",
        help="Include a separate same-quarter passive export estimate for the supervisor.",
    )
    options = parser.parse_args()
    prepare(
        options.schema,
        options.output,
        options.first,
        options.last,
        options.calibration_end,
        meter_fit_end=options.meter_fit_end,
        weather_fit_end=options.weather_fit_end,
        passive_export_history=options.passive_export_history,
    )
