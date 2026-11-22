"""Causal short-horizon PV forecasts from metered production and issued weather."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY = 96
HISTORY = 22 * DAY
SOLAR_LEADS = 12
WEATHER_CHANNELS = ("direct_solar_irradiance", "diffuse_solar_irradiance")
WEATHER_COLUMNS = tuple(
    column
    for name in WEATHER_CHANNELS
    for column in (name, f"{name}_predicted_1", f"{name}_predicted_2")
)


def _features(pv, weather, issues, fit_end, horizon):
    """At issue t, PV is metered through t-1 and weather is available through t."""
    t = issues
    target = t + horizon
    last = pv[t - 1]
    slope = last - pv[t - 2]
    theta = 2 * np.pi * (target % DAY) / DAY
    theta_now = 2 * np.pi * (t % DAY) / DAY
    typical = np.median([pv[target - day * DAY] for day in range(1, 8)], axis=0)
    columns = [last, slope, pv[t - 2] - pv[t - 3], typical, pv[target - DAY]]
    for harmonic in (1, 2, 3):
        for phase in (np.sin(harmonic * theta), np.cos(harmonic * theta)):
            columns.extend((phase, phase * slope, phase * last))
    for day in (7, 14, 21):
        columns.extend((pv[target - day * DAY], last - pv[t - day * DAY]))
    for channel in WEATHER_CHANNELS:
        radiation = weather[channel].to_numpy()
        latest = radiation[t]
        change = latest - radiation[t - 1]
        previous = radiation[t - 1] - radiation[t - 2]
        ratio = np.minimum(2.0, last / (radiation[t - 1] + 10.0))
        columns.extend((latest, change, previous, change - previous, change * ratio))
        for harmonic in (1, 2):
            for phase in (np.sin(harmonic * theta_now), np.cos(harmonic * theta_now)):
                columns.extend((change * phase, previous * phase, latest * phase))
        for suffix, lead in ((1, 4), (2, 24)):
            # Use a weather record issued by t, even for a longer delivery lead.
            columns.append(
                weather[f"{channel}_predicted_{suffix}"].to_numpy()[np.minimum(t, target - lead)]
            )
    elapsed_years = (target - fit_end) / (365 * DAY)
    columns.extend((elapsed_years, last * elapsed_years))
    return np.column_stack(columns)


@dataclass(frozen=True)
class WeatherPV:
    fit_end: int
    horizon: int
    weights: np.ndarray
    intercepts: np.ndarray

    @classmethod
    def fit(cls, pv, weather: pd.DataFrame, fit_end: int, horizon: int = 1):
        """Fit four ridge models for the quarter's position in the weather reporting hour."""
        pv = np.asarray(pv, dtype=float)
        if not 1 <= horizon <= SOLAR_LEADS:
            raise ValueError("Weather forecast lead must be between 1 and 12 quarters")
        if not HISTORY + 96 <= fit_end <= min(len(pv), len(weather)):
            raise ValueError("Next-quarter PV fitting needs 22 days of lags and a day of labels")
        # Labels end strictly before fit_end. Keep the most recent 48 days.
        origins = np.arange(max(HISTORY, fit_end - 48 * DAY), fit_end - horizon)
        features = _features(pv, weather, origins, fit_end, horizon)
        change = pv[origins + horizon] - pv[origins - 1]
        if not np.isfinite(features).all() or not np.isfinite(change).all():
            raise ValueError("Nonfinite historical PV or weather during next-quarter fitting")
        weights, intercepts = [], []
        for quarter in range(4):
            keep = origins % 4 == quarter
            x, y = features[keep], change[keep]
            centre, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.025)
            z = (x - centre) / scale
            normal = z.T @ z + np.eye(z.shape[1]) * (0.001 * len(z))
            coef = np.linalg.solve(normal, z.T @ (y - y.mean())) / scale
            weights.append(coef)
            intercepts.append(y.mean() - centre @ coef)
        return cls(fit_end, horizon, np.stack(weights), np.array(intercepts))

    def predict(self, pv, weather, issues):
        issues = np.asarray(issues, dtype=int)
        if np.any(issues < self.fit_end) or np.any(issues >= min(len(pv), len(weather))):
            raise ValueError("Forecast issue outside available meter and weather history")
        pv = np.asarray(pv, dtype=float)
        x = _features(pv, weather, issues, self.fit_end, self.horizon)
        phases = issues % 4
        result = pv[issues - 1] + (x * self.weights[phases]).sum(axis=1) + self.intercepts[phases]
        if not np.isfinite(result).all():
            raise ValueError("Missing issued weather or PV for the next-quarter forecast")
        return np.maximum(0, result).round(6)
