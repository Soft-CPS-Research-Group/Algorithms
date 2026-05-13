"""KPI-aligned reward for offline RL v2.

The reward is a **weighted sum of five non-negative cost terms**:

    reward_v2_t = - ( w_cost     * C_t
                    + w_carbon   * G_t
                    + w_peak     * P_t
                    + w_ramp     * R_t
                    + w_unserved * U_t )

with

    C_t = price_t        * max(0, e_t)               # cost in $
    G_t = carbon_t       * max(0, e_t)               # carbon kg
    P_t = max(0, e_t - mu_t)                         # peak excursion above 24h mean
    R_t = |e_t - e_{t-1}|                            # ramping
    U_t = required_soc_gap_kWh   if EV departs at t+1 else 0

where ``e_t = obs_net_electricity_consumption`` (positive = grid import).

Weights are calibrated **once** against RBC rollouts (see
``scripts/calibrate_reward_v2.py``) and frozen into
``datasets/offline_rl_v2/derived/reward_v2_weights.json``. The same weights
are reused across behaviour-policy iterations so improvements are measured
on the same yardstick.

This module is intentionally dependency-light (numpy + stdlib). It does not
depend on the env, only on the columns in ``schema_v2``.
"""

from __future__ import annotations

import collections
import dataclasses
import json
from pathlib import Path
from typing import Any, Deque, Dict, Mapping, Optional

import numpy as np

from algorithms.offline_rl_v2 import schema_v2 as S

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEAK_WINDOW_HOURS: int = 24
"""Rolling window over which we compute the local mean for peak excursion.

24 hours = 1 day, matching ``daily_peak_average`` in the CityLearn KPI set.
"""

DEFAULT_WEIGHTS: Dict[str, float] = {
    "cost": 1.0,
    "carbon": 1.0,
    "peak": 2.0,
    "ramp": 1.0,
    "unserved": 50.0,
}
"""Initial-guess weights from ``docs/offline_rl_v2/reward_design_v2.md`` §4.2.

Used only if calibration is skipped or fails. The calibration script
overwrites these with empirically-fit values.
"""

TERM_NAMES: tuple[str, ...] = ("cost", "carbon", "peak", "ramp", "unserved")


# ---------------------------------------------------------------------------
# Per-rollout state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RewardState:
    """Mutable state carried across calls to :func:`compute_reward_v2`.

    One instance per rollout (or per seed). Reset between rollouts.
    """

    rolling_consumption: Deque[float] = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=PEAK_WINDOW_HOURS)
    )
    last_consumption: Optional[float] = None

    def reset(self) -> None:
        self.rolling_consumption.clear()
        self.last_consumption = None


# ---------------------------------------------------------------------------
# Term computation (deterministic, side-effect-free given state)
# ---------------------------------------------------------------------------


def _required_soc_gap_kwh(obs: Mapping[str, float], next_obs: Mapping[str, float]) -> float:
    """Energy still owed to the EV if it departs at t+1.

    Returns 0 unless:
      - a car is currently connected (we can only owe energy to a car that's
        plugged in this step), AND
      - a car is **not** connected at t+1 (i.e. it's leaving, or there's no
        car at all next step) — that's our proxy for "departure between t and
        t+1".

    The shortfall is ``max(0, required_soc - current_soc) * battery_capacity``.
    """
    connected_now = obs.get(
        S.obs_column(f"electric_vehicle_charger_{S.CHARGER_ID}_connected_state"),
        0.0,
    )
    if not connected_now:
        return 0.0

    connected_next = next_obs.get(
        S.next_obs_column(f"electric_vehicle_charger_{S.CHARGER_ID}_connected_state"),
        0.0,
    )
    if connected_next:
        # Same car still here next step → no departure event yet.
        return 0.0

    soc = obs.get(
        S.obs_column(f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_soc"),
        0.0,
    )
    required_soc = obs.get(
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_required_soc_departure"
        ),
        0.0,
    )
    capacity_kwh = obs.get(
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_battery_capacity"
        ),
        0.0,
    )
    gap = max(0.0, float(required_soc) - float(soc))
    return gap * float(capacity_kwh)


def compute_terms(
    obs: Mapping[str, float],
    next_obs: Mapping[str, float],
    *,
    state: RewardState,
) -> Dict[str, float]:
    """Compute the five non-negative cost terms for one transition.

    The function **mutates** ``state`` (appends to the rolling window and
    updates ``last_consumption``). Calling it twice on the same transition
    will double-count the rolling window.
    """
    e = float(obs.get(S.obs_column("net_electricity_consumption"), 0.0))
    price = float(obs.get(S.obs_column("electricity_pricing"), 0.0))
    carbon = float(obs.get(S.obs_column("carbon_intensity"), 0.0))

    # Cost & carbon: only positive (importing) consumption is penalised.
    pos_e = max(0.0, e)
    cost = price * pos_e
    carbon_t = carbon * pos_e

    # Peak excursion: compare to the rolling 24h mean. If we don't have any
    # history yet, mean = e itself → excursion = 0.
    if state.rolling_consumption:
        mu = sum(state.rolling_consumption) / len(state.rolling_consumption)
    else:
        mu = e
    peak = max(0.0, e - mu)

    # Ramping: |e_t - e_{t-1}|. Zero on the first step of a rollout.
    if state.last_consumption is None:
        ramp = 0.0
    else:
        ramp = abs(e - state.last_consumption)

    # Unserved energy at departure.
    unserved = _required_soc_gap_kwh(obs, next_obs)

    # Update state for next call.
    state.rolling_consumption.append(e)
    state.last_consumption = e

    return {
        "cost": cost,
        "carbon": carbon_t,
        "peak": peak,
        "ramp": ramp,
        "unserved": unserved,
    }


def compute_reward_v2(
    obs: Mapping[str, float],
    action: Any,  # noqa: ARG001 — not used; reserved for future shaping
    next_obs: Mapping[str, float],
    *,
    weights: Mapping[str, float],
    state: RewardState,
) -> tuple[float, Dict[str, float]]:
    """Return (reward, term_breakdown) for one transition.

    Pure given (obs, action, next_obs, weights, state).
    """
    terms = compute_terms(obs, next_obs, state=state)
    weighted_sum = sum(float(weights.get(k, 0.0)) * terms[k] for k in TERM_NAMES)
    return -weighted_sum, terms


# ---------------------------------------------------------------------------
# Vectorised version (for batch processing / calibration)
# ---------------------------------------------------------------------------


def compute_terms_vectorised(df) -> Dict[str, np.ndarray]:
    """Compute the five terms for an entire rollout dataframe at once.

    ``df`` must contain the ``obs_*`` and ``next_obs_*`` columns from the
    schema, sorted by ``timestep``. Returns a dict of 1-D numpy arrays of
    length ``len(df)``.

    Equivalent to looping :func:`compute_terms` over the rows, but ~100×
    faster.
    """
    e = df[S.obs_column("net_electricity_consumption")].to_numpy(dtype=np.float64)
    price = df[S.obs_column("electricity_pricing")].to_numpy(dtype=np.float64)
    carbon = df[S.obs_column("carbon_intensity")].to_numpy(dtype=np.float64)

    pos_e = np.maximum(0.0, e)
    cost = price * pos_e
    carbon_t = carbon * pos_e

    # Rolling 24h mean. ``min_periods=1`` so the first 23 steps still get
    # a defined mean (using whatever history exists), matching the loop.
    import pandas as pd

    rolling_mean = pd.Series(e).rolling(PEAK_WINDOW_HOURS, min_periods=1).mean().to_numpy()
    # In the loop, mu is computed from history *before* appending e_t. To
    # match exactly, shift the rolling mean by one step (with a 0th-step
    # value of e[0] so peak[0] = 0).
    mu = np.empty_like(e)
    mu[0] = e[0]
    mu[1:] = rolling_mean[:-1]
    peak = np.maximum(0.0, e - mu)

    # Ramping: zero at t=0.
    ramp = np.zeros_like(e)
    ramp[1:] = np.abs(e[1:] - e[:-1])

    # Unserved: row-by-row using the same logic as the loop. Vectorise the
    # boolean mask, then compute the gap only on the masked rows.
    connected_now = df[
        S.obs_column(f"electric_vehicle_charger_{S.CHARGER_ID}_connected_state")
    ].to_numpy(dtype=np.float64)
    connected_next = df[
        S.next_obs_column(f"electric_vehicle_charger_{S.CHARGER_ID}_connected_state")
    ].to_numpy(dtype=np.float64)
    leaving = (connected_now > 0) & (connected_next <= 0)
    soc = df[
        S.obs_column(f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_soc")
    ].to_numpy(dtype=np.float64)
    required_soc = df[
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_required_soc_departure"
        )
    ].to_numpy(dtype=np.float64)
    capacity = df[
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{S.CHARGER_ID}_battery_capacity"
        )
    ].to_numpy(dtype=np.float64)
    gap = np.maximum(0.0, required_soc - soc) * capacity
    unserved = np.where(leaving, gap, 0.0)

    return {
        "cost": cost,
        "carbon": carbon_t,
        "peak": peak,
        "ramp": ramp,
        "unserved": unserved,
    }


def compute_reward_vectorised(
    df, *, weights: Mapping[str, float]
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply weights to vectorised terms. Returns (reward_array, term_dict)."""
    terms = compute_terms_vectorised(df)
    weighted = np.zeros(len(df), dtype=np.float64)
    for k in TERM_NAMES:
        weighted += float(weights.get(k, 0.0)) * terms[k]
    return -weighted, terms


# ---------------------------------------------------------------------------
# Weights I/O
# ---------------------------------------------------------------------------


def load_weights(path: Path) -> Dict[str, float]:
    """Load a frozen weights file. Validates structure."""
    with open(path) as f:
        data = json.load(f)
    if "weights" not in data:
        raise ValueError(f"weights file {path} missing 'weights' key")
    weights = data["weights"]
    missing = [k for k in TERM_NAMES if k not in weights]
    if missing:
        raise ValueError(f"weights file {path} missing terms: {missing}")
    return {k: float(weights[k]) for k in TERM_NAMES}


def save_weights(
    path: Path,
    weights: Mapping[str, float],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist weights + metadata."""
    data: Dict[str, Any] = {
        "weights": {k: float(weights[k]) for k in TERM_NAMES},
        "term_names": list(TERM_NAMES),
        "peak_window_hours": PEAK_WINDOW_HOURS,
    }
    if metadata:
        data["metadata"] = dict(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
