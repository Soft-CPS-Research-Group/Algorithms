"""Unit tests for ``algorithms.offline_rl.reward``.

Implements the test list from ``docs/offline_rl/reward_design.md`` §5.2.
Each test is intentionally small and uses hand-crafted observations so the
expected behaviour is unambiguous.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from algorithms.offline_rl import reward as R
from algorithms.offline_rl import schema as S


# ---------------------------------------------------------------------------
# Helpers — build minimally valid obs / next_obs dicts
# ---------------------------------------------------------------------------

CHARGER = S.CHARGER_ID


def make_obs(
    *,
    net: float = 0.0,
    price: float = 0.20,
    carbon: float = 0.30,
    connected: bool = False,
    soc: float = 0.5,
    required_soc: float = 0.8,
    capacity_kwh: float = 60.0,
) -> dict:
    """Build an obs dict with all schema columns populated to defaults."""
    obs = {S.obs_column(name): 0.0 for name in S.OBSERVATION_NAMES}
    obs[S.obs_column("net_electricity_consumption")] = net
    obs[S.obs_column("electricity_pricing")] = price
    obs[S.obs_column("carbon_intensity")] = carbon
    obs[S.obs_column(f"electric_vehicle_charger_{CHARGER}_connected_state")] = (
        1.0 if connected else 0.0
    )
    obs[S.obs_column(f"connected_electric_vehicle_at_charger_{CHARGER}_soc")] = soc
    obs[
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{CHARGER}_required_soc_departure"
        )
    ] = required_soc
    obs[
        S.obs_column(
            f"connected_electric_vehicle_at_charger_{CHARGER}_battery_capacity"
        )
    ] = capacity_kwh
    return obs


def make_next_obs(*, connected: bool = False, **kwargs) -> dict:
    """Build a next_obs dict (only the fields we read need to be non-default)."""
    nobs = {S.next_obs_column(name): 0.0 for name in S.OBSERVATION_NAMES}
    nobs[
        S.next_obs_column(f"electric_vehicle_charger_{CHARGER}_connected_state")
    ] = (1.0 if connected else 0.0)
    for k, v in kwargs.items():
        nobs[k] = v
    return nobs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_finite_on_zero_action():
    """Reward is finite when action=0 (and obs are zero-ish)."""
    state = R.RewardState()
    obs = make_obs(net=0.0)
    nobs = make_next_obs()
    r, terms = R.compute_reward(obs, np.zeros(2), nobs, weights=R.DEFAULT_WEIGHTS, state=state)
    assert math.isfinite(r)
    assert all(math.isfinite(v) for v in terms.values())
    assert r == 0.0  # all terms are zero


def test_sign_cost_negative():
    """Importing energy at positive price → reward strictly decreases."""
    state = R.RewardState()
    obs = make_obs(net=5.0, price=0.30, carbon=0.0)
    nobs = make_next_obs()
    r, terms = R.compute_reward(obs, None, nobs, weights={"cost": 1.0, "carbon": 0, "peak": 0, "ramp": 0, "unserved": 0}, state=state)
    assert terms["cost"] == pytest.approx(0.30 * 5.0)
    assert r < 0


def test_sign_carbon_negative():
    """Importing energy at positive carbon intensity → reward strictly decreases."""
    state = R.RewardState()
    obs = make_obs(net=5.0, price=0.0, carbon=0.40)
    nobs = make_next_obs()
    r, terms = R.compute_reward(
        obs, None, nobs,
        weights={"cost": 0, "carbon": 1.0, "peak": 0, "ramp": 0, "unserved": 0},
        state=state,
    )
    assert terms["carbon"] == pytest.approx(0.40 * 5.0)
    assert r < 0


def test_peak_only_above_mean():
    """If e_t < rolling mean, peak term = 0."""
    state = R.RewardState()
    # Warm up the rolling window with high values.
    for _ in range(10):
        obs = make_obs(net=10.0)
        R.compute_terms(obs, make_next_obs(), state=state)
    # Now a low value; should produce zero peak excursion.
    low_obs = make_obs(net=1.0)
    terms = R.compute_terms(low_obs, make_next_obs(), state=state)
    assert terms["peak"] == 0.0


def test_ramp_zero_on_constant_load():
    """Constant net consumption for many steps → ramp ≈ 0 after the first step."""
    state = R.RewardState()
    constant = 5.0
    ramps = []
    for _ in range(24):
        terms = R.compute_terms(make_obs(net=constant), make_next_obs(), state=state)
        ramps.append(terms["ramp"])
    # First step ramp is 0 (no last_consumption); all subsequent should be 0 too.
    assert max(ramps) == 0.0


def test_unserved_only_at_departure():
    """EV not at required SoC, but car still connected next step → unserved=0."""
    state = R.RewardState()
    obs = make_obs(connected=True, soc=0.3, required_soc=0.9, capacity_kwh=60.0)
    # Car still connected next step.
    nobs = make_next_obs(connected=True)
    terms = R.compute_terms(obs, nobs, state=state)
    assert terms["unserved"] == 0.0


def test_unserved_fires_at_departure():
    """If car connected now, gone next step, with SoC < required → unserved > 0."""
    state = R.RewardState()
    obs = make_obs(connected=True, soc=0.3, required_soc=0.9, capacity_kwh=60.0)
    nobs = make_next_obs(connected=False)
    terms = R.compute_terms(obs, nobs, state=state)
    expected_kwh = (0.9 - 0.3) * 60.0
    assert terms["unserved"] == pytest.approx(expected_kwh)


def test_monotonic_in_each_term():
    """Holding others fixed, increasing any term decreases the reward."""
    weights = R.DEFAULT_WEIGHTS

    def reward_with_terms(terms):
        return -sum(weights[k] * terms[k] for k in R.TERM_NAMES)

    base_terms = {"cost": 1.0, "carbon": 1.0, "peak": 1.0, "ramp": 1.0, "unserved": 1.0}
    base = reward_with_terms(base_terms)
    for term in R.TERM_NAMES:
        higher = dict(base_terms)
        higher[term] = 2.0
        r_higher = reward_with_terms(higher)
        assert r_higher < base, f"raising {term} did not decrease reward"


def test_vectorised_matches_loop():
    """Vectorised path must match the per-step loop on the smoke dataset."""
    df = pd.read_parquet("datasets/offline_rl/rbc/seed_22.parquet")

    # Loop version
    state = R.RewardState()
    loop_terms = {k: [] for k in R.TERM_NAMES}
    for _, row in df.iterrows():
        obs = {col: row[col] for col in row.index if col.startswith("obs_")}
        nobs = {col: row[col] for col in row.index if col.startswith("next_obs_")}
        terms = R.compute_terms(obs, nobs, state=state)
        for k, v in terms.items():
            loop_terms[k].append(v)
    loop_terms = {k: np.array(v) for k, v in loop_terms.items()}

    # Vectorised version
    vec_terms = R.compute_terms_vectorised(df)

    for k in R.TERM_NAMES:
        np.testing.assert_allclose(
            loop_terms[k],
            vec_terms[k],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"vectorised disagrees with loop on term '{k}'",
        )


def test_reward_outperforms_baseline_on_peaks():
    """v1 reward (= negative net consumption only) doesn't penalise peaks
    relative to off-peak; reward does.
    """
    # Synthetic 'peak day': constant 1 kWh except a 10 kWh spike.
    state_full = R.RewardState()

    base_obs = make_obs(net=1.0, price=0.20, carbon=0.30)
    spike_obs = make_obs(net=10.0, price=0.20, carbon=0.30)

    # Warm up rolling window with base_obs so peak excursion is meaningful.
    for _ in range(R.PEAK_WINDOW_HOURS):
        R.compute_terms(base_obs, make_next_obs(), state=state_full)

    # v1-style "reward" = -net (cost only, equal price/carbon weight).
    v1_base_reward = -1.0
    v1_spike_reward = -10.0

    # Full reward: same scenario, all terms active.
    base_r, _ = R.compute_reward(base_obs, None, make_next_obs(), weights=R.DEFAULT_WEIGHTS, state=R.RewardState())
    # warm reward state then spike
    state_full = R.RewardState()
    for _ in range(R.PEAK_WINDOW_HOURS):
        R.compute_terms(base_obs, make_next_obs(), state=state_full)
    spike_r, spike_terms = R.compute_reward(
        spike_obs, None, make_next_obs(), weights=R.DEFAULT_WEIGHTS, state=state_full
    )

    # The spike must be punished more in absolute terms by than the
    # ratio implied by v1 (-10 vs -1 = factor 10).
    cost_only_ratio = v1_spike_reward / v1_base_reward  # = 10
    full_ratio = spike_r / base_r if base_r != 0 else float("inf")
    assert full_ratio > cost_only_ratio, (
        f"full reward should penalise peaks more than cost-only ratio. "
        f"cost_only_ratio={cost_only_ratio}, full_ratio={full_ratio}, spike_terms={spike_terms}"
    )
    # The peak term must contribute > 0 at the spike step.
    assert spike_terms["peak"] > 0


# ---------------------------------------------------------------------------
# Weights I/O
# ---------------------------------------------------------------------------


def test_save_and_load_weights(tmp_path):
    weights = {"cost": 1.5, "carbon": 0.7, "peak": 3.0, "ramp": 0.4, "unserved": 100.0}
    path = tmp_path / "weights.json"
    R.save_weights(path, weights, metadata={"source": "test"})
    loaded = R.load_weights(path)
    assert loaded == weights


def test_load_weights_rejects_missing_terms(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"weights": {"cost": 1.0}}')
    with pytest.raises(ValueError, match="missing terms"):
        R.load_weights(path)
