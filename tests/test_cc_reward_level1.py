"""Tests for CCRewardLevel1.

Each test verifies one term of the reward in isolation, then a few tests
check interactions between terms to confirm the combined signal makes sense.
"""

from __future__ import annotations

import pytest

from reward_function.cc_reward_level1 import CCRewardLevel1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reward(**kwargs) -> CCRewardLevel1:
    """Return a CCRewardLevel1 with fixed reference values for predictable math."""
    defaults = dict(
        w_cost=1.0,
        w_peak=0.6,
        w_ramp=0.4,
        w_export=0.05,
        w_violation=2.0,
        target_import=4.0,       # kWh — community import above this is penalised
        reference_cost=1.0,      # makes cost_norm = community_cost directly
        reference_peak=1.0,      # makes peak_norm = peak_excess² directly
        reference_ramping=1.0,   # makes ramp_norm = |Δimport| directly
        reference_export=1.0,    # makes export_norm = export directly
        reference_violation=1.0, # makes violation_norm = violation_kwh directly
    )
    defaults.update(kwargs)
    return CCRewardLevel1(env_metadata={"central_agent": False}, **defaults)


def obs(net: float, price: float = 0.10, violation: float = 0.0) -> dict:
    return {
        "net_electricity_consumption": net,
        "electricity_pricing": price,
        "charging_constraint_violation_kwh": violation,
    }


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_returns_one_value_per_building():
    reward = make_reward()
    result = reward.calculate([obs(1.0), obs(2.0), obs(0.5)])
    assert len(result) == 3


def test_all_buildings_receive_identical_value():
    """CC reward is a community scalar split equally — all buildings get the same."""
    reward = make_reward()
    result = reward.calculate([obs(1.0), obs(2.0), obs(0.5)])
    assert result[0] == pytest.approx(result[1])
    assert result[1] == pytest.approx(result[2])


def test_empty_observations_returns_empty():
    reward = make_reward()
    assert reward.calculate([]) == []


# ---------------------------------------------------------------------------
# Cost term
# ---------------------------------------------------------------------------

def test_cost_term_penalises_community_import():
    """Higher import at the same price → more negative reward."""
    reward = make_reward(w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0)

    low_import  = sum(reward.calculate([obs(1.0, price=0.10), obs(1.0, price=0.10)]))
    high_import = sum(reward.calculate([obs(3.0, price=0.10), obs(3.0, price=0.10)]))

    assert high_import < low_import


def test_cost_term_penalises_higher_price():
    """Same import but higher price → more negative reward."""
    reward = make_reward(w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0)

    cheap     = sum(reward.calculate([obs(2.0, price=0.10)]))
    expensive = sum(reward.calculate([obs(2.0, price=0.20)]))

    assert expensive < cheap


def test_cost_term_zero_when_no_import():
    """Pure export produces zero cost (no import cost)."""
    reward = make_reward(w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0)
    result = sum(reward.calculate([obs(-2.0, price=0.10)]))
    assert result == pytest.approx(0.0)


def test_cost_term_math():
    """community_cost = 4.0 kWh * €0.25 = €1.0. reference_cost=1.0 → cost_norm=1.0.
    reward = -w_cost * 1.0 / N = -1.0 / 2 = -0.5 per building."""
    reward = make_reward(w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
                         reference_cost=1.0)
    # 2 buildings × 2.0 kWh = 4.0 kWh community import at €0.25
    result = reward.calculate([obs(2.0, price=0.25), obs(2.0, price=0.25)])
    assert result[0] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Peak term
# ---------------------------------------------------------------------------

def test_peak_term_zero_when_import_below_target():
    """No penalty when community import is at or below target."""
    reward = make_reward(w_cost=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
                         target_import=4.0)
    # 2 buildings × 2.0 kWh = 4.0 kWh = exactly at target
    result = sum(reward.calculate([obs(2.0), obs(2.0)]))
    assert result == pytest.approx(0.0)


def test_peak_term_penalises_excess_above_target():
    """Import above target → negative reward."""
    reward = make_reward(w_cost=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
                         target_import=4.0)
    # 2 buildings × 3.0 kWh = 6.0 kWh → excess = 2.0 → peak_norm = 4.0
    result = sum(reward.calculate([obs(3.0), obs(3.0)]))
    assert result < 0.0


def test_peak_term_grows_quadratically():
    """Doubling the excess quadruples the penalty."""
    reward = make_reward(w_cost=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
                         target_import=0.0, reference_peak=1.0)

    # excess = 2.0 → penalty ∝ 4.0
    small = abs(sum(reward.calculate([obs(1.0), obs(1.0)])))
    # excess = 4.0 → penalty ∝ 16.0
    large = abs(sum(reward.calculate([obs(2.0), obs(2.0)])))

    assert large == pytest.approx(4.0 * small, rel=1e-4)


# ---------------------------------------------------------------------------
# Ramping term
# ---------------------------------------------------------------------------

def test_ramp_term_penalises_from_zero_baseline():
    """First step: prev_import=0, so ramp = import, which is penalised."""
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_export=0.0, w_violation=0.0)
    result = sum(reward.calculate([obs(2.0), obs(2.0)]))
    # prev_import=0, import=4.0 kWh, ramp=4.0 → penalty = -w_ramp * 4.0 / N
    assert result < 0.0


def test_ramp_term_penalises_large_step_change():
    """A sudden spike in import is penalised more than a smooth increase."""
    reward_spike  = make_reward(w_cost=0.0, w_peak=0.0, w_export=0.0, w_violation=0.0)
    reward_smooth = make_reward(w_cost=0.0, w_peak=0.0, w_export=0.0, w_violation=0.0)

    # Spike: 0 → 10 kWh in one step
    reward_spike.calculate([obs(5.0), obs(5.0)])   # step 1: prev=0, import=10
    spike = sum(reward_spike.calculate([obs(5.0), obs(5.0)]))  # step 2: no change, ramp=0

    # Smooth: 0 → 5 → 10 kWh
    reward_smooth.calculate([obs(2.5), obs(2.5)])   # step 1: prev=0, import=5
    smooth = sum(reward_smooth.calculate([obs(5.0), obs(5.0)]))  # step 2: import=10, ramp=5

    # Step 2 of spike has zero ramp (already at 10); smooth still has ramp=5
    assert spike > smooth


def test_ramp_term_penalises_decrease_too():
    """Ramping is absolute — a drop in import is also penalised."""
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_export=0.0, w_violation=0.0)
    reward.calculate([obs(5.0), obs(5.0)])     # step 1: import=10
    result = sum(reward.calculate([obs(0.0), obs(0.0)]))   # step 2: import drops to 0
    assert result < 0.0


def test_ramp_term_zero_when_import_unchanged():
    """No ramp penalty when community import is identical step-to-step."""
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_export=0.0, w_violation=0.0)
    reward.calculate([obs(2.0), obs(2.0)])     # step 1
    result = sum(reward.calculate([obs(2.0), obs(2.0)]))   # step 2: same import
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Export term
# ---------------------------------------------------------------------------

def test_export_term_zero_when_no_export():
    """No community export → no export penalty."""
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_violation=0.0)
    result = sum(reward.calculate([obs(1.0), obs(1.0)]))
    assert result == pytest.approx(0.0)


def test_export_term_penalises_community_export():
    """Net export → negative reward (discourages uncontrolled grid export)."""
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_violation=0.0)
    result = sum(reward.calculate([obs(-2.0), obs(-1.0)]))
    assert result < 0.0


def test_export_penalty_less_severe_than_cost():
    """w_export (0.05) is much smaller than w_cost (1.0): export should hurt far less."""
    reward = make_reward(w_ramp=0.0, w_peak=0.0, w_violation=0.0,
                         reference_cost=1.0, reference_export=1.0)

    import_penalty = abs(sum(reward.calculate([obs(1.0, price=1.0)])))
    reward2 = make_reward(w_ramp=0.0, w_peak=0.0, w_violation=0.0, w_cost=0.0,
                          reference_export=1.0)
    export_penalty = abs(sum(reward2.calculate([obs(-1.0)])))

    assert import_penalty > export_penalty


# ---------------------------------------------------------------------------
# Violation term
# ---------------------------------------------------------------------------

def test_violation_term_zero_when_no_violation():
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)
    result = sum(reward.calculate([obs(1.0, violation=0.0), obs(1.0, violation=0.0)]))
    assert result == pytest.approx(0.0)


def test_violation_term_penalises_positive_violation():
    reward = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)
    result = sum(reward.calculate([obs(1.0, violation=1.0), obs(1.0, violation=0.5)]))
    assert result < 0.0


def test_violation_term_scales_with_total_kwh():
    """2× the violation kWh → 2× the penalty."""
    reward_small = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)
    reward_large = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)

    small = abs(sum(reward_small.calculate([obs(0.0, violation=1.0)])))
    large = abs(sum(reward_large.calculate([obs(0.0, violation=2.0)])))

    assert large == pytest.approx(2.0 * small, rel=1e-4)


def test_violation_term_sums_across_buildings():
    """Total violation = sum across all buildings."""
    reward_split    = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)
    reward_combined = make_reward(w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0)

    # 1.0 kWh spread across two buildings
    r_split    = sum(reward_split.calculate([obs(0.0, violation=0.5), obs(0.0, violation=0.5)]))
    # 1.0 kWh in one building only (but still 2 buildings for normalisation)
    r_combined = sum(reward_combined.calculate([obs(0.0, violation=1.0), obs(0.0, violation=0.0)]))

    assert r_split == pytest.approx(r_combined, rel=1e-4)


def test_violation_dominates_cost_when_large():
    """A large violation should outweigh the cost term — it's the hardest constraint."""
    reward = make_reward(reference_cost=1.0, reference_violation=1.0)

    # Same import/price but with 10 kWh of violations
    no_viol   = sum(reward.calculate([obs(2.0, price=0.10, violation=0.0)]))
    with_viol = sum(reward.calculate([obs(2.0, price=0.10, violation=10.0)]))

    assert with_viol < no_viol
    assert abs(with_viol - no_viol) > abs(no_viol)  # violation penalty > cost penalty


# ---------------------------------------------------------------------------
# Interaction / combined signal sanity checks
# ---------------------------------------------------------------------------

def test_idle_community_is_better_than_importing():
    """Zero net consumption should be rewarded more than importing."""
    reward = make_reward()
    reward.calculate([obs(0.0), obs(0.0)])  # step 1 to seed prev_import

    idle      = sum(reward.calculate([obs(0.0), obs(0.0)]))
    importing = sum(reward.calculate([obs(2.0), obs(2.0)]))

    assert idle > importing


def test_smooth_profile_better_than_spiky():
    """Two steps with constant import should score better than a spike then zero."""
    reward_smooth = make_reward()
    reward_spiky  = make_reward()

    # Smooth: 2 kWh/step both steps
    reward_smooth.calculate([obs(1.0), obs(1.0)])
    smooth = sum(reward_smooth.calculate([obs(1.0), obs(1.0)]))

    # Spiky: 0 then 4 kWh (same total, but spike)
    reward_spiky.calculate([obs(0.0), obs(0.0)])
    spiky = sum(reward_spiky.calculate([obs(2.0), obs(2.0)]))

    assert smooth > spiky


def test_below_target_import_with_no_violations_is_best_case():
    """Low import, no export, no violation, no ramp → highest possible reward for given cost."""
    reward = make_reward()
    reward.calculate([obs(1.0), obs(1.0)])  # seed prev_import to same value → ramp=0

    best  = sum(reward.calculate([obs(1.0), obs(1.0)]))          # stable, below target
    worse = sum(reward.calculate([obs(3.0), obs(3.0)]))          # above target

    assert best > worse
