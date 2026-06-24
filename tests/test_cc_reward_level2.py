"""Tests for CCRewardLevel2.

CCRewardLevel2 = CCRewardLevel1 community scalar  −  w_ev · ev_penalty.

The first section proves the inheritance refactor preserved the Level-1
community term exactly (with w_ev=0, Level 2 must equal Level 1). The second
section verifies the EV service term.
"""

from __future__ import annotations

import pytest

from reward_function.cc_reward_level1 import CCRewardLevel1
from reward_function.cc_reward_level2 import CCRewardLevel2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMUNITY_KW = dict(
    w_cost=1.0,
    w_peak=0.6,
    w_ramp=0.4,
    w_export=0.05,
    w_violation=2.0,
    target_import=4.0,
    reference_cost=1.0,
    reference_peak=1.0,
    reference_ramping=1.0,
    reference_export=1.0,
    reference_violation=1.0,
)


def make_l1(**kwargs) -> CCRewardLevel1:
    d = dict(_COMMUNITY_KW)
    d.update(kwargs)
    return CCRewardLevel1(env_metadata={"central_agent": False}, **d)


def make_l2(**kwargs) -> CCRewardLevel2:
    d = dict(_COMMUNITY_KW)
    d.update(kwargs)
    return CCRewardLevel2(env_metadata={"central_agent": False}, **d)


def obs(net: float, price: float = 0.10, violation: float = 0.0, evs=None) -> dict:
    o = {
        "net_electricity_consumption": net,
        "electricity_pricing": price,
        "charging_constraint_violation_kwh": violation,
    }
    if evs is not None:
        o["electric_vehicles_chargers_dict"] = evs
    return o


def ev(connected=True, battery_soc=0.5, required_soc=0.9, hours_until_departure=1.0) -> dict:
    """One charger entry for electric_vehicles_chargers_dict."""
    return {
        "connected": connected,
        "battery_soc": battery_soc,
        "required_soc": required_soc,
        "hours_until_departure": hours_until_departure,
    }


# ---------------------------------------------------------------------------
# Inheritance: community term must equal CCRewardLevel1 exactly when w_ev=0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("observations", [
    [{"net_electricity_consumption": 2.0, "electricity_pricing": 0.25,
      "charging_constraint_violation_kwh": 0.0}],
    [{"net_electricity_consumption": 5.0, "electricity_pricing": 0.30,
      "charging_constraint_violation_kwh": 1.5},
     {"net_electricity_consumption": -3.0, "electricity_pricing": 0.30,
      "charging_constraint_violation_kwh": 0.0}],
    [{"net_electricity_consumption": -4.0, "electricity_pricing": 0.10,
      "charging_constraint_violation_kwh": 0.0}],
])
def test_community_term_matches_level1_when_no_ev_weight(observations):
    """With w_ev=0 and no EVs, Level 2 reduces exactly to Level 1."""
    l1 = make_l1()
    l2 = make_l2(w_ev=0.0)
    assert l2.calculate(observations) == pytest.approx(l1.calculate(observations))


def test_community_term_matches_level1_across_sequence():
    """Ramp term depends on previous import; verify parity holds over a sequence
    (proves _prev_import state is advanced identically)."""
    l1 = make_l1()
    l2 = make_l2(w_ev=0.0)
    for net in (2.0, 5.0, 1.0, 6.0):
        observations = [obs(net, price=0.2), obs(net, price=0.2)]
        assert l2.calculate(observations) == pytest.approx(l1.calculate(observations))


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_returns_one_value_per_building():
    reward = make_l2()
    assert len(reward.calculate([obs(1.0), obs(2.0), obs(0.5)])) == 3


def test_all_buildings_receive_identical_value():
    reward = make_l2()
    result = reward.calculate([obs(1.0), obs(2.0), obs(0.5)])
    assert result[0] == pytest.approx(result[1]) == pytest.approx(result[2])


def test_empty_observations_returns_empty():
    assert make_l2().calculate([]) == []


# ---------------------------------------------------------------------------
# EV service term
# ---------------------------------------------------------------------------

def test_no_ev_means_no_ev_penalty():
    """No EV dict → reward equals the pure community term."""
    l2 = make_l2(w_ev=0.5)
    community_only = make_l2(w_ev=0.0)
    observations = [obs(2.0, price=0.2), obs(2.0, price=0.2)]
    assert l2.calculate(observations) == pytest.approx(community_only.calculate(observations))


def test_disconnected_ev_contributes_no_harm():
    """An EV that is not connected adds no penalty."""
    reward = make_l2(w_ev=0.5)
    with_disconnected = reward.calculate([obs(0.0, evs={"c1": ev(connected=False)})])
    reward2 = make_l2(w_ev=0.5)
    no_ev = reward2.calculate([obs(0.0)])
    assert with_disconnected == pytest.approx(no_ev)


def test_connected_urgent_ev_with_gap_lowers_reward():
    """A connected EV that is urgent with an SoC gap is penalised."""
    reward_ev   = make_l2(w_ev=0.5)
    reward_none = make_l2(w_ev=0.5)

    # urgency: hours=1, horizon=4 → 1 - 1/4 = 0.75; gap = 0.9 - 0.5 = 0.4 → harm = 0.3
    with_ev = sum(reward_ev.calculate([obs(0.0, evs={"c1": ev(hours_until_departure=1.0)})]))
    without = sum(reward_none.calculate([obs(0.0)]))

    assert with_ev < without


def test_ev_penalty_zero_when_soc_already_met():
    """No gap (soc >= required) → no EV penalty even if urgent."""
    reward = make_l2(w_ev=0.5)
    met = sum(reward.calculate([obs(0.0, evs={"c1": ev(battery_soc=0.95, required_soc=0.9)})]))
    reward2 = make_l2(w_ev=0.5)
    no_ev = sum(reward2.calculate([obs(0.0)]))
    assert met == pytest.approx(no_ev)


def test_ev_penalty_zero_when_far_from_departure():
    """Far from departure (hours >= horizon) → urgency 0 → no penalty."""
    reward = make_l2(w_ev=0.5, urgency_horizon=4.0)
    far = sum(reward.calculate([obs(0.0, evs={"c1": ev(hours_until_departure=8.0)})]))
    reward2 = make_l2(w_ev=0.5)
    no_ev = sum(reward2.calculate([obs(0.0)]))
    assert far == pytest.approx(no_ev)


def test_ev_penalty_grows_with_urgency():
    """Closer to departure → larger penalty (higher urgency)."""
    far_reward  = make_l2(w_ev=0.5, urgency_horizon=4.0)
    near_reward = make_l2(w_ev=0.5, urgency_horizon=4.0)

    far  = sum(far_reward.calculate([obs(0.0, evs={"c1": ev(hours_until_departure=3.0)})]))
    near = sum(near_reward.calculate([obs(0.0, evs={"c1": ev(hours_until_departure=0.5)})]))

    assert near < far  # nearer departure = more harm = more negative


def test_ev_penalty_math():
    """Exact EV-term math with community weights zeroed.

    harm = urgency * gap = (1 - 1/4) * (0.9 - 0.5) = 0.75 * 0.4 = 0.30
    ev_penalty = harm_sum / N = 0.30 / 1 = 0.30
    scalar = -w_ev * ev_penalty = -0.5 * 0.30 = -0.15  (community terms = 0)
    """
    reward = make_l2(
        w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
        w_ev=0.5, urgency_horizon=4.0,
    )
    result = reward.calculate([obs(0.0, evs={"c1": ev(hours_until_departure=1.0)})])
    assert result[0] == pytest.approx(-0.15)


def test_ev_penalty_averaged_over_buildings():
    """ev_penalty divides harm_sum by N buildings (keeps EV term on per-building scale)."""
    reward = make_l2(
        w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
        w_ev=0.5, urgency_horizon=4.0,
    )
    # One urgent EV (harm 0.30) across two buildings → ev_penalty = 0.30/2 = 0.15
    # scalar = -0.5 * 0.15 = -0.075, split over 2 buildings → -0.0375 each
    result = reward.calculate([
        obs(0.0, evs={"c1": ev(hours_until_departure=1.0)}),
        obs(0.0),
    ])
    assert result[0] == pytest.approx(-0.0375)


def test_multiple_chargers_in_building_are_averaged():
    """A building with two chargers averages harm across them."""
    reward = make_l2(
        w_cost=0.0, w_peak=0.0, w_ramp=0.0, w_export=0.0, w_violation=0.0,
        w_ev=0.5, urgency_horizon=4.0,
    )
    # c1 harm = 0.75*0.4 = 0.30 ; c2 connected but no gap → 0 ; avg = 0.15
    # ev_penalty = 0.15 / 1 building ; scalar = -0.5*0.15 = -0.075
    result = reward.calculate([obs(0.0, evs={
        "c1": ev(hours_until_departure=1.0, battery_soc=0.5, required_soc=0.9),
        "c2": ev(hours_until_departure=1.0, battery_soc=0.95, required_soc=0.9),
    })])
    assert result[0] == pytest.approx(-0.075)
