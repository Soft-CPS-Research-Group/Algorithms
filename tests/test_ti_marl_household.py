from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.configuration.agents import TIMARLRewardPenalties
from algorithms.ti_marl.experiments.household import car_features
from algorithms.ti_marl.learning.operating_reward import TERM_WEIGHTS, operating_reward


def test_each_port_uses_its_own_service_and_efficiency():
    g = SimpleNamespace(max_charge_power_kw=4, reservation_profile={"battery_efficiency": 0.81})
    curve = np.array([[0.25, 0.25, 0.1, 0.1], [0.2] * 4, [0, 0, 2, 2]])
    car = dict(
        hours_until_departure=1.0,
        connected_state=1,
        connected_ev_required_soc_departure=0.6,
        connected_ev_soc=0.5,
        connected_ev_battery_capacity_kwh=20,
        charger_efficiency_ratio=0.9,
    )
    a = car_features(g, car, curve, 900)
    b = car_features(g, {**car, "connected_ev_soc": 0.55, "hours_until_departure": 0.5}, curve, 900)
    assert a[0] == pytest.approx(2 / 0.81)
    assert b[0] == pytest.approx(1 / 0.81)
    assert a[9] == 0 and a[8] > 0 and b[8] == 0
    assert car_features(g, {**car, "hours_until_departure": 2.0}, curve, 900) is None
    assert car_features(g, {**car, "connected_state": 0}, curve, 900) is None


def test_excess_requires_measured_delivery_only_when_priced():
    ledger = dict.fromkeys(TERM_WEIGHTS, 0.0)
    info = {"thesis_reward_terms": ledger}
    assert operating_reward(info, reference_sites=1).value == 0
    fee = TIMARLRewardPenalties(ev_excess_kwh=0.05)
    with pytest.raises(ValueError, match="ev_excess_kwh"):
        operating_reward(info, reference_sites=1, penalties=fee)
    ledger["ev_excess_kwh"] = 0.4
    assert operating_reward(info, reference_sites=1, penalties=fee).value == pytest.approx(-0.02)


def test_post_action_excess_excludes_arrival_energy_and_carried_excess():
    from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

    # The vehicle arrives above target. Only subsequent physical additions are priced.
    schedule = SimpleNamespace(
        electric_vehicle_charger_state=[1, 1, 1, 3],
        electric_vehicle_session_id=["s", "s", "s", ""],
        electric_vehicle_id=["car", "car", "car", ""],
        electric_vehicle_required_soc_departure=[0.6] * 4,
        electric_vehicle_departure_time=[3, 2, 1, 0],
    )
    car = SimpleNamespace(
        battery=SimpleNamespace(
            capacity=10,
            soc=[0.7, 0.745, 0.745],
            gross_energy_charge=[0.0, 0.5, 0.0],
            round_trip_efficiency=0.9,
        )
    )
    port = SimpleNamespace(
        charger_simulation=schedule,
        past_connected_evs=[car] * 3,
        charger_id="c",
        max_charging_power=4.0,
    )
    audit = SimpleNamespace(
        _normal_ev_id=str, _normalize_soc_target=float, _ev_departure_service_tolerance=lambda: 0.05
    )
    env = SimpleNamespace(
        topology_mode="static",
        seconds_per_time_step=900,
        _kpi_service=audit,
        buildings=[SimpleNamespace(name="house", electric_vehicle_chargers=[port])],
    )
    feedback = SoftCPSRewardFeedback(measure_ev_excess=True)
    assert [feedback._ev_deficits(env, t)["ev_excess_kwh"] for t in range(3)] == pytest.approx(
        [0, 0.45, 0]
    )


def test_terminal_refill_is_once_and_gives_no_credit_for_surplus():
    from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

    battery = SimpleNamespace(nominal_power=4, capacity=10, initial_soc=0.5, soc=[0.3, 0.6])
    env = SimpleNamespace(
        topology_mode="static",
        terminated=False,
        truncated=False,
        buildings=[SimpleNamespace(electrical_storage=battery)],
    )
    assert SoftCPSRewardFeedback._terminal_stock_deficit(env, 0) == 0
    env.terminated = True
    assert SoftCPSRewardFeedback._terminal_stock_deficit(env, 0) == pytest.approx(2)
    assert SoftCPSRewardFeedback._terminal_stock_deficit(env, 1) == 0


def test_appliance_distinguishes_solar_at_start_from_same_energy_later(monkeypatch):
    from algorithms.ti_marl.experiments.household import HouseholdActor
    from algorithms.ti_marl.policy.joint import JointActor
    from algorithms.ti_marl.runtime.forecasts import SiteOutlook

    monkeypatch.setattr(JointActor, "_contexts", lambda self, snapshot: {})
    actor = SimpleNamespace(step_seconds=900, observe_programme_start_solar=True)
    group = SimpleNamespace(
        module_id="washer", group_id="A:washer", group_type="deferrable", max_charge_power_kw=0.0
    )
    reading = SimpleNamespace(
        valid=True, values=(2.0,), sensor_id="washer", observation_id="cycle_duration_steps"
    )

    def state(pv):
        curve = SiteOutlook("A", ((0.2, 0.2), (0.0, 0.0), pv), ((2, 2),) * 3)
        view = SimpleNamespace(
            agent_ids=("A",),
            outlooks=(curve,),
            parts_for=lambda site: [reading],
            groups_for=lambda site: [group],
        )
        HouseholdActor._contexts(actor, view)
        return actor.household_inputs["A:washer"][1]

    early, late = state((4.0, 0.0)), state((0.0, 4.0))
    assert early[4] == late[4] == 0.5
    assert early[5] == 1.0 and late[5] == 0.0


def test_ev_head_receives_power_interval_in_its_original_nominal_units():
    from algorithms.ti_marl.experiments.household import dose_bounds

    interval = SimpleNamespace(lower=0.5, upper=1.0, scale_kw=2.0)
    assert dose_bounds(interval, 4.0) == (0.25, 0.5)
    assert dose_bounds(interval, 2.0) == (0.5, 1.0)


def test_partial_ev_forecast_uses_announced_departure_and_only_issued_prices():
    group = SimpleNamespace(max_charge_power_kw=4, reservation_profile={"battery_efficiency": 1.0})
    car = dict(connected_state=1, hours_until_departure=48, connected_ev_required_soc_departure=0.8,
               connected_ev_soc=0.7, connected_ev_battery_capacity_kwh=20, charger_efficiency_ratio=1)
    curve = np.array([[0.2, 0.1], [1.0, 1.0], [0.0, 0.0]])
    assert car_features(group, car, curve, 900) is None
    features = car_features(group, car, curve, 900, use_partial_outlook=True)
    assert features[:5] == pytest.approx([2.0, 48, 47.5, 2.0, 0.5])
    assert car_features(group, car, curve[:, :0], 900, use_partial_outlook=True) is None
    short = {**car, "hours_until_departure": 0.5}
    assert car_features(group, short, curve, 900, use_partial_outlook=True) == car_features(
        group, short, curve, 900
    )
