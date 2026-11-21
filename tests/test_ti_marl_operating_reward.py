import pytest

from algorithms.ti_marl.learning.operating_reward import (
    SMOKE_PROFILE,
    TERM_WEIGHTS,
    operating_reward,
    proposal_exceedance_kwh,
)


def test_signed_community_account_fixed_reference_and_physical_service_terms():
    # Two active houses from a fixed population of 30. The DR revenue is EUR 2.
    # A 3 kWh loss below the EV minimum is distinct from its 1 kWh tolerance band.
    info = {
        "reward_components": {
            "community": {"community_settlement_cost_total": 7.0},
            "per_agent": [{"battery_throughput": 1.0}, {"battery_throughput": 2.0}],
        },
        "thesis_reward_terms": {
            "dr_cost_eur": -2,
            "proposal_exceedance_kwh": proposal_exceedance_kwh(2, [1, 2], [2], 900),
            "ev_deficit_kwh": 3.0,
            "programme_deficit_kwh": 0.5,
            "fixed_deficit_kwh": 0.25,
            "ev_tolerance_kwh": 1.0,
        },
    }
    result = operating_reward(info, reference_sites=30)
    assert result.value == pytest.approx(-(7 - 2 + 60 * 0.75 + 120 * 3.75 + 1 + 0.03) / 30)
    assert result.missing_terms == ()
    assert set(result.terms) == set(TERM_WEIGHTS)


def test_attributed_missing_terms_never_treated_as_zero_or_as_soc_penalties():
    info = {
        "reward_components": {
            "community": {"community_settlement_cost_total": 1.0},
            "per_agent": [
                {"battery_throughput": 2, "service_violation": 0.0, "ev_soc_deficit_sum": 8}
            ],
        }
    }
    with pytest.raises(ValueError, match="proposal_exceedance_kwh"):
        operating_reward(info, reference_sites=30)
    result = operating_reward(info, reference_sites=30, profile=SMOKE_PROFILE)
    assert result.value == pytest.approx(-1.02 / 30)
    assert len(result.missing_terms) == 6
    info["thesis_reward_terms"] = {"ev_deficit_kwh": 5.0}
    assert operating_reward(info, reference_sites=30, profile=SMOKE_PROFILE).value == result.value
    info["thesis_reward_terms"]["energy_cost_eur"] = 2.0
    with pytest.raises(ValueError, match="disagrees"):
        operating_reward(info, reference_sites=30, profile=SMOKE_PROFILE)


def test_rejects_corrupt_or_incomplete_optional_measurement():
    info = {"thesis_reward_terms": {"energy_cost_eur": 1.0, "stationary_throughput_kwh": 0.0}}
    with pytest.raises(ValueError, match="positive"):
        operating_reward(info, reference_sites=0, profile=SMOKE_PROFILE)
    info["thesis_reward_terms"]["ev_deficit_kwh"] = float("nan")
    with pytest.raises(ValueError, match="ev_deficit_kwh"):
        operating_reward(info, reference_sites=30, profile=SMOKE_PROFILE)
    with pytest.raises(ValueError, match="group excess"):
        proposal_exceedance_kwh(0, [0, 0, 0], [-1], 900)


def test_measured_ev_shortfall_is_counted_once_through_departure():
    from types import SimpleNamespace

    from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

    ns = SimpleNamespace
    schedule = ns(
        electric_vehicle_charger_state=[1, 1, 1, 3],
        electric_vehicle_session_id=["s", "s", "s", ""],
        electric_vehicle_id=["ev", "ev", "ev", ""],
        electric_vehicle_required_soc_departure=[0.8] * 4,
        electric_vehicle_departure_time=[3, 2, 1, 0],
    )
    ev = ns(battery=ns(capacity=10, soc=[0.2, 0.2, 0.2]))
    charger = ns(
        charger_id="c",
        past_connected_evs=[ev] * 3,
        charger_simulation=schedule,
        max_charging_power=4,
    )
    audit = ns(
        _normal_ev_id=str, _normalize_soc_target=float, _ev_departure_service_tolerance=lambda: 0.05
    )
    env = ns(
        topology_mode="static",
        seconds_per_time_step=900,
        _kpi_service=audit,
        buildings=[ns(name="A", electric_vehicle_chargers=[charger], deferrable_appliances=[])],
    )
    ledger = SoftCPSRewardFeedback()
    outcomes = []
    for step in range(3):
        env.time_step = step + 1
        actual = ledger(env, step)
        assert ledger(env, step) == actual  # repeated feedback reads are idempotent
        outcomes.append(actual)
    assert [row["ev_deficit_kwh"] for row in outcomes] == pytest.approx([3.5, 1.0, 1.0])
    assert [row["ev_tolerance_kwh"] for row in outcomes] == pytest.approx([0.0, 0.0, 0.5])
    assert sum(row["ev_deficit_kwh"] for row in outcomes) == pytest.approx(5.5)
    assert outcomes[0]["programme_deficit_kwh"] == 0
    assert "fixed_deficit_kwh" not in outcomes[0]
    assert "dr_cost_eur" not in outcomes[0]


def test_native_evidence_is_preserved_and_conflicts_are_visible():
    from algorithms.ti_marl.runtime.reward_feedback import merge_reward_evidence

    reported = {"dr_cost_eur": -2.0, "ev_deficit_kwh": 1.5}
    measured = {"dr_cost_eur": -2.0, "fixed_deficit_kwh": 0.0}
    assert merge_reward_evidence(reported, measured) == {
        "dr_cost_eur": -2.0,
        "ev_deficit_kwh": 1.5,
        "fixed_deficit_kwh": 0.0,
    }
    with pytest.raises(ValueError, match="dr_cost_eur"):
        merge_reward_evidence(reported, {"dr_cost_eur": 2.0})


def test_only_verified_supervisor_replaces_raw_proposal_for_prelimit_evidence(monkeypatch):
    from types import SimpleNamespace

    from algorithms.ti_marl.operational import OperatingControl
    from algorithms.ti_marl.runtime import reservation_metrics

    ns = SimpleNamespace
    calls = []

    def measure(current, bundle, **kwargs):
        calls.append((bundle.agent_id, bundle.kwh))
        return bundle.kwh

    monkeypatch.setattr(reservation_metrics, "prelimit_exceedance_kwh", measure)
    terms = dict.fromkeys(TERM_WEIGHTS, 0.0)
    terms["proposal_exceedance_kwh"] = 7.0
    pending = {
        "raw_bundles": [ns(agent_id="A", kwh=5), ns(agent_id="B", kwh=None)],
        # Reverse the ordering to check the site binding.
        "final_bundles": [ns(agent_id="B", kwh=2), ns(agent_id="A", kwh=0)],
        "prefix_traces": {"A": ns(supervised=False), "B": ns(supervised=True)},
        "reservations": {"B": {"verified": True}},
    }
    agent = ns(
        reward_profile="thesis_common_reward_v1",
        reward_penalties=None,
        realised_reward=None,
        _transition_info={"thesis_reward_terms": terms},
        _pending=pending,
        forecasts=ns(step_seconds=15),
        reference_sites=2,
        _latest_diagnostics={},
        compiler=ns(interface_registry=ns(for_agent=lambda _: ns(constraints={}))),
    )
    assert OperatingControl._real_reward(agent, None, []) == pytest.approx(-210)
    assert calls == [("A", 5), ("B", 2)]
    agent.realised_reward = None
    terms["proposal_exceedance_kwh"] = 6
    with pytest.raises(ValueError, match="disagree"):
        OperatingControl._real_reward(agent, None, [])
    pending["reservations"]["B"]["verified"] = False
    with pytest.raises(ValueError, match="Unverified"):
        OperatingControl._real_reward(agent, None, [])


def test_programme_energy_is_charged_once_at_irreversibility_even_before_deadline():
    from types import SimpleNamespace

    from citylearn.internal.deferrable_service_audit import ServiceLedger

    from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

    class Appliance:
        pass

    programme = Appliance()
    cycle = dict(
        cycle_id="wash",
        earliest_start_time_step=0,
        latest_start_time_step=1,
        duration_steps=2,
        deadline_time_step=3,
        total_energy_kwh=3.0,
        load_profile=[1.0, 2.0],
    )
    programme.deferrable_appliance_simulation = SimpleNamespace(flexibility_schedule=[cycle])
    env = SimpleNamespace(
        buildings=[SimpleNamespace(deferrable_appliances=[programme])], topology_mode="static"
    )
    for started, expected in [(False, [0, 3, 0, 0]), (True, [0, 0, 1, 0])]:
        ledger = programme._service_ledger = ServiceLedger(0, 5)
        feedback = SoftCPSRewardFeedback()
        values = []
        for step in range(4):
            if started and step == 1:
                ledger.start(cycle, step)
            ledger.record(step, float(started and step in (1, 2)))
            values.append(feedback._programme_deficit(env))
        assert values == expected


def test_dr_revenue_is_signed_and_missing_active_settlement_is_unknown():
    from types import SimpleNamespace

    from algorithms.ti_marl.runtime.reward_feedback import SoftCPSRewardFeedback

    ns = SimpleNamespace
    dr = ns(
        enabled=True,
        _settlement_by_time_step={},
        requests=[ns(start_time_step=101, end_time_step=101)],
    )
    env = ns(
        topology_mode="dynamic",
        _demand_response_service=dr,
        episode_tracker=ns(episode_start_time_step=100),
    )
    feedback = SoftCPSRewardFeedback()
    env.time_step = 1
    assert feedback(env, 0) == {"dr_cost_eur": 0.0}
    env.time_step = 2
    assert "dr_cost_eur" not in feedback(env, 1)
    dr._settlement_by_time_step[1] = {
        "baseline_valid": True,
        "penalty_eur": 1.0,
        "revenue_eur": 3.0,
    }
    feedback = SoftCPSRewardFeedback()
    env.time_step = 1
    feedback(env, 0)
    env.time_step = 2
    assert feedback(env, 1) == {"dr_cost_eur": -2.0}


def test_service_penalties_can_be_selected_without_changing_physical_costs():
    from algorithms.configuration.agents import TIMARLRewardPenalties

    ledger = dict.fromkeys(TERM_WEIGHTS, 0.0)
    ledger.update(energy_cost_eur=4.0, ev_tolerance_kwh=2.0)
    selected = TIMARLRewardPenalties(ev_tolerance_kwh=3.0)
    result = operating_reward(
        {"thesis_reward_terms": ledger}, reference_sites=2, penalties=selected
    )
    assert result.value == -5.0
    assert result.terms["energy_cost_eur"] == 4.0
    with pytest.raises(ValueError):
        TIMARLRewardPenalties(ev_deficit_kwh=-1)
