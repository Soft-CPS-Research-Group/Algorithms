"""TI-MARL §6.6, including a physically checked prefix of the running example."""

from dataclasses import replace

import pytest

from algorithms.ti_marl.runtime.reservation import SiteReservation
from algorithms.ti_marl.runtime.reservation_inputs import (
    Battery,
    Equipment,
    PowerLimit,
    PowerMode,
    SiteInputs,
)

IDLE = PowerMode("IDLE", 0, 0.0, 0.0, 0.0)
CHARGE = PowerMode("CHARGE_EV", 1, 1.4, 7.0, 7.0)
START = PowerMode("START", 0, 0.0, 0.0, 0.0)


def example():
    a = Equipment("A", (IDLE, CHARGE), Battery(60.0, 30.0), 900.0, 31.5)
    b = Equipment("B", (IDLE, CHARGE), Battery(50.0, 20.0), 3600.0, 22.0)
    w = Equipment("W", (IDLE, START), deadline_seconds=7200.0, programme_kw=(2.0,) * 4)
    site = PowerLimit("site", {"A": 1.0, "B": 1.0, "W": 1.0}, "import", 10.0, constant=True)
    cc = PowerLimit("CC", {"A": 1.0, "B": 1.0}, "import", 9.0, constant=True)
    return SiteInputs((a, b, w), (site, cc))


def test_two_chargers_and_washer_running_example():
    site = SiteReservation(example())
    assert site.admissible_modes("A") == ("CHARGE_EV",)
    site.select_mode("A", "CHARGE_EV")
    assert set(site.admissible_modes("B")) == {"IDLE", "CHARGE_EV"}
    site.select_mode("B", "CHARGE_EV")
    site.select_mode("W", "START")
    assert tuple(x * 7.0 for x in site.fraction_bounds("A")) == pytest.approx((6.0, 6.6))
    site.select_fraction("A", 6.4 / 7.0)
    assert tuple(x * 7.0 for x in site.fraction_bounds("B")) == pytest.approx((1.4, 1.6))
    site.select_fraction("B", 1.5 / 7.0)
    site.select_fraction("W", 1.0)
    assert site.diagnostics["verified"]


def test_same_deadline_fairness_is_fixed_before_policy_choice():
    a, b, _ = example().equipment
    b = replace(b, deadline_seconds=900.0, target_kwh=21.5)
    inputs = SiteInputs(
        (a, b), (PowerLimit("site", {"A": 1.0, "B": 1.0}, "import", 6.0, constant=True),)
    )
    site = SiteReservation(inputs)
    assert site.protected == pytest.approx({"A": 0.5, "B": 0.5})
    for key in site.group_order:
        site.select_mode(key, "CHARGE_EV")
    for key in site.group_order:
        assert site.fraction_bounds(key) == pytest.approx((3 / 7.0, 3 / 7.0))
        site.select_fraction(key, 3 / 7.0)


def test_missing_future_headroom_cannot_be_carried_forward():
    a, b, _ = example().equipment
    site = SiteReservation(SiteInputs((b,), (PowerLimit("site", {"B": 1.0}, "import", 7.0),)))
    assert site.protected["B"] == pytest.approx(0.875)
    assert site.admissible_modes("B") == ("CHARGE_EV",)


def test_cannot_start_cycle_with_future_capacity_gap():
    _, _, w = example().equipment
    cap = (10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    site = SiteReservation(SiteInputs((w,), (PowerLimit("site", {"W": 1.0}, "import", 10.0, cap),)))
    assert site.protected["W"] == 1.0
    assert site.admissible_modes("W") == ("IDLE",)


def test_small_timestep_preserves_physical_service_window():
    a, _, _ = example().equipment
    for seconds in (15.0, 60.0):
        site = SiteReservation(SiteInputs((a,), ()), step_seconds=seconds)
        assert site.edges == [0.0, seconds, 900.0]
        assert site.admissible_modes("A") == ("IDLE", "CHARGE_EV")


def test_subminute_capacity_uses_civil_quarter_boundary():
    # 10:14:00, followed by a different envelope at 10:15. Treating forecasts
    # as a fresh quarter hour at each local decision would incorrectly allow IDLE.
    car = Equipment("A", (IDLE, CHARGE), Battery(60.0, 30.0), 960.0, 30.0 + 7.0 / 60.0)
    cap = PowerLimit("site", {"A": 1.0}, "import", 7.0, (7.0, 0.0))
    site = SiteReservation(
        SiteInputs((car,), (cap,)), step_seconds=60.0, block_offset_seconds=840.0
    )
    assert site.edges == [0.0, 60.0, 960.0]
    assert site.admissible_modes("A") == ("CHARGE_EV",)
    site.select_mode("A", "CHARGE_EV")
    assert site.fraction_bounds("A") == pytest.approx((1.0, 1.0))


def test_taper_at_high_soc_matches_appendix_and_preserves_connector_minimum():
    # Appendix C, Member_025: 10.488 kW battery terminals, charger eta=.95.
    import math

    mode = PowerMode("CHARGE_EV", 1, 4.14, 11.04, 11.04)
    unit = Equipment(
        "EV",
        (IDLE, mode),
        Battery(50.0, 47.5, 0.95 * math.sqrt(0.96), taper_rating_kw=10.488 / 0.95),
    )
    site = SiteReservation(SiteInputs((unit,), ()))
    assert site.equipment_fraction_bounds("EV", "CHARGE_EV") == pytest.approx(
        (4.14 / 11.04, 6.9 / 11.04)
    )
    site.select_mode("EV", "CHARGE_EV")
    assert site.fraction_bounds("EV") == pytest.approx((4.14 / 11.04, 6.9 / 11.04))


def test_reconciliation_preserves_admissible_proposal_and_minimises_altered_modes():
    from algorithms.ti_marl.contracts.models import ActionDecision, LocalActionBundle

    def bundle(a_mode, a_fraction, b_fraction):
        return LocalActionBundle(
            "site",
            (
                ActionDecision("A", a_mode, a_fraction, int(a_mode != "IDLE")),
                ActionDecision("B", "CHARGE_EV", b_fraction, 1),
                ActionDecision("W", "START", 1.0, 1),
            ),
        )

    site = SiteReservation(example())
    valid = bundle("CHARGE_EV", 6.4 / 7.0, 1.5 / 7.0)
    applied = site.reconcile(valid)
    assert [d.mode for d in applied.decisions] == [d.mode for d in valid.decisions]
    assert [d.fraction for d in applied.decisions] == pytest.approx(
        [d.fraction for d in valid.decisions]
    )
    site = SiteReservation(example())
    changed = site.reconcile(bundle("IDLE", 0.0, 1.5 / 7.0))
    assert [d.mode for d in changed.decisions] == ["CHARGE_EV", "CHARGE_EV", "START"]
    assert site.diagnostics["verified"]


def test_declared_phase_topology_and_shared_circuit_compile(tmp_path):
    import yaml

    from algorithms.ti_marl.compiler import TypedInterfaceCompiler
    from scripts.generate_typed_interfaces import _building_constraints, _phase_routing
    from tests.ti_marl_fixtures import entity_payload, entity_specs, typed_interface_payload

    schema = {
        "buildings": {
            "Building_1": {
                "electrical_service": {
                    "mode": "three_phase",
                    "default_split": "L2",
                    "limits": {
                        "per_phase": {
                            "L1": {"import_kw": 0.0},
                            "L2": {"import_kw": 6.9},
                            "L3": {"import_kw": 0.0},
                        }
                    },
                },
                "chargers": {"c": {"attributes": {"phase_connection": "L3"}}},
            }
        }
    }
    bg = _building_constraints(schema, "Building_1")["background_phase_weights"]
    assert bg == {"L1": 0.0, "L2": 1.0, "L3": 0.0}
    assert _phase_routing(schema, "Building_1", "deferrable_appliance", "w") == bg
    assert _phase_routing(schema, "Building_1", "charger", "c") == {"L1": 0.0, "L2": 0.0, "L3": 1.0}
    assert _phase_routing(schema, "Building_1", "storage") == dict.fromkeys(bg, 1.0 / 3.0)
    payload = typed_interface_payload("Building_1")
    actuator_names = tuple(payload["actuators"])
    payload["constraints"].update(
        background_phase_weights=bg,
        charger_cluster={"members": [actuator_names[0]], "unit": "kW", "max": 4.0},
    )
    (tmp_path / "Building_1.yaml").write_text(yaml.safe_dump(payload))
    compiler = TypedInterfaceCompiler(contract_version="ti_marl_v1", typed_interfaces_dir=tmp_path)
    specs = entity_specs(("Building_1",))
    compiler.attach_entity_specs(specs)
    snap = compiler.compile(entity_payload(("Building_1",)))
    limit = next(c for c in snap.constraints if c.constraint_type == "group_import_limit_kw")
    assert limit.upper_bound == 4.0 and limit.member_group_ids == (
        f"Building_1:{actuator_names[0]}",
    )


def test_measured_exceedance_replaces_previous_signed_setpoint_and_avoids_double_counting():
    from types import SimpleNamespace

    from algorithms.ti_marl.contracts.models import (
        ActionDecision,
        LocalActionBundle,
        LocalConstraint,
    )
    from algorithms.ti_marl.runtime.reservation_metrics import prelimit_exceedance_kwh

    site = "house"
    key = site + ":battery"

    def reading(sensor, name, value):
        return SimpleNamespace(sensor_id=sensor, observation_id=name, values=(value,), valid=True)

    parts = [
        reading("self", "net_power_kw", 6.0),
        reading("self", "charging_phase_L1_power_kw", 6.0),
        reading("battery", "last_applied_power_kw", -2.0),
    ]
    group = SimpleNamespace(
        group_id=key,
        module_id="battery",
        group_type="stationary_storage",
        max_charge_power_kw=4.0,
        max_discharge_power_kw=4.0,
        ports=(SimpleNamespace(mode="CHARGE_STATIONARY", upper_bound=1.0),),
    )
    lim = LocalConstraint(
        site + ":charging_phase_L1_headroom_kw",
        site,
        "charging_phase_headroom_kw",
        upper_bound=4.0,
        member_group_ids=(key,),
        member_group_coefficients=((key, 1.0),),
    )
    snap = SimpleNamespace(
        groups_for=lambda _: (group,), parts_for=lambda _: parts, constraints=(lim,)
    )
    proposal = LocalActionBundle(site, (ActionDecision(key, "CHARGE_STATIONARY", 1.0, 1),))
    declared = {"grid_import": {"max": 10.0}, "phase_import": {"max": {"L1": 10.0}}}
    # New meter = 6 - (-2) + 4 = 12. Both the site and its phase exceed
    # by 2 kW; count the same event once, over 15 min.
    assert prelimit_exceedance_kwh(
        snap, proposal, step_seconds=900, declared=declared
    ) == pytest.approx(0.5)
    parts.pop()
    assert prelimit_exceedance_kwh(snap, proposal, step_seconds=900, declared=declared) is None


def test_oracle_and_reservation_share_the_native_thread_pool():
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint

    from algorithms.oracles.perfect_foresight_milp import _milp_primal_only

    def oracle():
        return _milp_primal_only(
            c=np.array([1.0]),
            integrality=np.array([1]),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint([[1.0]], 1.0, 1.0),
            options={"threads": 2},
        )

    assert oracle().status == 0
    constructor = SiteReservation(example())
    assert constructor.admissible_modes("A") == ("CHARGE_EV",)
    assert oracle().status == 0


def test_continuous_completion_tiebreak_is_released_after_fixing_modes():
    _, b, _ = example().equipment
    site = SiteReservation(SiteInputs((b,), ()))
    site.select_mode("B", "CHARGE_EV")
    first = site.edges.index(site.diagnostics["earliest_completion_seconds"]["B"])
    assert site.model.bounds[site.energy["B", first]][0] == 0.0
    assert site.diagnostics["protected_fractions"]["B"] == pytest.approx(1.0)


def test_disconnected_charger_replaces_previous_charge_on_its_declared_phase():
    from types import SimpleNamespace

    from algorithms.ti_marl.contracts.models import ActionDecision, LocalActionBundle
    from algorithms.ti_marl.runtime.reservation_metrics import prelimit_exceedance_kwh

    def reading(sensor, name, value):
        return SimpleNamespace(sensor_id=sensor, observation_id=name, values=(value,), valid=True)

    key = "Member_003:charger_1"
    # No valid live charger routing after the EV leaves. The previously
    # measured charge is on L3 and must be removed once from that phase.
    group = SimpleNamespace(
        group_id=key,
        module_id="charger_1",
        group_type="charger",
        phase_weights=(("L3", 1.0),),
        ports=(SimpleNamespace(mode="IDLE", upper_bound=1.0),),
    )
    parts = [
        reading("self", "net_power_kw", 8.0),
        reading("self", "charging_phase_L1_power_kw", 0.0),
        reading("self", "charging_phase_L3_power_kw", 8.0),
        reading("charger_1", "last_applied_power_kw", 4.75),
    ]
    snapshot = SimpleNamespace(
        groups_for=lambda _: (group,), parts_for=lambda _: parts, constraints=()
    )
    raw = LocalActionBundle("Member_003", (ActionDecision(key, "IDLE", 0.0, 0),))
    limits = {"grid_import": {"max": 20.0}, "phase_import": {"max": {"L1": 0.0, "L3": 3.0}}}
    assert prelimit_exceedance_kwh(
        snapshot, raw, step_seconds=900, declared=limits
    ) == pytest.approx(0.25 * 0.25)
    group.phase_weights = ()
    assert prelimit_exceedance_kwh(snapshot, raw, step_seconds=900, declared=limits) is None


def test_long_deadline_stays_in_ledger_and_subinterval_deadline_does_not_borrow_capacity():
    a, b, _ = example().equipment
    site = SiteReservation(
        SiteInputs((replace(a, deadline_seconds=172800.0), replace(b, deadline_seconds=30.0)), ()),
        step_seconds=60.0,
    )
    assert site.diagnostics["requested_service"] == {"A": 1.5, "B": 2.0}
    assert site.diagnostics["service_status"] == {
        "A": "beyond_reservation_horizon",
        "B": "deadline_before_next_command",
    }
    assert site.protected == {"B": 0.0}
    site.select_mode("A", "IDLE")
    assert site.admissible_modes("B") == ("IDLE",)


def test_held_setpoint_cannot_cross_quarter_hour_above_the_next_limit():
    car = replace(example().equipment[0], deadline_seconds=1800.0, target_kwh=30.01)
    cap = PowerLimit("site", {"A": 1.0}, "import", 7.0, (7.0, 0.0, 7.0))
    site = SiteReservation(
        SiteInputs((car,), (cap,)), step_seconds=60.0, block_offset_seconds=870.0
    )
    assert site.admissible_modes("A") == ("IDLE",)


def test_expired_budget_keeps_only_the_independently_checked_supervised_command():
    from algorithms.ti_marl.runtime.reservation_solver import ReservationUnavailable

    site = SiteReservation(example())
    site.model.finish = 0.0
    with pytest.raises(ReservationUnavailable, match="time budget expired"):
        site.admissible_modes("A")
    bundle = site.verified_fallback("house")
    assert (
        len(bundle.decisions) == 3
        and bundle.interventions[0]["reason"] == "reservation_supervised_fallback"
    )
    site._initial_witness[0] = 1e8
    with pytest.raises(ReservationUnavailable, match="no longer verifies"):
        site.verified_fallback("house")


def test_lost_command_reserves_the_full_device_rating_until_stop_is_confirmed():
    _, b, washer = example().equipment
    lost = Equipment("A", (IDLE,), unconfirmed_import_kw=7.0)
    inputs = SiteInputs((lost, b, washer), example().limits)
    site = SiteReservation(inputs)
    site.select_mode("A", "IDLE")
    site.select_mode("B", "CHARGE_EV")
    assert site.remaining_margin_kw("B") == (3.0, 2.0)
    assert site.admissible_modes("W") == ("IDLE",)
    site.select_mode("W", "IDLE")
    # B needs 2 kWh in the remaining four quarters. The reserved
    # circuit allows exactly 2 kW throughout when A may still be active.
    assert tuple(v * 7.0 for v in site.fraction_bounds("B")) == pytest.approx((2.0, 2.0))
    recovered = SiteReservation(
        replace(inputs, equipment=(replace(lost, unconfirmed_import_kw=0.0), b, washer))
    )
    recovered.select_mode("A", "IDLE")
    recovered.select_mode("B", "CHARGE_EV")
    assert "START" in recovered.admissible_modes("W")


def test_unconfirmed_output_metadata_survives_checkpoint_and_requires_fresh_zero(tmp_path):
    from tests.test_ti_marl_compiler import compiler
    from tests.ti_marl_fixtures import entity_payload, stuck_sensor_status

    tic = compiler(tmp_path)
    status = stuck_sensor_status(duration=1, age=1)
    status["actuator_channels"] = [
        {
            "event_id": "loss",
            "event_ids": ["loss"],
            "target_type": "storage",
            "target_id": "Building_1/electrical_storage",
            "target_feature": "both",
            "availability": "UNAVAILABLE",
            "quality": "INVALID",
            "fault_mode": "dropout",
        }
    ]
    snapshot = tic.compile(entity_payload(runtime_status=status))
    battery = next(
        group for group in snapshot.action_groups if group.group_id == "Building_1:battery_1"
    )
    assert battery.output_unconfirmed
    state = tic.checkpoint_state()
    tic.reset_runtime_state()
    assert not tic.checkpoint_state()["unconfirmed_outputs"]
    tic.load_checkpoint_state(state)
    # This older fixture provides no timestamped last-applied-power feedback.
    # Disappearance of the injected fault alone cannot release its capacity.
    later = tic.compile(entity_payload(time_step=2))
    battery = next(
        group for group in later.action_groups if group.group_id == "Building_1:battery_1"
    )
    assert battery.output_unconfirmed


@pytest.mark.parametrize(
    "margin,expected_modes", [(0.0, ("IDLE", "CHARGE_EV")), (900.0, ("CHARGE_EV",))]
)
def test_optional_departure_margin_protects_loss_of_last_charging_interval(margin, expected_modes):
    car = replace(example().equipment[0], deadline_seconds=1800.0, target_kwh=31.0)
    site = SiteReservation(SiteInputs((car,), ()), departure_margin_seconds=margin)
    assert site.admissible_modes("A") == expected_modes
    assert site.protected["A"] == 1.0
    if margin:
        assert site.diagnostics["ev_early_reserve"]["A"]["protected_kwh"] == pytest.approx(31.0)
        site.select_mode("A", "CHARGE_EV")
        assert site.fraction_bounds("A")[0] * 7 == pytest.approx(4.0)


def test_unachievable_early_target_keeps_full_service_on_real_deadline():
    car = replace(example().equipment[0], deadline_seconds=1800.0, target_kwh=32.0)
    site = SiteReservation(SiteInputs((car,), ()), departure_margin_seconds=900.0)
    assert site.protected["A"] == 1.0
    record = site.diagnostics["ev_early_reserve"]["A"]
    assert record["shortfall_at_margin_kwh"] == pytest.approx(0.25)
    assert record["protected_kwh"] == pytest.approx(31.75)
    site.select_mode("A", "CHARGE_EV")
    assert site.fraction_bounds("A") == pytest.approx((1.0, 1.0))


def test_early_buffer_never_preempts_another_real_deadline():
    a, b, _ = example().equipment
    a = replace(a, target_kwh=31.75)
    b = replace(b, target_kwh=21.75, deadline_seconds=1800.0)
    site = SiteReservation(
        SiteInputs(
            (a, b), (PowerLimit("site", {"A": 1.0, "B": 1.0}, "import", 7.0, constant=True),)
        ),
        departure_margin_seconds=900.0,
    )
    assert site.protected == {"A": 1.0, "B": 1.0}
    assert site.early_reserve["B"]["shortfall_at_margin_kwh"] == pytest.approx(1.75)
    site.select_mode("A", "CHARGE_EV")
    assert site.admissible_modes("B") == ("IDLE",)


def test_another_native_solve_cannot_block_site_beyond_its_deadline():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from algorithms.ti_marl.runtime.reservation_solver import LinearModel, ReservationUnavailable
    from algorithms.utils.highs_scheduler import highs_scheduler

    # Occupy the common scheduler and keep it until after the assertion. A
    # blocking lock here would wait for the external oracle rather than falling back.
    entered, finish = Event(), Event()

    def other_solve():
        with highs_scheduler(None, 2):
            entered.set()
            assert finish.wait(timeout=2.0)

    with ThreadPoolExecutor(1) as pool:
        job = pool.submit(other_solve)
        try:
            assert entered.wait(timeout=1.0)
            model = LinearModel(0.05)
            model.var()
            with pytest.raises(ReservationUnavailable, match="time budget expired"):
                model.solve()
            assert not job.done()
        finally:
            finish.set()


def test_retry_contradictory_infeasibility_requires_a_verified_witness(monkeypatch):
    import highspy

    from algorithms.ti_marl.runtime.reservation_solver import LinearModel, ReservationInfeasible

    model = LinearModel(5.0)
    col = model.var()
    previous = model.solve({col: -1.0})
    assert previous[col] == pytest.approx(1.0)
    actual_status = highspy.Highs.getModelStatus
    statuses = []

    def spurious_infeasibility(solver):
        statuses.append(solver.getOptionValue("presolve")[1])
        if len(statuses) == 1:
            return highspy.HighsModelStatus.kInfeasible
        return actual_status(solver)

    monkeypatch.setattr(highspy.Highs, "getModelStatus", spurious_infeasibility)
    optimum = model.solve({col: 1.0}, known_feasible=previous)
    assert optimum[col] == pytest.approx(0.0)
    assert statuses == ["choose", "off"]
    assert model.solver.getOptionValue("presolve")[1] == "choose"

    model.bound(col, 0.0, 0.0)
    statuses.clear()
    with pytest.raises(ReservationInfeasible):
        model.solve(known_feasible=previous)
    assert statuses == ["choose"]


def test_forecast_export_alert_keeps_measured_limits_and_complete_programmes():
    washer = Equipment("W", (IDLE, START), deadline_seconds=3600.0, programme_kw=(1.0,) * 3)
    export = PowerLimit("site", {"W": 1.0}, "export", 4.0, (4.0, 4.0, -3.0, 4.0), scope="site")
    data = SiteInputs((washer,), (export,))
    site = SiteReservation.plan(data)
    assert site.diagnostics["forecast_export_alert"] == {"at_seconds": 1800.0, "limit": "site"}
    assert site.diagnostics["verified"]
    assert site.edges[-1] == 1800
    assert site.admissible_modes("W") == ("IDLE",)
    from algorithms.ti_marl.runtime.reservation_solver import ReservationInfeasible

    # A present violation cannot be hidden by a forecast alert.
    with pytest.raises(ReservationInfeasible):
        SiteReservation.plan(replace(data, limits=(replace(export, current_kw=-3.0),)))
    # A held, irreversible action cannot be cut short for certification.
    with pytest.raises(ReservationInfeasible):
        SiteReservation.plan(replace(data, equipment=(replace(washer, running=True),)))


def test_available_absorption_preserves_the_full_export_witness():
    car = Equipment("A", (IDLE, CHARGE), Battery(60, 10), deadline_seconds=1800, target_kwh=11)
    export = PowerLimit("phase", {"A": 1.0}, "export", 1, (1, -3), scope="phase")
    site = SiteReservation.plan(SiteInputs((car,), (export,)))
    assert site.diagnostics["forecast_export_alert"] is None
    assert site.edges[-1] == 1800
    assert site.diagnostics["verified"]


def test_softcps_programme_uses_last_occupied_slot_and_separate_last_start():
    from types import SimpleNamespace

    from algorithms.ti_marl.runtime.reservation_inputs import _profiles

    group = SimpleNamespace(
        group_id="W",
        group_type="deferrable",
        reservation_profile={},
        ports=[],
        max_charge_power_kw=0,
        max_discharge_power_kw=0,
        output_unconfirmed=False,
    )
    values = dict(
        pending=1,
        cycle_duration_steps=4,
        cycle_peak_power_kw=1,
        hours_until_deadline=0.75,
        hours_until_latest_start=0,
    )
    for deadline in (0.75, 4.0):
        unit = _profiles(group, dict(values, hours_until_deadline=deadline), 900, 0.05, [])
        assert unit.deadline_seconds == 3600
        site = SiteReservation(SiteInputs((replace(unit, modes=(IDLE, START)),), ()))
        assert site.admissible_modes("W") == ("START",)


def test_programme_peak_is_not_guaranteed_absorption_of_forecast_solar():
    washer = Equipment("W", (IDLE, START), deadline_seconds=1800, programme_kw=(1, 4))
    cap = PowerLimit("phase", {"W": 1.0}, "export", 3, (3, -3), scope="phase")
    assert SiteReservation.plan(SiteInputs((washer,), (cap,))).admissible_modes("W") == ("START",)
    unknown_shape = replace(washer, programme_min_kw=(1, 0))
    domain = SiteReservation.plan(SiteInputs((unknown_shape,), (cap,)))
    assert domain.diagnostics["forecast_export_alert"]["at_seconds"] == 900
    assert domain.admissible_modes("W") == ("IDLE",)


def test_absorbable_forecast_export_cannot_hide_an_unrelated_import_conflict():
    from algorithms.ti_marl.runtime.reservation_solver import ReservationInfeasible

    car = Equipment("A", (IDLE, CHARGE), Battery(60, 10), deadline_seconds=1800, target_kwh=11)
    lost = Equipment("L", (IDLE,), unconfirmed_import_kw=3)
    export = PowerLimit("site", {"A": 1.0}, "export", 5, (5, -3), scope="site")
    branch = PowerLimit("branch", {"L": 1.0}, "import", 4, (4, 1), scope="group")
    with pytest.raises(ReservationInfeasible):
        SiteReservation.plan(SiteInputs((car, lost), (export, branch)))


def test_consecutive_solar_alerts_cannot_use_the_same_programme_start_twice():
    washer = Equipment("W", (IDLE, START), deadline_seconds=3600, programme_kw=(1,))
    cap = PowerLimit("phase", {"W": 1.0}, "export", 3, (3, -0.5, -0.5, 3), scope="phase")
    domain = SiteReservation.plan(SiteInputs((washer,), (cap,)))
    assert domain.diagnostics["forecast_export_alert"]["at_seconds"] == 900
    assert domain.edges == [0, 900]


def test_passive_import_forecast_rechecks_before_overload_and_preserves_committed_loads():
    from algorithms.ti_marl.runtime.reservation_solver import ReservationInfeasible

    washer = Equipment("W", (IDLE, START), deadline_seconds=3600, programme_kw=(1, 1))
    limit = PowerLimit("phase", {"W": 1.0}, "import", 3, (3, -0.8, 2), scope="phase")
    domain = SiteReservation.plan(SiteInputs((washer,), (limit,)))
    assert domain.diagnostics["forecast_import_alert"] == {"at_seconds": 900, "limit": "phase"}
    assert domain.diagnostics["forecast_export_alert"] is None
    assert domain.admissible_modes("W") == ("IDLE",)
    for unit, cap in [
        (replace(washer, running=True), limit),
        (washer, replace(limit, current_kw=-1)),
    ]:
        with pytest.raises(ReservationInfeasible):
            SiteReservation.plan(SiteInputs((unit,), (cap,)))


def test_certified_window_protects_a_complete_cycle_without_relaxing_export():
    washer = Equipment("W", (IDLE, START), deadline_seconds=3600, programme_kw=(1.0, 1.0))
    export = PowerLimit("site", {"W": 1.0}, "export", 4.0, (4.0, 4.0, -3.0, 4.0), scope="site")
    inputs = SiteInputs((washer,), (export,))
    legacy = SiteReservation.plan(inputs)
    assert set(legacy.admissible_modes("W")) == {"IDLE", "START"}
    enabled = SiteReservation.plan(inputs, reserve_programmes_in_certified_window=True)
    assert enabled.edges == legacy.edges and enabled.edges[-1] == 1800
    assert enabled.units["W"].deadline_seconds == 3600
    assert enabled.service_status["W"] == "in_certified_window"
    assert enabled.admissible_modes("W") == ("START",)
    too_long = replace(inputs, equipment=(replace(washer, programme_kw=(1.0,) * 3),))
    blocked = SiteReservation.plan(too_long, reserve_programmes_in_certified_window=True)
    assert blocked.admissible_modes("W") == ("IDLE",)
    assert blocked.service_status["W"] == "beyond_reservation_horizon"


def test_running_programme_can_certify_its_tail_from_metered_energy():
    from types import SimpleNamespace

    from algorithms.ti_marl.runtime.reservation_inputs import _profiles
    from algorithms.ti_marl.runtime.reservation_solver import ReservationInfeasible

    group = SimpleNamespace(
        group_id="washer",
        group_type="deferrable",
        reservation_profile={},
        ports=[],
        max_charge_power_kw=0,
        max_discharge_power_kw=0,
        output_unconfirmed=False,
    )
    # A non-interruptible three-quarter programme has 0.10 kWh left after
    # this quarter. All of it could fall in either of the two later quarters.
    values = dict(
        running=1,
        remaining_duration_steps=3,
        cycle_peak_power_kw=2,
        hours_until_deadline=0.5,
        current_step_power_kw=1.6,
        current_step_energy_kwh=0.4,
        remaining_energy_kwh=0.5,
    )

    def equipment(observations):
        return replace(_profiles(group, observations, 900, 0.05, []), modes=(IDLE, START))

    washer = equipment(values)
    limit = PowerLimit("site", {"washer": 1}, "import", 1.0, (1, 0.5, 0.5), scope="site")
    inputs = SiteInputs((washer,), (limit,))
    reservation = SiteReservation.plan(inputs)
    assert reservation.diagnostics["verified"]
    assert reservation.edges == [0, 900, 1800, 2700]
    assert reservation.units["washer"].programme_kw == pytest.approx((1.6, 0.4, 0.4), abs=1e-5)
    assert reservation.units["washer"].programme_min_kw == (1.6, 0.0, 0.0)
    assert reservation.limits == (limit,)
    assert "washer:measured_running_energy_envelope" in reservation.notes

    loose = replace(limit, future_kw=(1, 3, 3))
    already_feasible = SiteReservation.plan(replace(inputs, limits=(loose,)))
    assert already_feasible.units["washer"].programme_kw == (1.6, 2, 2)

    # 0.10 kWh could all be used in one slot. An average over the two slots
    # would incorrectly admit this 0.30 kW limit.
    tight = replace(limit, future_kw=(1, 0.3, 0.3))
    with pytest.raises(ReservationInfeasible):
        SiteReservation.plan(replace(inputs, limits=(tight,)))
    export = replace(limit, direction="export", future_kw=(1, -0.1, -0.1))
    with pytest.raises(ReservationInfeasible):
        SiteReservation.plan(replace(inputs, limits=(limit, export)))

    for inconsistent in (
        {k: v for k, v in values.items() if k != "remaining_energy_kwh"},
        dict(values, remaining_energy_kwh=0.1),
        dict(values, current_step_energy_kwh=0.2),
        dict(values, remaining_energy_kwh=2.0),
    ):
        unverified = equipment(inconsistent)
        assert unverified.running_tail_max_kw is None
        with pytest.raises(ReservationInfeasible):
            SiteReservation.plan(SiteInputs((unverified,), (limit,)))
    group.output_unconfirmed = True
    assert equipment(values).running_tail_max_kw is None


def test_pending_programme_reserves_energy_for_the_announced_peak():
    from algorithms.ti_marl.runtime.reservation_inputs import _pending_programme_upper_kw

    # Same 15-minute first sample and peak, very little energy in the other slots.
    upper = (1.6, 2.0, 2.0, 2.0)
    summary = dict(cycle_peak_step_offset_ratio=2 / 3, cycle_energy_kwh=0.925)
    bounded = _pending_programme_upper_kw(upper, 1.6, 2, summary, 900)
    assert bounded[0] == 1.6 and bounded[2] == 2
    assert 0.1 <= bounded[1] == bounded[3] < 0.101
    for other in (
        {},
        dict(summary, cycle_energy_kwh=0.2),
        dict(summary, cycle_peak_step_offset_ratio=0.5),
        dict(summary, cycle_peak_step_offset_ratio=0),
    ):
        assert _pending_programme_upper_kw(upper, 1.6, 2, other, 900) == upper
    assert _pending_programme_upper_kw(upper, None, 2, summary, 900) == upper

    # When the peak is at the start its energy can be counted only once.
    upper = (2.0, 2.0, 2.0)
    peak_first = dict(cycle_peak_step_offset_ratio=0, cycle_energy_kwh=0.6)
    bounded = _pending_programme_upper_kw(upper, 2, 2, peak_first, 900)
    assert bounded[0] == 2 and 0.4 <= bounded[1] == bounded[2] < 0.401


def test_measured_pending_energy_opens_a_verified_start_under_a_later_power_limit():
    from types import SimpleNamespace

    from algorithms.ti_marl.runtime.reservation_inputs import _profiles

    group = SimpleNamespace(
        group_id="dryer",
        group_type="deferrable",
        reservation_profile={},
        ports=[],
        max_charge_power_kw=0,
        max_discharge_power_kw=0,
        output_unconfirmed=False,
    )
    values = dict(
        pending=1,
        can_start=1,
        cycle_duration_steps=3,
        cycle_peak_power_kw=2.0,
        start_power_kw=1.7,
        cycle_peak_step_offset_ratio=0.5,
        cycle_energy_kwh=0.95,
        hours_until_deadline=0.5,
        hours_until_latest_start=0,
    )
    limit = PowerLimit("site", {"dryer": 1}, "import", 2.1, (2.1, 2.1, 0.2), scope="site")

    def modes(readings):
        unit = replace(_profiles(group, readings, 900, 0.05, []), modes=(IDLE, START))
        return SiteReservation.plan(SiteInputs((unit,), (limit,))).admissible_modes("dryer")

    assert modes(values) == ("START",)
    assert modes({k: v for k, v in values.items() if k != "cycle_energy_kwh"}) == ("IDLE",)
    assert modes(dict(values, cycle_energy_kwh=1.1)) == ("IDLE",)
    assert modes(dict(values, cycle_peak_step_offset_ratio=1.0)) == ("IDLE",)


def test_current_programme_preview_cannot_justify_delaying_a_certified_start():
    unit = Equipment(
        "dryer",
        (IDLE, START),
        programme_kw=(1.0, 2.0, 2.0),
        programme_min_kw=(1.0, 0.0, 0.0),
        start_now_upper_kw=(1.0, 2.0, 0.1),
        programme_step_seconds=900,
        deadline_seconds=3600,
    )
    # The first three quarters can cover the full peak. A delay would require
    # assuming that the last low-power slot is still certified at the next issue.
    limit = PowerLimit("site", {"dryer": 1}, "import", 2.1, (2.1, 2.1, 2.1, 0.2), scope="site")
    domain = SiteReservation.plan(SiteInputs((unit,), (limit,)))
    assert domain.admissible_modes("dryer") == ("START",)
