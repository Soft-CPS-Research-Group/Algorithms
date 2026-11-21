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
