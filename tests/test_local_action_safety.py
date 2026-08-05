from __future__ import annotations

import pytest

from algorithms.utils.local_action_safety import (
    ActionConstraint,
    ActionKind,
    ActionSpec,
    DeferrableConstraint,
    ElectricalHeadroom,
    EVConstraint,
    InfeasibleCode,
    InterventionReason,
    LocalActionProjectionRequest,
    LocalActionSafetyProjector,
    StorageConstraint,
    project_local_actions,
)


def _reason_codes(result, action_index: int) -> set[InterventionReason]:
    intervention = next(row for row in result.interventions if row.action_index == action_index)
    return set(intervention.reason_codes)


def _building_15_request(proposed_actions):
    """Battery plus two same-phase EV chargers sharing one building board."""

    return LocalActionProjectionRequest(
        building_id="Building_15",
        proposed_actions=proposed_actions,
        action_specs=(
            ActionSpec(
                "electrical_storage",
                ActionKind.STORAGE,
                low=-1.0,
                high=1.0,
                import_kw_per_action=10.0,
                phases=("L1", "L2", "L3"),
            ),
            ActionSpec(
                "charger_15_1",
                ActionKind.EV,
                low=-1.0,
                high=1.0,
                import_kw_per_action=7.0,
                phases=("L1",),
            ),
            ActionSpec(
                "charger_15_2",
                ActionKind.EV,
                low=-1.0,
                high=1.0,
                import_kw_per_action=7.0,
                phases=("L1",),
            ),
        ),
        action_constraints=(
            ActionConstraint(storage=StorageConstraint(soc=0.5)),
            ActionConstraint(ev=EVConstraint(connected=True, min_required_action=0.5)),
            ActionConstraint(ev=EVConstraint(connected=True, min_required_action=0.5)),
        ),
        headroom=ElectricalHeadroom(
            total_import_kw=10.0,
            phase_import_kw={"L1": 8.0, "L2": 10.0, "L3": 10.0},
        ),
    )


def test_building_15_reserves_two_ev_minima_before_discretionary_storage() -> None:
    result = project_local_actions(_building_15_request((0.6, 1.0, 1.0)))

    # Both 3.5 kW EV minima are reserved first.  The battery receives the
    # remaining 3 kW; no charger can independently consume the same L1 budget.
    assert result.executed_actions == pytest.approx((0.3, 0.5, 0.5))
    assert result.remaining_headroom.total_import_kw == pytest.approx(0.0)
    assert result.remaining_headroom.phase_import_kw["L1"] == pytest.approx(0.0)
    assert result.feasible
    assert not result.infeasible_reasons
    assert InterventionReason.IMPORT_HEADROOM_TOTAL in _reason_codes(result, 0)
    assert InterventionReason.IMPORT_HEADROOM_PHASE in _reason_codes(result, 1)
    assert InterventionReason.IMPORT_HEADROOM_PHASE in _reason_codes(result, 2)


def test_projection_is_idempotent_for_the_same_physical_envelope() -> None:
    projector = LocalActionSafetyProjector()
    first = projector.project(_building_15_request((0.6, 1.0, 1.0)))
    second = projector.project(_building_15_request(first.executed_actions))

    assert second.executed_actions == pytest.approx(first.executed_actions)
    assert second.feasible


def test_bounds_availability_and_storage_soc_are_enforced() -> None:
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(2.0, -2.0, 0.9),
            action_specs=tuple(
                ActionSpec(
                    f"battery_{index}",
                    ActionKind.STORAGE,
                    low=-1.0,
                    high=1.0,
                )
                for index in range(3)
            ),
            action_constraints=(
                ActionConstraint(storage=StorageConstraint(soc=1.0)),
                ActionConstraint(storage=StorageConstraint(soc=0.0)),
                ActionConstraint(
                    storage=StorageConstraint(soc=0.5, available_charge_action=0.4)
                ),
            ),
        )
    )

    assert result.executed_actions == pytest.approx((0.0, 0.0, 0.4))
    assert {InterventionReason.BOUNDS_HIGH, InterventionReason.STORAGE_SOC_MAX} <= _reason_codes(
        result, 0
    )
    assert {InterventionReason.BOUNDS_LOW, InterventionReason.STORAGE_SOC_MIN} <= _reason_codes(
        result, 1
    )
    assert InterventionReason.IMPORT_AVAILABILITY in _reason_codes(result, 2)


def test_urgent_ev_minimum_overrides_v2g_request() -> None:
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(-0.8,),
            action_specs=(
                ActionSpec(
                    "charger",
                    ActionKind.EV,
                    low=-1.0,
                    high=1.0,
                    import_kw_per_action=7.0,
                    export_kw_per_action=7.0,
                    phases=("L1",),
                ),
            ),
            action_constraints=(
                ActionConstraint(
                    ev=EVConstraint(
                        connected=True,
                        min_required_action=0.6,
                        available_charge_action=0.8,
                        available_discharge_action=1.0,
                    )
                ),
            ),
            headroom=ElectricalHeadroom(total_import_kw=7.0, phase_import_kw={"L1": 7.0}),
        )
    )

    assert result.executed_actions == pytest.approx((0.6,))
    assert InterventionReason.EV_MINIMUM in _reason_codes(result, 0)
    assert result.feasible


def test_urgent_ev_reports_infeasible_when_headroom_cannot_supply_minimum() -> None:
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(0.0,),
            action_specs=(
                ActionSpec(
                    "charger",
                    ActionKind.EV,
                    low=-1.0,
                    high=1.0,
                    import_kw_per_action=7.0,
                    phases=("L1",),
                ),
            ),
            action_constraints=(
                ActionConstraint(ev=EVConstraint(connected=True, min_required_action=0.6)),
            ),
            headroom=ElectricalHeadroom(total_import_kw=2.0, phase_import_kw={"L1": 2.0}),
        )
    )

    assert result.executed_actions == pytest.approx((2.0 / 7.0,))
    assert not result.feasible
    assert [reason.code for reason in result.infeasible_reasons] == [
        InfeasibleCode.EV_MINIMUM_HEADROOM
    ]
    assert result.infeasible_reasons[0].action_index == 0
    assert {InterventionReason.IMPORT_HEADROOM_TOTAL, InterventionReason.IMPORT_HEADROOM_PHASE} <= (
        _reason_codes(result, 0)
    )


def test_deferrable_must_start_is_reserved_and_unavailable_is_infeasible() -> None:
    specs = (
        ActionSpec(
            "dishwasher",
            ActionKind.DEFERRABLE,
            low=0.0,
            high=1.0,
            import_kw_per_action=2.0,
            phases=("L2",),
        ),
        ActionSpec("dryer", ActionKind.DEFERRABLE, low=0.0, high=1.0),
    )
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(0.0, 0.0),
            action_specs=specs,
            action_constraints=(
                ActionConstraint(
                    deferrable=DeferrableConstraint(
                        pending=True,
                        running=False,
                        can_start=True,
                        must_start=True,
                    )
                ),
                ActionConstraint(
                    deferrable=DeferrableConstraint(
                        pending=True,
                        running=False,
                        can_start=False,
                        must_start=True,
                    )
                ),
            ),
            headroom=ElectricalHeadroom(total_import_kw=2.0, phase_import_kw={"L2": 2.0}),
        )
    )

    assert result.executed_actions == pytest.approx((1.0, 0.0))
    assert InterventionReason.DEFERRABLE_MUST_START in _reason_codes(result, 0)
    assert InterventionReason.DEFERRABLE_UNAVAILABLE in _reason_codes(result, 1)
    assert [reason.code for reason in result.infeasible_reasons] == [
        InfeasibleCode.DEFERRABLE_MUST_START_LOCAL_LIMIT
    ]
    assert result.infeasible_reasons[0].action_index == 1


def test_deferrable_start_is_binary_and_never_partially_allocated() -> None:
    request = LocalActionProjectionRequest(
        proposed_actions=(0.8,),
        action_specs=(
            ActionSpec(
                "dishwasher",
                ActionKind.DEFERRABLE,
                low=0.0,
                high=1.0,
                import_kw_per_action=2.0,
            ),
        ),
        action_constraints=(
            ActionConstraint(
                deferrable=DeferrableConstraint(
                    pending=True,
                    running=False,
                    can_start=True,
                    must_start=True,
                )
            ),
        ),
        headroom=ElectricalHeadroom(total_import_kw=1.0),
    )

    result = project_local_actions(request)

    assert result.executed_actions == pytest.approx((0.0,))
    assert result.remaining_headroom.total_import_kw == pytest.approx(1.0)
    assert [reason.code for reason in result.infeasible_reasons] == [
        InfeasibleCode.DEFERRABLE_MUST_START_HEADROOM
    ]


@pytest.mark.parametrize(("proposal", "expected"), [(0.5, 0.0), (0.5001, 1.0)])
def test_discretionary_deferrable_command_uses_simulator_threshold(
    proposal: float,
    expected: float,
) -> None:
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(proposal,),
            action_specs=(
                ActionSpec(
                    "dishwasher",
                    ActionKind.DEFERRABLE,
                    low=0.0,
                    high=1.0,
                    import_kw_per_action=2.0,
                ),
            ),
            action_constraints=(
                ActionConstraint(
                    deferrable=DeferrableConstraint(
                        pending=True,
                        running=False,
                        can_start=True,
                    )
                ),
            ),
            headroom=ElectricalHeadroom(total_import_kw=2.0),
        )
    )

    assert result.executed_actions == pytest.approx((expected,))


def test_import_and_export_use_independent_total_and_phase_ledgers() -> None:
    result = project_local_actions(
        LocalActionProjectionRequest(
            proposed_actions=(1.0, -1.0),
            action_specs=(
                ActionSpec(
                    "importing_battery",
                    ActionKind.STORAGE,
                    low=-1.0,
                    high=1.0,
                    import_kw_per_action=10.0,
                    phases=("L1",),
                ),
                ActionSpec(
                    "exporting_battery",
                    ActionKind.STORAGE,
                    low=-1.0,
                    high=1.0,
                    import_kw_per_action=10.0,
                    export_kw_per_action=10.0,
                    phases=("L2",),
                ),
            ),
            action_constraints=(
                ActionConstraint(storage=StorageConstraint(soc=0.5)),
                ActionConstraint(storage=StorageConstraint(soc=0.5)),
            ),
            headroom=ElectricalHeadroom(
                total_import_kw=4.0,
                total_export_kw=8.0,
                phase_import_kw={"L1": 6.0},
                phase_export_kw={"L2": 3.0},
            ),
        )
    )

    assert result.executed_actions == pytest.approx((0.4, -0.3))
    assert InterventionReason.IMPORT_HEADROOM_TOTAL in _reason_codes(result, 0)
    assert InterventionReason.EXPORT_HEADROOM_PHASE in _reason_codes(result, 1)
    assert result.remaining_headroom.total_import_kw == pytest.approx(0.0)
    assert result.remaining_headroom.total_export_kw == pytest.approx(5.0)
    assert result.remaining_headroom.phase_import_kw["L1"] == pytest.approx(2.0)
    assert result.remaining_headroom.phase_export_kw["L2"] == pytest.approx(0.0)
