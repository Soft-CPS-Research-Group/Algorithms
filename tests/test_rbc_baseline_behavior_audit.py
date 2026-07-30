from scripts.audit_rbc_baseline_behavior import (
    _behavior_flags,
    _hard_gate_decision,
    _learning_gate_decision,
    _projection_tolerant_hard_gate_decision,
    _projection_tolerant_learning_gate_decision,
)


def _gate_row() -> dict[str, float]:
    return {
        "community_market_cost_present": 1.0,
        "battery_charge_community_surplus_share": 1.0,
        "battery_discharge_community_import_share": 1.0,
        "battery_discharge_during_community_surplus_share": 0.0,
        "ev_v2g_kwh": 0.0,
        "ev_charge_no_surplus_nonurgent_share": 0.0,
        "ev_min_acceptable_feasible_rate": 1.0,
        "ev_within_tolerance_feasible_rate": 1.0,
        "electrical_violation_kwh": 0.0,
        "electrical_violation_events": 0.0,
        "deferrable_missed_cycles_count": 0.0,
        "deferrable_unserved_energy_kwh": 0.0,
        "deferrable_service_level_rate": 1.0,
        "storage_soc_violation_count": 0.0,
        "outage_unserved_energy_normalized_rate": 0.0,
    }


def test_scorecard_hard_gates_pass_complete_safe_run():
    row = _gate_row()
    row.update(_behavior_flags(row))

    assert _hard_gate_decision(row) == "PASS_HARD_GATES"
    assert _learning_gate_decision(row) == "PASS_LEARNING_GATES"


def test_scorecard_hard_gates_report_service_failures():
    row = _gate_row()
    row.update(
        {
            "ev_min_acceptable_feasible_rate": 0.98,
            "deferrable_missed_cycles_count": 1.0,
            "deferrable_unserved_energy_kwh": 2.0,
            "deferrable_service_level_rate": 0.5,
        }
    )
    row.update(_behavior_flags(row))

    assert _hard_gate_decision(row) == "REJECT_ev_service+deferrable_service"
    assert _learning_gate_decision(row) == "REJECT_ev_service+deferrable_service"


def test_learning_gate_applies_ev_precision_after_hard_gates():
    row = _gate_row()
    row["ev_within_tolerance_feasible_rate"] = 0.39
    row.update(_behavior_flags(row))

    assert _hard_gate_decision(row) == "PASS_HARD_GATES"
    assert _learning_gate_decision(row) == "REJECT_ev_precision"


def test_projection_tolerant_gate_preserves_strict_reject_and_reports_separate_pass():
    row = _gate_row()
    row.update(
        {
            "electrical_violation_kwh": 0.02,
            "electrical_violation_events": 3.0,
            "executed_electrical_safety_certified": 1,
            "projection_request_within_tolerance": 1,
        }
    )
    row.update(_behavior_flags(row))

    assert _hard_gate_decision(row) == "REJECT_electrical_energy+electrical_events"
    assert _projection_tolerant_hard_gate_decision(row) == "PASS_WITH_SAFETY_PROJECTION"
    assert _projection_tolerant_learning_gate_decision(row) == "PASS_WITH_SAFETY_PROJECTION"


def test_projection_tolerant_gate_does_not_hide_other_service_failure():
    row = _gate_row()
    row.update(
        {
            "electrical_violation_kwh": 0.02,
            "electrical_violation_events": 3.0,
            "executed_electrical_safety_certified": 1,
            "deferrable_missed_cycles_count": 1.0,
        }
    )
    row.update(_behavior_flags(row))

    assert _projection_tolerant_hard_gate_decision(row) == "REJECT_deferrable_service"
