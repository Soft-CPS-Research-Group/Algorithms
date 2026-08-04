import pytest

from scripts.build_cc_frozen_leaf_scorecard import build_scorecard


def _aggregate(run: str, *, cost: float, gate: str = "PASS_WITH_SAFETY_PROJECTION", **updates):
    row = {
        "run": run,
        "community_cost_eur": cost,
        "projection_tolerant_learning_gate_decision": gate,
        "community_import_kwh": 100.0,
        "peak_daily_ratio_to_bau": 1.0,
        "peak_all_time_ratio_to_bau": 1.0,
        "ramping_ratio_to_bau": 1.0,
        "load_factor_penalty_daily_ratio_to_bau": 1.0,
        "emissions_kgco2": 100.0,
        "community_solar_self_consumption_rate": 0.8,
        "community_export_kwh": 20.0,
        "community_net_exchange_kwh": 80.0,
        "battery_throughput_kwh": 40.0,
        "battery_throughput_ratio_to_bau": 1.1,
        "v2g_export_kwh": 0.0,
    }
    row.update(updates)
    return row


def _buildings(run: str):
    return {
        "run": run,
        "building_count": 17,
        "projection_tolerant_gate_pass_count": 17,
        "buildings_beating_baseline_projection_tolerant_count": 12,
        "all_buildings_pass_projection_tolerant_gates": 1,
    }


def test_cc_scorecard_passes_cost_candidate_without_material_secondary_regression():
    rows = build_scorecard(
        [
            _aggregate("neutral", cost=100.0),
            _aggregate("candidate", cost=99.0, emissions_kgco2=100.5, ramping_ratio_to_bau=0.98),
        ],
        {name: _buildings(name) for name in ("neutral", "candidate")},
        baseline_name="neutral",
    )

    candidate = rows[1]
    assert candidate["decision"] == "PASS_CC_SCORECARD"
    assert candidate["cost_delta_to_baseline_eur"] == pytest.approx(-1.0)
    assert "ramping_ratio_to_bau" in candidate["secondary_improvements"]


def test_cc_scorecard_exposes_material_tradeoff_instead_of_hiding_it_with_cost():
    rows = build_scorecard(
        [_aggregate("neutral", cost=100.0), _aggregate("candidate", cost=99.0, emissions_kgco2=102.0)],
        {name: _buildings(name) for name in ("neutral", "candidate")},
        baseline_name="neutral",
    )

    candidate = rows[1]
    assert candidate["decision"] == "PASS_COST_WITH_TRADEOFFS"
    assert candidate["secondary_regressions"] == "emissions_kgco2"


@pytest.mark.parametrize(
    ("candidate", "decision"),
    [
        (_aggregate("candidate", cost=101.0), "REJECT_COST"),
        (_aggregate("candidate", cost=99.0, gate="REJECT_electrical"), "REJECT_HARD_GATES"),
    ],
)
def test_cc_scorecard_rejects_failed_primary_requirements(candidate, decision):
    rows = build_scorecard(
        [_aggregate("neutral", cost=100.0), candidate],
        {name: _buildings(name) for name in ("neutral", "candidate")},
        baseline_name="neutral",
    )

    assert rows[1]["decision"] == decision
