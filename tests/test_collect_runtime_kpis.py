"""Current runtime KPIs remain collectable through CSV-only orchestrators."""

import pytest

from scripts.collect_remote_results import _matrix_csv, _parse_kpi_matrix, _runtime_kpi_matrix


def test_runtime_kpis_round_trip_without_recomputing_values_or_invalid_cells():
    payload = {
        "schema_version": "algorithms.kpis.v1",
        "eligible_for_scorecard": True,
        "metrics": [
            {
                "name": "district_cost_community_market_settled_total_eur",
                "entity": "District",
                "value": 48.425029061473175,
                "valid": True,
            },
            {"name": "battery_ratio", "entity": "Member_001", "value": 0.3, "valid": False},
            {"name": "battery_ratio", "entity": "Member_002", "value": 0.0, "valid": True},
        ],
    }
    matrix = _runtime_kpi_matrix(payload)
    restored = _parse_kpi_matrix(_matrix_csv(matrix))
    assert (
        restored["district_cost_community_market_settled_total_eur"]["District"]
        == 48.425029061473175
    )
    assert restored["battery_ratio"]["Member_001"] is None
    assert restored["battery_ratio"]["Member_002"] == 0.0


@pytest.mark.parametrize(
    "extra",
    [
        {"schema_version": "unknown"},
        {"eligible_for_scorecard": False},
    ],
)
def test_collection_refuses_unqualified_runtime_kpis(extra):
    with pytest.raises(ValueError):
        _runtime_kpi_matrix(
            {
                "schema_version": "algorithms.kpis.v1",
                "eligible_for_scorecard": True,
                "metrics": [],
                **extra,
            }
        )
