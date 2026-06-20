import csv

import pytest

from scripts.build_phase10_opportunity_windows import build_opportunity_windows


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_opportunity_windows_rank_peak_and_solar_spill(tmp_path):
    community = tmp_path / "community.csv"
    pricing = tmp_path / "pricing.csv"
    community_fields = [
        "timestamp",
        "Net Electricity Consumption-kWh",
        "Self Consumption-kWh",
        "Stored energy by community- kWh",
        "Total Solar Generation-kWh",
        "Price-$",
    ]
    pricing_fields = ["timestamp", "electricity_pricing-$/kWh"]
    community_rows = []
    pricing_rows = []
    for step in range(8):
        timestamp = f"2024-01-01T{step:02d}:00:00"
        if step < 4:
            net = 2.0
            solar = 0.0
            self_consumption = 0.0
            cost = 1.0
            price = 0.1
        else:
            net = -8.0 if step < 7 else 15.0
            solar = 12.0
            self_consumption = 2.0
            cost = 3.0 if net > 0 else 0.0
            price = 0.3
        community_rows.append(
            {
                "timestamp": timestamp,
                "Net Electricity Consumption-kWh": net,
                "Self Consumption-kWh": self_consumption,
                "Stored energy by community- kWh": 0.0,
                "Total Solar Generation-kWh": solar,
                "Price-$": cost,
            }
        )
        pricing_rows.append({"timestamp": timestamp, "electricity_pricing-$/kWh": price})
    _write_csv(community, community_fields, community_rows)
    _write_csv(pricing, pricing_fields, pricing_rows)

    rows = build_opportunity_windows(
        community,
        baseline_pricing_path=pricing,
        window_steps=4,
        stride_steps=4,
    )

    assert rows[0]["start_step"] == 4
    assert rows[0]["opportunity_type"] in {
        "peak_shaving+solar_absorption",
        "peak_shaving+solar_absorption+price_arbitrage",
        "solar_absorption+price_arbitrage",
    }
    assert rows[0]["export_kwh"] == pytest.approx(24.0)
    assert rows[0]["peak_import_kwh"] == pytest.approx(15.0)
    assert rows[0]["solar_spill_kwh"] == pytest.approx(40.0)


def test_opportunity_windows_add_candidate_deltas(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    fields = [
        "timestamp",
        "Net Electricity Consumption-kWh",
        "Self Consumption-kWh",
        "Stored energy by community- kWh",
        "Total Solar Generation-kWh",
        "Price-$",
    ]
    baseline_rows = [
        {
            "timestamp": f"2024-01-01T{step:02d}:00:00",
            "Net Electricity Consumption-kWh": 10.0,
            "Self Consumption-kWh": 0.0,
            "Stored energy by community- kWh": 0.0,
            "Total Solar Generation-kWh": 0.0,
            "Price-$": 2.0,
        }
        for step in range(4)
    ]
    candidate_rows = [
        {
            **row,
            "Net Electricity Consumption-kWh": 12.0,
            "Price-$": 2.5,
        }
        for row in baseline_rows
    ]
    _write_csv(baseline, fields, baseline_rows)
    _write_csv(candidate, fields, candidate_rows)

    row = build_opportunity_windows(
        baseline,
        candidate_community_path=candidate,
        window_steps=4,
        stride_steps=4,
    )[0]

    assert row["candidate_cost_delta_eur"] == pytest.approx(2.0)
    assert row["candidate_import_delta_kwh"] == pytest.approx(8.0)
    assert row["candidate_peak_import_delta_kwh"] == pytest.approx(2.0)
    assert "candidate_regression" in row["opportunity_type"]


def test_opportunity_windows_treat_negative_solar_as_generation(tmp_path):
    community = tmp_path / "community.csv"
    fields = [
        "timestamp",
        "Net Electricity Consumption-kWh",
        "Self Consumption-kWh",
        "Stored energy by community- kWh",
        "Total Solar Generation-kWh",
        "Price-$",
    ]
    rows = [
        {
            "timestamp": f"2024-01-01T{step:02d}:00:00",
            "Net Electricity Consumption-kWh": -3.0,
            "Self Consumption-kWh": 1.0,
            "Stored energy by community- kWh": 0.5,
            "Total Solar Generation-kWh": -5.0,
            "Price-$": 0.0,
        }
        for step in range(4)
    ]
    _write_csv(community, fields, rows)

    row = build_opportunity_windows(community, window_steps=4, stride_steps=4)[0]

    assert row["solar_generation_kwh"] == pytest.approx(20.0)
    assert row["self_consumption_kwh"] == pytest.approx(4.0)
    assert row["solar_spill_kwh"] == pytest.approx(14.0)
