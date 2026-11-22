"""Causal forecasts, two physical clocks and the two economic accounts."""

import json
from dataclasses import replace

import pytest
import torch

from algorithms.configuration.agents import TIMARLHierarchyConfig, TIMARLHyperparameters
from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.learning.community_signal import CommunitySignal, ImportPeak
from algorithms.ti_marl.policy.forecast_encoder import ForecastEncoder
from algorithms.ti_marl.runtime.forecasts import (
    ESTIMATED,
    HORIZON,
    PUBLISHED,
    ForecastInputs,
    IssuedForecastArchive,
)
from tests.test_ti_marl_policy_runtime import compile_snapshot


def issued(site, **changes):
    return dict(
        {
            "site": site,
            "issued_at_seconds": 0,
            "received_at_seconds": 10,
            "valid_from_seconds": 0,
            "interval_seconds": 900,
            "price_eur_per_kwh": [0.30, 0.10] * 48,
            "published_price": [True] * 48 + [False] * 48,
            "solar_kw": [0.0] * HORIZON,
            "non_shiftable_kw": [2.0] * HORIZON,
        },
        **changes,
    )


def test_delivery_clock_revision_and_unknown_are_preserved(tmp_path):
    _, snapshot = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = snapshot.agent_ids[0]
    feed = tmp_path / "predictor.jsonl"
    records = [
        issued(site),
        issued(site, issued_at_seconds=12, received_at_seconds=15),
        issued(site, issued_at_seconds=4, received_at_seconds=17),
    ]
    feed.write_text("\n".join(json.dumps(record) for record in reversed(records)))
    archive = IssuedForecastArchive(feed)
    assert not archive.at(9)
    assert archive.at(10)[site] == records[0]
    assert archive.at(18)[site] == records[1]
    frame = ForecastInputs(15).snapshot(snapshot, archive.at(18), 18)
    curve = frame.outlooks[0]
    assert curve.values[0][:2] == (0.30, 0.10)
    assert curve.quality[0][0] == PUBLISHED and curve.quality[0][95] == ESTIMATED
    shifted = replace(curve, values=(curve.values[0][::-1], *curve.values[1:]))
    encoder = ForecastEncoder(16)
    before = encoder(frame, [site])
    after = encoder(replace(frame, outlooks=(shifted,)), [site])
    assert not torch.allclose(before, after)
    missing = ForecastInputs(15).snapshot(replace(snapshot, observation_parts=()), {}, 18)
    assert sum(sum(row) for row in missing.outlooks[0].quality) == 0


@pytest.mark.parametrize("dt", [15, 60, 900])
def test_meter_uses_interval_energy_and_manager_holds_curve_for_quarter_hour(tmp_path, dt):
    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    one = ForecastInputs(dt).snapshot(base, {}, 0)
    options = TIMARLHierarchyConfig(signal="learned", training_phase="coordinator").model_dump()
    manager = CommunitySignal(options, 16, "cpu", 21, 0.99)
    first = manager.curve_for(one, True)
    for second in range(0, 900, dt):
        snapshot = replace(one, clock_seconds=second)
        assert manager.curve_for(snapshot, True) == first
        increment = manager.peak.observe(second, dt, dt / 3600 * 4)
        manager.observe_reward(-dt / 3600, dt)
    assert increment == pytest.approx(4)
    assert len(manager.decisions) == 1 and manager.decisions[0].duration == 900
    next_curve = manager.curve_for(replace(one, clock_seconds=900), True)
    assert next_curve != first
    manager.observe_reward(-1.0, dt)
    before = manager.network.location.weight.detach().clone()
    assert manager.finish()["TI_MARL/manager_blocks"] == 2
    assert not torch.equal(manager.network.location.weight, before)


def test_partial_quarter_is_not_a_peak_and_new_day_resets():
    peak = ImportPeak()
    assert peak.observe(86340, 60, 1.0) == 0.0
    assert peak.observe(86400, 900, 1.5) == pytest.approx(6.0)
    assert peak.observe(87300, 900, 1.0) == 0.0
    assert peak.observe(172800, 900, 0.5) == pytest.approx(2.0)


def test_local_virtual_cost_is_incremental_and_manager_settles_real_tariff():
    from types import SimpleNamespace

    controller = object.__new__(TIMARL)
    controller.reward_profile = "settlement_smoke"
    controller.reward_penalties = None
    controller.reference_sites = 2
    controller.manager = SimpleNamespace(phase="local")
    controller._transition_info = {
        "reward_components": {
            "community": {"community_settlement_cost_total": 0.4},
            "per_agent": [
                {"net_electricity_consumption": 2.0, "battery_throughput": 0},
                {"net_electricity_consumption": -1.0, "battery_throughput": 0},
            ],
        }
    }
    controller._latest_diagnostics = {}
    controller._pending = None
    controller.realised_reward = None
    current = SimpleNamespace(incentive=(0.05,), agent_ids=("A", "B"))
    real, shaping = controller._thesis_rewards(current, [-0.4, 0])
    assert real == -0.2 and shaping == {"A": -0.05, "B": 0.0}
    controller.manager.phase = "coordinator"
    assert controller._thesis_rewards(current, [-0.4, 0]) == (real, {})


def test_net_quarter_hour_counts_export_before_setting_peak():
    meter = ImportPeak()
    assert meter.observe(0, 450, 0.75, 0) == 0
    assert meter.observe(450, 450, 0, 0.75) == 0
    assert meter.peak == 0


def test_relative_day_keeps_both_partial_quarters(tmp_path):
    _, snapshot = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = snapshot.agent_ids[0]
    record = issued(site, valid_from_seconds=840, solar_kw=[6.0] * 96)
    frame = ForecastInputs(60).snapshot(snapshot, {site: record}, 840)
    assert len(frame.outlooks[0].values[2]) == 97
    assert frame.outlooks[0].values[2][-1] == 6.0
    assert ForecastEncoder(8)(frame, [site]).shape == (1, 8)


def test_negative_physical_margin_on_wired_phase_and_zero_on_unconnected_phase(tmp_path):
    from types import SimpleNamespace

    from algorithms.ti_marl.contracts.models import LocalConstraint
    from algorithms.ti_marl.runtime.forecasts import reservation_envelopes

    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = base.agent_ids[0]
    record = issued(site, non_shiftable_kw=[3.0] * 96)
    curve = ForecastInputs(900).snapshot(base, {site: record}, 900)
    ids = ["Building_1:charging_phase_L1_headroom_kw", "Building_1:charging_phase_L2_headroom_kw"]
    limits = tuple(LocalConstraint(i, site, "charging_phase_headroom_kw") for i in ids)
    registry = SimpleNamespace(
        for_agent=lambda _: SimpleNamespace(
            constraints={
                "phase_import": {"max": {"L1": 2.0, "L2": 0.0}},
                "background_phase_weights": {"L1": 1.0, "L2": 0.0},
            }
        )
    )
    result = reservation_envelopes(replace(curve, constraints=limits), registry, 0.1)
    assert result[ids[0]][0] == pytest.approx(-1.1)
    assert result[ids[1]][0] == 0.0


def test_thesis_reward_default_and_explicit_environment_control():
    base = {"typed_interfaces_dir": "interfaces"}
    assert TIMARLHyperparameters(**base).operating_reward == "environment"
    assert (
        TIMARLHyperparameters(**base, method_variant="v2").operating_reward
        == "thesis_common_reward_v1"
    )
    hierarchical = TIMARLHyperparameters(**base, method_variant="v2", hierarchy={})
    assert hierarchical.operating_reward == "thesis_common_reward_v1"
    assert (
        TIMARLHyperparameters(
            **base, method_variant="v2", operating_reward="environment"
        ).operating_reward
        == "environment"
    )


def test_archive_replays_previous_episode_after_recent_reads(tmp_path):
    path = tmp_path / "deliveries.jsonl"
    rows = [issued("A"), issued("B"), issued("A", issued_at_seconds=400, received_at_seconds=410)]
    path.write_text("\n\n".join(json.dumps(row) for row in rows))
    archive = IssuedForecastArchive(path)
    assert archive.at(411) == {"A": rows[2], "B": rows[1]}
    assert archive.at(410) == archive.at(411)
    assert archive.at(100) == {"A": rows[0], "B": rows[1]}
    assert archive.at(0) == {}


def test_archive_is_published_only_after_complete_generation(tmp_path):
    from scripts.prepare_ti_marl_forecasts import new_archive

    path = tmp_path / "deliveries.jsonl"
    with pytest.raises(ValueError):
        with new_archive(path) as stream:
            stream.write("partial")
            raise ValueError("invalid physical record")
    assert not list(tmp_path.iterdir())
    with new_archive(path) as stream:
        stream.write("complete")
    with pytest.raises(FileExistsError):
        with new_archive(path):
            pass
    assert path.read_text() == "complete"


@pytest.mark.parametrize("dt", [15, 60, 900])
def test_measured_forecast_error_widens_a_calibrated_reservation(tmp_path, dt):
    from types import SimpleNamespace

    from algorithms.ti_marl.contracts.models import LocalConstraint
    from algorithms.ti_marl.runtime.forecasts import reservation_envelopes

    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = base.agent_ids[0]
    template = base.observation_parts[0]
    start = 900
    limit = LocalConstraint("site_import", site, "charging_headroom_kw")
    base = replace(base, constraints=(limit,))
    registry = SimpleNamespace(
        for_agent=lambda _: SimpleNamespace(constraints={"grid_import": {"max": 12}})
    )
    # Only estimates seen before delivery are eligible. The new issue at
    # 00:15 revises its own current quarter downward, after the cutoff.
    first = issued(
        site,
        issued_at_seconds=0,
        received_at_seconds=0,
        non_shiftable_kw=[1.0] * 96,
        solar_kw=[3.0] * 96,
        calibration_error_kw=[0.5, 0.5],
        calibration_end_seconds=0,
    )
    revised = issued(
        site,
        issued_at_seconds=start,
        received_at_seconds=start,
        non_shiftable_kw=[4.0] * 96,
        solar_kw=[1.0] * 96,
        calibration_error_kw=[0.1, 0.2],
        calibration_end_seconds=0,
    )
    feed = ForecastInputs(dt)
    for now in range(0, 3 * 900 + 1, dt):
        demand = 4.0
        # Metered values vary within the quarter. Their time average is 4 kW.
        if dt < 900 and start <= now < 2 * start:
            demand += (now - start + dt / 2 - 450) / 450
        values = {"non_shiftable_load": demand * dt / 3600, "pv_power_kw": 1.0}
        parts = tuple(
            replace(
                template,
                owner_agent_id=site,
                sensor_id="self",
                observation_id=key,
                values=(val,),
                valid=True,
                age_seconds=0,
                estimated=False,
            )
            for key, val in values.items()
        )
        base = replace(base, observation_parts=parts)
        issue = first if now < start else revised
        snapshot = feed.snapshot(base, {site: issue}, now)
        # The transition and next decision can carry exactly the same reading.
        assert feed.snapshot(base, {site: issue}, now).outlooks == snapshot.outlooks
        row = snapshot.outlooks[0]
        if now < 2 * start:
            assert row.calibration_error_kw == (0.5, 0.5)
        elif now == 2 * start:
            assert row.residual_block_start_seconds == start
            assert row.prediction_as_of_seconds == start - dt
            assert row.residual_kw == pytest.approx((3.0, -2.0))
            assert row.calibration_error_kw == pytest.approx((3.0, 2.0))
            # The new 4 kW forecast with 3 kW uncertainty and zero guaranteed PV.
            assert reservation_envelopes(snapshot, registry, None)["site_import"][
                0
            ] == pytest.approx(5.0)
        elif now == 3 * start:
            assert row.residual_kw == pytest.approx((0.0, 0.0))
            assert row.calibration_error_kw == pytest.approx((3.0, 2.0))


def test_no_physical_feedback_from_missing_measurements_or_predictions(tmp_path):
    from algorithms.ti_marl.runtime.forecasts import MISSING

    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = base.agent_ids[0]
    template = base.observation_parts[0]
    first = issued(site, received_at_seconds=0, non_shiftable_kw=[1.0] * 96, solar_kw=[None] * 96)
    feed = ForecastInputs(60)
    for now in range(0, 1801, 60):
        # Lose one load reading midway through the target quarter. Estimated
        # and stale readings do not complete the observed energy.
        observed = replace(
            template,
            owner_agent_id=site,
            sensor_id="self",
            observation_id="non_shiftable_load",
            values=(0.1,),
            valid=True,
            estimated=now == 1140,
            age_seconds=60 if now == 1200 else 0,
        )
        base = replace(base, observation_parts=(observed,))
        row = feed.snapshot(base, {site: first}, now).outlooks[0]
    assert row.residual_kw == (None, None)
    assert row.quality[2][0] == MISSING
    assert row.calibration_error_kw is None
    assert row.max_abs_residual_kw == (0.0, 0.0)
    # Measuring a real error still cannot establish a calibrated bound.
    for now in range(1860, 2701, 60):
        row = feed.snapshot(base, {site: first}, now).outlooks[0]
    assert row.residual_kw[0] == pytest.approx(5.0)
    assert row.calibration_error_kw is None
    assert ForecastInputs(60).snapshot(base, {site: first}, 0).outlooks[0].max_abs_residual_kw == (
        0.0,
        0.0,
    )


@pytest.mark.parametrize("dt", [15, 60, 900])
def test_lagged_softcps_pv_belongs_to_the_completed_interval(tmp_path, dt):
    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = base.agent_ids[0]
    meter = replace(
        base.observation_parts[0],
        owner_agent_id=site,
        sensor_id="self",
        observation_id="pv_power_kw",
        valid=True,
        age_seconds=0,
        estimated=False,
    )
    record = issued(site, received_at_seconds=0, solar_kw=[2.0] * 96)
    feed = ForecastInputs(dt)
    for now in range(0, 1801, dt):
        # The burst in the last interval of the quarter appears at 00:30.
        value = 900.0 / dt if now == 1800 else 0.0
        snapshot = replace(base, observation_parts=(replace(meter, values=(value,)),))
        curve = feed.snapshot(
            snapshot, {site: record}, now, pv_lag_steps=1 if now else None
        ).outlooks[0]
    assert curve.residual_block_start_seconds == 900
    assert curve.residual_kw == (None, -1.0)


def test_manager_discounts_on_the_same_configured_physical_clock_as_the_local_policy():
    from types import SimpleNamespace

    options = TIMARLHierarchyConfig(signal="learned", training_phase="coordinator").model_dump()
    manager = CommunitySignal(options, 16, "cpu", 42, 0.81, timebase_seconds=60)
    decision = SimpleNamespace(duration=0.0, reward=0.0)
    manager.decisions = [decision]
    manager.observe_reward(1.0, 30)
    manager.observe_reward(1.0, 30)
    assert decision.reward == pytest.approx(1.9)


def test_prepared_meter_forecast_is_causal_and_keeps_the_price_feed(tmp_path):
    import numpy as np
    import pandas as pd

    from scripts.prepare_ti_marl_forecasts import MeterForecaster, prepare

    issue, fit_end, calibration_end = 864, 800, 830
    steps = np.arange(issue + 98)
    daylight = np.maximum(0, np.sin((steps % 96 - 22) * np.pi / 48))
    physical = pd.DataFrame(
        {
            "non_shiftable_load": 0.1 + (steps % 21) / 80,
            "solar_generation": daylight * (1 + (steps // 96) / 4),
        }
    ).astype("float32")
    meter_file = tmp_path / "meters.parquet"
    physical.to_parquet(meter_file)
    tariff = {"electricity_pricing": steps / 900}
    tariff.update({f"electricity_pricing_predicted_{i}": steps / 1100 + i for i in (1, 2, 3)})
    pd.DataFrame(tariff).to_parquet(tmp_path / "price.parquet")
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "seconds_per_time_step": 900,
                "simulation_end_time_step": int(steps[-1]),
                "buildings": {
                    "site": {
                        "energy_simulation": meter_file.name,
                        "pricing": "price.parquet",
                        "pv": {"attributes": {"generation_mode": "absolute"}},
                    }
                },
            }
        )
    )

    def issue_record(filename, **kwargs):
        path = tmp_path / filename
        prepare(schema, path, issue, issue, **kwargs)
        return json.loads(path.read_text())

    prior = issue_record("prior.jsonl")
    fitted = issue_record("fitted.jsonl", meter_fit_end=fit_end, calibration_end=calibration_end)
    assert IssuedForecastArchive(tmp_path / "fitted.jsonl").at(issue * 900)["site"] == fitted
    assert fitted["price_eur_per_kwh"] == prior["price_eur_per_kwh"]
    np.testing.assert_allclose(
        prior["solar_kw"], 4 * physical.solar_generation.iloc[issue - 96 : issue]
    )
    assert len(fitted["solar_kw"]) == len(fitted["non_shiftable_kw"]) == 96
    assert min(fitted["solar_kw"] + fitted["non_shiftable_kw"]) >= 0
    assert fitted["solar_kw"] != prior["solar_kw"]
    meters = physical.to_numpy() * 4
    model = MeterForecaster.fit(meters[:fit_end], fit_end)
    origins = np.arange(fit_end, calibration_end - 1)
    errors = meters[origins + 1] - model.predict(meters, origins, 1)
    np.testing.assert_allclose(fitted["calibration_error_kw"], 1.1 * np.abs(errors).max(axis=0))
    physical.loc[issue:, :] = 75.0
    physical.to_parquet(meter_file)
    assert (
        issue_record("changed.jsonl", meter_fit_end=fit_end, calibration_end=calibration_end)
        == fitted
    )
    with pytest.raises(ValueError, match="before the first issue"):
        issue_record("future.jsonl", meter_fit_end=issue + 1)
    with pytest.raises(ValueError, match="after fitting"):
        issue_record("overlap.jsonl", meter_fit_end=fit_end, calibration_end=fit_end + 1)


def test_weather_nowcast_obeys_issue_and_fit_cutoffs(tmp_path):
    import json

    import numpy as np
    import pandas as pd

    from scripts.prepare_ti_marl_forecasts import prepare
    from scripts.ti_marl_weather_pv import WEATHER_COLUMNS, WeatherPV

    fit_end, issue = 2304, 2330
    t = np.arange(issue + 100)
    daylight = np.maximum(0, np.sin(2 * np.pi * (t % 96 - 24) / 96))
    pv = daylight * (3 + t % 5 / 10)
    w = pd.DataFrame({col: daylight * (100 + i) for i, col in enumerate(WEATHER_COLUMNS)})
    model = WeatherPV.fit(pv[:fit_end], w.iloc[:fit_end], fit_end)
    reference = model.predict(pv, w, [issue])
    pv_moved = pv.copy()
    pv_moved[issue:] += 100
    later_weather = w.copy()
    later_weather.iloc[issue + 1 :] += 100
    extended = WeatherPV.fit(pv_moved, later_weather, fit_end)
    np.testing.assert_equal(model.weights, extended.weights)
    np.testing.assert_equal(extended.predict(pv_moved, later_weather, [issue]), reference)

    series = pd.DataFrame({"non_shiftable_load": 0.2 + t % 4, "solar_generation": pv / 4})
    series.to_parquet(tmp_path / "meter.parquet")
    w.to_parquet(tmp_path / "weather.parquet")
    tariff = pd.DataFrame({"electricity_pricing": t * 0.001})
    for j in (1, 2, 3):
        tariff[f"electricity_pricing_predicted_{j}"] = t * 0.002
    tariff.to_parquet(tmp_path / "tariff.parquet")
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "seconds_per_time_step": 900,
                "simulation_end_time_step": int(t[-1]),
                "buildings": {
                    "home": {
                        "energy_simulation": "meter.parquet",
                        "weather": "weather.parquet",
                        "pricing": "tariff.parquet",
                        "pv": {"attributes": {"generation_mode": "absolute"}},
                    }
                },
            }
        )
    )
    previous = tmp_path / "previous.jsonl"
    updated = tmp_path / "updated.jsonl"
    prepare(schema, previous, issue, issue, meter_fit_end=900)
    prepare(
        schema,
        updated,
        issue,
        issue,
        meter_fit_end=900,
        weather_fit_end=fit_end,
        calibration_end=issue,
    )
    before, after = (json.loads(p.read_text()) for p in (previous, updated))
    assert before["price_eur_per_kwh"] == after["price_eur_per_kwh"]
    for channel in ("non_shiftable_kw", "solar_kw"):
        assert before[channel][0] == after[channel][0]
        assert before[channel][13:] == after[channel][13:]
    assert before["non_shiftable_kw"][2:] == after["non_shiftable_kw"][2:]
    medium = WeatherPV.fit(pv, w, fit_end, 8)
    expectation = medium.predict(pv, w, [issue])[0]
    assert after["solar_kw"][8] == pytest.approx(
        (before["solar_kw"][8] + expectation) / 2, abs=1e-5
    )
    assert medium.predict(pv_moved, later_weather, [issue])[0] == pytest.approx(expectation)
    assert after["non_shiftable_kw"][1] == pytest.approx(series.non_shiftable_load.iat[issue] * 4)
    assert after["solar_kw"][1] == pytest.approx(reference[0], abs=1e-5)
    origins = np.arange(fit_end, issue - 1)
    err_load = 4 * (
        series.non_shiftable_load.to_numpy()[origins + 1]
        - series.non_shiftable_load.to_numpy()[origins]
    )
    err_pv = pv[origins + 1] - model.predict(pv, w, origins)
    np.testing.assert_allclose(
        after["calibration_error_kw"], 1.1 * np.abs([err_load, err_pv]).max(axis=1), atol=1e-5
    )
    with pytest.raises(ValueError, match="after fitting"):
        prepare(
            schema,
            tmp_path / "invalid",
            issue,
            issue,
            weather_fit_end=fit_end,
            calibration_end=fit_end,
        )


def test_paired_export_requires_contiguous_same_lead_evidence_and_preserves_floors(tmp_path):
    from types import SimpleNamespace

    from algorithms.ti_marl.contracts.models import LocalConstraint
    from algorithms.ti_marl.runtime.forecasts import reservation_envelopes

    _, base = compile_snapshot(tmp_path, buildings=("Building_1",))
    site = base.agent_ids[0]
    template = base.observation_parts[0]
    feed = ForecastInputs(900, metered_export_reserve=True)
    limits = tuple(
        LocalConstraint(key, site, kind)
        for key, kind in (
            ("export", "export_headroom_kw"),
            ("site_import", "charging_headroom_kw"),
            ("s:export_phase_L1_kw", "export_phase_headroom_kw"),
            ("s:export_phase_L2_kw", "export_phase_headroom_kw"),
        )
    )
    spec = {
        "grid_export": {"max": 8},
        "grid_import": {"max": 10},
        "phase_export": {"max": {"L1": 3, "L2": 5}},
        "background_phase_weights": {"L1": 1 / 3},
    }
    registry = SimpleNamespace(for_agent=lambda _: SimpleNamespace(constraints=spec))
    # Both meters jump together. Underprediction of net export is zero at lead
    # one, and 2 kW at lead two. The individual absolute errors reach 6 kW.
    for block in range(103):
        now = 900 * block
        demand = 4 if block % 2 == 0 else 10
        lagged_solar = 5 if (block - 1) % 2 == 0 else 11
        obs = tuple(
            replace(
                template,
                sensor_id="self",
                observation_id=key,
                owner_agent_id=site,
                values=(value,),
                valid=not (block == 101 and key == "pv_power_kw"),
                estimated=False,
                age_seconds=0,
            )
            for key, value in (("non_shiftable_load", demand / 4), ("pv_power_kw", lagged_solar))
        )
        site_data = replace(base, constraints=limits, observation_parts=obs)
        solar = [5.0] * 96
        solar[2] = 3.0
        record = issued(
            site,
            issued_at_seconds=now,
            received_at_seconds=now,
            valid_from_seconds=now,
            non_shiftable_kw=[4.0] * 96,
            solar_kw=solar,
            calibration_error_kw=[0.2, 0.3],
            calibration_end_seconds=0,
        )
        snap = feed.snapshot(site_data, {site: record}, now, error_kw=0.1, pv_lag_steps=1)
        curve = snap.outlooks[0]
        if block == 96:
            assert curve.passive_export_error_kw[0] is None
        elif block == 99:
            assert curve.passive_export_error_kw[:2] == pytest.approx((0.5, 2.0))
            again = feed.snapshot(site_data, {site: record}, now, error_kw=0.1, pv_lag_steps=1)
            assert again.outlooks == snap.outlooks
            bounds = reservation_envelopes(snap, registry, 0.1)
            assert bounds["export"][1:3] == pytest.approx((0.0, 0.0))
            assert bounds["s:export_phase_L1_kw"][1] == pytest.approx(0.0)
            # Even higher certified capacity never increases ordinary export headroom.
            spec["grid_export"]["max"] = 15
            assert reservation_envelopes(snap, registry, 0.1)["export"][1] == pytest.approx(4.0)
            assert bounds["s:export_phase_L2_kw"][1] == pytest.approx(-6)
            assert bounds["site_import"][1] == pytest.approx(0)
        elif block == 100:
            missing = dict(record, solar_kw=[5.0, None] + solar[2:])
            assert (
                feed.snapshot(site_data, {site: missing}, now, error_kw=0.1, pv_lag_steps=1)
                .outlooks[0]
                .passive_export_error_kw[0]
                is None
            )
        elif block > 100:
            assert all(v is None for v in curve.passive_export_error_kw)
    assert ForecastInputs(60, metered_export_reserve=True).export_feedback is None
