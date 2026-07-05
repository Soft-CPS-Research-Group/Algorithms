"""Tests for scripts.curate_initiative_figures."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse(argv):
    import scripts.curate_initiative_figures as m
    return m._build_parser().parse_args(argv)


def test_curate_default_output_dir_is_iql_cql_figures():
    args = _parse(["--run-dir", "runs/foo"])
    assert str(args.output_dir).endswith("docs/offline_rl/iql_cql_figures")


def test_curate_default_showcase_group_is_obs627_act1():
    args = _parse(["--run-dir", "runs/foo"])
    assert args.showcase_group == "obs627_act1"


def test_curate_default_groups_are_four_production_groups():
    args = _parse(["--run-dir", "runs/foo"])
    assert args.groups == ["obs627_act1", "obs706_act2", "obs749_act3", "obs785_act3"]


def test_curate_run_dir_is_required():
    import pytest
    with pytest.raises(SystemExit):
        _parse([])


def test_curate_showcase_group_override():
    args = _parse(["--run-dir", "runs/foo", "--showcase-group", "obs163_act1"])
    assert args.showcase_group == "obs163_act1"


# -----------------------------------------------------------------------------
# Task 2: _copy_feature_analysis_figures
# -----------------------------------------------------------------------------

import pytest

SMOKE_DIR = REPO_ROOT / "runs" / "smoke_pipeline_phase9"


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "curated"
    d.mkdir()
    return d


@pytest.fixture
def smoke_available():
    if not SMOKE_DIR.exists():
        pytest.skip(f"smoke fixture not present at {SMOKE_DIR}")
    if not (SMOKE_DIR / "feature_analysis" / "figures").exists():
        pytest.skip("smoke feature_analysis/figures not present")


def test_copy_feature_analysis_figures_produces_five_renamed_pngs(smoke_available, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._copy_feature_analysis_figures(
        run_dir=SMOKE_DIR,
        showcase_group="obs163_act1",  # smoke uses hourly group keys
        output_dir=output_dir,
    )
    assert sorted(p.name for p in produced) == [
        "02_dataset_stats.png",
        "03_action_coverage_group_a.png",
        "04_reward_by_regime.png",
        "05_correlations_group_a.png",
        "06_temporal_patterns.png",
    ]
    for p in produced:
        assert p.exists()
        assert p.stat().st_size > 5_000  # non-empty plot


def test_copy_feature_analysis_figures_missing_dir_returns_empty(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._copy_feature_analysis_figures(
        run_dir=tmp_path,           # empty dir, no feature_analysis
        showcase_group="obs627_act1",
        output_dir=output_dir,
    )
    assert produced == []


# -----------------------------------------------------------------------------
# Task 3: _render_pipeline_diagram
# -----------------------------------------------------------------------------


def test_render_pipeline_diagram_produces_png(output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_pipeline_diagram(output_dir=output_dir)
    assert produced is not None
    assert produced.name == "01_pipeline_overview.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000


# -----------------------------------------------------------------------------
# Task 4: _render_training_curves
# -----------------------------------------------------------------------------


@pytest.fixture
def smoke_has_metrics():
    iql = SMOKE_DIR / "models-iql"
    cql = SMOKE_DIR / "models-cql"
    if not iql.exists() or not cql.exists():
        pytest.skip("smoke models-iql/models-cql not present")


def test_render_training_curves_produces_three_pngs(smoke_available, smoke_has_metrics, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_training_curves(
        run_dir=SMOKE_DIR,
        showcase_group="obs163_act1",
        groups=["obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3"],
        output_dir=output_dir,
    )
    names = sorted(p.name for p in produced)
    assert "07_training_loss_group_a.png" in names
    assert "08_training_valmse_all.png" in names
    assert "09_training_cql_penalty.png" in names
    for p in produced:
        assert p.stat().st_size > 5_000


# -----------------------------------------------------------------------------
# Task 5: _render_benchmark_kpi_bars
# -----------------------------------------------------------------------------


@pytest.fixture
def smoke_results_json():
    p = SMOKE_DIR / "benchmark" / "results.json"
    if not p.exists():
        pytest.skip(f"smoke benchmark results.json not present at {p}")
    return p


def test_render_benchmark_kpi_bars_produces_png(smoke_results_json, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_benchmark_kpi_bars(
        results_json=smoke_results_json,
        output_dir=output_dir,
    )
    assert produced is not None
    assert produced.name == "10_benchmark_kpi_bars.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000


def test_render_benchmark_kpi_bars_missing_file_returns_none(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_benchmark_kpi_bars(
        results_json=tmp_path / "nope.json",
        output_dir=output_dir,
    )
    assert produced is None


# -----------------------------------------------------------------------------
# Task 6: _render_iql_vs_cql_scatter
# -----------------------------------------------------------------------------


def test_render_iql_vs_cql_scatter_produces_png_with_n_equals_one(smoke_results_json, output_dir):
    """Smoke has only one eval seed; the scatter must still render with n=1 annotation."""
    import scripts.curate_initiative_figures as m
    produced = m._render_iql_vs_cql_scatter(
        results_json=smoke_results_json,
        output_dir=output_dir,
    )
    assert produced is not None
    assert produced.name == "11_iql_vs_cql_scatter.png"
    assert produced.exists()
    assert produced.stat().st_size > 5_000


def test_render_iql_vs_cql_scatter_missing_file_returns_none(tmp_path, output_dir):
    import scripts.curate_initiative_figures as m
    produced = m._render_iql_vs_cql_scatter(
        results_json=tmp_path / "nope.json",
        output_dir=output_dir,
    )
    assert produced is None


# -----------------------------------------------------------------------------
# Task 7: _write_sentinel + main() end-to-end
# -----------------------------------------------------------------------------


def test_write_sentinel_atomic_json(output_dir):
    import scripts.curate_initiative_figures as m
    sentinel = m._write_sentinel(
        output_dir=output_dir,
        run_dir=Path("runs/foo"),
        produced=[output_dir / "fake.png"],
    )
    assert sentinel.name == ".curation.done"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["n_figures"] == 1
    assert payload["run_dir"] == "runs/foo"
    assert payload["output_dir"] == str(output_dir)
    assert payload["figures"] == ["fake.png"]
    assert "generated_at" in payload


def test_main_end_to_end_smoke(smoke_available, smoke_has_metrics, smoke_results_json, tmp_path):
    """End-to-end: run main() against smoke artifacts; expect 11 PNGs + sentinel."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"
    rc = m.main([
        "--run-dir", str(SMOKE_DIR),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 11, f"expected 11 PNGs, got {len(pngs)}: {pngs}"
    sentinel = out / ".curation.done"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["n_figures"] == 11


# -----------------------------------------------------------------------------
# Task 8: error-handling regressions
# -----------------------------------------------------------------------------


def test_main_missing_benchmark_still_produces_nine_figures(smoke_available, smoke_has_metrics, tmp_path):
    """When benchmark/results.json is missing, figs 10-11 are skipped but
    the other 9 figures are still produced."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"

    # Stage a smoke copy without benchmark/
    staged = tmp_path / "staged_run"
    shutil.copytree(SMOKE_DIR, staged, ignore=shutil.ignore_patterns("benchmark"))

    rc = m.main([
        "--run-dir", str(staged),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 9, f"expected 9 PNGs without benchmark, got {len(pngs)}: {pngs}"
    payload = json.loads((out / ".curation.done").read_text())
    assert payload["n_figures"] == 9


def test_main_missing_feature_analysis_still_produces_six_figures(
    smoke_available, smoke_has_metrics, smoke_results_json, tmp_path
):
    """When feature_analysis/figures/ is missing, figs 02-06 are skipped
    but figs 01 + 07-11 are still produced (=6 figures)."""
    import scripts.curate_initiative_figures as m
    out = tmp_path / "curated"

    staged = tmp_path / "staged_run"
    shutil.copytree(SMOKE_DIR, staged, ignore=shutil.ignore_patterns("feature_analysis"))

    rc = m.main([
        "--run-dir", str(staged),
        "--output-dir", str(out),
        "--showcase-group", "obs163_act1",
        "--groups", "obs163_act1", "obs225_act2", "obs257_act3", "obs287_act3",
    ])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 6, f"expected 6 PNGs without feature_analysis, got {len(pngs)}: {pngs}"
    payload = json.loads((out / ".curation.done").read_text())
    assert payload["n_figures"] == 6


def test_main_empty_run_dir_produces_only_pipeline_diagram(tmp_path):
    """With an empty run-dir, only fig 01 (pipeline diagram, no run-dir
    dependency) is produced and the sentinel records n_figures==1."""
    import scripts.curate_initiative_figures as m
    empty = tmp_path / "empty_run"
    empty.mkdir()
    out = tmp_path / "curated"

    rc = m.main([
        "--run-dir", str(empty),
        "--output-dir", str(out),
        "--showcase-group", "obs627_act1",
    ])
    # Pipeline diagram has no run-dir dependency, so exactly 1 PNG produced.
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["01_pipeline_overview.png"]
    payload = json.loads((out / ".curation.done").read_text())
    assert payload["n_figures"] == 1


# -----------------------------------------------------------------------------
# Task 9 (Phase 13): Wilcoxon report helpers
# -----------------------------------------------------------------------------
#
# Bug 9 rerun retrospective added a machine-readable ``wilcoxon_report.json``
# alongside the curated PNGs so every p-value cited in section 7 of
# docs/offline_rl/iql_cql_initiative.md is regeneratable from a single
# curate call. The helpers are pure functions of the benchmark results.json
# dict; tests use a synthetic 10-seed dict rather than requiring the actual
# full-year artifacts to be present.


def _synthetic_10seed_results():
    """Fabricated benchmark dict with 10 seeds and deterministic KPIs.

    Designed so that IQL beats RBC on cost/carbon/etc. (all-negative delta)
    and CQL is slightly worse than IQL but still better than RBC. Provides
    a signed, non-degenerate signal for the Wilcoxon paired test.

    Unserved-energy KPI is all-zero across algos to exercise the
    identical-series shortcut in the Wilcoxon helper.
    """
    import numpy as np
    rbc_vals = [2.1 + 0.1 * i for i in range(10)]
    iql_vals = [1.1 + 0.05 * i for i in range(10)]
    cql_vals = [1.5 + 0.2 * i for i in range(10)]

    def _mk(name, vals):
        return {
            "runs": [
                {
                    "env_seed": 200 + i,
                    "label": name,
                    "district": {
                        "cost_total": v,
                        "carbon_emissions_total": v,
                        "daily_peak_average": 1.0,
                        "ramping_average": v * 1.5,
                        "annual_normalized_unserved_energy_total": 0.0,
                        "zero_net_energy": v - 1.0,
                    },
                    "steps": 35039,
                }
                for i, v in enumerate(vals)
            ],
            "aggregate": {
                "cost_total": {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "n": 10},
                "carbon_emissions_total": {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "n": 10},
                "daily_peak_average": {"mean": 1.0, "std": 0.0, "n": 10},
                "ramping_average": {"mean": float(np.mean(vals) * 1.5), "std": float(np.std(vals, ddof=1) * 1.5), "n": 10},
                "annual_normalized_unserved_energy_total": {"mean": 0.0, "std": 0.0, "n": 10},
                "zero_net_energy": {"mean": float(np.mean(vals) - 1.0), "std": float(np.std(vals, ddof=1)), "n": 10},
            },
        }

    return {
        "eval_seeds": list(range(200, 210)),
        "iql_root": "/dummy",
        "cql_root": "/dummy",
        "RBCSmart": _mk("RBCSmart", rbc_vals),
        "IQL": _mk("IQL", iql_vals),
        "CQL": _mk("CQL", cql_vals),
    }


def test_compute_wilcoxon_report_shape():
    """The report has the expected top-level shape and covers all six
    KPIs cited in section 7 of the initiative doc."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    report = m._compute_wilcoxon_report(data)
    assert report["eval_seeds"] == list(range(200, 210))
    expected_kpis = {
        "cost_total",
        "carbon_emissions_total",
        "daily_peak_average",
        "ramping_average",
        "annual_normalized_unserved_energy_total",
        "zero_net_energy",
    }
    assert set(report["kpis"]) == expected_kpis
    for kpi, entry in report["kpis"].items():
        assert set(entry["algos"]) == {"RBCSmart", "IQL", "CQL"}, kpi
        assert set(entry["pvalues"]) == {"IQL_vs_RBCSmart", "CQL_vs_RBCSmart", "IQL_vs_CQL"}, kpi


def test_compute_wilcoxon_report_deltas_have_correct_sign():
    """IQL and CQL synthetic means are lower than RBC's -> deltas_pct_vs_rbc
    must be negative for cost_total. Guards against sign-flip regressions
    in the delta arithmetic which would corrupt section 7 language."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    report = m._compute_wilcoxon_report(data)
    deltas = report["kpis"]["cost_total"]["deltas_pct_vs_rbc"]
    assert deltas["IQL"] < 0
    assert deltas["CQL"] < 0
    # IQL is strictly better than CQL (lower cost) -> more-negative delta
    assert deltas["IQL"] < deltas["CQL"]


def test_compute_wilcoxon_report_pvalues_are_significant_for_signal():
    """Synthetic IQL vs RBC are strictly monotonically different -> the
    paired Wilcoxon must produce p well below 0.05. Regression guard
    for accidentally swapping test statistics or misordered pairs."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    report = m._compute_wilcoxon_report(data)
    p_iql_rbc = report["kpis"]["cost_total"]["pvalues"]["IQL_vs_RBCSmart"]["p"]
    assert p_iql_rbc is not None
    assert p_iql_rbc < 0.05, f"expected significant IQL_vs_RBCSmart p, got {p_iql_rbc}"


def test_compute_wilcoxon_report_identical_series_handled_gracefully():
    """All-zero unserved-energy across algos -> Wilcoxon degenerate case.
    The helper must not raise and must record either p=1.0 (shortcut) or
    p=None with an explanatory note. Guards against noisy CI failures
    when a KPI happens to be constant across seeds."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    report = m._compute_wilcoxon_report(data)
    pvs = report["kpis"]["annual_normalized_unserved_energy_total"]["pvalues"]
    for pair, info in pvs.items():
        assert info["p"] in (1.0, None), f"{pair}: {info}"
        # When p is None, an explanatory note must be present.
        if info["p"] is None:
            assert info.get("note"), f"{pair}: missing note for degenerate case"


def test_persist_wilcoxon_report_writes_json(tmp_path):
    """_persist_wilcoxon_report writes wilcoxon_report.json under output_dir
    (atomic rename), returning the destination path. Downstream tooling
    (doc §7 rewrite) depends on this filename being stable."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    report = m._compute_wilcoxon_report(data)
    dst = m._persist_wilcoxon_report(report, tmp_path)
    assert dst.name == "wilcoxon_report.json"
    assert dst.exists()
    loaded = json.loads(dst.read_text())
    assert loaded["eval_seeds"] == list(range(200, 210))
    assert "cost_total" in loaded["kpis"]


def test_render_benchmark_kpi_bars_emits_wilcoxon_report(tmp_path):
    """Integration: rendering fig 10 must have the side effect of
    persisting wilcoxon_report.json in the same output dir. This is the
    single-call reproducibility guarantee for section 7 p-values."""
    import scripts.curate_initiative_figures as m
    data = _synthetic_10seed_results()
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(data))
    out = tmp_path / "out"
    out.mkdir()
    dst = m._render_benchmark_kpi_bars(results_json=results_json, output_dir=out)
    assert dst is not None
    assert (out / "10_benchmark_kpi_bars.png").exists()
    assert (out / "wilcoxon_report.json").exists(), (
        "fig 10 render must also emit wilcoxon_report.json for section 7 reproducibility"
    )
