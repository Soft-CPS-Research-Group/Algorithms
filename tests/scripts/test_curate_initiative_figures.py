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
