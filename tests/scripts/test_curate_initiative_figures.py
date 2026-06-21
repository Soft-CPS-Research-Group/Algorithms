"""Tests for scripts.curate_initiative_figures."""
from __future__ import annotations

from pathlib import Path
import sys

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
