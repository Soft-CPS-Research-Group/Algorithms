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
