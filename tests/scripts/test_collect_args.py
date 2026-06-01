from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY_SCHEMA = str(REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo" / "schema.json")


def _parse(argv):
    import scripts.collect_rbcsmart_dataset as m
    return m._build_parser().parse_args(argv)


def test_collect_default_schema_is_15s_parquet():
    args = _parse([])
    assert "15s_parquet" in str(args.schema)

def test_collect_schema_override():
    args = _parse(["--schema", HOURLY_SCHEMA])
    assert args.schema == HOURLY_SCHEMA

def test_collect_episode_steps_default_none():
    args = _parse([])
    assert args.episode_steps is None  # auto-detect

def test_collect_episode_steps_explicit():
    args = _parse(["--episode-steps", "24"])
    assert args.episode_steps == 24

def test_collect_offline_default_true():
    args = _parse([])
    assert args.offline is True

def test_collect_no_offline_flag():
    args = _parse(["--no-offline"])
    assert args.offline is False
