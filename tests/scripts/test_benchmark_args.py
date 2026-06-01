# tests/scripts/test_benchmark_args.py
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY_SCHEMA = str(REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo" / "schema.json")


def _parse(argv):
    import scripts.benchmark_entity_agents as m
    return m._build_parser().parse_args(argv)


def test_benchmark_default_schema_is_15s_parquet():
    args = _parse([])
    assert "15s_parquet" in str(args.schema)

def test_benchmark_schema_override():
    args = _parse(["--schema", HOURLY_SCHEMA])
    assert args.schema == HOURLY_SCHEMA

def test_benchmark_episode_steps_default_none():
    args = _parse([])
    assert args.episode_steps is None

def test_benchmark_episode_steps_explicit():
    args = _parse(["--episode-steps", "24"])
    assert args.episode_steps == 24

def test_benchmark_offline_default_true():
    args = _parse([])
    assert args.offline is True

def test_benchmark_no_offline_flag():
    args = _parse(["--no-offline"])
    assert args.offline is False

def test_benchmark_default_eval_seeds():
    args = _parse([])
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    assert len(seeds) == 10
    assert seeds[0] == 200

def test_benchmark_main_accepts_argv():
    import scripts.benchmark_entity_agents as m
    with pytest.raises(SystemExit) as exc:
        m.main(["--help"])
    assert exc.value.code == 0
