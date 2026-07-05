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


# ---------------------------------------------------------------------------
# Phase 13 / Bug 9: --skip-rbc and --merge-existing flags
# ---------------------------------------------------------------------------


def test_benchmark_skip_rbc_default_false():
    """--skip-rbc defaults to False so existing pipelines continue to
    include a fresh RBC baseline in every benchmark run."""
    args = _parse([])
    assert args.skip_rbc is False


def test_benchmark_skip_rbc_flag_parses():
    """--skip-rbc sets a truthy attribute so main() can gate the
    rbc_rollout loop. Used by Phase 13 to reuse existing full-year RBC
    runs when reprocessing IQL/CQL after Bug 9 fix."""
    args = _parse(["--skip-rbc"])
    assert args.skip_rbc is True


def test_benchmark_merge_existing_default_none():
    """--merge-existing defaults to None so results.json only reflects
    freshly-computed runs unless the caller opts in to splicing."""
    args = _parse([])
    assert args.merge_existing is None


def test_benchmark_merge_existing_flag_parses(tmp_path):
    """--merge-existing PATH resolves to a Path attribute so main() can
    load a prior RBCSmart block and inject it into the output JSON."""
    prev = tmp_path / "prev.json"
    prev.write_text("{}")
    args = _parse(["--merge-existing", str(prev)])
    assert args.merge_existing is not None
    assert Path(args.merge_existing).name == "prev.json"
