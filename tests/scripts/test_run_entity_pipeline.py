# tests/scripts/test_run_entity_pipeline.py
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY_SCHEMA = str(REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo" / "schema.json")


def _parse(argv):
    import scripts.run_entity_pipeline as m
    return m._build_parser().parse_args(argv)


def test_pipeline_default_schema_is_15s_parquet():
    args = _parse(["--output", "/tmp/x"])
    assert "15s_parquet" in str(args.schema)

def test_pipeline_all_steps_default():
    args = _parse(["--output", "/tmp/x"])
    assert set(args.steps.split(",")) == {"collect", "train-iql", "train-cql", "benchmark"}

def test_pipeline_steps_subset():
    args = _parse(["--output", "/tmp/x", "--steps", "collect,benchmark"])
    assert args.steps == "collect,benchmark"

def test_pipeline_buildings_default_none():
    args = _parse(["--output", "/tmp/x"])
    assert args.buildings is None

def test_pipeline_buildings_explicit():
    args = _parse(["--output", "/tmp/x", "--buildings", "Building_5,Building_1"])
    assert args.buildings == "Building_5,Building_1"

def test_pipeline_algorithm_default_both():
    args = _parse(["--output", "/tmp/x"])
    assert args.algorithm == "both"

def test_pipeline_algorithm_iql_only():
    args = _parse(["--output", "/tmp/x", "--algorithm", "iql"])
    assert args.algorithm == "iql"

def test_pipeline_train_seeds_default():
    args = _parse(["--output", "/tmp/x"])
    assert args.train_seeds == "22,23,24,25,26,27,28,29,30"

def test_pipeline_eval_seeds_default():
    args = _parse(["--output", "/tmp/x"])
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    assert len(seeds) == 10 and seeds[0] == 200

def test_pipeline_requires_output():
    with pytest.raises(SystemExit):
        _parse([])
