# tests/scripts/test_run_entity_pipeline.py
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY_SCHEMA = str(REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo" / "schema.json")


def _parse(argv):
    import scripts.run_entity_pipeline as m
    return m._build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_pipeline_default_schema_is_15min_parquet():
    """Phase 1: default schema switched from 15-sec to 15-min parquet."""
    args = _parse([])
    assert "15min_parquet" in str(args.schema)


def test_pipeline_default_gradient_steps_150k():
    """Phase 1: default gradient_steps raised from 50k to 150k."""
    args = _parse([])
    assert args.gradient_steps == 150_000


def test_pipeline_default_output_is_initiative_15min():
    """Phase 1: default output points at the initiative directory."""
    args = _parse([])
    assert "offline_iql_cql_initiative_15min" in str(args.output)


def test_pipeline_output_not_required():
    """Phase 1: --output no longer required (now defaulted)."""
    args = _parse([])  # should not raise
    assert args.output is not None


def test_pipeline_output_override_still_honoured():
    args = _parse(["--output", "/tmp/custom"])
    assert str(args.output) == "/tmp/custom"


def test_pipeline_all_steps_default():
    """Phase 5: feature-analysis is now part of the default ALL_STEPS."""
    args = _parse([])
    assert set(args.steps.split(",")) == {
        "collect", "train-iql", "train-cql", "benchmark", "feature-analysis"
    }


def test_pipeline_steps_subset():
    args = _parse(["--steps", "collect,benchmark"])
    assert args.steps == "collect,benchmark"


def test_pipeline_buildings_default_none():
    args = _parse([])
    assert args.buildings is None


def test_pipeline_buildings_explicit():
    args = _parse(["--buildings", "Building_5,Building_1"])
    assert args.buildings == "Building_5,Building_1"


def test_pipeline_algorithm_default_both():
    args = _parse([])
    assert args.algorithm == "both"


def test_pipeline_algorithm_iql_only():
    args = _parse(["--algorithm", "iql"])
    assert args.algorithm == "iql"


def test_pipeline_train_seeds_default():
    args = _parse([])
    assert args.train_seeds == "22,23,24,25,26,27,28,29,30"


def test_pipeline_eval_seeds_default():
    args = _parse([])
    seeds = [int(s) for s in args.eval_seeds.split(",")]
    assert len(seeds) == 10 and seeds[0] == 200


# ---------------------------------------------------------------------------
# New Phase 1 flags
# ---------------------------------------------------------------------------

def test_pipeline_cql_alpha_default_0_2():
    """Phase 1: --cql-alpha defaults to 0.2."""
    args = _parse([])
    assert args.cql_alpha == pytest.approx(0.2)


def test_pipeline_cql_alpha_override():
    args = _parse(["--cql-alpha", "1.0"])
    assert args.cql_alpha == pytest.approx(1.0)


def test_pipeline_hidden_layers_default_256_256():
    """Phase 1: --hidden-layers defaults to '256,256'."""
    args = _parse([])
    assert args.hidden_layers == "256,256"


def test_pipeline_hidden_layers_override():
    args = _parse(["--hidden-layers", "128,128,64"])
    assert args.hidden_layers == "128,128,64"


def test_pipeline_checkpoint_every_default_5000():
    """Phase 1: --checkpoint-every defaults to 5000 (forwarding lands in Phase 2)."""
    args = _parse([])
    assert args.checkpoint_every == 5000


def test_pipeline_checkpoint_every_override():
    args = _parse(["--checkpoint-every", "1000"])
    assert args.checkpoint_every == 1000


def test_pipeline_force_default_empty():
    """Phase 1: --force defaults to no forced stages (sentinel-skip lands in Phase 5)."""
    args = _parse([])
    assert args.force in (None, "")


def test_pipeline_force_explicit_stages():
    args = _parse(["--force", "collect,train-iql"])
    assert args.force == "collect,train-iql"


# ---------------------------------------------------------------------------
# _resolve_groups_arg: must always probe schema (production: --buildings unset)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path(HOURLY_SCHEMA).exists(),
    reason="hourly schema dataset not present",
)
def test_resolve_groups_arg_buildings_none_returns_all_groups():
    """Regression: when --buildings is unset, the orchestrator must still probe
    the schema and pass ALL groups to trainers. Falling back to the trainer's
    hardcoded AGENT_GROUPS defaults caused production runs to crash with
    'No rows for group (obs_dim=627, action_dim=1)' when the schema produced
    different obs_dims (e.g. 163/225/257/287).
    """
    import scripts.run_entity_pipeline as m

    out = m._resolve_groups_arg(HOURLY_SCHEMA, None)
    assert out is not None, "expected a comma-separated groups string, got None"
    pairs = [p for p in out.split(",") if p]
    assert len(pairs) >= 1, f"expected ≥1 group pair, got {pairs!r}"
    for p in pairs:
        a, b = p.split(":")
        assert int(a) > 0 and int(b) > 0, f"non-positive dims in {p!r}"


@pytest.mark.skipif(
    not Path(HOURLY_SCHEMA).exists(),
    reason="hourly schema dataset not present",
)
def test_resolve_groups_arg_buildings_set_filters():
    """When --buildings is set, _resolve_groups_arg must filter to matching groups
    (existing behaviour; covers Building_5 → obs225_act2 on the hourly schema).
    """
    import scripts.run_entity_pipeline as m

    all_out = m._resolve_groups_arg(HOURLY_SCHEMA, None)
    filtered = m._resolve_groups_arg(HOURLY_SCHEMA, "Building_5")
    assert filtered is not None
    # Building_5 belongs to exactly one group, so result has 1 pair.
    assert len([p for p in filtered.split(",") if p]) == 1
    # Filtered set must be a subset of full set.
    assert set(filtered.split(",")).issubset(set(all_out.split(",")))


# ---------------------------------------------------------------------------
# Forwarding: --hidden-layers and --cql-alpha must reach subscripts
# ---------------------------------------------------------------------------

def _run_main_capture(monkeypatch, argv):
    """Invoke main() with _run patched to record subprocess argv lists.

    Also patches _resolve_groups_arg → None to skip the slow env probe; tests
    that need to verify --groups forwarding patch it explicitly.
    """
    import scripts.run_entity_pipeline as m

    captured: list[list[str]] = []
    monkeypatch.setattr(m, "_run", lambda cmd: captured.append([str(c) for c in cmd]))
    monkeypatch.setattr(m, "_resolve_groups_arg", lambda schema, buildings: None)
    rc = m.main(argv)
    return rc, captured


def _cmds_for(captured, script_name):
    return [c for c in captured if any(script_name in part for part in c)]


def test_forward_hidden_layers_to_iql_and_cql(monkeypatch, tmp_path):
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "train-iql,train-cql",
            "--hidden-layers", "128,64",
        ],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    cql_cmds = _cmds_for(captured, "train_cql_entity")
    assert iql_cmds, "expected IQL train command"
    assert cql_cmds, "expected CQL train command"
    for cmd in iql_cmds + cql_cmds:
        assert "--hidden-layers" in cmd
        idx = cmd.index("--hidden-layers")
        assert cmd[idx + 1] == "128,64"


def test_forward_cql_alpha_to_cql_only(monkeypatch, tmp_path):
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "train-iql,train-cql",
            "--cql-alpha", "0.75",
        ],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    cql_cmds = _cmds_for(captured, "train_cql_entity")
    assert iql_cmds and cql_cmds
    # IQL must NOT receive --cql-alpha
    for cmd in iql_cmds:
        assert "--cql-alpha" not in cmd
    # CQL must receive --cql-alpha with the override value
    for cmd in cql_cmds:
        assert "--cql-alpha" in cmd
        idx = cmd.index("--cql-alpha")
        assert cmd[idx + 1] == "0.75"


def test_forward_checkpoint_every_to_iql_and_cql(monkeypatch, tmp_path):
    """Phase 2: --checkpoint-every must be forwarded to both train scripts."""
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "train-iql,train-cql",
            "--checkpoint-every", "2500",
        ],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    cql_cmds = _cmds_for(captured, "train_cql_entity")
    assert iql_cmds, "expected IQL train command"
    assert cql_cmds, "expected CQL train command"
    for cmd in iql_cmds + cql_cmds:
        assert "--checkpoint-every" in cmd, (
            f"--checkpoint-every missing from command: {cmd}"
        )
        idx = cmd.index("--checkpoint-every")
        assert cmd[idx + 1] == "2500"


def test_forward_checkpoint_every_default_value(monkeypatch, tmp_path):
    """Default --checkpoint-every (5000) is forwarded when flag is omitted."""
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "train-iql,train-cql",
        ],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    cql_cmds = _cmds_for(captured, "train_cql_entity")
    assert iql_cmds and cql_cmds
    for cmd in iql_cmds + cql_cmds:
        assert "--checkpoint-every" in cmd
        idx = cmd.index("--checkpoint-every")
        assert cmd[idx + 1] == "5000"


# ---------------------------------------------------------------------------
# Phase 5 — feature-analysis stage in default pipeline
# ---------------------------------------------------------------------------


def test_pipeline_feature_analysis_in_default_steps():
    """Phase 5: feature-analysis is the last stage in the default --steps."""
    args = _parse([])
    assert "feature-analysis" in args.steps.split(",")


def test_feature_analysis_invokes_analyzer(monkeypatch, tmp_path):
    """Phase 5: feature-analysis stage invokes scripts.analyze_entity_dataset."""
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "feature-analysis",
        ],
    )
    assert rc == 0
    fa_cmds = _cmds_for(captured, "analyze_entity_dataset")
    assert fa_cmds, f"expected analyzer invocation; captured={captured}"


def test_feature_analysis_passes_data_dir(monkeypatch, tmp_path):
    """Analyzer subprocess receives the orchestrator's data_dir as --data-dir."""
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "feature-analysis",
        ],
    )
    assert rc == 0
    fa_cmds = _cmds_for(captured, "analyze_entity_dataset")
    assert fa_cmds
    cmd = fa_cmds[0]
    assert "--data-dir" in cmd
    idx = cmd.index("--data-dir")
    assert cmd[idx + 1] == str(tmp_path / "data")


def test_feature_analysis_passes_output_dir(monkeypatch, tmp_path):
    """Analyzer subprocess receives --output-dir = orchestrator output base.

    Regression: scripts.analyze_entity_dataset.main() declares --output-dir
    as required=True; without it the analyzer aborts with
    ``error: the following arguments are required: --output-dir``.
    """
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "feature-analysis",
        ],
    )
    assert rc == 0
    fa_cmds = _cmds_for(captured, "analyze_entity_dataset")
    assert fa_cmds
    cmd = fa_cmds[0]
    assert "--output-dir" in cmd, f"expected --output-dir in {cmd}"
    idx = cmd.index("--output-dir")
    assert cmd[idx + 1] == str(tmp_path)


# ---------------------------------------------------------------------------
# Phase 5 — sentinel-skip per stage
# ---------------------------------------------------------------------------


def _touch_sentinel(output_dir, stage: str) -> None:
    """Create the orchestrator-level .{stage}.done sentinel."""
    import json as _json
    sentinel = output_dir / f".{stage}.done"
    sentinel.write_text(_json.dumps({"stage": stage, "completed_at": "test"}))


def test_pipeline_skips_collect_when_sentinel_exists(monkeypatch, tmp_path):
    """If runs/<output>/.collect.done exists, collect subprocess is NOT invoked."""
    _touch_sentinel(tmp_path, "collect")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert not collect_cmds, (
        f"collect must be skipped when sentinel exists; got {collect_cmds}"
    )


def test_pipeline_skips_train_iql_when_sentinel_exists(monkeypatch, tmp_path):
    _touch_sentinel(tmp_path, "train-iql")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "train-iql"],
    )
    assert rc == 0
    cmds = _cmds_for(captured, "train_iql_entity")
    assert not cmds, f"train-iql must be skipped when sentinel exists; got {cmds}"


def test_pipeline_skips_train_cql_when_sentinel_exists(monkeypatch, tmp_path):
    _touch_sentinel(tmp_path, "train-cql")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "train-cql"],
    )
    assert rc == 0
    cmds = _cmds_for(captured, "train_cql_entity")
    assert not cmds


def test_pipeline_skips_benchmark_when_sentinel_exists(monkeypatch, tmp_path):
    _touch_sentinel(tmp_path, "benchmark")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "benchmark"],
    )
    assert rc == 0
    cmds = _cmds_for(captured, "benchmark_entity_agents")
    assert not cmds


def test_pipeline_skips_feature_analysis_when_sentinel_exists(monkeypatch, tmp_path):
    _touch_sentinel(tmp_path, "feature-analysis")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "feature-analysis"],
    )
    assert rc == 0
    cmds = _cmds_for(captured, "analyze_entity_dataset")
    assert not cmds


def test_pipeline_runs_stage_when_sentinel_missing(monkeypatch, tmp_path):
    """No sentinel → subprocess IS invoked."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds, f"collect must run when sentinel missing; captured={captured}"


def test_pipeline_writes_sentinel_after_successful_stage(monkeypatch, tmp_path):
    """After a successful stage, the orchestrator writes .{stage}.done."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    assert (tmp_path / ".collect.done").exists(), (
        f"sentinel missing after successful stage; contents={list(tmp_path.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Phase 5 — --force handling
# ---------------------------------------------------------------------------


def test_force_collect_bypasses_sentinel(monkeypatch, tmp_path):
    """--force collect ignores existing .collect.done sentinel."""
    _touch_sentinel(tmp_path, "collect")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect", "--force", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds, "--force collect must bypass sentinel"


def test_force_collect_passes_no_skip_existing(monkeypatch, tmp_path):
    """--force collect passes --no-skip-existing to collect_rbcsmart_dataset."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect", "--force", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds
    assert "--no-skip-existing" in collect_cmds[0], (
        f"--force collect must pass --no-skip-existing; cmd={collect_cmds[0]}"
    )


def test_no_force_collect_omits_no_skip_existing(monkeypatch, tmp_path):
    """Without --force collect, --no-skip-existing is NOT passed."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds
    assert "--no-skip-existing" not in collect_cmds[0]


# ---------------------------------------------------------------------------
# --episode-steps: full-year episodes (35040 for 15-min) must reach collector
# ---------------------------------------------------------------------------


def test_pipeline_episode_steps_default_none():
    """--episode-steps defaults to None so the collector picks its own default
    (steps-per-day from the schema)."""
    args = _parse([])
    assert args.episode_steps is None


def test_pipeline_episode_steps_override():
    args = _parse(["--episode-steps", "35040"])
    assert args.episode_steps == 35040


def test_episode_steps_forwarded_to_collector(monkeypatch, tmp_path):
    """When --episode-steps is set, the orchestrator forwards it to the collector.

    Regression: without this, production defaults to 1-day episodes
    (~96 steps for 15-min, ~24 steps for hourly), producing a tiny dataset
    that would overfit on 150k gradient steps.
    """
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "collect",
            "--episode-steps", "35040",
        ],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds, f"expected collect invocation; captured={captured}"
    cmd = collect_cmds[0]
    assert "--episode-steps" in cmd, f"missing --episode-steps in {cmd}"
    idx = cmd.index("--episode-steps")
    assert cmd[idx + 1] == "35040"


def test_episode_steps_not_forwarded_when_unset(monkeypatch, tmp_path):
    """When --episode-steps is unset, the orchestrator must NOT pass any value,
    so the collector falls back to its own schema-derived default."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    collect_cmds = _cmds_for(captured, "collect_rbcsmart_dataset")
    assert collect_cmds
    assert "--episode-steps" not in collect_cmds[0]


def test_force_train_iql_passes_force_to_trainer(monkeypatch, tmp_path):
    """--force train-iql passes --force to train_iql_entity."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "train-iql", "--force", "train-iql"],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    assert iql_cmds
    assert "--force" in iql_cmds[0], (
        f"--force train-iql must pass --force to trainer; cmd={iql_cmds[0]}"
    )


def test_no_force_train_iql_omits_force(monkeypatch, tmp_path):
    """Without --force train-iql, --force is NOT passed to trainer."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "train-iql"],
    )
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    assert iql_cmds
    assert "--force" not in iql_cmds[0]


def test_force_train_cql_passes_force_to_trainer(monkeypatch, tmp_path):
    """--force train-cql passes --force to train_cql_entity."""
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "train-cql", "--force", "train-cql"],
    )
    assert rc == 0
    cql_cmds = _cmds_for(captured, "train_cql_entity")
    assert cql_cmds
    assert "--force" in cql_cmds[0]


def test_force_feature_analysis_passes_force_to_analyzer(monkeypatch, tmp_path):
    """--force feature-analysis passes --force to analyze_entity_dataset."""
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "feature-analysis",
            "--force", "feature-analysis",
        ],
    )
    assert rc == 0
    fa_cmds = _cmds_for(captured, "analyze_entity_dataset")
    assert fa_cmds
    assert "--force" in fa_cmds[0]


def test_force_benchmark_bypasses_sentinel(monkeypatch, tmp_path):
    """--force benchmark bypasses the benchmark sentinel (no extra sub-flag)."""
    _touch_sentinel(tmp_path, "benchmark")
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "benchmark", "--force", "benchmark"],
    )
    assert rc == 0
    bench_cmds = _cmds_for(captured, "benchmark_entity_agents")
    assert bench_cmds


def test_force_all_bypasses_every_sentinel(monkeypatch, tmp_path):
    """--force all bypasses sentinels and passes sub-flags everywhere applicable."""
    for stage in ("collect", "train-iql", "train-cql", "benchmark", "feature-analysis"):
        _touch_sentinel(tmp_path, stage)
    rc, captured = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--force", "all"],
    )
    assert rc == 0
    assert _cmds_for(captured, "collect_rbcsmart_dataset"), "collect must run"
    assert _cmds_for(captured, "train_iql_entity"), "iql must run"
    assert _cmds_for(captured, "train_cql_entity"), "cql must run"
    assert _cmds_for(captured, "benchmark_entity_agents"), "benchmark must run"
    assert _cmds_for(captured, "analyze_entity_dataset"), "feature-analysis must run"

    # Sub-flag forwarding
    assert "--no-skip-existing" in _cmds_for(captured, "collect_rbcsmart_dataset")[0]
    assert "--force" in _cmds_for(captured, "train_iql_entity")[0]
    assert "--force" in _cmds_for(captured, "train_cql_entity")[0]
    assert "--force" in _cmds_for(captured, "analyze_entity_dataset")[0]


def test_force_multiple_stages_csv(monkeypatch, tmp_path):
    """--force collect,train-iql forces both, leaves others alone."""
    for stage in ("collect", "train-iql", "train-cql"):
        _touch_sentinel(tmp_path, stage)
    rc, captured = _run_main_capture(
        monkeypatch,
        [
            "--output", str(tmp_path),
            "--steps", "collect,train-iql,train-cql",
            "--force", "collect,train-iql",
        ],
    )
    assert rc == 0
    assert _cmds_for(captured, "collect_rbcsmart_dataset")
    assert _cmds_for(captured, "train_iql_entity")
    # train-cql sentinel was present and NOT forced → must be skipped
    assert not _cmds_for(captured, "train_cql_entity")


# ---------------------------------------------------------------------------
# Phase 5 — status.json tracking
# ---------------------------------------------------------------------------


def test_status_json_created_after_run(monkeypatch, tmp_path):
    """status.json is created under output after pipeline runs."""
    rc, _ = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    assert (tmp_path / "status.json").exists()


def test_status_json_records_done_stage(monkeypatch, tmp_path):
    """A completed stage shows status='done' with duration and timestamp."""
    import json as _json
    rc, _ = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    status = _json.loads((tmp_path / "status.json").read_text())
    assert "stages" in status
    entry = status["stages"].get("collect")
    assert entry is not None, f"collect missing from status; got {status}"
    assert entry["status"] == "done"
    assert "duration_seconds" in entry
    assert entry["duration_seconds"] >= 0
    assert "completed_at" in entry


def test_status_json_records_skipped_stage(monkeypatch, tmp_path):
    """A stage skipped due to existing sentinel shows status='skipped'."""
    import json as _json
    _touch_sentinel(tmp_path, "collect")
    rc, _ = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect"],
    )
    assert rc == 0
    status = _json.loads((tmp_path / "status.json").read_text())
    entry = status["stages"].get("collect")
    assert entry is not None
    assert entry["status"] == "skipped"


def test_status_json_records_multiple_stages(monkeypatch, tmp_path):
    """status.json accumulates entries across multiple stages in one run."""
    import json as _json
    rc, _ = _run_main_capture(
        monkeypatch,
        ["--output", str(tmp_path), "--steps", "collect,train-iql"],
    )
    assert rc == 0
    status = _json.loads((tmp_path / "status.json").read_text())
    assert "collect" in status["stages"]
    assert "train-iql" in status["stages"]
    assert status["stages"]["collect"]["status"] == "done"
    assert status["stages"]["train-iql"]["status"] == "done"
