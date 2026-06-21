# tests/scripts/test_collect_rbcsmart_orchestration.py
"""Phase 3 tests for collect_rbcsmart_dataset.main() orchestration.

Verifies:
* ``write_manifest`` records the actual ``--schema`` value, not the hardcoded
  module-level ``SCHEMA_PATH`` constant.
* ``write_manifest`` records the actual ``--episode-steps`` value (or the
  resolved per-schema default), not the hardcoded ``EPISODE_STEPS`` constant.
* ``--skip-existing`` flag (default True) controls per-stage idempotency.
* ``.collect.done`` sentinel is written after a successful collection.
* When ``.collect.done`` exists and ``--skip-existing`` is True (default),
  ``main()`` short-circuits without touching the environment.
* ``--no-skip-existing`` bypasses both the per-seed and per-stage skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_module():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.collect_rbcsmart_dataset as m

    return m


def _create_minimal_seed_parquet(path: Path, *, seed: int = 22, n_steps: int = 4) -> None:
    """Write a tiny parquet matching collect_rbcsmart_dataset's expected schema.

    The parquet has obs_dim/action_dim columns (for ``_agent_group_summary_from_parquet``)
    and ``action__electrical_storage`` with non-zero std (for
    ``_action_variance_check_from_parquet``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(n_steps):
        rows.append(
            {
                "seed": int(seed),
                "episode": 0,
                "timestep": int(step),
                "agent_idx": 0,
                "obs_dim": 4,
                "action_dim": 1,
                "reward": -0.01,
                "terminated": 0,
                "truncated": int(step == n_steps - 1),
                "obs__feat_0": 0.5,
                # Non-zero std so assert_action_variance accepts it.
                "action__electrical_storage": float(step) * 0.25 + 0.1,
                "next_obs__feat_0": 0.4,
            }
        )
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


# ---------------------------------------------------------------------------
# Unit tests for write_manifest signature + values
# ---------------------------------------------------------------------------


def test_write_manifest_uses_provided_schema_path(tmp_path):
    """Phase 3: write_manifest must use the schema_path argument, not the hardcoded constant."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    fake_schema = "/fake/path/to/some_schema.json"

    manifest_path = m.write_manifest(
        out_dir,
        schema_path=fake_schema,
        episode_steps=1234,
        seeds=[22],
        episodes_per_seed=1,
        seed_files={22: out_dir / "seed_22.parquet"},
        n_rows_total=0,
        agent_groups={"obs4_act1": 0},
        schema_hash="cafebabe",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_path"] == fake_schema, (
        f"manifest.dataset_path must be the provided schema_path, got "
        f"{manifest['dataset_path']!r}"
    )


def test_write_manifest_uses_provided_episode_steps(tmp_path):
    """Phase 3: write_manifest must use the episode_steps argument, not EPISODE_STEPS."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    manifest_path = m.write_manifest(
        out_dir,
        schema_path="/x.json",
        episode_steps=35040,  # 15-min full year
        seeds=[22],
        episodes_per_seed=1,
        seed_files={22: out_dir / "seed_22.parquet"},
        n_rows_total=0,
        agent_groups={"obs4_act1": 0},
        schema_hash="cafebabe",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["episode_time_steps"] == 35040, (
        f"manifest.episode_time_steps must be the provided episode_steps, got "
        f"{manifest['episode_time_steps']!r}"
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_skip_existing_default_true():
    """Phase 3: --skip-existing defaults to True (preserves implicit skip behaviour)."""
    m = _collect_module()
    args = m._build_parser().parse_args([])
    assert args.skip_existing is True


def test_no_skip_existing_disables_skip():
    """Phase 3: --no-skip-existing turns the flag off."""
    m = _collect_module()
    args = m._build_parser().parse_args(["--no-skip-existing"])
    assert args.skip_existing is False


# ---------------------------------------------------------------------------
# Sentinel behaviour: writes .collect.done on success
# ---------------------------------------------------------------------------


def test_collect_done_sentinel_written_on_success(tmp_path):
    """Phase 3: after main() completes successfully, .collect.done exists in out_dir."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Pre-create seed_22.parquet so the per-seed loop short-circuits the
    # rollout (no env, no RBC, no real episode). Manifest + sentinel writes
    # still execute as the happy path.
    _create_minimal_seed_parquet(out_dir / "seed_22.parquet", seed=22)

    rc = m.main(["--output-dir", str(out_dir), "--seeds", "22"])
    assert rc == 0
    assert (out_dir / ".collect.done").exists(), "expected .collect.done sentinel"
    payload = json.loads((out_dir / ".collect.done").read_text())
    assert "completed_at" in payload


# ---------------------------------------------------------------------------
# Sentinel behaviour: short-circuits re-run
# ---------------------------------------------------------------------------


def test_collect_done_short_circuits_when_skip_existing_true(monkeypatch, tmp_path):
    """Phase 3: when .collect.done exists and --skip-existing is True,
    main() returns 0 without invoking the env."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".collect.done").write_text(json.dumps({"completed_at": "stub"}))

    # Sabotage: any attempt to start a rollout should fail loudly.
    def _no_env(*a, **kw):
        raise RuntimeError("env should not be created when .collect.done short-circuits")

    monkeypatch.setattr(m, "_make_env", _no_env)
    monkeypatch.setattr(m, "collect_episode", lambda **kw: _no_env())

    rc = m.main(["--output-dir", str(out_dir), "--seeds", "22"])
    assert rc == 0


def test_no_skip_existing_bypasses_collect_done_sentinel(monkeypatch, tmp_path):
    """Phase 3: --no-skip-existing forces collection even when .collect.done exists."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / ".collect.done").write_text(json.dumps({"completed_at": "stub"}))

    # Sabotage: collect_episode raises so we know the loop was entered.
    sentinel_error = RuntimeError("collect_episode SHOULD have been called")

    def _raise(**kw):
        raise sentinel_error

    monkeypatch.setattr(m, "collect_episode", _raise)

    with pytest.raises(RuntimeError, match="SHOULD have been called"):
        m.main(["--output-dir", str(out_dir), "--seeds", "22", "--no-skip-existing"])


# ---------------------------------------------------------------------------
# Per-seed skip behaviour gated by --skip-existing
# ---------------------------------------------------------------------------


def test_no_skip_existing_re_collects_when_seed_exists(monkeypatch, tmp_path):
    """Phase 3: with --no-skip-existing, even existing seed_*.parquet files trigger re-collection."""
    m = _collect_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-create seed; without --skip-existing this would normally be skipped.
    _create_minimal_seed_parquet(out_dir / "seed_22.parquet", seed=22)

    sentinel_error = RuntimeError("collect_episode SHOULD have been called for existing seed")

    def _raise(**kw):
        raise sentinel_error

    monkeypatch.setattr(m, "collect_episode", _raise)

    with pytest.raises(RuntimeError, match="SHOULD have been called"):
        m.main(["--output-dir", str(out_dir), "--seeds", "22", "--no-skip-existing"])
