# tests/scripts/test_analyze_entity_dataset.py
"""Phase 4 tests for scripts/analyze_entity_dataset.py.

The analyzer reads entity-interface RBCSmart parquet datasets and produces a
thesis-grade ``summary.md`` + per-section figures + ``.feature_analysis.done``
sentinel.

These tests use a small synthetic 2-seed × 1-group × 50-step dataset and verify:

* Output directory structure (figures/, summary.md, sentinel).
* Per-section figure files are produced for every analysis section.
* The ``--force`` flag overrides the sentinel.
* The sentinel short-circuits a re-run when present.

The analyzer does NOT depend on a real CityLearn env — it only reads parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_seed_parquet(
    path: Path,
    *,
    seed: int,
    n_steps: int = 50,
    obs_dim: int = 4,
    action_dim: int = 1,
    n_agents: int = 2,
    rng_seed: int = 0,
) -> None:
    """Write a tiny RBCSmart-style parquet for one seed.

    ``n_agents`` agents share the same (obs_dim, action_dim) group so the
    analyzer's per-group plots have multi-building data to work with.
    The ``action__electrical_storage`` column is included so the regime
    classifier (charge/idle/discharge) has data to bin.
    """
    rng = np.random.default_rng(rng_seed + seed)
    obs_names = [f"feat_{i}" for i in range(obs_dim)]
    act_names = ["electrical_storage"] + [f"act_{i}" for i in range(action_dim - 1)]
    rows = []
    for step in range(n_steps):
        for agent_idx in range(n_agents):
            row = {
                "seed": int(seed),
                "episode": 0,
                "timestep": int(step),
                "agent_idx": int(agent_idx),
                "obs_dim": int(obs_dim),
                "action_dim": int(action_dim),
                "reward": float(rng.normal(-0.02, 0.05)),
                "terminated": 0,
                "truncated": int(step == n_steps - 1),
            }
            for name, val in zip(obs_names, rng.uniform(-1.0, 1.0, size=obs_dim)):
                row[f"obs__{name}"] = float(val)
            for name, val in zip(act_names, rng.uniform(-1.0, 1.0, size=action_dim)):
                row[f"action__{name}"] = float(val)
            for name, val in zip(obs_names, rng.uniform(-1.0, 1.0, size=obs_dim)):
                row[f"next_obs__{name}"] = float(val)
            rows.append(row)
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


@pytest.fixture
def synth_data_dir(tmp_path):
    """Synthetic 2-seed × 2-agent × 1-group dataset."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for s in (22, 23):
        _make_seed_parquet(data_dir / f"seed_{s}.parquet", seed=s)
    # Also write a tiny manifest so the analyzer can extract metadata.
    manifest = {
        "behaviour_policy": "RBCSmartPolicy",
        "dataset_path": "/synthetic/schema.json",
        "interface": "entity",
        "topology_mode": "static",
        "entity_encoding": "minmax_space",
        "seeds": [22, 23],
        "episodes_per_seed": 1,
        "episode_time_steps": 50,
        "n_agents": 2,
        "agent_groups": {"obs4_act1": 100},
        "schema_hash": "fakeha5h",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return data_dir


def _collect_module():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import scripts.analyze_entity_dataset as m

    return m


def _run_main(data_dir, out_dir, *extra):
    m = _collect_module()
    return m.main(["--data-dir", str(data_dir), "--output-dir", str(out_dir), *extra])


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_analyzer_creates_feature_analysis_subdir(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    rc = _run_main(synth_data_dir, out)
    assert rc == 0
    fa_dir = out / "feature_analysis"
    assert fa_dir.exists() and fa_dir.is_dir()
    assert (fa_dir / "figures").is_dir()


def test_analyzer_writes_sentinel_on_success(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    rc = _run_main(synth_data_dir, out)
    assert rc == 0
    sentinel = out / "feature_analysis" / ".feature_analysis.done"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert "completed_at" in payload


def test_analyzer_writes_summary_md_with_all_sections(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    rc = _run_main(synth_data_dir, out)
    assert rc == 0
    summary = (out / "feature_analysis" / "summary.md").read_text()
    # Each numbered section header must appear (we accept either H1 or H2).
    expected_section_keywords = [
        "Dataset stats",
        "Observation distributions",
        "Action coverage",
        "Reward",  # "Reward distribution by regime"
        "Correlations",
        "Per-building",
        "Temporal",
    ]
    for kw in expected_section_keywords:
        assert kw.lower() in summary.lower(), (
            f"summary.md missing section keyword {kw!r}\n--- summary.md ---\n{summary}"
        )


# ---------------------------------------------------------------------------
# Per-section figure files
# ---------------------------------------------------------------------------


def test_analyzer_writes_dataset_stats_figure(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    assert (out / "feature_analysis" / "figures" / "01_dataset_stats_table.png").exists()


def test_analyzer_writes_obs_distributions_per_group(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    fig_dir = out / "feature_analysis" / "figures"
    matches = sorted(fig_dir.glob("02_obs_distributions_*.png"))
    assert matches, "expected at least one 02_obs_distributions_*.png"


def test_analyzer_writes_action_coverage_per_group(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    fig_dir = out / "feature_analysis" / "figures"
    matches = sorted(fig_dir.glob("03_action_coverage_*.png"))
    assert matches, "expected at least one 03_action_coverage_*.png"


def test_analyzer_writes_reward_by_regime_figure(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    assert (out / "feature_analysis" / "figures" / "04_reward_by_regime.png").exists()


def test_analyzer_writes_correlations_per_group(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    fig_dir = out / "feature_analysis" / "figures"
    matches = sorted(fig_dir.glob("05_correlations_*.png"))
    assert matches, "expected at least one 05_correlations_*.png"


def test_analyzer_writes_per_building_table_figure(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    assert (out / "feature_analysis" / "figures" / "06_per_building_table.png").exists()


def test_analyzer_writes_temporal_patterns_figure(synth_data_dir, tmp_path):
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    assert (out / "feature_analysis" / "figures" / "07_temporal_patterns.png").exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_analyzer_sentinel_short_circuits_rerun(synth_data_dir, tmp_path):
    """When .feature_analysis.done exists, a re-run short-circuits and does not
    regenerate figures (mtime unchanged)."""
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    sentinel = out / "feature_analysis" / ".feature_analysis.done"
    summary = out / "feature_analysis" / "summary.md"
    assert sentinel.exists()
    mtime_before = summary.stat().st_mtime_ns

    import time as _t

    _t.sleep(0.05)
    rc = _run_main(synth_data_dir, out)
    assert rc == 0
    mtime_after = summary.stat().st_mtime_ns
    assert mtime_after == mtime_before, "summary.md was rewritten despite sentinel"


def test_analyzer_force_overrides_sentinel(synth_data_dir, tmp_path):
    """``--force`` rewrites everything even when sentinel is present."""
    out = tmp_path / "run"
    _run_main(synth_data_dir, out)
    summary = out / "feature_analysis" / "summary.md"
    mtime_before = summary.stat().st_mtime_ns

    import time as _t

    _t.sleep(0.05)
    rc = _run_main(synth_data_dir, out, "--force")
    assert rc == 0
    mtime_after = summary.stat().st_mtime_ns
    assert mtime_after > mtime_before, "--force did not rewrite summary.md"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_analyzer_handles_missing_manifest(tmp_path):
    """If manifest.json is absent, the analyzer falls back to deriving metadata
    from the parquet files (rather than crashing)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for s in (22, 23):
        _make_seed_parquet(data_dir / f"seed_{s}.parquet", seed=s)
    # NB: no manifest.json
    out = tmp_path / "run"
    rc = _run_main(data_dir, out)
    assert rc == 0
    assert (out / "feature_analysis" / "summary.md").exists()
    assert (out / "feature_analysis" / ".feature_analysis.done").exists()


# ---------------------------------------------------------------------------
# Bug 8 — multi-group wide-sparse + memory-efficient loading
# ---------------------------------------------------------------------------
#
# Real production parquets are wide-sparse: every row has columns from ALL
# agent groups' schemas, but only the cols belonging to that row's group are
# non-NaN.  Loading everything via pd.concat([read_table(p).to_pandas() ...])
# materialises ~205 GB of float64 for the 17-building × 35040-step × 10-seed
# 15-min dataset, which crashes the kernel on 48 GB systems.
#
# The fix mirrors the column-selective per-group loader already proven in
# ``algorithms/offline_rl/entity_dataset.py`` (`_discover_group_cols_from_schema`
# + `pq.read_table(columns=needed_cols)`).  These tests pin the new contract:
#
# * ``_load_group(data_dir, obs_dim, action_dim)`` returns only the columns
#   belonging to that group (not the whole wide-sparse schema).
# * ``_load_group(..., max_rows_per_seed=N)`` caps loaded rows per seed for
#   subsampling on memory-bound environments.
# * The end-to-end analyzer still produces per-group figures for every group
#   in a multi-group wide-sparse dataset.


def _make_multi_group_seed_parquet(
    path: Path,
    *,
    seed: int,
    n_steps: int = 20,
    rng_seed: int = 0,
) -> None:
    """Write a wide-sparse multi-group seed parquet.

    Group A: obs_dim=4, action_dim=1, 2 agents (agent_idx 0, 1).
    Group B: obs_dim=6, action_dim=2, 1 agent (agent_idx 2).

    All rows share the same schema: 10 obs__ cols + 2 action__ cols +
    10 next_obs__ cols + 9 metadata cols.  Each row populates only the
    columns belonging to its group; the rest are NaN.  This mirrors the
    production layout where each parquet has 2150 obs cols total but each
    row touches only ~600-800 of them.
    """
    rng = np.random.default_rng(rng_seed + seed)
    obs_a = [f"obs__a_feat_{i}" for i in range(4)]
    obs_b = [f"obs__b_feat_{i}" for i in range(6)]
    # Group A shares ``action__electrical_storage`` with group B; group B
    # additionally controls ``action__ev_charge``.  This mirrors how real
    # CityLearn buildings share the storage action and only some have EV.
    act_a = ["action__electrical_storage"]
    act_b = ["action__electrical_storage", "action__ev_charge"]
    all_obs = obs_a + obs_b
    all_act = sorted(set(act_a) | set(act_b))
    all_next_obs = [c.replace("obs__", "next_obs__") for c in all_obs]

    def _nan_row(obs_dim: int, action_dim: int, agent_idx: int, step: int) -> dict:
        row: dict = {
            "seed": int(seed),
            "episode": 0,
            "timestep": int(step),
            "agent_idx": int(agent_idx),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "reward": float(rng.normal(-0.02, 0.05)),
            "terminated": 0,
            "truncated": int(step == n_steps - 1),
        }
        for c in all_obs + all_act + all_next_obs:
            row[c] = float("nan")
        return row

    rows: list = []
    for step in range(n_steps):
        # Group A rows
        for agent_idx in (0, 1):
            row = _nan_row(obs_dim=4, action_dim=1, agent_idx=agent_idx, step=step)
            for c in obs_a:
                row[c] = float(rng.uniform(-1.0, 1.0))
            for c in act_a:
                row[c] = float(rng.uniform(-1.0, 1.0))
            for c in obs_a:
                row[c.replace("obs__", "next_obs__")] = float(rng.uniform(-1.0, 1.0))
            rows.append(row)
        # Group B rows
        for agent_idx in (2,):
            row = _nan_row(obs_dim=6, action_dim=2, agent_idx=agent_idx, step=step)
            for c in obs_b:
                row[c] = float(rng.uniform(-1.0, 1.0))
            for c in act_b:
                row[c] = float(rng.uniform(-1.0, 1.0))
            for c in obs_b:
                row[c.replace("obs__", "next_obs__")] = float(rng.uniform(-1.0, 1.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


@pytest.fixture
def multi_group_data_dir(tmp_path):
    """Synthetic 2-seed × multi-group wide-sparse dataset.

    Per seed: 20 steps × (2 group-A agents + 1 group-B agent) = 60 rows.
    Across 2 seeds: 120 rows total; group A = 80 rows, group B = 40 rows.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for s in (22, 23):
        _make_multi_group_seed_parquet(data_dir / f"seed_{s}.parquet", seed=s)
    manifest = {
        "behaviour_policy": "RBCSmartPolicy",
        "dataset_path": "/synthetic/multi_group_schema.json",
        "interface": "entity",
        "topology_mode": "static",
        "entity_encoding": "minmax_space",
        "seeds": [22, 23],
        "episodes_per_seed": 1,
        "episode_time_steps": 20,
        "n_agents": 3,
        "agent_groups": {"obs4_act1": 2, "obs6_act2": 1},
        "schema_hash": "multifakeh4sh",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return data_dir


def test_load_group_returns_only_obs4_group_cols(multi_group_data_dir):
    """``_load_group(obs_dim=4, action_dim=1)`` returns only that group's
    columns (4 obs__ + 1 action__ + 4 next_obs__), not the full wide-sparse
    schema (10 obs__ + 2 action__ + 10 next_obs__)."""
    m = _collect_module()
    df = m._load_group(multi_group_data_dir, obs_dim=4, action_dim=1)
    obs_cols = [c for c in df.columns if c.startswith("obs__")]
    act_cols = [c for c in df.columns if c.startswith("action__")]
    next_obs_cols = [c for c in df.columns if c.startswith("next_obs__")]
    assert len(obs_cols) == 4, (
        f"expected 4 obs cols (not all 10 wide-sparse cols), got {len(obs_cols)}: {obs_cols}"
    )
    assert len(act_cols) == 1, (
        f"expected 1 action col, got {len(act_cols)}: {act_cols}"
    )
    assert len(next_obs_cols) == 4, (
        f"expected 4 next_obs cols, got {len(next_obs_cols)}: {next_obs_cols}"
    )
    # All rows belong to the obs_dim=4 group.
    assert (df["obs_dim"] == 4).all()
    assert (df["action_dim"] == 1).all()
    # 2 seeds × 20 steps × 2 group-A agents = 80 rows.
    assert len(df) == 80, f"expected 80 group-A rows, got {len(df)}"


def test_load_group_returns_only_obs6_group_cols(multi_group_data_dir):
    """``_load_group(obs_dim=6, action_dim=2)`` returns only group-B columns."""
    m = _collect_module()
    df = m._load_group(multi_group_data_dir, obs_dim=6, action_dim=2)
    obs_cols = [c for c in df.columns if c.startswith("obs__")]
    act_cols = [c for c in df.columns if c.startswith("action__")]
    next_obs_cols = [c for c in df.columns if c.startswith("next_obs__")]
    assert len(obs_cols) == 6, f"expected 6 obs cols, got {len(obs_cols)}: {obs_cols}"
    assert len(act_cols) == 2, f"expected 2 action cols, got {len(act_cols)}: {act_cols}"
    assert len(next_obs_cols) == 6
    assert (df["obs_dim"] == 6).all()
    assert (df["action_dim"] == 2).all()
    # 2 seeds × 20 steps × 1 group-B agent = 40 rows.
    assert len(df) == 40, f"expected 40 group-B rows, got {len(df)}"


def test_load_group_max_rows_per_seed_caps_loaded_rows(multi_group_data_dir):
    """``max_rows_per_seed=N`` caps loaded rows per (group, seed) for memory-
    bound environments.  Real production: 200000 rows/seed × 10 seeds ≈ 2M rows,
    far under the 5.96M-row full-dataset OOM trigger."""
    m = _collect_module()
    # Each seed has 40 group-A rows; cap at 15 → ≤ 30 total across 2 seeds.
    df = m._load_group(
        multi_group_data_dir, obs_dim=4, action_dim=1, max_rows_per_seed=15
    )
    assert len(df) <= 30, (
        f"expected ≤ 30 rows (15/seed × 2 seeds), got {len(df)}"
    )
    # Both seeds should still be represented.
    assert set(int(s) for s in df["seed"].unique()) == {22, 23}, (
        f"expected both seeds 22, 23 in capped load, got {df['seed'].unique()}"
    )


def test_analyzer_handles_multi_group_wide_sparse_e2e(multi_group_data_dir, tmp_path):
    """End-to-end: analyzer on multi-group wide-sparse fixture produces
    per-group figures for BOTH groups (regression guard for the refactor)."""
    out = tmp_path / "run"
    rc = _run_main(multi_group_data_dir, out)
    assert rc == 0
    fig_dir = out / "feature_analysis" / "figures"
    # Both groups must have section 2/3/5 figures.
    assert (fig_dir / "02_obs_distributions_obs4_act1.png").exists()
    assert (fig_dir / "02_obs_distributions_obs6_act2.png").exists()
    assert (fig_dir / "03_action_coverage_obs4_act1.png").exists()
    assert (fig_dir / "03_action_coverage_obs6_act2.png").exists()
    assert (fig_dir / "05_correlations_obs4_act1.png").exists()
    assert (fig_dir / "05_correlations_obs6_act2.png").exists()
    # Dataset-wide sections still produced.
    assert (fig_dir / "01_dataset_stats_table.png").exists()
    assert (fig_dir / "04_reward_by_regime.png").exists()
    assert (fig_dir / "06_per_building_table.png").exists()
    assert (fig_dir / "07_temporal_patterns.png").exists()
    # Sentinel + summary written.
    assert (out / "feature_analysis" / ".feature_analysis.done").exists()
    assert (out / "feature_analysis" / "summary.md").exists()
