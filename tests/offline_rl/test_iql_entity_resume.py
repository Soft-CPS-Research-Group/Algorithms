# tests/offline_rl/test_iql_entity_resume.py
"""Tests for trainer resume + checkpointing in iql_entity_trainer.

Phase 2 of the IQL+CQL 15-min initiative adds the following to
``train_entity_single_seed``:

* Mid-training checkpoint every ``checkpoint_every_n_steps`` (``checkpoint_latest.pt``).
* ``best_policy.pt`` written on each val MSE improvement.
* ``seed.done`` sentinel written on successful completion.
* If ``seed.done`` exists → early return (unless ``force=True``).
* If ``checkpoint_latest.pt`` exists → resume from that step (unless ``force``).
* ``status.json`` updated under ``output_root`` after each checkpoint.

The deterministic-resume test verifies that 200 steps in one shot vs.
100 → kill → resume → 100 more produce identical final network state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from algorithms.offline_rl.entity_schema import (
    ACTION_PREFIX,
    META_COLUMNS,
    NEXT_OBS_PREFIX,
    OBS_PREFIX,
)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers (copied from test_entity_schema_and_dataset)
# ---------------------------------------------------------------------------


def _make_synthetic_df(
    *,
    n_steps: int = 200,
    seed: int = 1,
    episode: int = 0,
    groups: List[Tuple[int, int]] = None,
) -> pd.DataFrame:
    if groups is None:
        groups = [(4, 1)]
    rng = np.random.default_rng(seed * 1000 + episode)
    rows = []
    for step in range(n_steps):
        for agent_idx, (obs_d, act_d) in enumerate(groups):
            obs_names = [f"obs_feat_{i}" for i in range(obs_d)]
            act_names = [f"act_{i}" for i in range(act_d)]
            row = {
                "seed": seed,
                "episode": episode,
                "timestep": step,
                "agent_idx": agent_idx,
                "obs_dim": obs_d,
                "action_dim": act_d,
                "reward": float(rng.normal(-0.02, 0.01)),
                "terminated": 0,
                "truncated": int(step == n_steps - 1),
            }
            for name, val in zip(obs_names, rng.uniform(-1.0, 1.0, size=obs_d)):
                row[f"{OBS_PREFIX}{name}"] = float(val)
            for name, val in zip(act_names, rng.uniform(0.0, 1.0, size=act_d)):
                row[f"{ACTION_PREFIX}{name}"] = float(val)
            for name, val in zip(obs_names, rng.uniform(-1.0, 1.0, size=obs_d)):
                row[f"{NEXT_OBS_PREFIX}{name}"] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


@pytest.fixture
def tiny_data_dir(tmp_path):
    """Create a minimal entity dataset with 2 seeds × 200 rows × 1 group."""
    groups = [(4, 1)]
    data = tmp_path / "data"
    data.mkdir()
    for s in (22, 23):
        df = _make_synthetic_df(seed=s, n_steps=200, groups=groups)
        _write_parquet(df, data / f"seed_{s}.parquet")
    return data


def _tiny_config(gradient_steps: int, checkpoint_every: int = 50):
    """Build a fast IQLTrainingConfig for tests."""
    from algorithms.offline_rl.iql_trainer import IQLTrainingConfig

    return IQLTrainingConfig(
        hidden_layers=[16, 16],
        dropout=0.0,
        log_std_init=-2.3025850929940455,
        tau_expectile=0.7,
        beta_advantage=3.0,
        advantage_clip=100.0,
        gamma=0.99,
        tau_target=0.005,
        learning_rate=3e-4,
        weight_decay=1e-5,
        gradient_clip_norm=1.0,
        batch_size=32,
        gradient_steps=gradient_steps,
        eval_every_n_steps=max(1, checkpoint_every),
        checkpoint_every_n_steps=checkpoint_every,
        device="cpu",
    )


def _run_seed(data_dir, output_dir, *, gradient_steps, checkpoint_every=50, force=False, seed=22):
    from algorithms.offline_rl.iql_entity_trainer import train_entity_single_seed

    cfg = _tiny_config(gradient_steps=gradient_steps, checkpoint_every=checkpoint_every)
    return train_entity_single_seed(
        data_dir=data_dir,
        output_dir=output_dir,
        obs_dim=4,
        action_dim=1,
        seed=seed,
        val_seeds=[23],
        config=cfg,
        force=force,
    )


# ---------------------------------------------------------------------------
# IQLTrainingConfig now carries checkpoint_every_n_steps
# ---------------------------------------------------------------------------


def test_config_has_checkpoint_every_n_steps_field():
    from algorithms.offline_rl.iql_trainer import IQLTrainingConfig

    cfg = IQLTrainingConfig()
    assert hasattr(cfg, "checkpoint_every_n_steps")
    assert cfg.checkpoint_every_n_steps == 5000


def test_config_checkpoint_every_n_steps_override():
    from algorithms.offline_rl.iql_trainer import IQLTrainingConfig

    cfg = IQLTrainingConfig(checkpoint_every_n_steps=123)
    assert cfg.checkpoint_every_n_steps == 123


# ---------------------------------------------------------------------------
# Checkpoint behaviour
# ---------------------------------------------------------------------------


def test_checkpoint_latest_written_every_n_steps(tiny_data_dir, tmp_path):
    """After 100 steps with checkpoint-every=50, checkpoint_latest.pt exists with step=100."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    ckpt_path = out / "checkpoint_latest.pt"
    assert ckpt_path.exists(), "expected checkpoint_latest.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert state["step"] == 100


def test_best_policy_pt_written(tiny_data_dir, tmp_path):
    """best_policy.pt is created on val-MSE improvement."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "best_policy.pt").exists()


def test_seed_done_sentinel_written_on_completion(tiny_data_dir, tmp_path):
    """seed.done sentinel is written after the loop completes successfully."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "seed.done").exists()


def test_seed_done_sentinel_skips_rerun(tiny_data_dir, tmp_path):
    """A second call with seed.done present should early-return without retraining."""
    out = tmp_path / "run"
    first = _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)

    # Sabotage policy.pt so we'd know if training had retrained.
    sentinel_text = "TOUCHED"
    sabotage = out / "policy.pt"
    original = sabotage.read_bytes()
    sabotage.write_bytes(sabotage.read_bytes())  # rewrite to bump mtime
    mtime_before = sabotage.stat().st_mtime_ns

    second = _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)

    # Same summary contract — no retraining happened.
    assert second["seed"] == first["seed"]
    # Path not modified by the skipped run.
    assert sabotage.stat().st_mtime_ns == mtime_before


def test_force_overrides_seed_done(tiny_data_dir, tmp_path):
    """force=True ignores seed.done and reruns training."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "seed.done").exists()

    # Touch a marker file to detect rerun.
    sabotage = out / "policy.pt"
    mtime_before = sabotage.stat().st_mtime_ns

    # Sleep to give mtime resolution room.
    import time as _t

    _t.sleep(0.05)

    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50, force=True)
    mtime_after = sabotage.stat().st_mtime_ns
    assert mtime_after > mtime_before, "force=True should have rewritten policy.pt"


def test_resume_picks_up_from_checkpoint(tiny_data_dir, tmp_path):
    """After resuming, training advances past the checkpoint step."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    # Remove seed.done so resume kicks in instead of early-return.
    (out / "seed.done").unlink()
    _run_seed(tiny_data_dir, out, gradient_steps=200, checkpoint_every=50)
    state = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 200


def test_resume_deterministic_matches_continuous(tiny_data_dir, tmp_path):
    """100 + 100 (with resume) produces identical final weights to a single 200-step run.

    This is the critical correctness test: RNG state and optimiser state must be
    captured in the checkpoint and restored exactly on resume.
    """
    # Continuous baseline
    out_a = tmp_path / "continuous"
    _run_seed(tiny_data_dir, out_a, gradient_steps=200, checkpoint_every=100)
    state_a = torch.load(out_a / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert state_a["step"] == 200

    # Interrupted: train 100, drop seed.done, train 200
    out_b = tmp_path / "interrupted"
    _run_seed(tiny_data_dir, out_b, gradient_steps=100, checkpoint_every=100)
    (out_b / "seed.done").unlink()
    _run_seed(tiny_data_dir, out_b, gradient_steps=200, checkpoint_every=100)
    state_b = torch.load(out_b / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert state_b["step"] == 200

    # Compare weight tensors element-wise
    for net_key in ("policy_state", "qf1_state", "qf2_state", "vf_state"):
        pa = state_a[net_key]
        pb = state_b[net_key]
        assert set(pa.keys()) == set(pb.keys())
        for k in pa:
            torch.testing.assert_close(pa[k], pb[k], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# status.json side-channel
# ---------------------------------------------------------------------------


def test_status_json_written_after_checkpoint(tiny_data_dir, tmp_path):
    """A status.json file is written under output_root after each checkpoint."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    status_path = out / "status.json"
    assert status_path.exists()
    data = json.loads(status_path.read_text())
    # Must contain at minimum: step, val_mse and one of (group_key, seed).
    assert "step" in data
    assert "val_mse" in data
