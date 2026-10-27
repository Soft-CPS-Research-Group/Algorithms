# tests/offline_rl/test_cql_entity_resume.py
"""Tests for trainer resume + checkpointing in cql_entity_trainer.

Mirrors ``tests/offline_rl/test_iql_entity_resume.py``.  The CQL trainer
implements the same resume contract as the IQL trainer with one extra Q
penalty term, so these tests act as regression guards on the CQL mirror:

* Mid-training checkpoint every ``checkpoint_every_n_steps`` (``checkpoint_latest.pt``).
* ``best_policy.pt`` written on each val MSE improvement.
* ``seed.done`` sentinel written on successful completion.
* If ``seed.done`` exists → early return (unless ``force=True``).
* If ``checkpoint_latest.pt`` exists → resume from that step (unless ``force``).
* ``status.json`` updated under ``output_root`` after each checkpoint.

Additionally verifies CQL-specific persistence: the ``cql_penalty`` metric
appears in ``metrics.jsonl`` (asserting the CQL update path actually ran).

The deterministic-resume test verifies that 200 steps in one shot vs.
100 → kill → resume → 100 more produce identical final network state for
ALL five CQL networks (policy, Q1, Q2, Q1_target, Q2_target, V), with full
RNG capture/restore covering the CQL random-action sampler.
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
    NEXT_OBS_PREFIX,
    OBS_PREFIX,
)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers (mirrors IQL resume test)
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
    """Build a fast CQLTrainingConfig for tests.

    Uses very small ``cql_n_random_actions=2`` to keep tests fast while still
    exercising the CQL penalty code path.
    """
    from algorithms.offline_rl.cql_entity_trainer import CQLTrainingConfig

    return CQLTrainingConfig(
        hidden_layers=[16, 16],
        dropout=0.0,
        log_std_init=-2.3025850929940455,
        tau_expectile=0.7,
        beta_advantage=3.0,
        advantage_clip=100.0,
        gamma=0.99,
        tau_target=0.005,
        cql_alpha=0.2,
        cql_n_random_actions=2,
        cql_min_q_weight=0.0,
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
    from algorithms.offline_rl.cql_entity_trainer import train_cql_single_seed

    cfg = _tiny_config(gradient_steps=gradient_steps, checkpoint_every=checkpoint_every)
    return train_cql_single_seed(
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
# CQLTrainingConfig inherits checkpoint_every_n_steps from IQLTrainingConfig
# ---------------------------------------------------------------------------


def test_cql_config_has_checkpoint_every_n_steps_field():
    from algorithms.offline_rl.cql_entity_trainer import CQLTrainingConfig

    cfg = CQLTrainingConfig()
    assert hasattr(cfg, "checkpoint_every_n_steps")
    assert cfg.checkpoint_every_n_steps == 5000


def test_cql_config_checkpoint_every_n_steps_override():
    from algorithms.offline_rl.cql_entity_trainer import CQLTrainingConfig

    cfg = CQLTrainingConfig(checkpoint_every_n_steps=123)
    assert cfg.checkpoint_every_n_steps == 123


# ---------------------------------------------------------------------------
# Checkpoint behaviour
# ---------------------------------------------------------------------------


def test_cql_checkpoint_latest_written_every_n_steps(tiny_data_dir, tmp_path):
    """After 100 steps with checkpoint-every=50, checkpoint_latest.pt exists with step=100."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    ckpt_path = out / "checkpoint_latest.pt"
    assert ckpt_path.exists(), "expected checkpoint_latest.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert state["step"] == 100


def test_cql_checkpoint_contains_all_cql_networks(tiny_data_dir, tmp_path):
    """CQL checkpoint must carry all five Q/V/policy net states + 3 opt states."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    state = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    # Networks: policy + Q1 + Q2 + Q1_target + Q2_target + V
    for key in (
        "policy_state",
        "qf1_state",
        "qf2_state",
        "qf1_target_state",
        "qf2_target_state",
        "vf_state",
    ):
        assert key in state, f"checkpoint missing {key}"
    # Optimisers: policy, Q (shared), V
    for key in ("policy_opt_state", "qf_opt_state", "vf_opt_state"):
        assert key in state, f"checkpoint missing {key}"
    # RNG: torch + numpy + generator
    for key in ("rng_state_torch", "rng_state_numpy", "rng_state_gen"):
        assert key in state, f"checkpoint missing {key}"
    # Best-so-far bookkeeping
    for key in ("best_val_mse", "best_step", "best_policy_state", "wall_clock_seconds"):
        assert key in state, f"checkpoint missing {key}"


def test_cql_best_policy_pt_written(tiny_data_dir, tmp_path):
    """best_policy.pt is created on val-MSE improvement."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "best_policy.pt").exists()


def test_cql_seed_done_sentinel_written_on_completion(tiny_data_dir, tmp_path):
    """seed.done sentinel is written after the loop completes successfully."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "seed.done").exists()


def test_cql_seed_done_sentinel_skips_rerun(tiny_data_dir, tmp_path):
    """A second call with seed.done present should early-return without retraining."""
    out = tmp_path / "run"
    first = _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)

    sabotage = out / "policy.pt"
    mtime_before = sabotage.stat().st_mtime_ns

    second = _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)

    # Same summary contract — no retraining happened.
    assert second["seed"] == first["seed"]
    # Path not modified by the skipped run.
    assert sabotage.stat().st_mtime_ns == mtime_before


def test_cql_force_overrides_seed_done(tiny_data_dir, tmp_path):
    """force=True ignores seed.done and reruns training."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    assert (out / "seed.done").exists()

    sabotage = out / "policy.pt"
    mtime_before = sabotage.stat().st_mtime_ns

    import time as _t

    _t.sleep(0.05)

    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50, force=True)
    mtime_after = sabotage.stat().st_mtime_ns
    assert mtime_after > mtime_before, "force=True should have rewritten policy.pt"


def test_cql_resume_picks_up_from_checkpoint(tiny_data_dir, tmp_path):
    """After resuming, training advances past the checkpoint step."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    # Remove seed.done so resume kicks in instead of early-return.
    (out / "seed.done").unlink()
    _run_seed(tiny_data_dir, out, gradient_steps=200, checkpoint_every=50)
    state = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 200


def test_cql_resume_deterministic_matches_continuous(tiny_data_dir, tmp_path):
    """100 + 100 (with resume) produces identical final weights to a single 200-step run.

    Critical correctness test: RNG state (including the CQL random-action
    sampler ``rng_gen``) and optimiser states must be captured in the
    checkpoint and restored exactly on resume.

    Compares ALL CQL nets: policy, Q1, Q2, Q1_target, Q2_target, V.
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

    # Compare weight tensors element-wise across all five CQL nets
    for net_key in (
        "policy_state",
        "qf1_state",
        "qf2_state",
        "qf1_target_state",
        "qf2_target_state",
        "vf_state",
    ):
        pa = state_a[net_key]
        pb = state_b[net_key]
        assert set(pa.keys()) == set(pb.keys()), f"{net_key} key mismatch"
        for k in pa:
            torch.testing.assert_close(pa[k], pb[k], atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# CQL-specific: cql_penalty present in metrics
# ---------------------------------------------------------------------------


def test_cql_metrics_contain_cql_penalty(tiny_data_dir, tmp_path):
    """metrics.jsonl must include the CQL conservative penalty value."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    metrics_path = out / "metrics.jsonl"
    assert metrics_path.exists()
    lines = [json.loads(ln) for ln in metrics_path.read_text().splitlines() if ln.strip()]
    assert lines, "expected at least one metrics row"
    last = lines[-1]
    assert "cql_penalty" in last, f"cql_penalty missing from metrics: {last}"
    # Sanity: cql_penalty is a finite float
    val = last["cql_penalty"]
    assert isinstance(val, float)
    assert np.isfinite(val), f"cql_penalty must be finite, got {val}"


# ---------------------------------------------------------------------------
# status.json side-channel
# ---------------------------------------------------------------------------


def test_cql_status_json_written_after_checkpoint(tiny_data_dir, tmp_path):
    """A status.json file is written under output_root after each checkpoint."""
    out = tmp_path / "run"
    _run_seed(tiny_data_dir, out, gradient_steps=100, checkpoint_every=50)
    status_path = out / "status.json"
    assert status_path.exists()
    data = json.loads(status_path.read_text())
    assert "step" in data
    assert "val_mse" in data


# ---------------------------------------------------------------------------
# Phase 5 prerequisite: force propagation through multi_seed / all_groups
# ---------------------------------------------------------------------------


def _fake_single_seed_summary(*, seed: int) -> dict:
    return {
        "seed": int(seed),
        "best_val_policy_mse": 0.1,
        "best_step": 1,
        "duration_seconds": 0.0,
        "final_metrics": {
            "policy_loss": 0.0,
            "cql_penalty": 0.0,
        },
    }


def test_cql_multi_seed_propagates_force_to_single_seed(monkeypatch, tmp_path):
    """train_cql_multi_seed must forward ``force`` to train_cql_single_seed.

    Regression guard: in Phase 2 ``force`` was added to ``train_cql_single_seed``
    but never propagated through ``train_cql_multi_seed`` or ``train_all_groups``,
    leaving ``--force train-cql`` on the orchestrator unable to actually clear
    seed.done sentinels.
    """
    from algorithms.offline_rl import cql_entity_trainer

    captured_force: list = []

    def fake_single_seed(**kwargs):
        captured_force.append(kwargs.get("force", "MISSING"))
        return _fake_single_seed_summary(seed=kwargs["seed"])

    monkeypatch.setattr(cql_entity_trainer, "train_cql_single_seed", fake_single_seed)

    cfg = _tiny_config(gradient_steps=10, checkpoint_every=10)
    cql_entity_trainer.train_cql_multi_seed(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "out",
        obs_dim=4,
        action_dim=1,
        seeds=[22, 23],
        val_seeds=[31],
        config=cfg,
        force=True,
    )
    assert captured_force == [True, True], (
        f"force=True must propagate to every single-seed call; got {captured_force}"
    )


def test_cql_multi_seed_force_defaults_to_false(monkeypatch, tmp_path):
    """When ``force`` is omitted, multi_seed must forward force=False."""
    from algorithms.offline_rl import cql_entity_trainer

    captured_force: list = []

    def fake_single_seed(**kwargs):
        captured_force.append(kwargs.get("force", "MISSING"))
        return _fake_single_seed_summary(seed=kwargs["seed"])

    monkeypatch.setattr(cql_entity_trainer, "train_cql_single_seed", fake_single_seed)

    cfg = _tiny_config(gradient_steps=10, checkpoint_every=10)
    cql_entity_trainer.train_cql_multi_seed(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "out",
        obs_dim=4,
        action_dim=1,
        seeds=[22],
        val_seeds=[31],
        config=cfg,
    )
    assert captured_force == [False], (
        f"force default must be False; got {captured_force}"
    )


def test_cql_all_groups_propagates_force_to_multi_seed(monkeypatch, tmp_path):
    """train_all_groups must forward ``force`` to train_cql_multi_seed."""
    from algorithms.offline_rl import cql_entity_trainer

    captured_force: list = []

    def fake_multi_seed(**kwargs):
        captured_force.append(kwargs.get("force", "MISSING"))
        return {
            "group_key": f"obs{kwargs['obs_dim']}_act{kwargs['action_dim']}",
            "obs_dim": kwargs["obs_dim"],
            "action_dim": kwargs["action_dim"],
            "n_seeds": 1,
            "seeds": [22],
            "best_val_policy_mse_mean": 0.1,
            "best_val_policy_mse_std": 0.0,
            "duration_seconds": 0.0,
            "data_dir": str(kwargs["data_dir"]),
            "config": {},
            "per_seed": [],
        }

    monkeypatch.setattr(cql_entity_trainer, "train_cql_multi_seed", fake_multi_seed)

    cfg = _tiny_config(gradient_steps=10, checkpoint_every=10)
    cql_entity_trainer.train_all_groups(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "out",
        seeds=[22],
        val_seeds=[31],
        config=cfg,
        groups=[(4, 1), (6, 2)],
        force=True,
    )
    assert captured_force == [True, True], (
        f"force=True must propagate to every multi_seed call; got {captured_force}"
    )


def test_cql_all_groups_force_defaults_to_false(monkeypatch, tmp_path):
    """When ``force`` is omitted, train_all_groups must forward force=False."""
    from algorithms.offline_rl import cql_entity_trainer

    captured_force: list = []

    def fake_multi_seed(**kwargs):
        captured_force.append(kwargs.get("force", "MISSING"))
        return {
            "group_key": f"obs{kwargs['obs_dim']}_act{kwargs['action_dim']}",
            "obs_dim": kwargs["obs_dim"],
            "action_dim": kwargs["action_dim"],
            "n_seeds": 1,
            "seeds": [22],
            "best_val_policy_mse_mean": 0.1,
            "best_val_policy_mse_std": 0.0,
            "duration_seconds": 0.0,
            "data_dir": str(kwargs["data_dir"]),
            "config": {},
            "per_seed": [],
        }

    monkeypatch.setattr(cql_entity_trainer, "train_cql_multi_seed", fake_multi_seed)

    cfg = _tiny_config(gradient_steps=10, checkpoint_every=10)
    cql_entity_trainer.train_all_groups(
        data_dir=tmp_path / "data",
        output_root=tmp_path / "out",
        seeds=[22],
        val_seeds=[31],
        config=cfg,
        groups=[(4, 1)],
    )
    assert captured_force == [False], (
        f"force default must be False; got {captured_force}"
    )
