"""Tests for IQLEntityAgent.

Covers:
  * Loading from a synthetic seed dir
  * Loading from a synthetic multi-group output root
  * ``predict`` dispatches by obs_dim and returns correct shapes
  * ``update`` is a no-op
  * ``attach_environment`` validates groups
  * ``export_artifacts`` saves all required files
  * Registry registration
  * ``_pick_best_seed_dir`` selects the lowest-mse seed
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent, _pick_best_seed_dir
from algorithms.offline_rl.iql_networks import GaussianPolicy


# ---------------------------------------------------------------------------
# Fixtures — build tiny in-memory seed dirs on disk
# ---------------------------------------------------------------------------

_GROUPS = [
    (8, 1),   # stand-in for obs627_act1
    (12, 2),  # stand-in for obs706_act2
]


def _make_seed_dir(tmp_path: Path, obs_dim: int, action_dim: int, seed: int = 0) -> Path:
    """Write policy.pt / architecture.json / obs_standardiser.npz for one seed."""
    seed_dir = tmp_path / f"obs{obs_dim}_act{action_dim}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Architecture
    arch = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_layers": [16, 16],
        "dropout": 0.0,
        "log_std_init": math.log(0.1),
        "group_key": f"obs{obs_dim}_act{action_dim}",
    }
    (seed_dir / "architecture.json").write_text(json.dumps(arch))

    # Policy
    torch.manual_seed(seed)
    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden=[16, 16],
        dropout=0.0,
        log_std_init=math.log(0.1),
    )
    torch.save(policy.state_dict(), seed_dir / "policy.pt")

    # Standardiser
    rng = np.random.default_rng(seed)
    mean = rng.random(obs_dim).astype(np.float32)
    std = rng.random(obs_dim).astype(np.float32) + 0.1
    std_obj = ObservationStandardiser(
        mean=mean,
        std=std,
        feature_names=[f"f{i}" for i in range(obs_dim)],
    )
    std_obj.save(seed_dir / "obs_standardiser.npz")
    return seed_dir


def _make_multi_seed_root(
    tmp_path: Path,
    obs_dim: int,
    action_dim: int,
    n_seeds: int = 2,
) -> Path:
    """Write a full group directory with n_seeds and multi_seed_summary.json."""
    group_key = f"obs{obs_dim}_act{action_dim}"
    group_root = tmp_path / group_key

    per_seed = []
    for s in range(n_seeds):
        seed_dir = _make_seed_dir(group_root.parent / "SEED_TMP", obs_dim, action_dim, s)
        # Move into the group root
        final_dir = group_root / f"seed_{s}"
        final_dir.mkdir(parents=True, exist_ok=True)
        for f in seed_dir.iterdir():
            f.rename(final_dir / f.name)
        per_seed.append(
            {
                "seed": s,
                "output_dir": str(final_dir),
                "best_val_policy_mse": float(n_seeds - s),  # seed 0 = worst, last = best
            }
        )

    summary = {
        "group_key": group_key,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_seeds": n_seeds,
        "per_seed": per_seed,
    }
    (group_root / "multi_seed_summary.json").write_text(json.dumps(summary))
    return group_root


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------


def test_load_from_single_seed_dir(tmp_path: Path) -> None:
    """IQLEntityAgent loads when given a single seed directory."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)
    assert 8 in agent._policy_map
    assert 8 in agent._standardiser_map


def test_load_from_multi_group_root(tmp_path: Path) -> None:
    """IQLEntityAgent loads all groups from a multi-group output root."""
    model_root = tmp_path / "model_root"
    for obs_dim, action_dim in _GROUPS:
        _make_multi_seed_root(model_root, obs_dim, action_dim, n_seeds=2)
    agent = IQLEntityAgent.from_model_dir(model_root)
    assert len(agent._policy_map) == len(_GROUPS)
    for obs_dim, _ in _GROUPS:
        assert obs_dim in agent._policy_map


def test_load_raises_if_model_dir_missing(tmp_path: Path) -> None:
    """FileNotFoundError when model_dir does not exist."""
    with pytest.raises(FileNotFoundError):
        IQLEntityAgent.from_model_dir(tmp_path / "nonexistent")


def test_load_raises_if_architecture_missing(tmp_path: Path) -> None:
    """FileNotFoundError when seed dir is missing architecture.json."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    (seed_dir / "architecture.json").unlink()
    with pytest.raises(FileNotFoundError):
        IQLEntityAgent.from_model_dir(seed_dir)


# ---------------------------------------------------------------------------
# Tests: predict
# ---------------------------------------------------------------------------


def test_predict_returns_correct_shape(tmp_path: Path) -> None:
    """predict returns one action list per agent with correct action_dim."""
    model_root = tmp_path / "model_root"
    for obs_dim, action_dim in _GROUPS:
        _make_multi_seed_root(model_root, obs_dim, action_dim, n_seeds=1)

    agent = IQLEntityAgent.from_model_dir(model_root)

    # Two agents: one per group
    observations = [np.random.randn(obs_dim).astype(np.float64) for obs_dim, _ in _GROUPS]
    actions = agent.predict(observations)

    assert len(actions) == len(_GROUPS)
    for i, (_, action_dim) in enumerate(_GROUPS):
        assert len(actions[i]) == action_dim, (
            f"group {i}: expected {action_dim} actions, got {len(actions[i])}"
        )


def test_predict_values_are_finite(tmp_path: Path) -> None:
    """predict outputs are finite (no NaN/inf) even with NaN input obs."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)

    obs_with_nan = np.full(8, np.nan, dtype=np.float64)
    obs_with_nan[0] = 1.0  # at least one non-nan

    actions = agent.predict([obs_with_nan])
    assert len(actions) == 1
    assert all(np.isfinite(v) for v in actions[0])


def test_predict_unknown_obs_dim_raises(tmp_path: Path) -> None:
    """predict raises ValueError when obs_dim has no loaded policy."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)

    obs_wrong = np.random.randn(99).astype(np.float64)
    with pytest.raises(ValueError, match="not in loaded policies"):
        agent.predict([obs_wrong])


# ---------------------------------------------------------------------------
# Tests: update is a no-op
# ---------------------------------------------------------------------------


def test_update_is_noop(tmp_path: Path) -> None:
    """update() must return None without raising."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)
    result = agent.update(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Tests: attach_environment
# ---------------------------------------------------------------------------


def test_attach_environment_validates_groups(tmp_path: Path) -> None:
    """attach_environment succeeds when all agent obs_dims have loaded policies."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)

    # Two agents, both obs_dim=8
    observation_names: List[List[str]] = [[f"f{i}" for i in range(8)] for _ in range(2)]
    action_names: List[List[str]] = [["a"] for _ in range(2)]
    agent.attach_environment(
        observation_names=observation_names,
        action_names=action_names,
        action_space=[None, None],
        observation_space=[None, None],
    )
    assert agent._obs_dims_per_agent == [8, 8]


def test_attach_environment_raises_on_unknown_group(tmp_path: Path) -> None:
    """attach_environment raises ValueError when an agent's obs_dim is not loaded."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)

    observation_names = [[f"f{i}" for i in range(99)]]  # obs_dim=99, not loaded
    action_names = [["a"]]
    with pytest.raises(ValueError, match="no policy was loaded"):
        agent.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=[None],
            observation_space=[None],
        )


# ---------------------------------------------------------------------------
# Tests: export_artifacts
# ---------------------------------------------------------------------------


def test_export_artifacts_writes_expected_files(tmp_path: Path) -> None:
    """export_artifacts saves policy.pt, obs_standardiser.npz, and manifest."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = IQLEntityAgent.from_model_dir(seed_dir)
    out_dir = tmp_path / "exported"

    manifest = agent.export_artifacts(str(out_dir))

    assert manifest["artifact_type"] == "iql_entity_agent"
    assert "groups" in manifest
    assert len(manifest["groups"]) == 1

    group_key = "obs8_act1"
    assert group_key in manifest["groups"]
    assert Path(manifest["groups"][group_key]["policy_path"]).exists()
    assert Path(manifest["groups"][group_key]["standardiser_path"]).exists()
    assert (out_dir / "iql_entity_manifest.json").exists()


# ---------------------------------------------------------------------------
# Tests: registry
# ---------------------------------------------------------------------------


def test_registry_contains_iql_entity_agent() -> None:
    """IQLEntityAgent must be present in the algorithm registry."""
    from algorithms.registry import ALGORITHM_REGISTRY

    assert "IQLEntityAgent" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["IQLEntityAgent"] is IQLEntityAgent


def test_registry_can_instantiate_via_config(tmp_path: Path) -> None:
    """build_execution_unit can instantiate IQLEntityAgent from a pipeline config."""
    from algorithms.registry import build_execution_unit

    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    config = {
        "pipeline": [
            {
                "algorithm": "IQLEntityAgent",
                "count": 1,
                "hyperparameters": {
                    "model_dir": str(seed_dir),
                    "device": "cpu",
                },
            }
        ]
    }
    agent = build_execution_unit(config)
    assert isinstance(agent, IQLEntityAgent)


# ---------------------------------------------------------------------------
# Tests: _pick_best_seed_dir
# ---------------------------------------------------------------------------


def test_pick_best_seed_dir_selects_lowest_mse(tmp_path: Path) -> None:
    """_pick_best_seed_dir should return the seed with the lowest best_val_policy_mse."""
    group_root = tmp_path / "obs8_act1"
    # Create two seed dirs
    for s in range(2):
        seed_dir = group_root / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "architecture.json").write_text("{}")  # placeholder

    multi = {
        "per_seed": [
            {"seed": 0, "output_dir": str(group_root / "seed_0"), "best_val_policy_mse": 0.9},
            {"seed": 1, "output_dir": str(group_root / "seed_1"), "best_val_policy_mse": 0.1},
        ]
    }
    multi_path = group_root / "multi_seed_summary.json"
    multi_path.write_text(json.dumps(multi))

    best = _pick_best_seed_dir(group_root, multi_path)
    assert best == group_root / "seed_1"
