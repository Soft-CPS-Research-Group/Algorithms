"""Tests for CQLEntityAgent.

CQLEntityAgent inherits all inference logic from IQLEntityAgent.
These tests verify:
  * Basic loading and predict still work
  * export_artifacts reports artifact_type = 'cql_entity_agent'
  * Registry registration
  * from_model_dir classmethod
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.cql_entity_agent import CQLEntityAgent
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent
from algorithms.offline_rl.iql_networks import GaussianPolicy


# ---------------------------------------------------------------------------
# Fixture helper (copy from test_iql_entity_agent to avoid test coupling)
# ---------------------------------------------------------------------------


def _make_seed_dir(tmp_path: Path, obs_dim: int, action_dim: int, seed: int = 0) -> Path:
    seed_dir = tmp_path / f"obs{obs_dim}_act{action_dim}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    arch = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_layers": [16, 16],
        "dropout": 0.0,
        "log_std_init": math.log(0.1),
        "group_key": f"obs{obs_dim}_act{action_dim}",
    }
    (seed_dir / "architecture.json").write_text(json.dumps(arch))
    torch.manual_seed(seed)
    policy = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=[16, 16])
    torch.save(policy.state_dict(), seed_dir / "policy.pt")
    rng = np.random.default_rng(seed)
    std_obj = ObservationStandardiser(
        mean=rng.random(obs_dim).astype(np.float32),
        std=rng.random(obs_dim).astype(np.float32) + 0.1,
        feature_names=[f"f{i}" for i in range(obs_dim)],
    )
    std_obj.save(seed_dir / "obs_standardiser.npz")
    return seed_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cql_agent_is_subclass_of_iql_entity_agent() -> None:
    """CQLEntityAgent must extend IQLEntityAgent."""
    assert issubclass(CQLEntityAgent, IQLEntityAgent)


def test_cql_agent_load_and_predict(tmp_path: Path) -> None:
    """CQLEntityAgent loads and predicts the correct action shape."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = CQLEntityAgent.from_model_dir(seed_dir)
    obs = [np.random.randn(8).astype(np.float64)]
    actions = agent.predict(obs)
    assert len(actions) == 1
    assert len(actions[0]) == 1


def test_cql_agent_export_artifact_type(tmp_path: Path) -> None:
    """export_artifacts must report artifact_type = 'cql_entity_agent'."""
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    agent = CQLEntityAgent.from_model_dir(seed_dir)
    manifest = agent.export_artifacts(str(tmp_path / "export"))
    assert manifest["artifact_type"] == "cql_entity_agent"


def test_cql_agent_registry() -> None:
    """CQLEntityAgent must be in the algorithm registry."""
    from algorithms.registry import ALGORITHM_REGISTRY
    assert "CQLEntityAgent" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["CQLEntityAgent"] is CQLEntityAgent


def test_cql_agent_instantiate_via_config(tmp_path: Path) -> None:
    """build_execution_unit instantiates CQLEntityAgent from pipeline config."""
    from algorithms.registry import build_execution_unit
    seed_dir = _make_seed_dir(tmp_path, obs_dim=8, action_dim=1)
    config = {
        "pipeline": [
            {
                "algorithm": "CQLEntityAgent",
                "count": 1,
                "hyperparameters": {"model_dir": str(seed_dir), "device": "cpu"},
            }
        ]
    }
    agent = build_execution_unit(config)
    assert isinstance(agent, CQLEntityAgent)
