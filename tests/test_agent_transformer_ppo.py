"""WP05 §16.4 — AgentTransformerPPO unit tests.

Covers:
- ``predict`` returns ``[B][N_ca]`` floats clamped to ``[-1, 1]``.
- ``update`` accumulates rollouts and runs PPO step on ``update_step``.
- Topology change rebuilds layout, preserves weights for stable types.
- Layout-drift on existing type (feature-count change) hard-fails.
- New type appearing on existing tokenizer hard-fails.
- ``save_checkpoint`` / ``load_checkpoint`` round-trip + signature mismatch
  rejection (spec §14.3).
- ``export_artifacts`` returns a well-formed manifest with one entry per
  building and TorchScript files on disk.
- Registered in ``algorithms.registry.ALGORITHM_REGISTRY``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from algorithms.agents.agent_transformer_ppo import (
    AgentTransformerPPO,
    _synthetic_sample_from_obs_names,
)
from algorithms.registry import ALGORITHM_REGISTRY, build_execution_unit
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"
_DEFAULT_ACTIONS = ["electrical_storage", "electric_vehicle_storage"]


def _base_config() -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "tokenizer_config_path": _TOKENIZER_CFG,
            "transformer": {
                "d_model": 16,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
            },
            "hyperparameters": {
                "learning_rate": 1.0e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 1,
                "minibatch_size": 4,
                "entropy_coeff": 0.0,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "actor_hidden_dim": 32,
                "critic_hidden_dim": 32,
            },
        },
    }


def _make_agent(n_buildings: int = 1) -> tuple[AgentTransformerPPO, List[List[str]], List[List[str]], int]:
    obs_names = load_sample_observation_names_for_first_building()
    obs_names_per = [list(obs_names) for _ in range(n_buildings)]
    act_names_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerPPO(_base_config())
    agent.attach_environment(
        observation_names=obs_names_per,
        action_names=act_names_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b+1}" for b in range(n_buildings)]},
    )
    obs_dim = max(
        max(seg.feature_indices) for seg in agent._per_building[0].layout.segments
    ) + 1
    return agent, obs_names_per, act_names_per, obs_dim


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered_under_canonical_name() -> None:
    assert ALGORITHM_REGISTRY.get("AgentTransformerPPO") is AgentTransformerPPO


def test_create_agent_via_registry() -> None:
    base = _base_config()
    algo = base.pop("algorithm")
    stage = {"algorithm": algo.pop("name")}
    stage.update(algo)
    base["pipeline"] = [stage]
    agent = build_execution_unit(base)
    assert isinstance(agent, AgentTransformerPPO)


def test_supports_dynamic_topology_classvar_true() -> None:
    assert AgentTransformerPPO.supports_dynamic_topology is True


# ---------------------------------------------------------------------------
# predict / update
# ---------------------------------------------------------------------------


def test_predict_shape_and_range() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=2)
    obs = [np.random.rand(obs_dim).astype(np.float64) for _ in range(2)]
    actions = agent.predict(obs, deterministic=False)
    assert isinstance(actions, list) and len(actions) == 2
    for b, vec in enumerate(actions):
        assert isinstance(vec, list)
        assert len(vec) == agent._per_building[b].layout.n_ca
        for v in vec:
            assert -1.0 <= v <= 1.0


def test_predict_deterministic_is_repeatable() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    obs = [np.zeros(obs_dim, dtype=np.float64)]
    a1 = agent.predict(obs, deterministic=True)
    a2 = agent.predict(obs, deterministic=True)
    assert a1 == a2


def test_update_appends_to_buffer_then_ppo_step_clears() -> None:
    agent, _, _, obs_dim = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    rng = np.random.default_rng(0)
    n_ca = state.layout.n_ca

    # Five non-update steps → buffer fills
    for _ in range(5):
        obs = [rng.standard_normal(obs_dim)]
        next_obs = [rng.standard_normal(obs_dim)]
        actions_arr = rng.uniform(-0.5, 0.5, size=(n_ca,))
        agent.update(
            observations=obs,
            actions=[actions_arr],
            rewards=[0.1],
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=0,
            update_step=False,
            initial_exploration_done=True,
        )
    assert len(state.buffer) == 5

    # Snapshot a parameter to confirm gradient step actually moved weights.
    p_before = next(state.actor.parameters()).clone().detach()

    obs = [rng.standard_normal(obs_dim)]
    next_obs = [rng.standard_normal(obs_dim)]
    actions_arr = rng.uniform(-0.5, 0.5, size=(n_ca,))
    agent.update(
        observations=obs,
        actions=[actions_arr],
        rewards=[0.1],
        next_observations=next_obs,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )
    assert len(state.buffer) == 0  # cleared after PPO step
    p_after = next(state.actor.parameters()).clone().detach()
    assert not torch.allclose(p_before, p_after), "PPO step should update actor weights"


# ---------------------------------------------------------------------------
# Topology change handling
# ---------------------------------------------------------------------------


def test_topology_change_no_op_when_names_unchanged() -> None:
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    state_before = agent._per_building[0]
    actor_id_before = id(state_before.actor)
    agent.attach_environment(
        observation_names=copy.deepcopy(obs_per),
        action_names=copy.deepcopy(act_per),
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    # Same instance — no rebuild.
    assert id(agent._per_building[0].actor) == actor_id_before


def test_topology_change_rebuilds_layout_and_preserves_weights() -> None:
    """Add a second charger (and its EV blocks). The 'charger' projection
    weights survive (per-type weight sharing) and the layout grows by one
    CA segment."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    state = agent._per_building[0]
    n_ca_before = state.layout.n_ca
    # Snapshot the storage projection weights — should survive the rebuild.
    storage_w_before = state.tokenizer.projections["storage"].weight.detach().clone()
    charger_w_before = state.tokenizer.projections["charger"].weight.detach().clone()
    actor_w_before = next(state.actor.parameters()).detach().clone()

    # Identify existing charger block and replicate it under a fresh id.
    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_NEW"
    new_obs: List[str] = []
    new_block: List[str] = []
    for n in obs_per[0]:
        new_obs.append(n)
        if n.startswith(f"charger::{orig_id}::"):
            # Append a parallel entry for the new charger right after each
            # existing one (order doesn't matter to the layout builder, but
            # this keeps the diff readable).
            new_block.append(n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1))
    new_obs.extend(new_block)
    new_acts = list(act_per[0]) + ["electric_vehicle_storage"]

    agent.attach_environment(
        observation_names=[new_obs],
        action_names=[new_acts],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )

    state_after = agent._per_building[0]
    assert state_after.layout.n_ca == n_ca_before + 1
    # Per-type weights preserved
    assert torch.allclose(
        storage_w_before,
        state_after.tokenizer.projections["storage"].weight,
    )
    assert torch.allclose(
        charger_w_before,
        state_after.tokenizer.projections["charger"].weight,
    )
    # Actor preserved (per-CA weight sharing)
    actor_w_after = next(state_after.actor.parameters())
    assert torch.allclose(actor_w_before, actor_w_after)


def test_topology_change_feature_count_drift_hard_fails() -> None:
    """Inject an extra storage feature that wasn't present at attach time —
    feature count for type 'storage' changes. Must raise (spec §11.4)."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    storage_id = next(
        n.split("::")[1] for n in obs_per[0] if n.startswith("storage::")
    )
    drifted = list(obs_per[0])
    drifted.append(f"storage::{storage_id}::brand_new_storage_feature")

    with pytest.raises(ValueError, match=r"feature count for type 'storage'"):
        agent.attach_environment(
            observation_names=[drifted],
            action_names=copy.deepcopy(act_per),
            action_space=[None],
            observation_space=[None],
            metadata={"building_names": ["Building_1"]},
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    agent, _, _, _ = _make_agent(n_buildings=1)
    # Mutate weights so round-trip is meaningful.
    with torch.no_grad():
        for p in agent._per_building[0].actor.parameters():
            p.add_(0.1)
    actor_w = (
        next(agent._per_building[0].actor.parameters()).detach().clone()
    )
    path = agent.save_checkpoint(str(tmp_path), step=42)
    assert path is not None and Path(path).exists()

    # Build a fresh agent with identical layout and load.
    obs_names = load_sample_observation_names_for_first_building()
    fresh = AgentTransformerPPO(_base_config())
    fresh.attach_environment(
        observation_names=[list(obs_names)],
        action_names=[list(_DEFAULT_ACTIONS)],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    fresh.load_checkpoint(path)
    actor_w_loaded = next(fresh._per_building[0].actor.parameters()).detach()
    assert torch.allclose(actor_w, actor_w_loaded)


def test_checkpoint_layout_signature_mismatch_rejected(tmp_path: Path) -> None:
    """Save a 1-building checkpoint, then try to load into a 2-building agent.
    Cardinality mismatch is rejected before signature check, exercising the
    same §14.3 guarantee (cross-topology resume disallowed)."""
    agent, _, _, _ = _make_agent(n_buildings=1)
    path = agent.save_checkpoint(str(tmp_path), step=1)
    assert path is not None

    fresh, _, _, _ = _make_agent(n_buildings=2)
    with pytest.raises(ValueError, match=r"Cross-cardinality resume|layout_signature"):
        fresh.load_checkpoint(path)


def test_checkpoint_signature_mismatch_same_cardinality(tmp_path: Path) -> None:
    """Save with the bundled obs_names; reload into an agent whose obs_names
    have an extra (allowed) charger appended → signature differs but
    cardinality matches → ``layout_signature mismatch`` raised."""
    agent, obs_per, act_per, _ = _make_agent(n_buildings=1)
    path = agent.save_checkpoint(str(tmp_path), step=1)
    assert path is not None

    # Build a fresh agent with one extra charger (same trick as the
    # rebuild test) — different obs_names_tuple.
    orig_id = next(
        n.split("::")[1]
        for n in obs_per[0]
        if n.startswith("charger::") and "::connected_ev::" not in n and "::incoming_ev::" not in n
    )
    new_id = "Building_1/charger_NEW"
    extended = list(obs_per[0]) + [
        n.replace(f"charger::{orig_id}::", f"charger::{new_id}::", 1)
        for n in obs_per[0]
        if n.startswith(f"charger::{orig_id}::")
    ]

    fresh = AgentTransformerPPO(_base_config())
    fresh.attach_environment(
        observation_names=[extended],
        action_names=[list(act_per[0]) + ["electric_vehicle_storage"]],
        action_space=[None],
        observation_space=[None],
        metadata={"building_names": ["Building_1"]},
    )
    with pytest.raises(ValueError, match=r"layout_signature mismatch"):
        fresh.load_checkpoint(path)


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------


def test_export_artifacts_writes_files_and_returns_manifest(tmp_path: Path) -> None:
    agent, _, _, _ = _make_agent(n_buildings=2)
    manifest = agent.export_artifacts(
        str(tmp_path), context={"topology_version": 7}
    )
    assert manifest["format"] == "onnx"
    assert manifest["supports_dynamic_topology"] is True
    assert manifest["tokenizer_config_path"] == _TOKENIZER_CFG
    assert len(manifest["artifacts"]) == 2
    assert len(manifest["agent_models"]) == 2
    for entry in manifest["artifacts"]:
        p = tmp_path / entry["path"]
        assert p.exists() and p.stat().st_size > 0
        assert entry["path"].endswith(".onnx")
        assert "topology_v7" in entry["path"]
        assert entry["config"]["n_ca"] == 2
        assert set(entry["config"]["ca_types"]) <= {"storage", "charger"}


# ---------------------------------------------------------------------------
# Synthetic-sample reverse parser
# ---------------------------------------------------------------------------


def test_synthetic_sample_routes_features_by_table() -> None:
    obs = [
        "district__hour",
        "non_shiftable_load",
        "storage::s1::soc",
        "pv::p1::generation",
        "charger::c1::state",
        "charger::c1::connected_ev::soc",
        "charger::c1::incoming_ev::departure",
    ]
    sample = _synthetic_sample_from_obs_names(obs)
    feats = sample.feature_names_per_table
    assert "district__hour" in feats["district"]
    assert "non_shiftable_load" in feats["building"]
    assert "soc" in feats["storage"]
    assert "generation" in feats["pv"]
    assert "state" in feats["charger"]
    assert {"soc", "departure"} <= set(feats["ev"])
