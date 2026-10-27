# tests/offline_rl/test_entity_schema_utils.py
import json
from pathlib import Path
import pytest
from algorithms.offline_rl.entity_schema import (
    episode_steps_for_schema,
    AgentGroupSpec,
    buildings_to_group_keys,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY_SCHEMA = REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo" / "schema.json"
PARQUET_SCHEMA = REPO_ROOT / "datasets" / "citylearn_three_phase_electrical_service_demo_15s_parquet" / "schema.json"

@pytest.mark.skipif(not HOURLY_SCHEMA.exists(), reason="hourly dataset not present")
def test_episode_steps_for_schema_hourly():
    steps = episode_steps_for_schema(HOURLY_SCHEMA)
    assert steps == 24  # 86400 // 3600

@pytest.mark.skipif(not PARQUET_SCHEMA.exists(), reason="15s parquet dataset not present")
def test_episode_steps_for_schema_15s():
    steps = episode_steps_for_schema(PARQUET_SCHEMA)
    assert steps == 5760  # 86400 // 15

def test_episode_steps_for_schema_missing_key(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({}))
    # Missing key → default 3600 s → 24 steps
    assert episode_steps_for_schema(schema) == 24

def test_buildings_to_group_keys_match():
    groups = [
        AgentGroupSpec(obs_dim=42, action_dim=1, buildings=["B1", "B2"]),
        AgentGroupSpec(obs_dim=50, action_dim=2, buildings=["B3"]),
    ]
    result = buildings_to_group_keys(["B3"], groups)
    assert result == ["obs50_act2"]

def test_buildings_to_group_keys_no_match():
    groups = [AgentGroupSpec(obs_dim=42, action_dim=1, buildings=["B1"])]
    assert buildings_to_group_keys(["B99"], groups) == []

def test_buildings_to_group_keys_all():
    groups = [
        AgentGroupSpec(obs_dim=42, action_dim=1, buildings=["B1"]),
        AgentGroupSpec(obs_dim=50, action_dim=2, buildings=["B2"]),
    ]
    result = buildings_to_group_keys(["B1", "B2"], groups)
    assert set(result) == {"obs42_act1", "obs50_act2"}

def test_agentgroupspec_has_buildings_field():
    g = AgentGroupSpec(obs_dim=10, action_dim=1, buildings=["X"])
    assert g.buildings == ["X"]
    # Backward compat: default empty
    g2 = AgentGroupSpec(obs_dim=10, action_dim=1)
    assert g2.buildings == []


# ---------------------------------------------------------------------------
# probe_agent_groups: end-to-end (instantiates CityLearn, unpacks env.reset)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HOURLY_SCHEMA.exists(), reason="hourly dataset not present")
def test_probe_agent_groups_returns_non_empty_specs():
    """probe_agent_groups must instantiate CityLearn, reset, and return AgentGroupSpec list.

    Regression guard for the bug where probe_agent_groups treated env.reset()'s
    (payload, info) tuple as the payload directly, crashing with
    ``AttributeError: 'tuple' object has no attribute 'get'`` inside
    EntityContractAdapter.to_agent_observations.
    """
    from algorithms.offline_rl.entity_schema import probe_agent_groups

    groups = probe_agent_groups(HOURLY_SCHEMA)
    assert len(groups) > 0, "expected at least one agent group from hourly schema"
    for g in groups:
        assert g.obs_dim > 0, f"obs_dim must be positive: {g}"
        assert g.action_dim > 0, f"action_dim must be positive: {g}"
        assert len(g.buildings) > 0, f"each group must list buildings: {g}"


@pytest.mark.skipif(not HOURLY_SCHEMA.exists(), reason="hourly dataset not present")
def test_probe_agent_groups_group_keys_unique():
    """No two AgentGroupSpec objects share the same (obs_dim, action_dim)."""
    from algorithms.offline_rl.entity_schema import probe_agent_groups

    groups = probe_agent_groups(HOURLY_SCHEMA)
    keys = [(g.obs_dim, g.action_dim) for g in groups]
    assert len(keys) == len(set(keys)), f"duplicate group keys: {keys}"
