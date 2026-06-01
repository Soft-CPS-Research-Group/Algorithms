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

def test_episode_steps_for_schema_hourly():
    steps = episode_steps_for_schema(HOURLY_SCHEMA)
    assert steps == 24  # 86400 // 3600

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
