from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.generate_ppo_cc_settlement_templates import generate
from utils.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/experiments/ppo_cc_settlement_annual_v1"
CONFIG_NAMES = (
    "smart_settlement_annual.yaml",
    "cc_smart_settlement_annual_seed123.yaml",
    "ppo_settlement_annual_seed789.yaml",
    "cc_ppo_settlement_annual_seed789.yaml",
)
CHECKPOINT_ROOT = ROOT / "artifacts/frozen_ppo/annual_v1/seed789"


def _load(name: str) -> dict:
    payload = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_settlement_protocol_templates_validate(name: str):
    validate_config(yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8")))


def test_settlement_protocol_common_contract_is_frozen():
    for name in CONFIG_NAMES:
        config = _load(name)
        simulator = config["simulator"]
        market = simulator["community_market"]
        export = simulator["export"]

        assert config["tracking"]["mlflow_enabled"] is False
        assert simulator["dataset_name"] == "citylearn_three_phase_electrical_service_demo_15min_parquet"
        assert simulator["dataset_path"] == (
            "./datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json"
        )
        assert simulator["simulation_start_time_step"] == 0
        assert simulator["simulation_end_time_step"] == 35039
        assert simulator["episode_time_steps"] == 35040
        assert simulator["deterministic_finish"] is True
        assert simulator["reward_function"] == "CCRewardLevel1"
        assert simulator["reward_function_kwargs"]["cost_aggregation"] == "community_net"
        assert market == {
            "enabled": True,
            "local_price_ratio_to_grid_import": 0.8,
            "intra_community_sell_ratio": 0.8,
            "grid_export_price": 0.0,
            "import_member_weights": {},
            "kpis": {
                "community_local_traded_enabled": True,
                "community_self_consumption_enabled": True,
            },
        }
        assert export["mode"] == "end"
        assert export["final_episode_only"] is True
        assert export["kpis_final_episode_only"] is True
        assert export["timeseries_final_episode_only"] is True
        assert export["include_business_as_usual"] is True


def test_neutral_and_learned_pairs_share_the_exact_same_frozen_leaf():
    smart = _load("smart_settlement_annual.yaml")
    cc_smart = _load("cc_smart_settlement_annual_seed123.yaml")
    ppo = _load("ppo_settlement_annual_seed789.yaml")
    cc_ppo = _load("cc_ppo_settlement_annual_seed789.yaml")

    assert smart["pipeline"][1] == cc_smart["pipeline"][1]
    assert ppo["pipeline"][1] == cc_ppo["pipeline"][1]
    assert smart["pipeline"][1]["frozen"] is True
    assert ppo["pipeline"][1]["frozen"] is True

    for neutral in (smart, ppo):
        assert neutral["simulator"]["episodes"] == 1
        assert neutral["pipeline"][0] == {
            "algorithm": "FixedPriceSignal",
            "count": 1,
            "frozen": True,
            "hyperparameters": {"multiplier": 1.0},
        }

    for learned in (cc_smart, cc_ppo):
        manager = learned["pipeline"][0]
        params = manager["hyperparameters"]
        assert learned["simulator"]["episodes"] == 4
        assert manager["algorithm"] == "CCLevel1"
        assert manager["frozen"] is False
        assert params["price_min"] == 0.5
        assert params["price_max"] == 1.5
        assert params["reference_multiplier"] == 1.0
        assert params["policy_residual_scale"] == 1.0
        assert params["cc_action_interval"] == 4
        assert params["bc_collect_steps"] == 8760
        assert params["bc_train_steps"] == 2000


def test_ppo_templates_require_only_the_portable_frozen_checkpoint_state():
    for name in ("ppo_settlement_annual_seed789.yaml", "cc_ppo_settlement_annual_seed789.yaml"):
        checkpointing = _load(name)["checkpointing"]
        assert checkpointing["resume_training"] is True
        assert checkpointing["stage_checkpoint_local_paths"] == {
            1: "./artifacts/frozen_ppo/annual_v1/seed789"
        }
        assert checkpointing["restore_optimizers"] is False
        assert checkpointing["restore_replay_buffer"] is False
        assert checkpointing["restore_exploration_state"] is False


def test_generated_templates_match_the_committed_templates(tmp_path: Path):
    generated = generate(tmp_path)
    assert {path.name for path in generated} == set(CONFIG_NAMES)
    for path in generated:
        committed = CONFIG_ROOT / path.name
        assert yaml.safe_load(path.read_text(encoding="utf-8")) == yaml.safe_load(
            committed.read_text(encoding="utf-8")
        )


def test_compact_seed789_checkpoint_manifest_and_payloads_are_complete():
    manifest = json.loads((CHECKPOINT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "ppo_frozen_inference_v1"
    assert manifest["seed"] == 789
    assert manifest["member_count"] == 17
    assert manifest["required_loader_flags"] == {
        "restore_exploration_state": False,
        "restore_optimizers": False,
        "restore_replay_buffer": False,
    }

    total_bytes = 0
    for index, member in enumerate(manifest["members"]):
        assert member["agent_index"] == index
        path = CHECKPOINT_ROOT / member["path"]
        assert path.is_file()
        assert _sha256(path) == member["sha256"]
        assert path.stat().st_size == member["bytes"]
        total_bytes += path.stat().st_size

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert set(checkpoint) == {
            "checkpoint_format",
            "source_sha256",
            "source_step",
            "actor_state_dict_0",
            "value_state_dict_0",
        }
        assert checkpoint["checkpoint_format"] == "ppo_frozen_inference_v1"
        assert checkpoint["source_sha256"] == member["source_sha256"]
        assert checkpoint["actor_state_dict_0"]
        assert checkpoint["value_state_dict_0"]

    assert total_bytes == manifest["total_bytes"]
