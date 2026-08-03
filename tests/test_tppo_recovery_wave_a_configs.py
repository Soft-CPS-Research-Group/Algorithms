"""Contracts for the Wave A TPPO recovery configurations."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from algorithms.utils.behavior_cloning import BehaviorCloningRegularizer
from utils.config_schema import validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]
WAVE_A_DIR = REPO_ROOT / "configs/recovery/tppo/wave_a"
FILENAMES = (
    "rbc_smart.yaml",
    "rbc_community.yaml",
    "tppo_plain.yaml",
    "tppo_plain_conservative.yaml",
    "tppo_bc_pretrain.yaml",
    "tppo_bc_auxiliary.yaml",
)
COMMON_SIMULATOR = {
    "dataset_name": "citylearn_three_phase_dynamic_assets_only_demo_15min_parquet",
    "dataset_path": "./datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json",
    "interface": "entity",
    "topology_mode": "dynamic",
    "reward_function": "CostHardConstraintReward",
    "reward_function_kwargs": {
        "export_credit_ratio": 0.0,
        "grid_violation_penalty": 60.0,
        "power_outage_penalty": 120.0,
        "ev_departure_window_hours": 1.0,
        "ev_departure_service_tolerance": 0.05,
        "ev_connected_deficit_penalty": 30.0,
        "ev_schedule_deficit_penalty": 120.0,
        "ev_departure_deficit_penalty": 120.0,
        "ev_departure_missed_penalty": 250.0,
        "battery_soc_min": 0.0,
        "battery_soc_max": 1.0,
        "use_observed_storage_soc_limits": True,
        "battery_soc_violation_penalty": 30.0,
        "battery_throughput_penalty": 0.2,
        "community_import_penalty": 0.01,
        "community_peak_import_penalty": 0.001,
        "community_penalty_divide_by_agents": True,
        "scale_state_penalties_by_time_step": True,
        "state_penalty_reference_seconds": 3600.0,
    },
    "simulation_start_time_step": 0,
    "simulation_end_time_step": 35039,
    "episode_time_steps": 35040,
}
SMART_BASELINE = REPO_ROOT / "configs/templates/baselines/rbc_smart_15min_local.yaml"
README_PATH = WAVE_A_DIR / "README.md"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    return {path.name: _load(path) for path in sorted(WAVE_A_DIR.glob("*.yaml"))}


def test_wave_a_contains_the_six_qualified_configurations(configs: dict[str, dict]) -> None:
    assert tuple(sorted(configs)) == tuple(sorted(FILENAMES))


def test_readme_has_blank_required_commit_image_cells_for_each_config() -> None:
    lines = README_PATH.read_text(encoding="utf-8").splitlines()

    assert "| Config path | UI run name | Purpose | Phases | Required commit/image |" in lines
    rows = [line for line in lines if line.startswith("| `")]
    assert len(rows) == len(FILENAMES)
    assert all(row.endswith("|  |") for row in rows)
    assert "Use the final handoff Wave A SHA." in "\n".join(lines)


def test_wave_a_configs_validate_and_share_the_qualified_scenario(configs: dict[str, dict]) -> None:
    for filename, config in configs.items():
        assert config["simulator"]["central_agent"] is False, filename
        assert config["simulator"]["entity_encoding"] == {
            "enabled": True,
            "normalization": "minmax_space",
            "clip": True,
        }, filename
        assert {key: config["simulator"][key] for key in COMMON_SIMULATOR} == COMMON_SIMULATOR, filename
        assert config["simulator"]["export"] == {
            "mode": "end",
            "export_kpis_on_episode_end": True,
            "final_episode_only": True,
            "kpis_final_episode_only": True,
            "timeseries_final_episode_only": True,
            "include_business_as_usual": True,
            "export_business_as_usual_timeseries": False,
        }, filename
        assert config["training"]["seed"] == 7, filename
        assert config["metadata"]["run_name"].startswith("tppo-recovery-wa-"), filename
        assert config["metadata"]["run_name"].endswith("-s7"), filename
        validate_config(config)


def test_rule_based_configs_are_deterministic_smart_policy_controls(configs: dict[str, dict]) -> None:
    smart_baseline = _load(SMART_BASELINE)
    smart = configs["rbc_smart.yaml"]
    community = configs["rbc_community.yaml"]

    for config in (smart, community):
        assert config["simulator"]["episodes"] == 1
        assert config["simulator"]["deterministic_finish"] is True
        assert config["simulator"]["community_market"] == smart_baseline["simulator"]["community_market"]

    assert smart["pipeline"][0]["algorithm"] == "RBCSmartPolicy"
    assert smart["pipeline"][0]["hyperparameters"] == smart_baseline["pipeline"][0]["hyperparameters"]
    assert community["pipeline"][0]["algorithm"] == "RBCCommunityPolicy"
    assert "hyperparameters" not in community["pipeline"][0]


def test_plain_tppo_controls_have_two_deterministic_episodes(configs: dict[str, dict]) -> None:
    plain = configs["tppo_plain.yaml"]
    conservative = configs["tppo_plain_conservative.yaml"]

    for config in (plain, conservative):
        assert config["simulator"]["episodes"] == 2
        assert config["simulator"]["deterministic_finish"] is True
        stage = config["pipeline"][0]
        assert stage["algorithm"] == "AgentTransformerPPO"
        assert stage["transformer"]["dropout"] == pytest.approx(0.0)
        assert stage["hyperparameters"]["require_cuda"] is True
        assert config["training"]["steps_between_training_updates"] == 256
        assert "behavior_cloning" not in stage

    assert plain["pipeline"][0]["hyperparameters"]["actor_log_std_init"] == pytest.approx(-0.5)
    assert conservative["pipeline"][0]["hyperparameters"]["actor_log_std_init"] == pytest.approx(-1.2)
    assert conservative["pipeline"][0]["hyperparameters"] == {
        **plain["pipeline"][0]["hyperparameters"],
        "actor_log_std_init": -1.2,
    }


def test_bc_tppo_configs_have_demo_ppo_eval_phases(configs: dict[str, dict]) -> None:
    pretrain = configs["tppo_bc_pretrain.yaml"]
    auxiliary = configs["tppo_bc_auxiliary.yaml"]

    for config in (pretrain, auxiliary):
        assert config["simulator"]["episodes"] == 3
        assert config["simulator"]["deterministic_finish"] is True
        stage = config["pipeline"][0]
        assert stage["algorithm"] == "AgentTransformerPPO"
        assert stage["transformer"]["dropout"] == pytest.approx(0.0)
        assert stage["hyperparameters"]["require_cuda"] is True
        assert config["training"]["steps_between_training_updates"] == 256
        bc = stage["behavior_cloning"]
        assert bc["enabled"] is True
        assert bc["demonstration_episodes"] == 1
        assert bc["teacher"] == {
            "policy": "RBCSmartPolicy",
            "deterministic": True,
            "hyperparameters": {},
        }

    assert pretrain["pipeline"][0]["behavior_cloning"]["weight"] == pytest.approx(0.0)
    auxiliary_bc = auxiliary["pipeline"][0]["behavior_cloning"]
    assert auxiliary_bc["weight"] > 0.0
    assert auxiliary_bc["min_weight"] == pytest.approx(0.0)
    assert auxiliary_bc["decay_start_step"] == auxiliary["simulator"]["episode_time_steps"]
    ppo_start_step = auxiliary["simulator"]["episode_time_steps"] + 1
    ppo_end_step = auxiliary["simulator"]["episode_time_steps"] * 2
    assert auxiliary_bc["decay_steps"] == auxiliary["simulator"]["episode_time_steps"]
    assert auxiliary_bc["ev_multiplier"] == auxiliary_bc["storage_multiplier"]
    assert auxiliary_bc["ev_multiplier"] != pytest.approx(24.0)

    regularizer = BehaviorCloningRegularizer(
        demonstration_episodes=auxiliary_bc["demonstration_episodes"],
        max_samples_per_building=auxiliary_bc["max_samples_per_building"],
        pretraining_epochs=auxiliary_bc["pretraining_epochs"],
        batch_size=auxiliary_bc["batch_size"],
        weight=auxiliary_bc["weight"],
        min_weight=auxiliary_bc["min_weight"],
        decay_start_step=auxiliary_bc["decay_start_step"],
        decay_steps=auxiliary_bc["decay_steps"],
        ev_multiplier=auxiliary_bc["ev_multiplier"],
        storage_multiplier=auxiliary_bc["storage_multiplier"],
        policy=auxiliary_bc["teacher"]["policy"],
        deterministic=auxiliary_bc["teacher"]["deterministic"],
        hyperparameters=auxiliary_bc["teacher"]["hyperparameters"],
        agent_config_template={},
        config_dict=auxiliary_bc,
    )
    assert regularizer.effective_weight(ppo_start_step) == pytest.approx(
        auxiliary_bc["weight"],
        abs=auxiliary_bc["weight"] / auxiliary_bc["decay_steps"],
    )
    assert regularizer.effective_weight(ppo_end_step) == pytest.approx(0.0)
