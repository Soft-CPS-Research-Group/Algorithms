"""Contracts for the local Transformer-PPO demonstration template."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from algorithms.registry import build_execution_unit
from tests.test_agent_transformer_ppo_wrapper_integration import _DummyEntityEnvForPPO
from tests.test_wrapper_entity_mode import _entity_config
from utils.config_schema import validate_config
from utils.wrapper_citylearn import Wrapper_CityLearn

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml"
)
_TOKENIZER_FIXTURE = "tests/fixtures/tokenizer_dummy_env.json"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _assert_no_legacy_bc_fields(value: object) -> None:
    legacy_fields = {
        "phaseout_steps",
        "phaseout_mode",
        "noise_scale",
        "warm_start",
    }
    if isinstance(value, dict):
        assert not legacy_fields.intersection(value)
        for child in value.values():
            _assert_no_legacy_bc_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_legacy_bc_fields(child)


def test_local_bc_template_uses_demonstrations_without_action_blending() -> None:
    config = _load_template()
    stage = config["pipeline"][0]
    transformer = stage["transformer"]
    hyperparameters = stage["hyperparameters"]
    behavior_cloning = stage["behavior_cloning"]

    assert transformer["dropout"] == pytest.approx(0.0)
    assert config["training"]["steps_between_training_updates"] == 256
    assert config["training"]["steps_between_training_updates"] >= hyperparameters["minibatch_size"]
    assert "actor_log_std_init" in hyperparameters
    assert behavior_cloning["enabled"] is True
    assert behavior_cloning["demonstration_episodes"] >= 1
    assert behavior_cloning["max_samples_per_building"] == 3400
    assert behavior_cloning["pretraining_epochs"] >= 1
    assert behavior_cloning["batch_size"] >= 1
    assert behavior_cloning["weight"] == pytest.approx(0.42)
    assert behavior_cloning["min_weight"] == pytest.approx(0.24)
    assert behavior_cloning["decay_start_step"] == 512
    assert behavior_cloning["decay_steps"] == 3584
    assert behavior_cloning["ev_multiplier"] == pytest.approx(24.0)
    assert behavior_cloning["storage_multiplier"] == pytest.approx(0.18)
    assert behavior_cloning["teacher"] == {
        "policy": "RBCSmartPolicy",
        "deterministic": True,
        "hyperparameters": {},
    }
    _assert_no_legacy_bc_fields(behavior_cloning)

    validate_config(config)


def test_dynamic_bc_template_preserves_layout_compatible_demonstrations() -> None:
    config = _load_template()
    stage = config["pipeline"][0]
    stage["tokenizer_config_path"] = _TOKENIZER_FIXTURE
    stage["transformer"] = {
        "d_model": 16,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
    }
    stage["hyperparameters"].update(
        {"minibatch_size": 4, "actor_hidden_dim": 32, "critic_hidden_dim": 32}
    )
    agent = build_execution_unit(config)
    assert isinstance(agent, AgentTransformerPPO)
    assert agent._bc is not None

    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(env=env, config=_entity_config(), job_id="tppo-bc-template-topology")
    wrapper.set_model(agent)
    observations = wrapper._apply_entity_layout(env._observation_payload(version=0), force_attach=False)
    agent._bc.record_demonstration(0, observations[0], agent._per_building[0].layout, [0.0, 0.0])

    env._version = 1
    wrapper._apply_entity_layout(env._observation_payload(version=1), force_attach=False)

    assert agent._bc.demonstration_count(0) == 1
    assert len(agent._bc.sample_demonstrations(0, agent._per_building[0].layout, batch_size=1)) == 1
    assert agent._bc.demonstration_count(1) == 0
