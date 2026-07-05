"""Template and smoke tests for Transformer-PPO behavior cloning."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from tests.test_agent_transformer_ppo_wrapper_integration import (
    _DummyEntityEnvForPPO,
)
from tests.test_wrapper_entity_mode import _entity_config
from utils.wrapper_citylearn import Wrapper_CityLearn

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = (
    REPO_ROOT / "configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml"
)
DOC_PATH = REPO_ROOT / "docs/transformer_ppo_spec.md"
AGENTS_PATH = REPO_ROOT / "AGENTS.md"
_TOKENIZER_FIXTURE = "tests/fixtures/tokenizer_dummy_env.json"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bc_ppo_algo_config() -> Dict[str, Any]:
    return {
        "name": "AgentTransformerPPO",
        "tokenizer_config_path": _TOKENIZER_FIXTURE,
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
        "behavior_cloning": {
            "enabled": True,
            "weight": 0.42,
            "min_weight": 0.24,
            "decay_start_step": 512,
            "decay_steps": 3584,
            "ev_multiplier": 24.0,
            "storage_multiplier": 0.18,
            "warm_start": {
                "policy": "RBCCommunityPolicy",
                "deterministic": True,
                "noise_scale": 0.0,
                "phaseout_steps": 6144,
                "phaseout_mode": "blend",
                "hyperparameters": {},
            },
        },
    }


def _bc_full_config() -> Dict[str, Any]:
    return {"algorithm": _bc_ppo_algo_config(), "training": {"seed": 7}}


def _rollout_transition(
    wrapper: Wrapper_CityLearn,
    agent: AgentTransformerPPO,
    observations: List[np.ndarray],
    *,
    step: int,
    update_step: bool,
) -> None:
    actions = wrapper.predict(observations, deterministic=False)
    next_observations = [
        np.asarray(obs, dtype=np.float64) + 0.01 for obs in observations
    ]
    agent.update(
        observations=observations,
        actions=[np.asarray(row, dtype=np.float64) for row in actions],
        rewards=[0.1 for _ in observations],
        next_observations=next_observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=step,
        update_step=update_step,
        initial_exploration_done=True,
    )


def test_template_passes_schema_validation_and_resolves_bc_block() -> None:
    from utils.config_schema import validate_config

    cfg = _load_template()
    resolved = validate_config(cfg)
    stage = resolved.pipeline[0]

    assert cfg["pipeline"][0]["algorithm"] == "AgentTransformerPPO"
    assert "bc" in cfg["metadata"]["experiment_name"].lower()
    assert "bc" in cfg["metadata"]["run_name"].lower()
    assert stage.behavior_cloning is not None
    assert stage.behavior_cloning.enabled is True
    assert stage.behavior_cloning.warm_start is not None
    assert stage.behavior_cloning.warm_start.policy == "RBCCommunityPolicy"
    assert stage.behavior_cloning.warm_start.phaseout_mode == "blend"
    assert stage.behavior_cloning.ev_multiplier == pytest.approx(24.0)
    assert stage.behavior_cloning.storage_multiplier == pytest.approx(0.18)


def test_transformer_ppo_bc_smoke_records_bc_metrics_across_topology_change() -> None:
    env = _DummyEntityEnvForPPO()
    cfg = _entity_config()
    cfg["pipeline"] = [
        {"algorithm": "AgentTransformerPPO", "count": 1, "hyperparameters": {}}
    ]
    wrapper = Wrapper_CityLearn(env=env, config=cfg, job_id="ppo-bc-entity-smoke")
    agent = AgentTransformerPPO(_bc_full_config())
    wrapper.set_model(agent)

    observations = wrapper._apply_entity_layout(
        env._observation_payload(version=0), force_attach=False
    )
    initial_topology_version = wrapper._entity_topology_version
    initial_total_action_count = sum(
        state.layout.n_ca for state in agent._per_building
    )

    for step in range(agent._minibatch_size):
        _rollout_transition(
            wrapper,
            agent,
            observations,
            step=step,
            update_step=step == agent._minibatch_size - 1,
        )

    assert agent._bc is not None
    diagnostics = agent.get_diagnostic_metrics()
    training_metrics = agent.consume_latest_training_metrics()
    assert diagnostics["behavior_cloning_teacher_enabled"] == pytest.approx(1.0)
    assert diagnostics["behavior_cloning_latest_teacher_available"] == pytest.approx(1.0)
    assert diagnostics["behavior_cloning_phaseout_probability"] > 0.0
    assert diagnostics["behavior_cloning_phaseout_used"] == pytest.approx(1.0)
    assert training_metrics["behavior_cloning_effective_weight"] > 0.0
    assert np.isfinite(training_metrics["behavior_cloning_effective_weight"])
    assert "behavior_cloning_loss" in training_metrics
    assert "behavior_cloning_valid_samples" in training_metrics
    assert training_metrics["behavior_cloning_valid_samples"] > 0.0

    env._version = 1
    changed_observations = wrapper._apply_entity_layout(
        env._observation_payload(version=1),
        force_attach=False,
    )

    assert wrapper._entity_topology_version == initial_topology_version + 1
    assert len(changed_observations) == 2
    assert len(agent._per_building) == 2
    assert sum(state.layout.n_ca for state in agent._per_building) != (
        initial_total_action_count
    )
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [0, 0]

    _rollout_transition(
        wrapper,
        agent,
        changed_observations,
        step=agent._minibatch_size,
        update_step=False,
    )
    assert [len(buffer) for buffer in agent._bc.teacher_action_buffers] == [1, 1]


def test_docs_describe_transformer_ppo_behavior_cloning() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "## 13. Behavior Cloning" in text
    assert "RBCCommunityPolicy" in text
    assert "phaseout" in text
    assert "ev_multiplier" in text
    assert "storage_multiplier" in text
    assert "deferred residual policy" in text


def test_agents_doc_mentions_transformer_ppo_bc_support() -> None:
    text = AGENTS_PATH.read_text(encoding="utf-8")

    assert "AgentTransformerPPO" in text
    assert "behavior cloning" in text
    assert "warm-start" in text
