"""Integration of AgentTransformerPPO with Wrapper_CityLearn
over the entity interface in dynamic-topology mode.

Reuses the dummy entity env from ``tests/test_wrapper_entity_mode.py`` but
overrides ``action_names`` so the per-building action list uses the bare
``action_field`` (matching the layout-builder contract). A purpose-built
tokenizer config under ``tests/fixtures/tokenizer_dummy_env.json`` matches
the dummy env's feature schema.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from gymnasium import spaces

from algorithms.transformer_ppo.agent import AgentTransformerPPO
from tests.test_wrapper_entity_mode import _DummyEntityEnv, _entity_config
from utils.wrapper_citylearn import Wrapper_CityLearn


_TOKENIZER_FIXTURE = "tests/fixtures/tokenizer_dummy_env.json"


class _DummyEntityEnvForPPO(_DummyEntityEnv):
    """Dummy env whose action_names use bare ``action_field`` strings.

    The base test fixture suffixes charger IDs onto the action field
    (``electric_vehicle_storage_C1``); the v2 layout builder matches
    action_field exactly, so we strip the suffix here.
    """

    @property
    def action_names(self) -> List[List[str]]:  # type: ignore[override]
        if self._version == 0:
            return [["electrical_storage", "electric_vehicle_storage"]]
        return [
            ["electrical_storage", "electric_vehicle_storage"],
            ["electrical_storage", "electric_vehicle_storage"],
        ]

    @property
    def flat_action_space(self) -> List[spaces.Box]:  # type: ignore[override]
        building_count = 1 if self._version == 0 else 2
        return [
            spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for _ in range(building_count)
        ]

class _TerminalTopologyChangeEntityEnvForPPO(_DummyEntityEnvForPPO):
    """Changes topology on the terminal transition of a two-step episode."""

    def __init__(self, *, truncated: bool = False) -> None:
        super().__init__()
        self._steps = 0
        self._truncated = truncated

    def reset(self):
        self._version = 0
        self._steps = 0
        return self._observation_payload(version=0), {}

    def step(self, _actions):
        self._steps += 1
        if self._steps == 2:
            self._version = 1
            return (
                self._observation_payload(version=1),
                [0.1],
                not self._truncated,
                self._truncated,
                {},
            )
        return self._observation_payload(version=0), [0.1], False, False, {}


def _ppo_algo_config() -> Dict[str, Any]:
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
    }


def _ppo_full_config() -> Dict[str, Any]:
    """Wrapper-shape config (the agent constructor expects ``cfg["algorithm"]``)."""
    return {"algorithm": _ppo_algo_config()}


def test_wrapper_attaches_transformer_ppo_with_entity_dynamic() -> None:
    """The dynamic-topology guardrail must accept ``AgentTransformerPPO``
    (it has ``supports_dynamic_topology=True``) and ``set_model`` must
    drive a single ``attach_environment`` call."""
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-entity"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    # One per-building stack initialised at version 0.
    assert len(agent._per_building) == 1
    state = agent._per_building[0]
    assert state.layout.n_ca == 2  # storage + charger


def test_wrapper_predict_returns_per_building_per_ca_actions() -> None:
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-entity-predict"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    payload = env._observation_payload(version=0)
    adapted = wrapper._apply_entity_layout(payload, force_attach=False)
    assert isinstance(adapted, list) and len(adapted) == 1

    actions = agent.predict(adapted, deterministic=True)
    assert len(actions) == 1
    assert len(actions[0]) == 2  # storage + charger CA
    for v in actions[0]:
        assert -1.0 <= v <= 1.0


@pytest.mark.parametrize("truncated", [False, True], ids=["terminal", "truncated"])
def test_demo_topology_transition_rejects_building_without_demonstrations(
    monkeypatch: pytest.MonkeyPatch,
    truncated: bool,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=truncated)
    wrapper_config = _entity_config()
    wrapper_config["training"]["steps_between_training_updates"] = 4
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-entity-terminal-topology"
    )
    agent_config = deepcopy(_ppo_full_config())
    agent_config["algorithm"]["hyperparameters"]["minibatch_size"] = 2
    agent_config["algorithm"]["behavior_cloning"] = {
        "enabled": True,
        "demonstration_episodes": 1,
        "max_samples_per_building": 8,
        "pretraining_epochs": 2,
        "batch_size": 2,
        "weight": 0.4,
        "min_weight": 0.1,
        "decay_start_step": 0,
        "decay_steps": 100,
        "ev_multiplier": 2.0,
        "storage_multiplier": 1.0,
        "teacher": {
            "policy": "RBCSmartPolicy",
            "deterministic": True,
            "hyperparameters": {},
        },
    }
    agent = AgentTransformerPPO(agent_config)
    wrapper.set_model(agent)
    assert agent._bc is not None

    ppo_updates: list[tuple[int, int, int, torch.Tensor]] = []
    original_update = agent._run_ppo_update_with_last_value

    def record_ppo_update(state, last_value, *, building_idx):
        ppo_updates.append(
            (agent._current_episode, building_idx, len(state.buffer), last_value.detach().clone())
        )
        return original_update(state, last_value, building_idx=building_idx)

    monkeypatch.setattr(agent, "_run_ppo_update_with_last_value", record_ppo_update)
    pretraining_calls = 0
    pretraining_metrics: list[dict[str, float]] = []
    original_pretraining = agent._run_bc_pretraining

    def record_pretraining() -> None:
        nonlocal pretraining_calls
        pretraining_calls += 1
        original_pretraining()
        pretraining_metrics.append(agent._bc.snapshot_metrics())

    monkeypatch.setattr(agent, "_run_bc_pretraining", record_pretraining)
    teacher_calls: list[int] = []

    def teacher_actions(observations):
        teacher_calls.append(agent._current_episode)
        return [
            [0.9] * state.layout.n_ca
            for state in agent._per_building[: len(observations)]
        ]

    agent._bc.compute_teacher_actions = teacher_actions
    with pytest.raises(RuntimeError, match=r"zero compatible demonstrations.*B2"):
        wrapper.learn(episodes=2, deterministic=False)

    assert pretraining_calls == 1
    assert teacher_calls == [0, 0]
    assert pretraining_metrics == []
    assert ppo_updates == []
    assert len(agent._per_building) == 2
    assert agent._bc.demonstration_count(0) == 2
    assert agent._bc.demonstration_count(1) == 0


def test_wrapper_topology_change_triggers_agent_rebuild() -> None:
    """Bump ``_version`` to add a second building; the wrapper re-attaches
    on the next ``_apply_entity_layout``, and the agent rebuilds its stacks
    accordingly."""
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-entity-topo"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)
    assert len(agent._per_building) == 1

    env._version = 1
    new_payload = env._observation_payload(version=1)
    adapted = wrapper._apply_entity_layout(new_payload, force_attach=False)
    assert len(adapted) == 2
    assert len(agent._per_building) == 2
    for state in agent._per_building:
        assert state.layout.n_ca == 2


def test_learn_rolls_back_wrapper_and_agent_when_deferred_attach_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO()
    wrapper_config = _entity_config()
    wrapper_config["training"]["steps_between_training_updates"] = 4
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-entity-rollback"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)
    original_attach = agent.attach_environment

    def fail_after_new_topology_is_attached(**kwargs):
        original_attach(**kwargs)
        if len(kwargs["observation_names"]) == 2:
            raise RuntimeError("deferred reattachment failed")

    monkeypatch.setattr(agent, "attach_environment", fail_after_new_topology_is_attached)

    with pytest.raises(RuntimeError, match="deferred reattachment failed"):
        wrapper.learn(episodes=1)

    assert wrapper._entity_topology_version == 0
    assert wrapper._entity_adapter is not None
    assert wrapper._entity_adapter.topology_version == 0
    assert len(wrapper.action_names) == 1
    assert len(agent._per_building) == 1
    assert agent._per_building[0].topology_version == 0
    assert len(agent._per_building[0].buffer) == 1
    assert agent._per_building[0].raw_rewards == [0.1]
    assert agent._pending_decisions[0] is not None

    monkeypatch.setattr(agent, "attach_environment", original_attach)
    adapted = wrapper._apply_entity_layout(
        env._observation_payload(version=1), force_attach=False
    )
    assert len(adapted) == 2
    assert len(agent._per_building) == 2


def test_learn_rolls_back_wrapper_when_agent_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO()
    wrapper_config = _entity_config()
    wrapper_config["training"]["steps_between_training_updates"] = 4
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-entity-snapshot-rollback"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    def fail_snapshot():
        raise RuntimeError("topology snapshot failed")

    monkeypatch.setattr(agent, "snapshot_topology_state", fail_snapshot)

    with pytest.raises(RuntimeError, match="topology snapshot failed"):
        wrapper.learn(episodes=1)

    assert wrapper._entity_topology_version == 0
    assert wrapper._entity_adapter is not None
    assert wrapper._entity_adapter.topology_version == 0
    assert len(wrapper.action_names) == 1
    assert len(agent._per_building) == 1
    assert len(agent._per_building[0].buffer) == 0
    assert agent._per_building[0].raw_rewards == []
    assert agent._pending_decisions[0] is None


def test_wrapper_to_env_actions_round_trips_ppo_output() -> None:
    """``predict`` -> ``_to_env_actions`` produces the entity-tabled action
    payload the simulator expects."""
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-entity-actions"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    payload = env._observation_payload(version=0)
    adapted = wrapper._apply_entity_layout(payload, force_attach=False)
    actions = agent.predict(adapted, deterministic=True)

    env_payload = wrapper._to_env_actions(actions)
    assert "tables" in env_payload
    # storage CA -> building action table; charger CA -> charger action table.
    assert env_payload["tables"]["building"].shape == (1, 1)
    assert env_payload["tables"]["charger"].shape == (1, 1)


def test_non_dynamic_agent_in_entity_dynamic_still_rejected_on_topology_change() -> None:
    """The flag-based guardrail must reject non-dynamic agents when the
    topology actually mutates."""

    class _NonDynamicModel:
        supports_dynamic_topology = False
        use_raw_observations = True

        def attach_environment(self, **_kwargs):
            pass

        def predict(self, observations, deterministic=None):
            return [[0.0, 0.0] for _ in observations]

        def update(self, **_kwargs):
            pass

        def is_initial_exploration_done(self, _):
            return True

    env = _DummyEntityEnvForPPO()
    cfg = _entity_config()
    cfg["pipeline"] = [{"algorithm": "MADDPG", "count": 1, "hyperparameters": {}}]
    wrapper = Wrapper_CityLearn(env=env, config=cfg, job_id="ppo-entity-guard")
    wrapper.set_model(_NonDynamicModel())

    env._version = 1
    with pytest.raises(ValueError, match=r"MADDPG|dynamic"):
        wrapper._apply_entity_layout(
            env._observation_payload(version=1), force_attach=False
        )
