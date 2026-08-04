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
import json
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from gymnasium import spaces

from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
from tests.test_wrapper_entity_mode import _DummyEntityEnv, _entity_config
from utils import wrapper_citylearn as wrapper_module
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
        return [
            spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for _ in self._building_ids(self._version)
        ]


class _PositiveOnlyChargerEntityEnvForPPO(_DummyEntityEnvForPPO):
    @property
    def flat_action_space(self) -> List[spaces.Box]:  # type: ignore[override]
        return [
            spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for _ in self._building_ids(self._version)
        ]


class _TerminalTopologyChangeEntityEnvForPPO(_DummyEntityEnvForPPO):
    """Changes topology on the terminal transition of a two-step episode."""

    def __init__(self, *, truncated: bool) -> None:
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


def test_watchdog_progress_brackets_tppo_lifecycle_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper_config = _entity_config()
    wrapper_config["runtime"]["log_dir"] = str(tmp_path)
    wrapper_config["tracking"].update(
        {
            "progress_updates_enabled": False,
            "stall_watchdog_enabled": True,
            "stall_watchdog_timeout_seconds": 1.0,
            "stall_watchdog_exit_on_timeout": False,
        }
    )
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-lifecycle-watchdog"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    events: list[tuple[str, str | None]] = []
    context_path = tmp_path / "ppo-lifecycle-watchdog_stall_watchdog.log.context.json"
    original_watchdog_update = wrapper._update_stall_watchdog_for_phase
    original_start = agent.on_episode_start
    original_end = agent.on_episode_end

    def record_watchdog_update(*, phase: str, **kwargs: Any) -> None:
        events.append(("watchdog_phase", phase))
        original_watchdog_update(phase=phase, **kwargs)

    def record_arm(_timeout, *, repeat=False, file=None, exit=False) -> None:
        _ = repeat, file, exit
        events.append(("watchdog_arm", json.loads(context_path.read_text())["phase"]))

    def record_cancel() -> None:
        events.append(("watchdog_cancel", wrapper._stall_watchdog_armed_phase))

    def record_start(*, episode: int, training: bool) -> None:
        events.append(("callback", "episode_start"))
        original_start(episode=episode, training=training)

    def record_end(*, episode: int, training: bool) -> None:
        events.append(("callback", "episode_end"))
        original_end(episode=episode, training=training)

    monkeypatch.setattr(wrapper, "_update_stall_watchdog_for_phase", record_watchdog_update)
    monkeypatch.setattr(wrapper_module.faulthandler, "dump_traceback_later", record_arm)
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", record_cancel)
    monkeypatch.setattr(agent, "on_episode_start", record_start)
    monkeypatch.setattr(agent, "on_episode_end", record_end)

    wrapper.learn(episodes=1, deterministic=False)

    for callback in ("episode_start", "episode_end"):
        phase_prefix = f"{callback}_callback"
        expected = [
            ("watchdog_phase", f"{phase_prefix}_start"),
            ("watchdog_arm", f"{phase_prefix}_start"),
            ("callback", callback),
            ("watchdog_phase", f"{phase_prefix}_end"),
            ("watchdog_cancel", f"{phase_prefix}_start"),
        ]
        positions = []
        next_position = 0
        for event in expected:
            next_position = events.index(event, next_position)
            positions.append(next_position)
            next_position += 1
        assert positions == sorted(positions)


def test_watchdog_brackets_out_of_loop_model_attachment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper_config = _entity_config()
    wrapper_config["runtime"]["log_dir"] = str(tmp_path)
    wrapper_config["tracking"].update(
        {
            "progress_updates_enabled": False,
            "stall_watchdog_enabled": True,
            "stall_watchdog_timeout_seconds": 1.0,
            "stall_watchdog_exit_on_timeout": False,
        }
    )
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-model-attach-watchdog"
    )
    agent = AgentTransformerPPO(_ppo_full_config())

    attachment_events: list[tuple[str, str]] = []
    watchdog_phases: list[str] = []
    attachment_phases: list[str | None] = []
    original_write_phase_progress = wrapper._write_phase_progress
    original_watchdog_update = wrapper._update_stall_watchdog_for_phase
    original_attach_environment = agent.attach_environment

    def collect_phase(*, phase: str, extra=None, **kwargs: Any) -> None:
        source = (extra or {}).get("attach_source")
        if source is not None:
            attachment_events.append(("phase", f"{phase}:{source}"))
        original_write_phase_progress(phase=phase, extra=extra, **kwargs)

    def collect_watchdog_phase(*, phase: str, **kwargs: Any) -> None:
        watchdog_phases.append(phase)
        original_watchdog_update(phase=phase, **kwargs)

    def collect_attach_environment(**kwargs: Any) -> None:
        attachment_phases.append(wrapper._stall_watchdog_armed_phase)
        if wrapper._stall_watchdog_armed_phase == "model_attach_start":
            attachment_events.append(("attach", "model_attach_start"))
        original_attach_environment(**kwargs)

    monkeypatch.setattr(wrapper, "_write_phase_progress", collect_phase)
    monkeypatch.setattr(wrapper, "_update_stall_watchdog_for_phase", collect_watchdog_phase)
    monkeypatch.setattr(agent, "attach_environment", collect_attach_environment)
    monkeypatch.setattr(wrapper_module.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", lambda: None)

    wrapper.set_model(agent)
    wrapper.learn(episodes=1, deterministic=False)

    assert attachment_events == [
        ("phase", "model_attach_start:set_model"),
        ("attach", "model_attach_start"),
        ("phase", "model_attach_end:set_model"),
        ("phase", "model_attach_start:topology_refresh"),
        ("attach", "model_attach_start"),
        ("phase", "model_attach_end:topology_refresh"),
    ]
    assert attachment_phases == ["model_attach_start", "episode_reset_start", "model_attach_start"]
    assert watchdog_phases.count("model_attach_start") == 2
    assert watchdog_phases.count("model_attach_end") == 2
    assert len(agent._per_building) == 2


def test_watchdog_keeps_start_phase_when_tppo_lifecycle_callback_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper_config = _entity_config()
    wrapper_config["runtime"]["log_dir"] = str(tmp_path)
    wrapper_config["tracking"].update(
        {
            "progress_updates_enabled": False,
            "stall_watchdog_enabled": True,
            "stall_watchdog_timeout_seconds": 1.0,
            "stall_watchdog_exit_on_timeout": False,
        }
    )
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-lifecycle-watchdog-error"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    phases: list[str] = []
    armed_phases: list[str] = []
    context_path = tmp_path / "ppo-lifecycle-watchdog-error_stall_watchdog.log.context.json"
    original_watchdog_update = wrapper._update_stall_watchdog_for_phase

    def record_watchdog_update(*, phase: str, **kwargs: Any) -> None:
        phases.append(phase)
        original_watchdog_update(phase=phase, **kwargs)

    def record_arm(_timeout, *, repeat=False, file=None, exit=False) -> None:
        _ = repeat, file, exit
        armed_phases.append(json.loads(context_path.read_text())["phase"])

    def raise_on_start(*, episode: int, training: bool) -> None:
        _ = episode, training
        raise RuntimeError("lifecycle start failure")

    monkeypatch.setattr(wrapper, "_update_stall_watchdog_for_phase", record_watchdog_update)
    monkeypatch.setattr(wrapper_module.faulthandler, "dump_traceback_later", record_arm)
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(agent, "on_episode_start", raise_on_start)

    with pytest.raises(RuntimeError, match="lifecycle start failure"):
        wrapper.learn(episodes=1, deterministic=False)

    assert "episode_start_callback_start" in phases
    assert "episode_start_callback_end" not in phases
    assert "episode_start_callback_start" in armed_phases
    assert wrapper._stall_watchdog_armed_phase == "episode_start_callback_start"


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


def test_topology_transition_records_encoded_observation_matching_tppo_pending_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper_config = _entity_config()
    wrapper_config["training"]["steps_between_training_updates"] = 4
    wrapper = Wrapper_CityLearn(
        env=env, config=wrapper_config, job_id="ppo-entity-encoded-transition"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    raw_observations = wrapper._apply_entity_layout(
        env._observation_payload(version=0), force_attach=False
    )
    encoded_observations = wrapper._encode_observations_for_model(raw_observations)
    assert not np.array_equal(encoded_observations[0], raw_observations[0])

    recorded_observations: list[np.ndarray] = []
    original_record = agent.record_topology_transition

    def record_transition(*, observations, **kwargs) -> None:
        pending = agent._pending_decisions[0]
        assert pending is not None
        np.testing.assert_allclose(
            observations[0], pending.observation.detach().cpu().numpy()
        )
        recorded_observations.extend(observations)
        original_record(observations=observations, **kwargs)

    monkeypatch.setattr(agent, "record_topology_transition", record_transition)

    wrapper.learn(episodes=1, deterministic=False)

    assert len(recorded_observations) == 1
    np.testing.assert_allclose(recorded_observations[0], encoded_observations[0])


def test_topology_transition_preserves_raw_observations_for_raw_unit() -> None:
    class _RawTopologyProbe:
        supports_dynamic_topology = True
        use_raw_observations = True

        def __init__(self) -> None:
            self.predict_observations: list[list[np.ndarray]] = []
            self.transition_observations: list[list[np.ndarray]] = []

        def attach_environment(self, **_kwargs) -> None:
            pass

        def predict(self, observations, deterministic=None):
            _ = deterministic
            self.predict_observations.append(
                [np.asarray(obs, dtype=np.float64).copy() for obs in observations]
            )
            return [[0.0, 0.0] for _ in observations]

        def update(self, **_kwargs) -> None:
            pass

        def record_topology_transition(self, *, observations, **_kwargs) -> None:
            self.transition_observations.append(
                [np.asarray(obs, dtype=np.float64).copy() for obs in observations]
            )

        def is_initial_exploration_done(self, _global_step: int) -> bool:
            return True

    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="raw-entity-transition"
    )
    model = _RawTopologyProbe()
    wrapper.set_model(model)

    wrapper.learn(episodes=1, deterministic=False)

    assert len(model.transition_observations) == 1
    raw_observations = model.predict_observations[-1]
    np.testing.assert_allclose(model.transition_observations[0][0], raw_observations[0])
    assert not np.array_equal(
        wrapper.get_all_encoded_observations(raw_observations)[0], raw_observations[0]
    )


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

    monkeypatch.setattr(
        agent, "_run_ppo_update_with_last_value", record_ppo_update
    )
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


def test_wrapper_clipped_float64_action_round_trip_validates_exact_ppo_decision() -> None:
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-entity-action-validation"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)

    observations = wrapper._apply_entity_layout(
        env._observation_payload(version=0), force_attach=False
    )
    actions = agent.predict(observations, deterministic=True)
    clipped_actions = wrapper._clip_actions(actions)
    env_actions = wrapper._to_env_actions(clipped_actions)
    received_actions = [
        [
            env_actions["tables"]["building"][0, 0],
            env_actions["tables"]["charger"][0, 0],
        ]
    ]

    assert np.asarray(received_actions[0]).dtype == np.float32
    agent.update(
        observations=observations,
        actions=[np.asarray(received_actions[0], dtype=np.float64)],
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    altered_actions = [list(received_actions[0])]
    altered_actions[0][0] += 5.0e-7
    agent.predict(observations, deterministic=True)
    with pytest.raises(ValueError, match="does not match the pending TPPO action"):
        agent.update(
            observations=observations,
            actions=[np.asarray(altered_actions[0], dtype=np.float64)],
            rewards=[0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=1,
            update_step=False,
            initial_exploration_done=True,
        )


def test_positive_only_charger_action_is_cached_as_the_executed_ppo_decision() -> None:
    env = _PositiveOnlyChargerEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env, config=_entity_config(), job_id="ppo-positive-only-action"
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)
    state = agent._per_building[0]

    # Force the actor's deterministic signed sample below zero for both CAs.
    for parameter in state.actor.parameters():
        parameter.data.zero_()
    state.actor.mlp[-1].bias.data.fill_(-1.0)

    observations = wrapper._apply_entity_layout(
        env._observation_payload(version=0), force_attach=False
    )
    actions = agent.predict(observations, deterministic=True)
    pending = agent._pending_decisions[0]
    assert pending is not None

    # The raw signed actor output is negative, but affine mapping puts the
    # positive-only charger action inside its executable interval.
    assert actions[0][0] < 0.0
    assert actions[0][1] == pytest.approx((np.tanh(-1.0) + 1.0) / 2.0)
    assert torch.equal(
        pending.action.cpu(),
        torch.as_tensor(actions[0], dtype=torch.float32).view(2, 1),
    )

    clipped_actions = wrapper._clip_actions(actions)
    assert clipped_actions == actions
    env_actions = wrapper._to_env_actions(clipped_actions)
    received_actions = [
        np.asarray(
            [
                env_actions["tables"]["building"][0, 0],
                env_actions["tables"]["charger"][0, 0],
            ],
            dtype=np.float64,
        )
    ]

    agent.update(
        observations=observations,
        actions=received_actions,
        rewards=[0.1],
        next_observations=observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=True,
    )

    changed_actions = [received_actions[0].copy()]
    changed_actions[0][1] = 0.1
    agent.predict(observations, deterministic=True)
    with pytest.raises(ValueError, match="does not match the pending TPPO action"):
        agent.update(
            observations=observations,
            actions=changed_actions,
            rewards=[0.1],
            next_observations=observations,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=1,
            update_step=False,
            initial_exploration_done=True,
        )


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


def test_wrapper_publishes_training_metrics_with_active_mlflow(monkeypatch) -> None:
    """Training diagnostics must be consumed and sent through the MLflow path."""

    class _Space:
        def __init__(self, low: float, high: float) -> None:
            self.low = np.array([low], dtype=np.float64)
            self.high = np.array([high], dtype=np.float64)

    class _Env:
        def __init__(self) -> None:
            self.observation_names = [["obs_0"]]
            self.observation_space = [_Space(0.0, 1.0)]
            self.action_names = [["action_0"]]
            self.action_space = [_Space(-1.0, 1.0)]
            self.reward_function = type("Reward", (), {})()
            self.time_steps = 1
            self.seconds_per_time_step = 3600
            self.time_step_ratio = 1.0
            self.random_seed = 0
            self.episode_tracker = type("Tracker", (), {"episode_time_steps": 1})()
            self.unwrapped = self

        def reset(self):
            return [np.array([0.0], dtype=np.float64)], {}

        def step(self, _actions):
            return [np.array([0.0], dtype=np.float64)], [1.0], True, False, {}

        def get_metadata(self):
            return {"buildings": [{}]}

    class _Model:
        use_raw_observations = True

        def __init__(self) -> None:
            self.consume_calls = 0

        def attach_environment(self, **_kwargs) -> None:
            pass

        def predict(self, observations, deterministic=None):
            return [[0.0] for _ in observations]

        def update(self, **_kwargs) -> None:
            pass

        def is_initial_exploration_done(self, _global_step: int) -> bool:
            return True

        def consume_latest_training_metrics(self):
            self.consume_calls += 1
            return {
                "TPPO/approx_kl": 0.1,
                "TPPO/ratio_error_max": 0.2,
                "TPPO/explained_variance": 0.3,
            }

    logged = []
    monkeypatch.setattr(wrapper_module.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(
        wrapper_module.mlflow,
        "log_metrics",
        lambda metrics, step=None: logged.append((metrics, step)),
    )
    wrapper = Wrapper_CityLearn(
        env=_Env(),
        model=_Model(),
        config={
            "training": {},
            "checkpointing": {},
            "tracking": {
                "mlflow_enabled": True,
                "log_frequency": 1,
                "mlflow_step_sample_interval": 1,
                "progress_updates_enabled": False,
            },
        },
        job_id="ppo-mlflow-metrics",
    )

    wrapper.learn(episodes=1)

    step_metrics = next(metrics for metrics, step in logged if step == 1)
    assert step_metrics["TPPO/approx_kl"] == 0.1
    assert step_metrics["TPPO/ratio_error_max"] == 0.2
    assert step_metrics["TPPO/explained_variance"] == 0.3
    assert wrapper.model.consume_calls == 1


def test_deterministic_final_episode_calls_lifecycle_without_agent_updates() -> None:
    class _Env:
        def __init__(self) -> None:
            self.observation_names = [["obs_0"]]
            self.observation_space = [
                type("Space", (), {"low": np.array([0.0]), "high": np.array([1.0])})()
            ]
            self.action_names = [["action_0"]]
            self.action_space = [type("Space", (), {"low": np.array([-1.0]), "high": np.array([1.0])})()]
            self.reward_function = type("Reward", (), {})()
            self.time_steps = 1
            self.seconds_per_time_step = 3600
            self.time_step_ratio = 1.0
            self.random_seed = 0
            self.episode_tracker = type("Tracker", (), {"episode_time_steps": 1})()
            self.unwrapped = self

        def reset(self):
            return [np.array([0.0], dtype=np.float64)], {}

        def step(self, _actions):
            return [np.array([0.0], dtype=np.float64)], [1.0], True, False, {}

        def get_metadata(self):
            return {"buildings": [{}]}

    class _LifecycleModel:
        use_raw_observations = True

        def __init__(self) -> None:
            self.starts: list[tuple[int, bool]] = []
            self.ends: list[tuple[int, bool]] = []
            self.update_calls = 0

        def attach_environment(self, **_kwargs) -> None:
            pass

        def on_episode_start(self, *, episode: int, training: bool) -> None:
            self.starts.append((episode, training))

        def on_episode_end(self, *, episode: int, training: bool) -> None:
            self.ends.append((episode, training))

        def predict(self, observations, deterministic=None):
            return [[0.0] for _ in observations]

        def update(self, **_kwargs) -> None:
            self.update_calls += 1

        def is_initial_exploration_done(self, _global_step: int) -> bool:
            return True

    model = _LifecycleModel()
    wrapper = Wrapper_CityLearn(
        env=_Env(),
        model=model,
        config={
            "training": {},
            "checkpointing": {},
            "tracking": {"mlflow_enabled": False, "progress_updates_enabled": False},
        },
        job_id="ppo-deterministic-final-lifecycle",
    )

    wrapper.learn(episodes=2, deterministic_finish=True)

    assert model.starts == [(0, True), (1, False)]
    assert model.ends == [(0, True), (1, False)]
    assert model.update_calls == 1
