from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithms.utils.price_multiplier_adapter import PRICE_NAMES
from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)
from tests.test_agent_transformer_matd3 import _ACTION_NAMES, _Box, _config
from tests.test_entity_tokenizer_config_schema import (
    _make_minimal_transformer_ppo_cfg,
)


def _stage() -> dict:
    return {
        "algorithm": "AgentTransformerMATD3",
        "count": 1,
        "frozen": False,
        "tokenizer_config_path": "configs/tokenizers/entity_default.json",
        "transformer": {
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.1,
        },
        "hyperparameters": {
            "learning_rate": 1.0e-3,
            "gamma": 0.95,
            "tau": 0.01,
            "batch_size": 2,
            "buffer_capacity": 16,
            "max_grad_norm": 1.0,
            "target_policy_noise": 0.2,
            "target_policy_noise_clip": 0.1,
            "sigma": 0.4,
            "sigma_decay": 0.99,
            "min_sigma": 0.1,
            "bias": 0.0,
        },
    }


def _project_config() -> dict:
    config = _make_minimal_transformer_ppo_cfg()
    config["pipeline"] = [_stage()]
    return config


def test_schema_accepts_matd3_and_defaults_n_step_gamma() -> None:
    from utils.config_schema import TransformerMATD3StageConfig, validate_config

    validated = validate_config(_project_config())
    stage = validated.pipeline[0]

    assert isinstance(stage, TransformerMATD3StageConfig)
    assert stage.hyperparameters.n_step_gamma == pytest.approx(0.95)
    assert stage.behavior_cloning.replay_based.enabled is False
    assert stage.behavior_cloning.demonstration_based.enabled is False


def test_schema_rejects_missing_matd3_tokenizer_file() -> None:
    from utils.config_schema import validate_config

    config = _project_config()
    config["pipeline"][0]["tokenizer_config_path"] = "/tmp/not-a-tokenizer.json"

    with pytest.raises(FileNotFoundError, match="not-a-tokenizer.json"):
        validate_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: cfg["pipeline"][0].update(count=2), "count=1"),
        (
            lambda cfg: cfg["pipeline"][0]["transformer"].update(d_model=7),
            "d_model must be divisible by nhead",
        ),
        (
            lambda cfg: cfg["pipeline"][0]["transformer"].update(dropout=1.0),
            "less than 1",
        ),
        (
            lambda cfg: cfg["pipeline"][0]["hyperparameters"].update(buffer_capacity=1),
            "buffer_capacity must be greater than or equal to batch_size",
        ),
        (
            lambda cfg: cfg["pipeline"][0]["hyperparameters"].update(min_sigma=0.5),
            "min_sigma must be less than or equal to sigma",
        ),
        (
            lambda cfg: cfg["pipeline"][0]["hyperparameters"].update(
                residual_policy_enabled=True
            ),
            "warm_start_policy_name",
        ),
    ],
)
def test_schema_rejects_invalid_matd3_stage(mutate, message: str) -> None:
    from utils.config_schema import validate_config

    config = _project_config()
    mutate(config)

    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_schema_rejects_bc_weights_above_initial_values() -> None:
    from utils.config_schema import validate_config

    config = _project_config()
    config["pipeline"][0]["behavior_cloning"] = {
        "replay_based": {"enabled": True, "weight": 0.1, "min_weight": 0.2},
        "demonstration_based": {
            "enabled": True,
            "weight": 0.1,
            "min_weight": 0.2,
        },
    }

    with pytest.raises(ValueError, match="min_weight"):
        validate_config(config)


def test_schema_requires_entity_interface_and_minmax_price_profile() -> None:
    from utils.config_schema import validate_config

    flat = _project_config()
    flat["simulator"].update(interface="flat", topology_mode="static")
    with pytest.raises(ValueError, match="requires simulator.interface='entity'"):
        validate_config(flat)

    wrong_profile = _project_config()
    wrong_profile["simulator"]["entity_encoding"]["profile"] = "maddpg_v1"
    wrong_profile["pipeline"][0]["hyperparameters"][
        "local_price_conditioning_enabled"
    ] = True
    with pytest.raises(ValueError, match="minmax_space"):
        validate_config(wrong_profile)


def test_schema_and_runtime_require_matd3_to_be_final_stage() -> None:
    from algorithms.registry import build_execution_unit
    from utils.config_schema import validate_config

    config = _project_config()
    config["pipeline"].append({"algorithm": "RuleBasedPolicy", "count": 1})

    with pytest.raises(ValueError, match="must be the final pipeline stage"):
        validate_config(config)
    with pytest.raises(ValueError, match="must be the final stage"):
        build_execution_unit(config)


def test_registry_constructs_transformer_matd3() -> None:
    from algorithms.registry import build_execution_unit
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3
    from utils.config_schema import validate_config

    validated = validate_config(_project_config())
    unit = build_execution_unit(validated.to_dict())

    assert isinstance(unit, AgentTransformerMATD3)


def _price_agent(*, forecast_mode: str = "real_unmodified"):
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    names = load_sample_observation_names_for_first_building()
    config = _config(
        local_price_conditioning_enabled=True,
        local_price_forecast_mode=forecast_mode,
    )
    agent = AgentTransformerMATD3(config)
    lows = np.zeros(len(names), dtype=np.float64)
    highs = np.ones(len(names), dtype=np.float64)
    for name in PRICE_NAMES:
        index = names.index(name)
        lows[index] = 0.1
        highs[index] = 0.5
    agent.attach_environment(
        observation_names=[list(names)],
        action_names=[list(_ACTION_NAMES)],
        action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
        observation_space=[None],
        metadata={
            "building_names": ["Building_1"],
            "raw_observation_names": [list(names)],
            "encoded_observation_names": [list(names)],
            "raw_observation_bounds": [
                {"low": lows.tolist(), "high": highs.tolist()}
            ],
        },
    )
    return agent, names


def test_price_conditioning_rewrites_a_copy_before_tokenization(monkeypatch) -> None:
    agent, names = _price_agent()
    observation = np.zeros(len(names), dtype=np.float32)
    for name in PRICE_NAMES:
        observation[names.index(name)] = 0.5
    original = observation.copy()
    captured = []

    def capture(state, values, *, target):
        del state, target
        captured.append(values.detach().cpu().numpy().copy())
        return torch.zeros((1, len(_ACTION_NAMES)), device=agent.device)

    monkeypatch.setattr(agent, "_actor_unit_action", capture)
    agent.predict([observation], deterministic=True, context=1.5)

    assert captured[0][0, names.index(PRICE_NAMES[0])] == pytest.approx(0.875)
    for name in PRICE_NAMES[1:]:
        assert captured[0][0, names.index(name)] == pytest.approx(0.5)
    assert np.array_equal(observation, original)
    metrics = agent.get_diagnostic_metrics()
    assert metrics["TransformerMATD3/local_price_context_non_neutral"] == 1.0


def test_neutral_price_context_is_an_exact_copy(monkeypatch) -> None:
    agent, names = _price_agent()
    observation = np.linspace(0.0, 1.0, len(names), dtype=np.float32)
    captured = []

    def capture(state, values, *, target):
        del state, target
        captured.append(values.detach().cpu().numpy().copy())
        return torch.zeros((1, len(_ACTION_NAMES)), device=agent.device)

    monkeypatch.setattr(agent, "_actor_unit_action", capture)
    agent.predict([observation], deterministic=True, context=1.0)

    assert np.array_equal(captured[0][0], observation)


def test_price_conditioning_stores_conditioned_current_and_successor_replay() -> None:
    agent, names = _price_agent()
    observation = np.zeros(len(names), dtype=np.float32)
    next_observation = np.zeros(len(names), dtype=np.float32)
    for value in (observation, next_observation):
        for name in PRICE_NAMES:
            value[names.index(name)] = 0.5

    actions = agent.predict([observation], deterministic=True, context=1.5)
    agent.update(
        [observation],
        actions,
        [0.0],
        [next_observation],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=False,
    )

    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert transition.observations[0][names.index(PRICE_NAMES[0])] == pytest.approx(
        0.875
    )
    assert transition.next_observations[0][names.index(PRICE_NAMES[0])] == pytest.approx(
        0.875
    )
    assert np.array_equal(observation, np.zeros(len(names), dtype=np.float32)) is False


def test_price_conditioning_accepts_distinct_successor_context() -> None:
    agent, names = _price_agent()
    observation = np.zeros(len(names), dtype=np.float32)
    next_observation = np.zeros(len(names), dtype=np.float32)
    for value in (observation, next_observation):
        for name in PRICE_NAMES:
            value[names.index(name)] = 0.5

    actions = agent.predict([observation], deterministic=True, context=1.5)
    agent.set_transition_context(
        encoded_observations=[observation],
        encoded_next_observations=[next_observation],
        price_context=1.5,
        next_price_context=0.5,
    )
    agent.update(
        [observation],
        actions,
        [0.0],
        [next_observation],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=False,
        initial_exploration_done=False,
    )

    transition = agent.replay_buffer.get_state()["transitions"][0]
    assert transition.observations[0][names.index(PRICE_NAMES[0])] == pytest.approx(
        0.875
    )
    assert transition.next_observations[0][names.index(PRICE_NAMES[0])] == pytest.approx(
        0.125
    )


def test_price_conditioning_rejects_missing_price_names() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    names = load_sample_observation_names_for_first_building()
    names.remove(PRICE_NAMES[-1])
    agent = AgentTransformerMATD3(
        _config(local_price_conditioning_enabled=True)
    )

    with pytest.raises(ValueError, match=PRICE_NAMES[-1]):
        agent.attach_environment(
            observation_names=[list(names)],
            action_names=[list(_ACTION_NAMES)],
            action_space=[_Box([-2.0, -0.5], [1.0, 0.75])],
            observation_space=[None],
            metadata={
                "building_names": ["Building_1"],
                "raw_observation_names": [list(names)],
                "encoded_observation_names": [list(names)],
                "raw_observation_bounds": [
                    {
                        "low": np.zeros(len(names)).tolist(),
                        "high": np.ones(len(names)).tolist(),
                    }
                ],
            },
        )


def test_price_conditioning_rejects_wrong_per_building_context_width() -> None:
    agent, names = _price_agent()

    with pytest.raises(ValueError, match="length must match"):
        agent.predict(
            [np.zeros(len(names), dtype=np.float32)],
            deterministic=True,
            context=[0.9, 1.1],
        )
