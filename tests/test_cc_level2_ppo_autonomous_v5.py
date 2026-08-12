from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from algorithms.agents.cc_level2_agent import (
    CCLevel2Agent,
    DeterministicVectorMultiplierPolicy,
    _CC_LEVEL2_BUILDING_FEATURES,
    _CC_LEVEL2_DISTRICT_FEATURES,
)
from scripts.generate_cc_level2_ppo_autonomous_v5 import (
    ANNUAL_EPISODES,
    VARIANTS,
    build_config,
    build_paired_neutral_config,
    generate,
)
from utils.config_schema import validate_config
from utils.wrapper_citylearn import Wrapper_CityLearn


def _names(building: int) -> list[str]:
    names = list(_CC_LEVEL2_DISTRICT_FEATURES)
    for feature in _CC_LEVEL2_BUILDING_FEATURES:
        if "::" in feature:
            prefix, tail = feature.split("::", 1)
            names.append(f"{prefix}::Building_{building}/asset::{tail}")
        else:
            names.append(f"charger::Building_{building}/charger::{feature}")
    return names


def _protocol_agent() -> tuple[CCLevel2Agent, list[np.ndarray]]:
    count = 2
    agent = CCLevel2Agent(
        {
            "algorithm": {
                "hyperparameters": {
                    "num_buildings": count,
                    "c_dim": len(_CC_LEVEL2_DISTRICT_FEATURES)
                    + len(_CC_LEVEL2_BUILDING_FEATURES) * count,
                    "hidden_dims": [8],
                    "price_min": 0.8,
                    "price_max": 1.0,
                    "reference_multipliers": [1.0, 1.0],
                    "policy_parameterization": "sparse_centered_residual",
                    "policy_deadband": 0.1,
                    "cc_action_interval": 1,
                    "num_steps": 8,
                    "credit_assignment": "member_decomposed",
                    "reward_normalization": "none",
                    "team_reward_mix": 0.0,
                    "w_factor": 0.0,
                    "w_smoothness": 0.0,
                    "bc_pretrain_enabled": False,
                    "neutral_baseline_enabled": True,
                    "counterfactual_baseline_weight": 1.0,
                    "training_episodes_per_validation": 1,
                    "rollback_rejected_validation": True,
                    "restore_best_policy_for_deterministic": True,
                    "train_log_std": False,
                }
            }
        }
    )
    names = [_names(1), _names(2)]
    agent.attach_environment(
        observation_names=names,
        action_names=[[], []],
        action_space=[None, None],
        observation_space=[None, None],
        metadata={},
    )
    observations = [
        np.zeros(len(member_names), dtype=np.float32)
        for member_names in names
    ]
    return agent, observations


def _update(
    agent: CCLevel2Agent,
    observations: list[np.ndarray],
    rewards: list[float],
    *,
    done: bool,
) -> None:
    agent.update(
        observations,
        [[], []],
        rewards,
        observations,
        terminated=done,
        truncated=False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )


def test_sparse_residual_has_exact_neutral_deadband_and_export_parity() -> None:
    agent, _ = _protocol_agent()
    inside = np.arctanh(np.asarray([0.05, -0.05], dtype=np.float32))
    outside = np.arctanh(np.asarray([-0.55, -0.30], dtype=np.float32))

    np.testing.assert_allclose(
        agent._raw_to_multipliers(inside),
        [1.0, 1.0],
        atol=1.0e-7,
    )
    mapped = agent._raw_to_multipliers(outside)
    assert mapped[0] < mapped[1] < 1.0

    inference = DeterministicVectorMultiplierPolicy(
        policy=agent.policy,
        price_min=agent._price_min,
        price_max=agent._price_max,
        reference_multipliers=agent._reference_multipliers,
        policy_residual_scale=agent._policy_residual_scale,
        policy_parameterization=agent._policy_parameterization,
        policy_deadband=agent._policy_deadband,
    )
    with torch.no_grad():
        agent.policy.mean_head.weight.zero_()
        agent.policy.mean_head.bias.copy_(torch.from_numpy(outside))
        exported = inference(torch.zeros((1, agent._c_dim))).squeeze(0).numpy()
    np.testing.assert_allclose(exported, mapped, atol=1.0e-6)


def test_sparse_actor_mask_excludes_samples_with_no_physical_price_effect() -> None:
    agent, observations = _protocol_agent()
    with torch.no_grad():
        agent.policy.mean_head.weight.zero_()
        # The first raw action discounts building 1. The second lies outside
        # the raw deadband but cannot exceed the reference/price_max of 1.0,
        # so it must not contribute a false policy-gradient sample.
        agent.policy.mean_head.bias.copy_(torch.tensor([-0.5, 0.5]))

    agent._sample_new_decision(observations, deterministic=True)

    assert agent._cached_multipliers[0] < 1.0
    assert agent._cached_multipliers[1] == pytest.approx(1.0)
    np.testing.assert_array_equal(agent._cached_actor_mask, [1.0, 0.0])


def test_repeat_episode_scenario_resets_citylearn_episode_index() -> None:
    class Tracker:
        calls = 0

        def reset_episode_index(self) -> None:
            self.calls += 1

    class Environment:
        episode_tracker = Tracker()

        @property
        def unwrapped(self):
            return self

    wrapper = object.__new__(Wrapper_CityLearn)
    wrapper.env = Environment()
    wrapper._repeat_episode_scenario = True
    wrapper._deferrable_wait_steps = {}

    wrapper._prepare_episode_reset()

    assert wrapper.env.episode_tracker.calls == 1


def test_neutral_episode_is_control_variate_for_member_rewards() -> None:
    agent, observations = _protocol_agent()

    agent.set_episode_context(episode_step=0)
    assert agent.predict(observations, deterministic=False) == [1.0, 1.0]
    _update(agent, observations, [-10.0, -2.0], done=True)

    assert agent._neutral_baseline_objective == pytest.approx(-12.0)
    assert len(agent._neutral_baseline_rewards) == 1
    assert agent.rollout_buffer._ptr == 0

    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-8.0, -1.5], done=True)

    np.testing.assert_allclose(agent.rollout_buffer.rewards[0], [2.0, 0.5])
    assert agent.rollout_buffer._ptr == 1


def test_protocol_places_cold_start_before_baseline_and_validations() -> None:
    agent, _ = _protocol_agent()
    agent._neutral_warmup_episodes = 1

    assert [agent._mode_for_protocol_episode(index) for index in range(6)] == [
        "neutral_warmup",
        "neutral_baseline",
        "training",
        "validation",
        "training",
        "validation",
    ]


def test_validation_selects_policy_and_final_episode_restores_without_learning() -> None:
    agent, observations = _protocol_agent()

    # Episode 0: exact neutral baseline and incumbent snapshot.
    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-10.0, -2.0], done=True)

    # Episode 1: training trajectory. Do not fill the rollout in this unit test.
    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-9.0, -1.5], done=True)

    # Episode 2: deterministic internal validation promotes this known bias.
    with torch.no_grad():
        agent.policy.mean_head.bias.fill_(-0.5)
    selected_bias = agent.policy.mean_head.bias.detach().clone()
    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-7.0, -1.0], done=True)
    assert agent._best_validation_episode == 2

    # Damage the live actor; the wrapper-deterministic final episode must
    # restore the selected validation policy and must not enter the rollout.
    with torch.no_grad():
        agent.policy.mean_head.bias.fill_(0.75)
    pointer_before = agent.rollout_buffer._ptr
    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=True)
    torch.testing.assert_close(agent.policy.mean_head.bias, selected_bias)
    _update(agent, observations, [-7.0, -1.0], done=True)
    assert agent.rollout_buffer._ptr == pointer_before


def test_rejected_validation_rolls_training_back_to_selected_policy() -> None:
    agent, observations = _protocol_agent()

    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-10.0, -2.0], done=True)
    selected_bias = agent.policy.mean_head.bias.detach().clone()

    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-9.0, -1.5], done=True)

    with torch.no_grad():
        agent.policy.mean_head.bias.fill_(-0.5)
    agent.set_episode_context(episode_step=0)
    agent.predict(observations, deterministic=False)
    _update(agent, observations, [-12.0, -3.0], done=True)

    assert agent._validation_history[-1]["promoted"] is False
    torch.testing.assert_close(agent.policy.mean_head.bias, selected_bias)


def test_autonomous_configs_use_no_level1_signal_and_validate(tmp_path: Path) -> None:
    paths = generate(tmp_path, pilot_steps=384)
    assert len(paths) == 1 + len(VARIANTS)

    neutral = build_paired_neutral_config(episodes=ANNUAL_EPISODES)
    validate_config(neutral)
    assert neutral["pipeline"][0]["algorithm"] == "FixedPriceSignal"

    for name in VARIANTS:
        config = build_config(name)
        validate_config(config)
        manager, leaf = config["pipeline"]
        params = manager["hyperparameters"]

        assert manager["algorithm"] == "CCLevel2"
        assert manager["frozen"] is False
        assert config["tracking"]["tags"]["uses_cc_level1_signal"] == "False"
        assert all(stage["algorithm"] != "CCLevel1" for stage in config["pipeline"])
        assert params["reference_multipliers"] == [1.0] * 17
        assert params["policy_parameterization"] == "sparse_centered_residual"
        assert params["neutral_baseline_enabled"] is True
        assert params["neutral_warmup_episodes"] == 1
        assert params["training_episodes_per_validation"] == 2
        assert params["rollback_rejected_validation"] is True
        assert params["restore_best_policy_for_deterministic"] is True
        assert params["bc_pretrain_enabled"] is False
        assert params["train_log_std"] is False
        assert config["simulator"]["episodes"] == ANNUAL_EPISODES
        assert config["simulator"]["repeat_episode_scenario"] is True
        assert config["simulator"]["reward_function_kwargs"][
            "credit_assignment"
        ] == "member_decomposed"
        assert leaf["frozen"] is True
        assert leaf["exploration"]["params"][
            "residual_base_price_conditioning_enabled"
        ] is True
        assert leaf["exploration"]["params"][
            "residual_base_policy_hyperparameters"
        ]["allow_v2g"] is False

    assert VARIANTS["cost_first_seed123"]["price_min"] == pytest.approx(0.60)
