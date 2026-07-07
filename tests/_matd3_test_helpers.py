from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from tests.test_agent_transformer_matd3_foundation import _matd3_config


def _matd3_config_full_training() -> dict:
    """Config with all training features enabled (BC, residual, reward normalization)."""
    cfg = _matd3_config()
    algo = cfg["algorithm"]
    algo["hyperparameters"]["reward_normalization"] = True
    algo["hyperparameters"]["reward_normalization_clip"] = 5.0
    algo["exploration"] = {
        "random_exploration_steps": 2,
        "end_initial_exploration_time_step": 4,
        "train_during_initial_exploration": False,
        "warm_start_policy": {
            "enabled": True,
            "phaseout_steps": 3,
        },
    }
    algo["residual"] = {
        "enabled": True,
        "initial_scale": 0.1,
        "growth_steps": 10,
        "max_scale": 1.0,
        "storage_scale_multiplier": 0.5,
        "ev_scale_multiplier": 0.8,
    }
    algo["behavior_cloning"] = {
        "enabled": True,
        "weight": 1.0,
        "min_weight": 0.0,
        "decay_start_step": 5,
        "decay_steps": 10,
        "ev_multiplier": 2.0,
        "storage_multiplier": 1.5,
    }
    return cfg


def _make_matd3_full(n_buildings: int = 2):
    """Create an agent with full training config."""
    from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building
    from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3

    obs_names = load_sample_observation_names_for_first_building()
    obs_per = [list(obs_names) for _ in range(n_buildings)]
    act_per = [["electrical_storage", "electric_vehicle_storage"] for _ in range(n_buildings)]
    agent = AgentTransformerMATD3(_matd3_config_full_training())
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
    )
    obs_dim = len(obs_names)
    return agent, obs_per, act_per, obs_dim


def _generate_transition(
    n_buildings: int, obs_dim: int
) -> Tuple[List[npt.NDArray], List[npt.NDArray], List[float], List[npt.NDArray], bool, bool]:
    """Generate a random transition tuple for update()."""
    obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(n_buildings)]
    actions = [np.random.uniform(-1, 1, size=2).astype(np.float64) for _ in range(n_buildings)]
    rewards = [float(np.random.randn()) for _ in range(n_buildings)]
    next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(n_buildings)]
    terminated = False
    truncated = False
    return obs, actions, rewards, next_obs, terminated, truncated


def _run_update_step(
    agent,
    obs,
    actions,
    rewards,
    next_obs,
    terminated,
    truncated,
    *,
    global_learning_step: int,
    update_step: bool = True,
    update_target_step: bool = False,
    initial_exploration_done: bool = True,
) -> None:
    """Run a single update step with context hooks."""
    agent.set_observation_context(
        raw_observations=obs,
        encoded_observations=obs,
    )
    agent.set_transition_context(
        raw_observations=obs,
        raw_next_observations=next_obs,
        encoded_observations=obs,
        encoded_next_observations=next_obs,
    )
    agent.update(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        terminated=terminated,
        truncated=truncated,
        update_target_step=update_target_step,
        global_learning_step=global_learning_step,
        update_step=update_step,
        initial_exploration_done=initial_exploration_done,
    )


def _add_charger_to_building_obs(
    obs_names: List[str],
    building_id: str,
    new_charger_id: str,
) -> Tuple[List[str], str]:
    """Extend obs_names with a full valid charger asset block."""
    charger_prefix = None
    for name in obs_names:
        if name.startswith("charger::") and "::" in name[len("charger::"):]:
            _, existing_id, _ = name.split("::", 2)
            charger_prefix = existing_id
            break
    if charger_prefix is None:
        raise RuntimeError("No existing charger in obs_names - cannot mirror feature block.")

    prefix = f"charger::{charger_prefix}::"
    suffixes = [name[len(prefix):] for name in obs_names if name.startswith(prefix)]
    if not suffixes:
        raise RuntimeError(f"No features found under prefix {prefix!r}")

    new_prefix = f"charger::{building_id}/{new_charger_id}::"
    new_names = list(obs_names) + [new_prefix + s for s in suffixes]
    new_action = f"electric_vehicle_storage_{new_charger_id}"
    return new_names, new_action
