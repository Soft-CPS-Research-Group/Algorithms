"""Tests for CCLevel2 factored per-building credit assignment.

Two layers:
  1. Network shape contract — CommunityMarketMakerNetV2 returns per-building
     log-prob / entropy / value when per_building_credit=True, and the original
     joint (summed) shapes when False.
  2. Agent end-to-end — a short predict/update loop fills the rollout buffer and
     triggers a PPO update in both modes without shape errors, proving the
     buffer / GAE / PPO plumbing handles (T, N) and (T,) uniformly.
"""

from __future__ import annotations

import numpy as np
import torch

from algorithms.agents.cc_level2_agent import (
    CCLevel2Agent,
    CommunityMarketMakerNetV2,
    _CC_LEVEL2_DISTRICT_FEATURES,
    _N_BUILDING_FEATS,
    _N_DISTRICT,
)


# ---------------------------------------------------------------------------
# Network shape contract
# ---------------------------------------------------------------------------

def _make_net(n: int, per_building: bool) -> CommunityMarketMakerNetV2:
    c_dim = _N_DISTRICT + _N_BUILDING_FEATS * n
    return CommunityMarketMakerNetV2(
        c_dim, n, [16, 16],
        n_district=_N_DISTRICT, n_building_feats=_N_BUILDING_FEATS,
        per_building_credit=per_building,
    )


def test_net_per_building_shapes():
    n, batch = 5, 3
    net = _make_net(n, per_building=True)
    action, log_prob, entropy, value = net.get_action_and_value(
        torch.randn(batch, net.n_district + _N_BUILDING_FEATS * n)
    )
    assert action.shape == (batch, n)
    assert log_prob.shape == (batch, n)   # NOT summed over buildings
    assert entropy.shape == (batch, n)
    assert value.shape == (batch, n)      # one value per building


def test_net_joint_shapes_unchanged():
    n, batch = 5, 3
    net = _make_net(n, per_building=False)
    action, log_prob, entropy, value = net.get_action_and_value(
        torch.randn(batch, net.n_district + _N_BUILDING_FEATS * n)
    )
    assert action.shape == (batch, n)
    assert log_prob.shape == (batch,)     # summed over buildings (joint action)
    assert entropy.shape == (batch,)
    assert value.shape == (batch,)        # single community value


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------

def _make_agent(n: int, per_building: bool) -> CCLevel2Agent:
    return CCLevel2Agent({"algorithm": {"hyperparameters": {
        "num_buildings": n,
        "hidden_dims": [16, 16],
        "num_steps": 6,
        "mini_batch_size": 3,
        "cc_action_interval": 1,
        "num_epochs": 2,
        "per_building_credit": per_building,
        "bc_pretrain_enabled": False,
        "price_min": 0.5,
        "price_max": 1.3,
    }}})


def _names_for(n: int) -> list[list[str]]:
    district = list(_CC_LEVEL2_DISTRICT_FEATURES)
    names = []
    for i in range(n):
        names.append(district + [
            f"storage::B{i}/electrical_storage::soc",
            f"pv::B{i}/pv::generation_power_kw",
            f"B{i}::net_power_kw",
            f"B{i}::connected_state",
            f"B{i}::connected_ev_soc_deficit",
            f"B{i}::connected_ev_departure_urgency_24h",
        ])
    return names


def _obs_for(n: int, names: list[list[str]]) -> list[np.ndarray]:
    """Build per-building obs; per-building net differs across buildings."""
    obs = []
    for i, names_i in enumerate(names):
        row = np.zeros(len(names_i), dtype=np.float32)
        for j, nm in enumerate(names_i):
            if nm == "district__electricity_pricing":
                row[j] = 0.5
            elif nm.startswith("district__electricity_pricing_predicted"):
                row[j] = 0.4
            elif nm == "district__community_import_power_kw":
                row[j] = 0.6
            elif nm == "district__community_export_power_kw":
                row[j] = 0.1
            elif nm == "district__community_building_headroom_kw":
                row[j] = 0.3
            elif nm.endswith("net_power_kw") and "community" not in nm:
                row[j] = 0.1 * (i + 1)          # per-building differentiation
            elif nm.endswith("::soc"):
                row[j] = 0.5
            elif nm.endswith("generation_power_kw") and "community" not in nm:
                row[j] = 0.2
        obs.append(row)
    return obs


def _run_loop(agent: CCLevel2Agent, n: int, steps: int = 13) -> None:
    names = _names_for(n)
    agent.attach_environment(
        observation_names=names,
        action_names=[["multiplier"]] * n,
        action_space=[None] * n,
        observation_space=[None] * n,
        metadata={},
    )
    obs = _obs_for(n, names)
    for step in range(steps):
        pred = agent.predict(obs, deterministic=False)
        assert len(pred) == n
        assert all(0.5 <= m <= 1.3 for m in pred)          # within [price_min, price_max]
        agent.update(
            obs,
            [np.zeros(1, dtype=np.float32)] * n,
            [-0.1 * (i + 1) for i in range(n)],            # per-building rewards
            obs,
            terminated=False,
            truncated=(step == steps - 1),
            update_target_step=False,
            global_learning_step=step,
            update_step=True,
            initial_exploration_done=True,
        )


def test_agent_per_building_runs_and_updates():
    n = 4
    agent = _make_agent(n, per_building=True)
    assert agent.rollout_buffer.rewards.ndim == 2          # (T, N)
    assert agent.rollout_buffer.values.ndim == 2
    _run_loop(agent, n)
    assert agent._ppo_update_count >= 1                     # PPO fired without shape errors


def test_agent_joint_runs_and_updates():
    n = 4
    agent = _make_agent(n, per_building=False)
    assert agent.rollout_buffer.rewards.ndim == 1          # (T,) — unchanged
    assert agent.rollout_buffer.values.ndim == 1
    _run_loop(agent, n)
    assert agent._ppo_update_count >= 1
