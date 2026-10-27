"""Tests for IQL (networks, dataset, trainer, agent).

Eight tests, written before implementation per the IQL implementation plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from algorithms.offline_rl import schema as S


REPO_ROOT = Path(__file__).resolve().parents[2]
RBC_PARQUET = (
    REPO_ROOT / "datasets" / "offline_rl" / "derived" / "rbc_with_reward.parquet"
)


# ---------------------------------------------------------------------------
# 1. Expectile loss correctness
# ---------------------------------------------------------------------------


def test_expectile_loss_correctness() -> None:
    from algorithms.offline_rl.iql_trainer import expectile_loss

    # Convention: diff = q_target - v_pred. tau=0.7 weights positive diff
    # (V under-predicts Q) by 0.7 and negative diff by 0.3.
    diff_pos = torch.tensor([1.0])
    diff_neg = torch.tensor([-1.0])
    assert torch.isclose(
        expectile_loss(diff_pos, tau=0.7), torch.tensor(0.7), atol=1e-7
    )
    assert torch.isclose(
        expectile_loss(diff_neg, tau=0.7), torch.tensor(0.3), atol=1e-7
    )
    # tau=0.5 → symmetric (=0.5 * u^2) → recovers half MSE.
    diff_arr = torch.tensor([2.0, -3.0, 0.5])
    half_mse = 0.5 * diff_arr.pow(2)
    assert torch.allclose(
        expectile_loss(diff_arr, tau=0.5), half_mse, atol=1e-7
    )


# ---------------------------------------------------------------------------
# 2. Bellman target with done flag
# ---------------------------------------------------------------------------


def test_bellman_target_with_done() -> None:
    from algorithms.offline_rl.iql_trainer import bellman_target

    reward = torch.tensor([2.0, -1.0])
    next_v = torch.tensor([10.0, 5.0])
    done = torch.tensor([1.0, 0.0])
    gamma = 0.99
    target = bellman_target(reward, gamma, next_v, done)
    # done=1 → target == reward; done=0 → target == r + gamma * V(s').
    expected = torch.tensor([2.0, -1.0 + 0.99 * 5.0])
    assert torch.allclose(target, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Advantage-weighted policy loss sign
# ---------------------------------------------------------------------------


def test_advantage_weighted_loss_pushes_policy_toward_action() -> None:
    """For a single (s, a) with positive advantage, one Adam step on the
    AWR loss should reduce |tanh(mean) - a|.
    """
    from algorithms.offline_rl.iql_networks import GaussianPolicy

    torch.manual_seed(0)
    policy = GaussianPolicy(
        obs_dim=4, action_dim=2, hidden=[16, 16], dropout=0.0
    )
    obs = torch.randn(8, 4)
    target_action = torch.full((8, 2), 0.5)
    advantage = torch.full((8,), 2.0)  # positive
    weight = torch.exp(3.0 * advantage).clamp(max=100.0)

    def _action_distance() -> float:
        with torch.no_grad():
            mean = policy.predict_deterministic(obs)
            return float((mean - target_action).abs().mean().item())

    before = _action_distance()
    optim = torch.optim.Adam(policy.parameters(), lr=3e-3)
    for _ in range(10):
        lp = policy.log_prob(obs, target_action)
        loss = -(weight * lp).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    after = _action_distance()
    assert after < before, f"AWR step did not move policy toward action: {before:.4f} → {after:.4f}"


# ---------------------------------------------------------------------------
# 4. Twin Q clipped min
# ---------------------------------------------------------------------------


def test_twin_q_clipped_min() -> None:
    from algorithms.offline_rl.iql_networks import QNetwork

    torch.manual_seed(1)
    q1 = QNetwork(obs_dim=4, action_dim=2, hidden=[8, 8], dropout=0.0)
    q2 = QNetwork(obs_dim=4, action_dim=2, hidden=[8, 8], dropout=0.0)
    obs = torch.randn(5, 4)
    act = torch.randn(5, 2)
    with torch.no_grad():
        v1 = q1(obs, act)
        v2 = q2(obs, act)
        clipped = torch.minimum(v1, v2)
    expected = torch.stack([v1, v2], dim=0).min(dim=0).values
    assert torch.allclose(clipped, expected, atol=1e-7)
    assert clipped.shape == (5,)


# ---------------------------------------------------------------------------
# 5. Policy output shape and range
# ---------------------------------------------------------------------------


def test_policy_output_shape_and_range() -> None:
    from algorithms.offline_rl.iql_networks import GaussianPolicy

    torch.manual_seed(2)
    policy = GaussianPolicy(
        obs_dim=35, action_dim=2, hidden=[32, 32], dropout=0.0
    )
    obs = torch.randn(7, 35)
    out = policy.predict_deterministic(obs)
    assert out.shape == (7, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


# ---------------------------------------------------------------------------
# 6. End-to-end smoke train: artefacts written and metrics finite
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_smoke_train_artifacts_finite(tmp_path: Path) -> None:
    from algorithms.offline_rl.iql_trainer import (
        IQLTrainingConfig,
        train_single_seed,
    )

    out = tmp_path / "seed_99"
    config = IQLTrainingConfig(
        hidden_layers=[32, 32],
        dropout=0.0,
        gradient_steps=200,
        eval_every_n_steps=50,
        batch_size=128,
        device="cpu",
    )
    summary = train_single_seed(RBC_PARQUET, out, seed=99, config=config)

    for name in (
        "policy.pt",
        "q1.pt",
        "q2.pt",
        "value.pt",
        "obs_standardiser.npz",
        "metrics.jsonl",
        "architecture.json",
        "seed_summary.json",
    ):
        assert (out / name).exists(), f"missing artefact: {name}"

    # All metrics-line scalars finite.
    lines = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert lines, "metrics.jsonl is empty"
    for line in lines:
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v), f"non-finite metric {k}={v} in {rec}"

    assert summary["obs_dim"] == 35
    assert summary["action_dim"] == 2
    assert summary["gradient_steps"] == 200
    assert np.isfinite(summary["best_val_policy_mse"])


# ---------------------------------------------------------------------------
# 7. Best-checkpoint round-trip: persisted policy.pt reproduces best val MSE
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_best_checkpoint_round_trip(tmp_path: Path) -> None:
    from algorithms.offline_rl.iql_dataset import load_iql_split
    from algorithms.offline_rl.iql_networks import GaussianPolicy
    from algorithms.offline_rl.iql_trainer import (
        IQLTrainingConfig,
        train_single_seed,
    )

    out = tmp_path / "seed_77"
    config = IQLTrainingConfig(
        hidden_layers=[16, 16],
        dropout=0.0,
        gradient_steps=300,
        eval_every_n_steps=50,
        batch_size=256,
        device="cpu",
    )
    summary = train_single_seed(RBC_PARQUET, out, seed=77, config=config)

    arch = json.loads((out / "architecture.json").read_text())
    policy = GaussianPolicy(
        obs_dim=int(arch["obs_dim"]),
        action_dim=int(arch["action_dim"]),
        hidden=list(arch["hidden_layers"]),
        dropout=float(arch.get("dropout", 0.0)),
    )
    state = torch.load(out / "policy.pt", map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()

    split = load_iql_split(RBC_PARQUET, val_fraction=config.val_fraction, seed=77)
    val_obs = split.val._obs  # type: ignore[attr-defined]
    val_act = split.val._act  # type: ignore[attr-defined]
    with torch.no_grad():
        pred = policy.predict_deterministic(val_obs)
        persisted = float(((pred - val_act) ** 2).mean().item())

    assert abs(persisted - summary["best_val_policy_mse"]) < 1e-6, (
        f"persisted policy MSE {persisted:.8f} != best_val_policy_mse "
        f"{summary['best_val_policy_mse']:.8f}"
    )


# ---------------------------------------------------------------------------
# 8. IQLAgent.predict returns 17 vectors; off-target buildings == fresh RBC
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_iql_agent_predict_returns_17(tmp_path: Path) -> None:
    from algorithms.offline_rl.iql_agent import IQLAgent
    from algorithms.offline_rl.iql_trainer import (
        IQLTrainingConfig,
        train_single_seed,
    )
    from algorithms.offline_rl.rbc import OfflineRBC

    out = tmp_path / "seed_99"
    config = IQLTrainingConfig(
        hidden_layers=[16, 16],
        dropout=0.0,
        gradient_steps=100,
        eval_every_n_steps=50,
        batch_size=128,
        device="cpu",
    )
    train_single_seed(RBC_PARQUET, out, seed=99, config=config)

    from scripts._benchmark_common import make_env

    env = make_env(seed=22)
    obs_list, _ = env.reset()

    agent = IQLAgent.from_seed_dir(out)
    agent.attach_environment(
        observation_names=env.observation_names,
        action_names=env.action_names,
        action_space=env.action_space,
        observation_space=env.observation_space,
        metadata={
            "interface": "flat",
            "topology_mode": "static",
            "topology_version": 0,
        },
    )
    actions = agent.predict(obs_list, deterministic=True)
    assert isinstance(actions, list)
    assert len(actions) == len(env.action_names)
    b5 = S.TARGET_BUILDING_INDEX
    assert len(actions[b5]) == len(S.ACTION_NAMES)
    for v in actions[b5]:
        assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6

    rbc_only = OfflineRBC(
        config={
            "algorithm": {"hyperparameters": {}},
            "simulator": {
                "dataset_path": "./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json"
            },
        }
    )
    rbc_only.attach_environment(
        observation_names=env.observation_names,
        action_names=env.action_names,
        action_space=env.action_space,
        observation_space=env.observation_space,
    )
    rbc_actions = rbc_only.predict(obs_list, deterministic=True)
    for i in range(len(actions)):
        if i == b5:
            continue
        assert actions[i] == rbc_actions[i], (
            f"IQLAgent changed off-target building {i}'s action; should defer to RBC"
        )
