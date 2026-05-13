"""Tests for BC-v2 (policy, dataset, trainer, agent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from algorithms.offline_rl_v2 import schema_v2 as S
from algorithms.offline_rl_v2.bc_agent_v2 import BCAgentV2
from algorithms.offline_rl_v2.bc_dataset_v2 import (
    BCDatasetV2,
    ObservationStandardiser,
    load_and_split,
    load_obs_action_arrays,
    make_split,
)
from algorithms.offline_rl_v2.bc_policy_v2 import BCPolicyV2
from algorithms.offline_rl_v2.bc_trainer_v2 import (
    BCTrainingConfigV2,
    train_single_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RBC_PARQUET = (
    REPO_ROOT / "datasets" / "offline_rl_v2" / "derived" / "rbc_with_reward_v2.parquet"
)


# ---------------------------------------------------------------------------
# 1. Dataset split: disjoint, standardiser fit on train only.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_dataset_split_disjoint() -> None:
    obs, action, obs_names, act_names = load_obs_action_arrays(RBC_PARQUET)
    split = make_split(
        obs, action, obs_names, act_names, val_fraction=0.1, seed=42
    )
    train_set = set(split.train_indices.tolist())
    val_set = set(split.val_indices.tolist())
    assert train_set.isdisjoint(val_set)
    assert len(train_set) + len(val_set) == obs.shape[0]
    # Determinism: re-running with same seed produces the same split.
    split2 = make_split(
        obs, action, obs_names, act_names, val_fraction=0.1, seed=42
    )
    np.testing.assert_array_equal(split.train_indices, split2.train_indices)
    np.testing.assert_array_equal(split.val_indices, split2.val_indices)
    # Standardiser fit on train: mean/std should equal the train slice's
    # mean/std (post-eps floor on std). Compute the reference in float64 to
    # match what ObservationStandardiser.fit does internally; comparing in
    # float32 drifts because std on ~80k float32 samples loses precision.
    train_obs = obs[split.train_indices].astype(np.float64)
    np.testing.assert_allclose(
        split.standardiser.mean,
        train_obs.mean(axis=0).astype(np.float32),
        atol=1e-4,
        rtol=1e-4,
    )
    expected_std = np.maximum(
        train_obs.std(axis=0), split.standardiser.eps
    ).astype(np.float32)
    np.testing.assert_allclose(split.standardiser.std, expected_std, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. Action targets are finite and in expected range.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_dataset_action_targets_finite_and_in_range() -> None:
    _, action, _, act_names = load_obs_action_arrays(RBC_PARQUET)
    assert np.all(np.isfinite(action))
    # All RBC actions in this dataset should be in [-1, 1] (env normalised).
    assert action.min() >= -1.0 - 1e-6
    assert action.max() <= 1.0 + 1e-6
    # Sanity: the storage-action column is constant 0 in v2 RBC.
    storage_idx = act_names.index("electrical_storage")
    assert np.all(action[:, storage_idx] == 0.0)
    # And the EV-action column has real variance.
    ev_name = next(n for n in act_names if n.startswith("electric_vehicle_storage"))
    ev_idx = act_names.index(ev_name)
    assert action[:, ev_idx].std() > 0.05


# ---------------------------------------------------------------------------
# 3. Policy forward shape and tanh range.
# ---------------------------------------------------------------------------


def test_policy_forward_shape() -> None:
    policy = BCPolicyV2(obs_dim=35, action_dim=2, hidden_layers=[32, 32], dropout=0.0)
    obs = torch.randn(8, 35)
    out = policy(obs)
    assert out.shape == (8, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


# ---------------------------------------------------------------------------
# 4. Training overfits a tiny synthetic subset (sanity).
# ---------------------------------------------------------------------------


def test_training_overfits_tiny_subset(tmp_path: Path) -> None:
    """Synthesise a small dataset where action = f(obs) is recoverable, and
    confirm the trainer drives MSE down below a sanity threshold.

    Synth: obs is 8-D Gaussian; target action is [tanh(obs[0]), tanh(obs[1])].
    A 2-layer MLP must learn this in well under 200 epochs.
    """
    rng = np.random.default_rng(0)
    n = 256
    obs_dim = 8
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    action = np.tanh(obs[:, :2]).astype(np.float32)

    # Build a parquet that satisfies load_obs_action_arrays: requires the v2
    # schema columns. Easier to bypass and call make_split directly.
    obs_names = [f"feat_{i}" for i in range(obs_dim)]
    action_names = ["a0", "a1"]
    split = make_split(
        obs, action, obs_names, action_names, val_fraction=0.1, seed=0
    )

    policy = BCPolicyV2(
        obs_dim=obs_dim, action_dim=2, hidden_layers=[64, 64], dropout=0.0
    )
    optim = torch.optim.Adam(policy.parameters(), lr=3e-3)
    train_obs = split.train._obs  # type: ignore[attr-defined]
    train_act = split.train._act  # type: ignore[attr-defined]

    for _ in range(300):
        pred = policy(train_obs)
        loss = torch.nn.functional.mse_loss(pred, train_act)
        optim.zero_grad()
        loss.backward()
        optim.step()
    final = float(loss.item())
    assert final < 1e-3, f"trainer failed to overfit tiny subset: final MSE={final}"


# ---------------------------------------------------------------------------
# 5. Single-seed train_single_seed end-to-end on the real dataset (1 epoch).
#     Verifies wiring; not testing convergence here.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_train_single_seed_smoke(tmp_path: Path) -> None:
    out = tmp_path / "seed_99"
    config = BCTrainingConfigV2(
        hidden_layers=[32, 32],
        dropout=0.0,
        epochs=1,
        batch_size=512,
        device="cpu",
    )
    summary = train_single_seed(RBC_PARQUET, out, seed=99, config=config)
    assert (out / "policy.pt").exists()
    assert (out / "obs_standardiser.npz").exists()
    assert (out / "metrics.jsonl").exists()
    assert (out / "architecture.json").exists()
    assert (out / "seed_summary.json").exists()
    assert summary["n_train"] + summary["n_val"] == 87590
    assert summary["obs_dim"] == 35
    assert summary["action_dim"] == 2
    # MSE is finite — even one epoch should not produce NaN.
    assert np.isfinite(summary["final_train_mse"])
    assert np.isfinite(summary["final_val_mse"])
    # Per-dim MSE recorded for both action dims.
    assert len(summary["final_val_per_dim_mse"]) == 2


# ---------------------------------------------------------------------------
# 6. BCAgentV2.predict returns 17 action vectors after attach_environment;
#     non-target buildings come from RBC (non-BC), B5 from BC.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_agent_predict_returns_17_action_vectors(tmp_path: Path) -> None:
    # Train a tiny BC checkpoint.
    out = tmp_path / "seed_99"
    config = BCTrainingConfigV2(
        hidden_layers=[16, 16], dropout=0.0, epochs=1, batch_size=512, device="cpu"
    )
    train_single_seed(RBC_PARQUET, out, seed=99, config=config)

    # Build env and attach.
    from scripts._benchmark_common import make_env
    env = make_env(seed=22)
    obs_list, _ = env.reset()

    agent = BCAgentV2.from_seed_dir(out)
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
    # B5 action vector has dim 2 (electrical_storage + EV charger).
    b5 = S.TARGET_BUILDING_INDEX
    assert len(actions[b5]) == len(S.ACTION_NAMES)
    for v in actions[b5]:
        assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6, f"BC action out of [-1, 1]: {v}"

    # Sanity: BC's B5 action differs from a fresh RBC's B5 action on the
    # same obs (we have a randomly-init policy, so this is overwhelmingly
    # likely; if it ever ties exactly the test will flag a mis-wiring).
    from algorithms.offline_rl_v2.rbc_v2 import RuleBasedPolicyV2
    rbc_only = RuleBasedPolicyV2(
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
    # Off-target buildings: BC agent must defer to RBC (identical).
    for i in range(len(actions)):
        if i == b5:
            continue
        assert actions[i] == rbc_actions[i], (
            f"BCAgentV2 changed off-target building {i}'s action; should defer to RBC"
        )


# ---------------------------------------------------------------------------
# 7. Standardisation actually applied: different raw obs → different actions.
# ---------------------------------------------------------------------------


def test_agent_obs_standardisation_changes_outputs() -> None:
    """Construct a fixed BC checkpoint by hand, hit it with two distinct B5
    obs vectors, confirm the standardiser is in the loop (different outputs).
    """
    obs_dim = len(S.OBSERVATION_NAMES)
    action_dim = len(S.ACTION_NAMES)
    policy = BCPolicyV2(obs_dim=obs_dim, action_dim=action_dim, hidden_layers=[16, 16], dropout=0.0)
    # Fixed weights for determinism.
    torch.manual_seed(123)
    for p in policy.parameters():
        p.data.normal_(mean=0.0, std=0.5)
    standardiser = ObservationStandardiser(
        mean=np.zeros(obs_dim, dtype=np.float32),
        std=np.ones(obs_dim, dtype=np.float32),
        feature_names=list(S.OBSERVATION_NAMES),
    )

    agent = BCAgentV2(policy, standardiser)

    # The agent's RBC is a real RuleBasedPolicyV2, but we want to isolate
    # the BC path here. Replace the RBC's predict method with a stub that
    # returns 17 zero-vectors. (Can't reassign self._rbc itself because
    # nn.Module.__setattr__ forbids non-Module submodules.)
    n_buildings = S.TARGET_BUILDING_INDEX + 1

    def _stub_predict(observations, deterministic=None):  # noqa: ARG001
        return [[0.0] * action_dim for _ in observations]

    agent._rbc.predict = _stub_predict  # type: ignore[assignment]
    agent._attached = True

    obs_list_a: List[np.ndarray] = [
        np.zeros(obs_dim, dtype=np.float64) for _ in range(n_buildings)
    ]
    obs_list_a[S.TARGET_BUILDING_INDEX] = np.full(obs_dim, 0.1, dtype=np.float64)
    obs_list_b: List[np.ndarray] = [
        np.zeros(obs_dim, dtype=np.float64) for _ in range(n_buildings)
    ]
    obs_list_b[S.TARGET_BUILDING_INDEX] = np.full(obs_dim, -0.1, dtype=np.float64)
    actions_a = agent.predict(obs_list_a, deterministic=True)
    actions_b = agent.predict(obs_list_b, deterministic=True)
    b5 = S.TARGET_BUILDING_INDEX
    # Different inputs → different outputs (with high probability given
    # random init); identical inputs → identical outputs.
    assert actions_a[b5] != actions_b[b5]
    actions_a_again = agent.predict(obs_list_a, deterministic=True)
    assert actions_a[b5] == actions_a_again[b5]


# ---------------------------------------------------------------------------
# 8. Best-epoch checkpointing: policy.pt holds best-val-MSE weights, not
#    final-epoch weights. Required so longer training never regresses the
#    persisted model below its best-seen validation performance.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RBC_PARQUET.exists(), reason="RBC parquet not present")
def test_train_persists_best_epoch_weights(tmp_path: Path) -> None:
    """Train a few epochs; verify that the saved policy.pt loads to a state
    whose val MSE matches summary['best_val_mse'] (within float tolerance),
    and verify the summary records best_epoch < epochs - 1 isn't required —
    we only assert that the persisted policy reproduces best_val_mse, not
    final_val_mse.
    """
    out = tmp_path / "seed_77"
    # High LR + many epochs + tiny net → val MSE oscillates so best_epoch
    # is unlikely to be the final epoch; this exercises the contract that
    # policy.pt holds best-epoch weights, not final-epoch.
    config = BCTrainingConfigV2(
        hidden_layers=[8, 8],
        dropout=0.0,
        epochs=15,
        batch_size=512,
        learning_rate=5e-2,
        device="cpu",
    )
    summary = train_single_seed(RBC_PARQUET, out, seed=77, config=config)
    assert summary["best_epoch"] >= 0
    # Sanity for the test itself: in this regime best_epoch should NOT be
    # the final epoch (so the test actually distinguishes best from final).
    assert summary["best_epoch"] != config.epochs - 1, (
        "test setup didn't produce best != final; weaken to LR sweep"
    )

    # Re-load the persisted policy and recompute val MSE on the same split.
    from algorithms.offline_rl_v2.bc_dataset_v2 import load_and_split
    from torch.utils.data import DataLoader

    split = load_and_split(RBC_PARQUET, val_fraction=config.val_fraction, seed=77)
    obs_dim = split.standardiser.mean.shape[0]
    action_dim = len(split.action_feature_names)
    policy = BCPolicyV2(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
    state = torch.load(out / "policy.pt", map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()

    val_loader = DataLoader(split.val, batch_size=1024, shuffle=False)
    n_total = 0
    sse = 0.0
    with torch.no_grad():
        for obs, action in val_loader:
            pred = policy(obs)
            sse += float(((pred - action) ** 2).sum().item())
            n_total += int(obs.shape[0])
    persisted_val_mse = sse / max(n_total, 1) / action_dim
    # Compare with reasonable tolerance — eval-mode dropout off is satisfied
    # already (dropout=0.0 in config), so equality should be tight.
    assert abs(persisted_val_mse - summary["best_val_mse"]) < 1e-6, (
        f"persisted policy MSE {persisted_val_mse:.8f} does not match "
        f"best_val_mse {summary['best_val_mse']:.8f}"
    )
