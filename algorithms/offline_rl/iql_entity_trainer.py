"""IQL trainer for the entity-interface multi-agent dataset.

Reuses the IQL update primitives from ``iql_trainer.py`` (expectile loss,
Bellman target, soft-update, hard-copy) but loads data via
``entity_dataset.load_entity_dataset`` instead of the old flat-schema loader.

One policy is trained *per agent group* (identified by ``obs_dim`` ×
``action_dim``).  Call ``train_entity_group`` for a single group, or
``train_all_groups`` to train all four groups sequentially.

Per-seed artefacts (under ``output_dir/<group_key>/seed_<N>/``):
  * ``policy.pt``           — best-checkpoint policy state_dict
  * ``q1.pt``, ``q2.pt``   — final Q state_dicts
  * ``value.pt``            — final V state_dict
  * ``obs_standardiser.npz``
  * ``metrics.jsonl``       — one line per eval step
  * ``architecture.json``
  * ``seed_summary.json``

Aggregated:
  * ``multi_seed_summary.json``
  * ``seeds_index.json``
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from algorithms.offline_rl.entity_dataset import EntityOfflineDataset, load_entity_dataset
from algorithms.offline_rl.entity_schema import AGENT_GROUPS, AgentGroupSpec
from algorithms.offline_rl.iql_networks import GaussianPolicy, QNetwork, ValueNetwork
from algorithms.offline_rl.iql_trainer import (
    IQLTrainingConfig,
    _hard_copy,
    _resolve_device,
    _soft_update,
    bellman_target,
    expectile_loss,
)


# ---------------------------------------------------------------------------
# Eval helper (adapted for EntityOfflineDataset)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_policy_mse_entity(
    policy: GaussianPolicy,
    val_ds: EntityOfflineDataset,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    policy.eval()
    n = len(val_ds)
    if n == 0:
        return float("nan")
    obs_all = val_ds._obs.to(device)
    act_all = val_ds._actions.to(device)
    sse = 0.0
    total = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred = policy.predict_deterministic(obs_all[start:end])
        sse += float(((pred - act_all[start:end]) ** 2).sum().item())
        total += end - start
    policy.train()
    return sse / max(total, 1) / max(int(act_all.shape[1]), 1)


# ---------------------------------------------------------------------------
# Single-seed training for one agent group
# ---------------------------------------------------------------------------


def train_entity_single_seed(
    data_dir: Path,
    output_dir: Path,
    *,
    obs_dim: int,
    action_dim: int,
    seed: int,
    val_seeds: Optional[Sequence[int]] = None,
    config: IQLTrainingConfig,
) -> Dict[str, Any]:
    """Train one IQL seed for a given agent group.

    Parameters
    ----------
    data_dir:
        Directory with ``seed_*.parquet`` files from collect_rbcsmart_dataset.
    output_dir:
        Where to save artefacts.
    obs_dim, action_dim:
        Target agent group.
    seed:
        Random seed for weight init and minibatch sampling.
    val_seeds:
        Dataset seeds to hold out for validation.  Default: last seed.
    config:
        IQL hyperparameters.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    train_ds, val_ds, spec, standardiser = load_entity_dataset(
        data_dir,
        obs_dim=obs_dim,
        action_dim=action_dim,
        val_seeds=val_seeds,
        standardise=True,
    )
    device = _resolve_device(config.device)

    # Move dataset to device
    train_ds._obs = train_ds._obs.to(device)
    train_ds._actions = train_ds._actions.to(device)
    train_ds._rewards = train_ds._rewards.to(device)
    train_ds._next_obs = train_ds._next_obs.to(device)
    train_ds._dones = train_ds._dones.to(device)

    # Networks
    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden=config.hidden_layers,
        dropout=config.dropout,
        log_std_init=config.log_std_init,
    ).to(device)
    q1 = QNetwork(obs_dim, action_dim, config.hidden_layers, config.dropout).to(device)
    q2 = QNetwork(obs_dim, action_dim, config.hidden_layers, config.dropout).to(device)
    q1_target = QNetwork(obs_dim, action_dim, config.hidden_layers, config.dropout).to(device)
    q2_target = QNetwork(obs_dim, action_dim, config.hidden_layers, config.dropout).to(device)
    value_net = ValueNetwork(obs_dim, config.hidden_layers, config.dropout).to(device)
    _hard_copy(q1_target, q1)
    _hard_copy(q2_target, q2)
    for p in q1_target.parameters():
        p.requires_grad_(False)
    for p in q2_target.parameters():
        p.requires_grad_(False)

    # Optimisers
    pi_opt = torch.optim.Adam(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    q_params = list(q1.parameters()) + list(q2.parameters())
    q_opt = torch.optim.Adam(q_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    v_opt = torch.optim.Adam(
        value_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")

    best_val_mse: float = float("inf")
    best_step: int = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_metrics: Dict[str, Any] = {}
    t0 = time.time()
    rng_gen = torch.Generator(device=device).manual_seed(int(seed))
    n_train = len(train_ds)

    for step in range(int(config.gradient_steps)):
        # Sample minibatch
        idx = torch.randint(0, n_train, (config.batch_size,), generator=rng_gen, device=device)
        s = train_ds._obs[idx]
        a = train_ds._actions[idx]
        r = train_ds._rewards[idx]
        s_next = train_ds._next_obs[idx]
        done = train_ds._dones[idx]

        # --- V update ---
        with torch.no_grad():
            q_target_min = torch.minimum(q1_target(s, a), q2_target(s, a))
        v_pred = value_net(s)
        v_loss = expectile_loss(q_target_min - v_pred, config.tau_expectile).mean()
        v_opt.zero_grad()
        v_loss.backward()
        if config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(value_net.parameters(), config.gradient_clip_norm)
        v_opt.step()

        # --- Q update ---
        with torch.no_grad():
            v_next = value_net(s_next)
            tgt = bellman_target(r, config.gamma, v_next, done)
        q1_pred = q1(s, a)
        q2_pred = q2(s, a)
        q_loss = ((q1_pred - tgt) ** 2).mean() + ((q2_pred - tgt) ** 2).mean()
        q_opt.zero_grad()
        q_loss.backward()
        if config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(q_params, config.gradient_clip_norm)
        q_opt.step()

        # --- Policy update ---
        with torch.no_grad():
            adv_q_min = torch.minimum(q1_target(s, a), q2_target(s, a))
            adv_v = value_net(s)
            adv = adv_q_min - adv_v
            raw_w = torch.exp(config.beta_advantage * adv)
            weight = raw_w.clamp(max=config.advantage_clip)
            clip_frac = float((raw_w >= config.advantage_clip).float().mean().item())
        log_prob = policy.log_prob(s, a)
        policy_loss = -(weight * log_prob).mean()
        pi_opt.zero_grad()
        policy_loss.backward()
        if config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
        pi_opt.step()

        # --- Soft target update ---
        _soft_update(q1_target, q1, config.tau_target)
        _soft_update(q2_target, q2, config.tau_target)

        # --- Eval / log ---
        eval_due = (step + 1) % max(int(config.eval_every_n_steps), 1) == 0
        is_last = step == int(config.gradient_steps) - 1
        if eval_due or is_last:
            val_mse = _eval_policy_mse_entity(policy, val_ds, device=device)
            record = {
                "step": int(step + 1),
                "v_loss": float(v_loss.item()),
                "q_loss": float(q_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "val_policy_mse": float(val_mse),
                "adv_mean": float(adv.mean().item()),
                "adv_std": float(adv.std().item()),
                "adv_clip_frac": float(clip_frac),
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            last_metrics = record
            if val_mse < best_val_mse:
                best_val_mse = float(val_mse)
                best_step = int(step + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    duration = time.time() - t0

    # Persist artefacts
    if best_state is not None:
        torch.save(best_state, output_dir / "policy.pt")
    else:
        torch.save(policy.state_dict(), output_dir / "policy.pt")
    torch.save(q1.state_dict(), output_dir / "q1.pt")
    torch.save(q2.state_dict(), output_dir / "q2.pt")
    torch.save(value_net.state_dict(), output_dir / "value.pt")
    standardiser.save(output_dir / "obs_standardiser.npz")

    arch = policy.architecture_summary()
    arch["obs_names"] = spec.obs_names
    arch["action_names"] = spec.action_names
    arch["obs_dim"] = obs_dim
    arch["action_dim"] = action_dim
    arch["group_key"] = spec.group_key
    (output_dir / "architecture.json").write_text(json.dumps(arch, indent=2))

    summary: Dict[str, Any] = {
        "seed": int(seed),
        "output_dir": str(output_dir),
        "group_key": spec.group_key,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "gradient_steps": int(config.gradient_steps),
        "duration_seconds": float(duration),
        "best_step": int(best_step),
        "best_val_policy_mse": float(best_val_mse),
        "final_metrics": last_metrics,
        "config": dataclasses.asdict(config),
        "action_names": spec.action_names,
        "obs_names": spec.obs_names,
    }
    (output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Multi-seed driver for one group
# ---------------------------------------------------------------------------


def train_entity_multi_seed(
    data_dir: Path,
    output_root: Path,
    *,
    obs_dim: int,
    action_dim: int,
    seeds: Sequence[int],
    val_seeds: Optional[Sequence[int]] = None,
    config: IQLTrainingConfig,
) -> Dict[str, Any]:
    """Train multiple IQL seeds for one agent group."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_key = f"obs{obs_dim}_act{action_dim}"
    seed_summaries: List[Dict[str, Any]] = []
    seeds_index: Dict[str, str] = {}
    t0 = time.time()

    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        print(f"[iql_entity] group={group_key} seed={seed} → {seed_dir}", flush=True)
        summary = train_entity_single_seed(
            data_dir=data_dir,
            output_dir=seed_dir,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(seed),
            val_seeds=val_seeds,
            config=config,
        )
        seed_summaries.append(summary)
        seeds_index[str(int(seed))] = str(seed_dir)
        fm = summary.get("final_metrics", {})
        print(
            f"  seed={seed} best_val_policy_mse={summary['best_val_policy_mse']:.6f} "
            f"best_step={summary['best_step']} "
            f"policy_loss={fm.get('policy_loss', float('nan')):.4f} "
            f"duration={summary['duration_seconds']:.1f}s",
            flush=True,
        )

    duration = time.time() - t0
    best = [s["best_val_policy_mse"] for s in seed_summaries]
    aggregate = {
        "group_key": group_key,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_seeds": len(seed_summaries),
        "seeds": [int(s["seed"]) for s in seed_summaries],
        "best_val_policy_mse_mean": float(np.mean(best)) if best else float("nan"),
        "best_val_policy_mse_std": float(np.std(best, ddof=0)) if best else float("nan"),
        "duration_seconds": float(duration),
        "data_dir": str(data_dir),
        "config": dataclasses.asdict(config),
        "per_seed": seed_summaries,
    }
    (output_root / "multi_seed_summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_root / "seeds_index.json").write_text(json.dumps(seeds_index, indent=2))
    return aggregate


# ---------------------------------------------------------------------------
# Train all four groups
# ---------------------------------------------------------------------------


def train_all_groups(
    data_dir: Path,
    output_root: Path,
    *,
    seeds: Sequence[int],
    val_seeds: Optional[Sequence[int]] = None,
    config: IQLTrainingConfig,
    groups: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Train IQL for all (or specified) agent groups.

    Parameters
    ----------
    groups:
        List of ``(obs_dim, action_dim)`` pairs.  Defaults to all four groups.
    """
    if groups is None:
        groups = list(AGENT_GROUPS)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    for obs_dim, action_dim in groups:
        group_key = f"obs{obs_dim}_act{action_dim}"
        group_out = output_root / group_key
        print(f"\n[iql_entity] === group {group_key} ===", flush=True)
        agg = train_entity_multi_seed(
            data_dir=data_dir,
            output_root=group_out,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seeds=seeds,
            val_seeds=val_seeds,
            config=config,
        )
        results[group_key] = agg

    (output_root / "all_groups_summary.json").write_text(json.dumps(results, indent=2))
    return results
