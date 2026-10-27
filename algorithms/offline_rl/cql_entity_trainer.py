"""CQL (Conservative Q-Learning) trainer for the entity-interface dataset.

Extends the IQL entity trainer by adding a conservative Q penalty to the Q
update.  Everything else (data loading, V network, policy update, artefacts)
is identical to :mod:`algorithms.offline_rl.iql_entity_trainer`.

CQL Conservative Q penalty
---------------------------
For each minibatch, N random actions are sampled uniformly from ``[-1, 1]``
and the CQL penalty is:

    L_CQL = cql_alpha * mean( logsumexp_rand(Q(s, a_rand)) - Q(s, a_data) )

Applied independently to Q1 and Q2, then added to the standard Bellman MSE.

This keeps Q values conservative at out-of-distribution actions, reducing
overestimation without requiring any on-policy roll-outs.

Config additions vs IQL
-----------------------
``cql_alpha`` — weight of the conservative penalty (default 0.2).
``cql_n_random_actions`` — random actions per state for logsumexp (default 10).
``cql_target_action_gap`` — optional min-Q clip (default disabled = −inf).

Artefacts (identical layout to IQL entity trainer)
---------------------------------------------------
Per-seed: ``policy.pt``, ``q1.pt``, ``q2.pt``, ``value.pt``,
          ``obs_standardiser.npz``, ``metrics.jsonl``, ``architecture.json``,
          ``seed_summary.json``.
Aggregated: ``multi_seed_summary.json``, ``seeds_index.json``.
All-groups: ``all_groups_summary.json``.
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

from algorithms.offline_rl.checkpoint_utils import atomic_save, write_status
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
# CQL config (extends IQLTrainingConfig with CQL-specific fields)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CQLTrainingConfig(IQLTrainingConfig):
    """Hyperparameters for CQL training.

    Inherits all IQL fields; adds conservative Q-learning params.
    """

    # CQL-specific
    cql_alpha: float = 0.2
    """Weight of the conservative Q penalty."""

    cql_n_random_actions: int = 10
    """Random actions sampled per state for the logsumexp approximation."""

    cql_min_q_weight: float = 0.0
    """If > 0, used as a lower bound on the CQL penalty (target action gap).
    Set to 0.0 to disable (standard CQL without gap target)."""


# ---------------------------------------------------------------------------
# Eval helper (shared with IQL entity trainer)
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
# CQL conservative penalty
# ---------------------------------------------------------------------------


def cql_conservative_loss(
    q_net: QNetwork,
    obs: torch.Tensor,
    actions_data: torch.Tensor,
    *,
    n_random: int,
    device: torch.device,
    rng_gen: torch.Generator,
) -> torch.Tensor:
    """CQL conservative penalty for one Q network.

    Returns a scalar: ``mean(logsumexp(Q(s, a_rand)) - Q(s, a_data))``.

    Parameters
    ----------
    q_net:   The Q network to evaluate.
    obs:     ``(B, obs_dim)`` — current observations.
    actions_data: ``(B, action_dim)`` — dataset actions.
    n_random: Number of random actions to sample per state.
    """
    batch_size, obs_dim = obs.shape
    action_dim = actions_data.shape[1]

    # Random actions: (B, N, action_dim) uniform in [-1, 1]
    rand_shape = (batch_size * n_random, action_dim)
    rand_actions = (
        torch.rand(rand_shape, device=device, generator=rng_gen) * 2.0 - 1.0
    )  # (B*N, action_dim)

    # Expand obs: (B, N, obs_dim) → (B*N, obs_dim)
    obs_expand = obs.unsqueeze(1).expand(-1, n_random, -1).reshape(-1, obs_dim)

    with torch.no_grad():
        q_rand = q_net(obs_expand, rand_actions).reshape(batch_size, n_random)

    # log Z ≈ logsumexp - log(N)
    log_z = torch.logsumexp(q_rand, dim=1) - math.log(n_random)

    # Q for dataset actions
    q_data = q_net(obs, actions_data)  # (B,)

    return (log_z - q_data).mean()


# ---------------------------------------------------------------------------
# Single-seed CQL training
# ---------------------------------------------------------------------------


def train_cql_single_seed(
    data_dir: Path,
    output_dir: Path,
    *,
    obs_dim: int,
    action_dim: int,
    seed: int,
    val_seeds: Optional[Sequence[int]] = None,
    config: CQLTrainingConfig,
    force: bool = False,
) -> Dict[str, Any]:
    """Train one CQL seed for a given agent group.

    Resume / idempotency
    --------------------
    Same contract as :func:`algorithms.offline_rl.iql_entity_trainer.train_entity_single_seed`:

    * ``seed.done`` → early-return with persisted summary (unless ``force``).
    * ``checkpoint_latest.pt`` → restore networks / optimisers / RNG and
      continue from saved step (unless ``force``).
    * Persists ``best_policy.pt`` on each val MSE improvement.
    * Writes ``status.json`` after every checkpoint.

    Parameters
    ----------
    data_dir:
        Directory with ``seed_*.parquet`` files.
    output_dir:
        Where to save artefacts.
    obs_dim, action_dim:
        Target agent group.
    seed:
        Random seed for weight init and minibatch sampling.
    val_seeds:
        Dataset seeds to hold out for validation.  Default: last seed.
    config:
        CQL hyperparameters.
    force:
        If ``True``, ignore both ``seed.done`` and ``checkpoint_latest.pt``
        and retrain from scratch.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Early-return if this seed already finished successfully.
    seed_done_path = output_dir / "seed.done"
    summary_path = output_dir / "seed_summary.json"
    if seed_done_path.exists() and summary_path.exists() and not force:
        print(
            f"[cql_entity] seed.done present at {output_dir} — skipping (force=False)",
            flush=True,
        )
        return json.loads(summary_path.read_text())

    if force and seed_done_path.exists():
        seed_done_path.unlink()

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

    # Networks (same as IQL)
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
    checkpoint_path = output_dir / "checkpoint_latest.pt"

    best_val_mse: float = float("inf")
    best_step: int = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_metrics: Dict[str, Any] = {}
    rng_gen = torch.Generator(device=device).manual_seed(int(seed))
    n_train = len(train_ds)

    # Resume bookkeeping
    start_step = 0
    wall_clock_resumed = 0.0
    resumed = False
    if checkpoint_path.exists() and not force:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy_state"])
        q1.load_state_dict(ckpt["qf1_state"])
        q2.load_state_dict(ckpt["qf2_state"])
        q1_target.load_state_dict(ckpt["qf1_target_state"])
        q2_target.load_state_dict(ckpt["qf2_target_state"])
        value_net.load_state_dict(ckpt["vf_state"])
        pi_opt.load_state_dict(ckpt["policy_opt_state"])
        q_opt.load_state_dict(ckpt["qf_opt_state"])
        v_opt.load_state_dict(ckpt["vf_opt_state"])
        torch.set_rng_state(ckpt["rng_state_torch"])
        np.random.set_state(ckpt["rng_state_numpy"])
        rng_gen.set_state(ckpt["rng_state_gen"].cpu())
        best_val_mse = float(ckpt["best_val_mse"])
        best_step = int(ckpt["best_step"])
        if ckpt.get("best_policy_state") is not None:
            best_state = {k: v.cpu().clone() for k, v in ckpt["best_policy_state"].items()}
        start_step = int(ckpt["step"])
        wall_clock_resumed = float(ckpt.get("wall_clock_seconds", 0.0))
        resumed = True
        print(
            f"[cql_entity] resuming at step={start_step}, "
            f"best_val_mse={best_val_mse:.6f} (best_step={best_step})",
            flush=True,
        )
    else:
        metrics_path.write_text("")

    t0 = time.time()
    checkpoint_every = max(1, int(config.checkpoint_every_n_steps))

    def _snapshot_checkpoint(step: int, val_mse: Optional[float]) -> None:
        wall_clock = wall_clock_resumed + (time.time() - t0)
        payload = {
            "step": int(step),
            "policy_state": policy.state_dict(),
            "qf1_state": q1.state_dict(),
            "qf2_state": q2.state_dict(),
            "qf1_target_state": q1_target.state_dict(),
            "qf2_target_state": q2_target.state_dict(),
            "vf_state": value_net.state_dict(),
            "policy_opt_state": pi_opt.state_dict(),
            "qf_opt_state": q_opt.state_dict(),
            "vf_opt_state": v_opt.state_dict(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_numpy": np.random.get_state(),
            "rng_state_gen": rng_gen.get_state(),
            "best_val_mse": float(best_val_mse),
            "best_step": int(best_step),
            "best_policy_state": (
                {k: v.detach().cpu().clone() for k, v in best_state.items()}
                if best_state is not None
                else None
            ),
            "wall_clock_seconds": float(wall_clock),
        }
        atomic_save(payload, checkpoint_path)
        write_status(
            output_dir / "status.json",
            {
                "group_key": spec.group_key,
                "seed": int(seed),
                "step": int(step),
                "val_mse": (float(val_mse) if val_mse is not None else None),
                "best_val_mse": float(best_val_mse),
                "best_step": int(best_step),
                "wall_clock_seconds": float(wall_clock),
            },
        )

    for step in range(start_step, int(config.gradient_steps)):
        # Sample minibatch
        idx = torch.randint(0, n_train, (config.batch_size,), generator=rng_gen, device=device)
        s = train_ds._obs[idx]
        a = train_ds._actions[idx]
        r = train_ds._rewards[idx]
        s_next = train_ds._next_obs[idx]
        done = train_ds._dones[idx]

        # --- V update (same as IQL) ---
        with torch.no_grad():
            q_target_min = torch.minimum(q1_target(s, a), q2_target(s, a))
        v_pred = value_net(s)
        v_loss = expectile_loss(q_target_min - v_pred, config.tau_expectile).mean()
        v_opt.zero_grad()
        v_loss.backward()
        if config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(value_net.parameters(), config.gradient_clip_norm)
        v_opt.step()

        # --- Q update with CQL penalty ---
        with torch.no_grad():
            v_next = value_net(s_next)
            tgt = bellman_target(r, config.gamma, v_next, done)
        q1_pred = q1(s, a)
        q2_pred = q2(s, a)
        bellman_mse = ((q1_pred - tgt) ** 2).mean() + ((q2_pred - tgt) ** 2).mean()

        # CQL conservative penalty
        cql1 = cql_conservative_loss(
            q1, s, a, n_random=config.cql_n_random_actions, device=device, rng_gen=rng_gen
        )
        cql2 = cql_conservative_loss(
            q2, s, a, n_random=config.cql_n_random_actions, device=device, rng_gen=rng_gen
        )
        cql_pen = config.cql_alpha * (cql1 + cql2)

        # Optional target action gap (clamp CQL penalty from below)
        if config.cql_min_q_weight > 0.0:
            cql_pen = torch.clamp(cql_pen, min=float(config.cql_min_q_weight))

        q_loss = bellman_mse + cql_pen
        q_opt.zero_grad()
        q_loss.backward()
        if config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(q_params, config.gradient_clip_norm)
        q_opt.step()

        # --- Policy update (AWR, same as IQL) ---
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
        completed_step = step + 1
        eval_due = completed_step % max(int(config.eval_every_n_steps), 1) == 0
        is_last = step == int(config.gradient_steps) - 1
        val_mse: Optional[float] = None
        if eval_due or is_last:
            val_mse = _eval_policy_mse_entity(policy, val_ds, device=device)
            record = {
                "step": int(completed_step),
                "v_loss": float(v_loss.item()),
                "q_loss": float(q_loss.item()),
                "bellman_mse": float(bellman_mse.item()),
                "cql_penalty": float(cql_pen.item()),
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
                best_step = int(completed_step)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                }
                atomic_save(best_state, output_dir / "best_policy.pt")

        # --- Checkpoint ---
        if completed_step % checkpoint_every == 0 or is_last:
            _snapshot_checkpoint(completed_step, val_mse)

    duration = wall_clock_resumed + (time.time() - t0)

    # Persist artefacts
    if best_state is not None:
        atomic_save(best_state, output_dir / "policy.pt")
    else:
        atomic_save(policy.state_dict(), output_dir / "policy.pt")
    atomic_save(q1.state_dict(), output_dir / "q1.pt")
    atomic_save(q2.state_dict(), output_dir / "q2.pt")
    atomic_save(value_net.state_dict(), output_dir / "value.pt")
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
        "resumed": bool(resumed),
        "best_step": int(best_step),
        "best_val_policy_mse": float(best_val_mse),
        "final_metrics": last_metrics,
        "config": dataclasses.asdict(config),
        "action_names": spec.action_names,
        "obs_names": spec.obs_names,
    }
    (output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))

    seed_done_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "group_key": spec.group_key,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    )
    return summary


# ---------------------------------------------------------------------------
# Multi-seed driver for one group
# ---------------------------------------------------------------------------


def train_cql_multi_seed(
    data_dir: Path,
    output_root: Path,
    *,
    obs_dim: int,
    action_dim: int,
    seeds: Sequence[int],
    val_seeds: Optional[Sequence[int]] = None,
    config: CQLTrainingConfig,
    force: bool = False,
) -> Dict[str, Any]:
    """Train multiple CQL seeds for one agent group.

    Parameters
    ----------
    force:
        If ``True``, ignore each seed's ``seed.done`` / ``checkpoint_latest.pt``
        sentinels and retrain from scratch.  Forwarded to
        :func:`train_cql_single_seed`.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    group_key = f"obs{obs_dim}_act{action_dim}"
    seed_summaries: List[Dict[str, Any]] = []
    seeds_index: Dict[str, str] = {}
    t0 = time.time()

    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        print(f"[cql_entity] group={group_key} seed={seed} → {seed_dir}", flush=True)
        summary = train_cql_single_seed(
            data_dir=data_dir,
            output_dir=seed_dir,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(seed),
            val_seeds=val_seeds,
            config=config,
            force=force,
        )
        seed_summaries.append(summary)
        seeds_index[str(int(seed))] = str(seed_dir)
        fm = summary.get("final_metrics", {})
        print(
            f"  seed={seed} best_val_policy_mse={summary['best_val_policy_mse']:.6f} "
            f"best_step={summary['best_step']} "
            f"policy_loss={fm.get('policy_loss', float('nan')):.4f} "
            f"cql_penalty={fm.get('cql_penalty', float('nan')):.4f} "
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
    config: CQLTrainingConfig,
    groups: Optional[Sequence[Tuple[int, int]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Train CQL for all (or specified) agent groups.

    Parameters
    ----------
    groups:
        List of ``(obs_dim, action_dim)`` pairs.  Defaults to all four groups.
    force:
        If ``True``, ignore ``seed.done`` / ``checkpoint_latest.pt`` sentinels
        and retrain every (group, seed) from scratch.  Forwarded to
        :func:`train_cql_multi_seed`.
    """
    if groups is None:
        groups = list(AGENT_GROUPS)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    for obs_dim, action_dim in groups:
        group_key = f"obs{obs_dim}_act{action_dim}"
        group_out = output_root / group_key
        print(f"\n[cql_entity] === group {group_key} ===", flush=True)
        agg = train_cql_multi_seed(
            data_dir=data_dir,
            output_root=group_out,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seeds=seeds,
            val_seeds=val_seeds,
            config=config,
            force=force,
        )
        results[group_key] = agg

    (output_root / "all_groups_summary.json").write_text(json.dumps(results, indent=2))
    return results
