"""IQL trainer.

Implements the IQL update loop (V expectile regression, twin Q Bellman MSE,
advantage-weighted policy regression, soft target updates) and a multi-seed
driver.

Per-seed artefacts (under ``output_dir``):
  * ``policy.pt``           — best-checkpoint policy state_dict
  * ``q1.pt``, ``q2.pt``    — final Q state_dicts (target nets discarded)
  * ``value.pt``            — final V state_dict
  * ``obs_standardiser.npz``
  * ``metrics.jsonl``       — one line per eval-step
  * ``architecture.json``
  * ``seed_summary.json``

Aggregated under ``output_root``:
  * ``multi_seed_summary.json``
  * ``seeds_index.json``

The "best" policy is selected by deterministic policy MSE on the held-out
validation split (same proxy as BC). True on-policy return is unobservable
offline; this proxy at least catches divergence.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from algorithms.offline_rl.iql_dataset import IQLDataset, IQLSplit, load_iql_split
from algorithms.offline_rl.iql_networks import (
    GaussianPolicy,
    QNetwork,
    ValueNetwork,
)


# ---------------------------------------------------------------------------
# Loss primitives (pure functions, testable without networks)
# ---------------------------------------------------------------------------


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared loss.

    ``L_tau(u) = |tau - 1{u < 0}| * u^2``.

    With ``diff = q_target - v_pred`` and ``tau = 0.7``, positive ``diff``
    (V under-predicts Q) is weighted ``0.7``; negative is weighted ``0.3``.
    Returns the per-element loss (no reduction).
    """
    if not (0.0 < float(tau) < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    weight = torch.where(diff > 0, float(tau), 1.0 - float(tau))
    return weight * diff.pow(2)


def bellman_target(
    reward: torch.Tensor,
    gamma: float,
    next_v: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """``r + γ * (1 - done) * V(s')`` (standard TD target)."""
    return reward + float(gamma) * (1.0 - done) * next_v


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IQLTrainingConfig:
    # Architecture
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    log_std_init: float = math.log(0.1)
    # IQL hyperparameters
    tau_expectile: float = 0.7
    beta_advantage: float = 3.0
    advantage_clip: float = 100.0
    gamma: float = 0.99
    tau_target: float = 0.005
    # Optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    batch_size: int = 256
    gradient_steps: int = 150_000
    eval_every_n_steps: int = 2_500
    checkpoint_every_n_steps: int = 5_000
    """Persist ``checkpoint_latest.pt`` every N gradient steps (Phase 2 resume)."""
    val_fraction: float = 0.1
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def _soft_update(target: nn.Module, source: nn.Module, tau_target: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau_target).add_(sp.data, alpha=tau_target)


@torch.no_grad()
def _hard_copy(target: nn.Module, source: nn.Module) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data)


@torch.no_grad()
def _eval_policy_mse(
    policy: GaussianPolicy,
    val_dataset: IQLDataset,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> float:
    policy.eval()
    n = len(val_dataset)
    if n == 0:
        return float("nan")
    obs_all = val_dataset._obs.to(device)  # type: ignore[attr-defined]
    act_all = val_dataset._act.to(device)  # type: ignore[attr-defined]
    sse = 0.0
    total = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pred = policy.predict_deterministic(obs_all[start:end])
        sse += float(((pred - act_all[start:end]) ** 2).sum().item())
        total += end - start
    policy.train()
    # Mean over (rows × action_dims) — same convention as BCTrainer's per-dim
    # collapse: matches ``mse_loss`` reduction.
    return sse / max(total, 1) / int(act_all.shape[1])


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------


def train_single_seed(
    parquet_path: Path,
    output_dir: Path,
    *,
    seed: int,
    config: IQLTrainingConfig,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    split: IQLSplit = load_iql_split(
        parquet_path, val_fraction=config.val_fraction, seed=int(seed)
    )
    obs_dim = int(split.standardiser.mean.shape[0])
    action_dim = len(split.action_feature_names)
    device = _resolve_device(config.device)

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
    q_opt = torch.optim.Adam(
        q_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    v_opt = torch.optim.Adam(
        value_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")

    # Move dataset tensors to device once (small enough — ~26 MB).
    train: IQLDataset = split.train
    train._obs = train._obs.to(device)        # type: ignore[attr-defined]
    train._act = train._act.to(device)        # type: ignore[attr-defined]
    train._rew = train._rew.to(device)        # type: ignore[attr-defined]
    train._next_obs = train._next_obs.to(device)  # type: ignore[attr-defined]
    train._done = train._done.to(device)      # type: ignore[attr-defined]

    best_val_mse: float = float("inf")
    best_step: int = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    last_metrics: Dict[str, Any] = {}
    t0 = time.time()
    rng_gen = torch.Generator(device=device).manual_seed(int(seed))

    for step in range(int(config.gradient_steps)):
        s, a, r, s_next, done = train.sample(config.batch_size, generator=rng_gen)

        # --- V update ---
        with torch.no_grad():
            q_target_min = torch.minimum(q1_target(s, a), q2_target(s, a))
        v_pred = value_net(s)
        v_loss = expectile_loss(q_target_min - v_pred, config.tau_expectile).mean()
        v_opt.zero_grad()
        v_loss.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.gradient_clip_norm)
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
            torch.nn.utils.clip_grad_norm_(q_params, config.gradient_clip_norm)
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
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
        pi_opt.step()

        # --- Soft target update ---
        _soft_update(q1_target, q1, config.tau_target)
        _soft_update(q2_target, q2, config.tau_target)

        # --- Eval / log ---
        eval_due = (step + 1) % max(int(config.eval_every_n_steps), 1) == 0
        is_last = step == int(config.gradient_steps) - 1
        if eval_due or is_last:
            val_mse = _eval_policy_mse(policy, split.val, device=device)
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
                best_state = {
                    k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
                }

    duration = time.time() - t0

    # Persist artefacts
    if best_state is not None:
        torch.save(best_state, output_dir / "policy.pt")
    else:
        torch.save(policy.state_dict(), output_dir / "policy.pt")
    torch.save(q1.state_dict(), output_dir / "q1.pt")
    torch.save(q2.state_dict(), output_dir / "q2.pt")
    torch.save(value_net.state_dict(), output_dir / "value.pt")
    split.standardiser.save(output_dir / "obs_standardiser.npz")

    arch = policy.architecture_summary()
    arch["action_feature_names"] = list(split.action_feature_names)
    arch["obs_feature_names"] = list(split.obs_feature_names)
    (output_dir / "architecture.json").write_text(json.dumps(arch, indent=2))

    summary: Dict[str, Any] = {
        "seed": int(seed),
        "output_dir": str(output_dir),
        "n_train": int(split.train_indices.shape[0]),
        "n_val": int(split.val_indices.shape[0]),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "gradient_steps": int(config.gradient_steps),
        "duration_seconds": float(duration),
        "best_step": int(best_step),
        "best_val_policy_mse": float(best_val_mse),
        "final_metrics": last_metrics,
        "config": dataclasses.asdict(config),
        "action_feature_names": list(split.action_feature_names),
    }
    (output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------


def train_multi_seed(
    parquet_path: Path,
    output_root: Path,
    *,
    seeds: Sequence[int],
    config: IQLTrainingConfig,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = Path(parquet_path)

    seed_summaries: List[Dict[str, Any]] = []
    seeds_index: Dict[str, str] = {}
    t0 = time.time()
    for seed in seeds:
        seed_dir = output_root / f"seed_{seed}"
        print(f"[iql] training seed={seed} → {seed_dir}", flush=True)
        summary = train_single_seed(
            parquet_path, seed_dir, seed=int(seed), config=config
        )
        seed_summaries.append(summary)
        seeds_index[str(int(seed))] = str(seed_dir)
        fm = summary.get("final_metrics", {})
        print(
            f"  seed={seed} best_val_policy_mse={summary['best_val_policy_mse']:.6f} "
            f"best_step={summary['best_step']} "
            f"final_policy_loss={fm.get('policy_loss', float('nan')):.4f} "
            f"duration={summary['duration_seconds']:.1f}s",
            flush=True,
        )

    duration = time.time() - t0
    best = [s["best_val_policy_mse"] for s in seed_summaries]
    aggregate = {
        "n_seeds": len(seed_summaries),
        "seeds": [int(s["seed"]) for s in seed_summaries],
        "best_val_policy_mse_mean": float(np.mean(best)) if best else float("nan"),
        "best_val_policy_mse_std": float(np.std(best, ddof=0)) if best else float("nan"),
        "duration_seconds": float(duration),
        "parquet_path": str(parquet_path),
        "config": dataclasses.asdict(config),
        "per_seed": seed_summaries,
    }
    (output_root / "multi_seed_summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_root / "seeds_index.json").write_text(json.dumps(seeds_index, indent=2))
    return aggregate
