"""Per-step behavioral analysis for RBCSmart vs IQLEntityAgent vs CQLEntityAgent.

Runs one evaluation episode per agent, logging per-step data (district net
consumption, electricity price, EV and storage action means) and generating
four diagnostic figures that explain the quantitative benchmark differences.

Usage
-----
# RBC only (no model files needed):
.venv/bin/python -m scripts.analyze_entity_policies \\
    --output-dir runs/analysis/seed200

# Full comparison (requires trained model artifacts):
.venv/bin/python -m scripts.analyze_entity_policies \\
    --iql-root runs/offline_iql_entity/run-001 \\
    --cql-root runs/offline_cql_entity/run-001 \\
    --output-dir runs/analysis/seed200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as _loguru_logger  # noqa: E402

_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

# Re-use env factory and helpers from benchmark script to avoid duplication
from scripts.benchmark_entity_agents import (  # noqa: E402
    _extract_kpis,
    _make_adapter,
    _make_env,
    _make_rbcsmart,
    SCHEMA_PATH,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--output-dir", "-o",
        required=True,
        type=Path,
        dest="output_dir",
        help="Directory for per-step CSVs and figures.",
    )
    p.add_argument(
        "--schema",
        default=SCHEMA_PATH,
        help="Path to CityLearn schema.json (default: 15s parquet dataset).",
    )
    p.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        dest="episode_steps",
        help="Steps per episode. Auto-detected from schema if not given.",
    )
    p.add_argument(
        "--no-offline",
        dest="offline",
        action="store_false",
        default=True,
        help="Disable offline=True for CSV-based datasets.",
    )
    p.add_argument("--seed", type=int, default=200,
                   help="Evaluation seed (default: 200).")
    p.add_argument("--iql-root", type=Path, default=None,
                   help="Output root from train_iql_entity.py")
    p.add_argument("--cql-root", type=Path, default=None,
                   help="Output root from train_cql_entity.py")
    p.add_argument("--device", default="cpu")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_idx(obs_names_flat: List[str]) -> Optional[int]:
    """Return index of the district electricity pricing feature, or None."""
    try:
        return obs_names_flat.index("district__electricity_pricing")
    except ValueError:
        return None


def _split_actions(
    actions: List,
    action_names_all: List[List[str]],
) -> Tuple[List[float], List[float]]:
    """Separate EV-charging actions from battery-storage actions.

    Returns ``(ev_actions, storage_actions)`` as flat lists of float.
    EV actions are identified by ``"electric_vehicle"`` in the action name.
    """
    ev_vals: List[float] = []
    storage_vals: List[float] = []
    for per_building_actions, names in zip(actions, action_names_all):
        arr = np.array(per_building_actions).flatten()
        for j, name in enumerate(names):
            if j < len(arr):
                if "electric_vehicle" in name:
                    ev_vals.append(float(arr[j]))
                else:
                    storage_vals.append(float(arr[j]))
    return ev_vals, storage_vals


# ---------------------------------------------------------------------------
# Logged rollouts
# ---------------------------------------------------------------------------


def _logged_rbc_rollout(
    *, env_seed: int, env_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """RBCSmart rollout with per-step logging."""
    env = _make_env(env_seed, **env_kwargs)
    adapter = _make_adapter(env)

    obs_payload, _ = env.reset()
    obs_list, obs_names, _ = adapter.to_agent_encoded_observations(obs_payload)
    action_names_all = [list(n) for n in env.action_names]

    rbc = _make_rbcsmart(obs_names, action_names_all, env)
    pidx = _price_idx(obs_names[0])
    sps = env.seconds_per_time_step

    steps: List[Dict[str, Any]] = []
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        actions = rbc.predict(obs_list, deterministic=True)
        ev_vals, storage_vals = _split_actions(actions, action_names_all)

        price_norm = float(obs_list[0][pidx]) if pidx is not None else 0.0
        district_net = float(
            sum(float(b.net_electricity_consumption[0]) for b in env.buildings)
        )

        steps.append({
            "step": step,
            "hour": (step * sps) / 3600.0,
            "price_norm": price_norm,
            "district_net_kwh": district_net,
            "ev_action_mean": float(np.mean(ev_vals)) if ev_vals else 0.0,
            "storage_action_mean": float(np.mean(storage_vals)) if storage_vals else 0.0,
        })

        env_actions = adapter.to_entity_actions(actions, action_names_all)
        next_payload, _, terminated, truncated, _ = env.step(env_actions)
        next_obs_list, _, _ = adapter.to_agent_encoded_observations(next_payload)
        obs_list = next_obs_list
        step += 1

    kpi_df = env.evaluate()
    return {
        "label": "RBCSmart",
        "env_seed": env_seed,
        "steps": steps,
        "district_kpis": _extract_kpis(kpi_df, level="district"),
    }


def _logged_entity_rollout(
    agent,
    *,
    env_seed: int,
    label: str,
    env_kwargs: Dict[str, Any],
    max_steps: int = 6000,
) -> Dict[str, Any]:
    """IQL/CQL entity rollout with per-step logging."""
    from gymnasium import spaces as gym_spaces

    env = _make_env(env_seed, **env_kwargs)
    adapter = _make_adapter(env)

    obs_payload, _ = env.reset()
    obs_list, obs_names, _ = adapter.to_agent_encoded_observations(obs_payload)
    action_names_all = [list(n) for n in env.action_names]

    obs_spaces = [
        gym_spaces.Box(low=-1e6, high=1e6, shape=(len(o),), dtype=np.float32)
        for o in obs_names
    ]
    agent.attach_environment(
        observation_names=obs_names,
        action_names=action_names_all,
        action_space=list(getattr(env, "flat_action_space", [])),
        observation_space=obs_spaces,
        metadata={
            "building_names": [b.name for b in env.buildings],
            "seconds_per_time_step": env.seconds_per_time_step,
        },
    )

    pidx = _price_idx(obs_names[0])
    sps = env.seconds_per_time_step

    steps: List[Dict[str, Any]] = []
    terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        actions = agent.predict(obs_list, deterministic=True)
        ev_vals, storage_vals = _split_actions(actions, action_names_all)

        price_norm = float(obs_list[0][pidx]) if pidx is not None else 0.0
        district_net = float(
            sum(float(b.net_electricity_consumption[0]) for b in env.buildings)
        )

        steps.append({
            "step": step,
            "hour": (step * sps) / 3600.0,
            "price_norm": price_norm,
            "district_net_kwh": district_net,
            "ev_action_mean": float(np.mean(ev_vals)) if ev_vals else 0.0,
            "storage_action_mean": float(np.mean(storage_vals)) if storage_vals else 0.0,
        })

        env_actions = adapter.to_entity_actions(actions, action_names_all)
        next_payload, _, terminated, truncated, _ = env.step(env_actions)
        next_obs_list, _, _ = adapter.to_agent_encoded_observations(next_payload)
        obs_list = next_obs_list
        step += 1
        if step % 1000 == 0:
            print(f"  [{label} seed={env_seed}] step {step}", flush=True)
        if step >= max_steps:
            print(f"  [{label} seed={env_seed}] WARN: max_steps={max_steps} reached",
                  flush=True)
            break

    kpi_df = env.evaluate()
    return {
        "label": label,
        "env_seed": env_seed,
        "steps": steps,
        "district_kpis": _extract_kpis(kpi_df, level="district"),
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

AGENT_COLORS = {"RBCSmart": "#666666", "IQL": "#1f77b4", "CQL": "#ff7f0e"}

HEADLINE_KPIs = [
    "cost_total",
    "carbon_emissions_total",
    "daily_peak_average",
    "ramping_average",
    "electricity_consumption_total",
    "zero_net_energy",
]

HIGHER_IS_BETTER = {"zero_net_energy"}


def _generate_figures(rollouts: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate 4 diagnostic figures from a list of rollout result dicts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = {r["label"]: pd.DataFrame(r["steps"]) for r in rollouts}

    # -------------------------------------------------------------------------
    # Figure 1: District net demand profile — mean ± std by hour of day
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    for r in rollouts:
        df = dfs[r["label"]].copy()
        df["hour_bin"] = df["hour"].apply(lambda h: int(h % 24))
        by_hour = df.groupby("hour_bin")["district_net_kwh"]
        mean_v = by_hour.mean()
        std_v = by_hour.std().fillna(0)
        color = AGENT_COLORS.get(r["label"])
        ax.plot(mean_v.index, mean_v.values, label=r["label"],
                color=color, linewidth=1.8)
        ax.fill_between(mean_v.index,
                        mean_v.values - std_v.values,
                        mean_v.values + std_v.values,
                        color=color, alpha=0.15)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("District Net Consumption (kWh/step)")
    ax.set_title("District Power Demand Profile — Mean ± Std by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_demand_profile.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Mean EV charging action by hour of day
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    for r in rollouts:
        df = dfs[r["label"]].copy()
        df["hour_bin"] = df["hour"].apply(lambda h: int(h % 24))
        by_hour = df.groupby("hour_bin")["ev_action_mean"].mean()
        color = AGENT_COLORS.get(r["label"])
        ax.plot(by_hour.index, by_hour.values, label=r["label"],
                color=color, linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean EV Charging Action (normalized)")
    ax.set_title("EV Charging Activity by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_ev_action_by_hour.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Electricity price vs. EV charging action (scatter + trend)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rollouts:
        df = dfs[r["label"]]
        sample = df.sample(min(800, len(df)), random_state=42)
        color = AGENT_COLORS.get(r["label"])
        ax.scatter(sample["price_norm"], sample["ev_action_mean"],
                   label=r["label"], color=color, alpha=0.25, s=6)
        # Linear trend line (suppress RankWarning for very short/constant series)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            z = np.polyfit(df["price_norm"], df["ev_action_mean"], 1)
        xs = np.linspace(df["price_norm"].min(), df["price_norm"].max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=color, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Electricity Price (normalized)")
    ax.set_ylabel("Mean EV Charging Action")
    ax.set_title("Price Response: EV Charging vs. Electricity Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_price_vs_action.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: KPI % improvement vs RBCSmart (only when >1 agent present)
    # -------------------------------------------------------------------------
    non_rbc = [r for r in rollouts if r["label"] != "RBCSmart"]
    if not non_rbc:
        return  # skip — nothing to compare against

    rbc_kpis = next(
        (r["district_kpis"] for r in rollouts if r["label"] == "RBCSmart"), {}
    )
    available_kpis = [k for k in HEADLINE_KPIs if k in rbc_kpis]

    fig, ax = plt.subplots(figsize=(8, max(4, len(available_kpis) * 0.6)))
    y_pos = np.arange(len(available_kpis))
    bar_height = 0.35

    for i, r in enumerate(non_rbc[:2]):
        deltas = []
        for k in available_kpis:
            rbc_v = rbc_kpis.get(k, float("nan"))
            agent_v = r["district_kpis"].get(k, float("nan"))
            if np.isfinite(rbc_v) and np.isfinite(agent_v) and rbc_v != 0:
                raw_delta = (agent_v - rbc_v) / abs(rbc_v) * 100
                # Flip sign so "improvement" (lower cost, higher ZNE) is positive
                deltas.append(raw_delta if k in HIGHER_IS_BETTER else -raw_delta)
            else:
                deltas.append(0.0)

        offset = (i - (len(non_rbc) - 1) / 2) * bar_height
        colors = [
            AGENT_COLORS.get(r["label"], "#333333") if d >= 0 else "#cc4444"
            for d in deltas
        ]
        ax.barh(y_pos + offset, deltas, bar_height,
                label=r["label"], color=colors, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([k.replace("_", "\n") for k in available_kpis], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% Improvement vs RBCSmart  (positive = better)")
    ax.set_title("KPI Improvement vs RBCSmart Baseline")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_kpi_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    import pandas as pd

    args = _build_parser().parse_args(argv)

    from algorithms.offline_rl.entity_schema import episode_steps_for_schema

    episode_steps = (
        args.episode_steps
        if args.episode_steps is not None
        else episode_steps_for_schema(args.schema)
    )
    env_kwargs: Dict[str, Any] = dict(
        schema_path=args.schema,
        episode_steps=episode_steps,
        offline=args.offline,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] seed={args.seed}  schema={args.schema}")
    print(f"[analyze] episode_steps={episode_steps}  output_dir={output_dir}")

    rollouts: List[Dict[str, Any]] = []

    print("[analyze] Running RBCSmart rollout...")
    rollouts.append(_logged_rbc_rollout(env_seed=args.seed, env_kwargs=env_kwargs))

    if args.iql_root is not None:
        from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent

        print(f"[analyze] Running IQL rollout from {args.iql_root}...")
        iql_agent = IQLEntityAgent.from_model_dir(args.iql_root, device=args.device)
        rollouts.append(
            _logged_entity_rollout(
                iql_agent, env_seed=args.seed, label="IQL", env_kwargs=env_kwargs
            )
        )

    if args.cql_root is not None:
        from algorithms.offline_rl.cql_entity_agent import CQLEntityAgent

        print(f"[analyze] Running CQL rollout from {args.cql_root}...")
        cql_agent = CQLEntityAgent.from_model_dir(args.cql_root, device=args.device)
        rollouts.append(
            _logged_entity_rollout(
                cql_agent, env_seed=args.seed, label="CQL", env_kwargs=env_kwargs
            )
        )

    # Save per-step CSVs
    for r in rollouts:
        df = pd.DataFrame(r["steps"])
        csv_path = output_dir / f"steps_{r['label']}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[analyze] {r['label']}: {len(r['steps'])} steps → {csv_path.name}")

    # Generate figures
    _generate_figures(rollouts, output_dir)
    print(f"[analyze] Figures saved to {output_dir}/")

    # Save KPI summary
    summary = {r["label"]: r["district_kpis"] for r in rollouts}
    (output_dir / "kpi_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[analyze] KPI summary → {output_dir}/kpi_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
