"""Benchmark RBCSmartPolicy vs IQLEntityAgent vs CQLEntityAgent.

Rolls out all three agents on the 15-second entity-interface CityLearn
environment (same reward: CostServiceCommunityFeasiblePrecisionRewardV46)
across multiple eval seeds and prints a KPI comparison table.

Usage
-----
Smoke (1 eval seed)::

    .venv/bin/python -m scripts.benchmark_entity_agents \\
        --iql-root runs/offline_iql_entity/run-001 \\
        --cql-root runs/offline_cql_entity/run-001 \\
        --eval-seeds 200 \\
        --output runs/benchmark_entity/smoke.json

Full (10 eval seeds)::

    .venv/bin/python -m scripts.benchmark_entity_agents \\
        --iql-root runs/offline_iql_entity/run-001 \\
        --cql-root runs/offline_cql_entity/run-001 \\
        --output runs/benchmark_entity/results.json

IQL only (no CQL yet)::

    .venv/bin/python -m scripts.benchmark_entity_agents \\
        --iql-root runs/offline_iql_entity/run-001 \\
        --no-cql \\
        --output runs/benchmark_entity/iql_only.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger as _loguru_logger  # noqa: E402
_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

from citylearn.citylearn import CityLearnEnv  # noqa: E402
from reward_function import CostServiceCommunityFeasiblePrecisionRewardV46  # noqa: E402
from utils.entity_adapter import EntityContractAdapter  # noqa: E402
from algorithms.agents.baseline_policies import RBCSmartPolicy  # noqa: E402
from algorithms.agents.base_agent import BaseAgent  # noqa: E402
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent  # noqa: E402
from algorithms.offline_rl.cql_entity_agent import CQLEntityAgent  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_PATH = str(
    REPO_ROOT
    / "datasets"
    / "citylearn_three_phase_electrical_service_demo_15s_parquet"
    / "schema.json"
)

DEFAULT_EVAL_SEEDS: Tuple[int, ...] = tuple(range(200, 210))

HEADLINE_KPIs = (
    "ev_departure_success_rate",   # Gate criterion (higher = better)
    "cost_total",                  # Primary KPI (lower = better)
    "carbon_emissions_total",      # Secondary
    "daily_peak_average",          # Community/peak
    "ramping_average",             # Ramping
    "annual_normalized_unserved_energy_total",  # Feasibility
    "bess_throughput_total_kwh",   # Battery utilisation
)

HIGHER_IS_BETTER = {"ev_departure_success_rate", "bess_throughput_total_kwh"}


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


EPISODE_STEPS: int = 5760  # 1 day at 15-second resolution (matches training data)


def _make_env(seed: int) -> CityLearnEnv:
    return CityLearnEnv(
        schema=SCHEMA_PATH,
        central_agent=False,
        interface="entity",
        topology_mode="static",
        reward_function=CostServiceCommunityFeasiblePrecisionRewardV46,
        random_seed=int(seed),
        episode_time_steps=EPISODE_STEPS,
        offline=True,
    )


def _make_adapter(env: CityLearnEnv) -> EntityContractAdapter:
    return EntityContractAdapter(
        env,
        normalization_enabled=True,
        clip=True,
        encoding_profile="minmax_space",
    )


# ---------------------------------------------------------------------------
# RBCSmart factory helper
# ---------------------------------------------------------------------------


def _make_rbcsmart(
    obs_names: List[List[str]],
    action_names: List[List[str]],
    env: CityLearnEnv,
) -> RBCSmartPolicy:
    from gymnasium import spaces as gym_spaces
    rbc = RBCSmartPolicy(
        config={
            "algorithm": {"name": "RBCSmartPolicy", "hyperparameters": {}},
            "simulator": {},
        }
    )
    obs_spaces = [
        gym_spaces.Box(low=-1e6, high=1e6, shape=(len(obs),), dtype=np.float32)
        for obs in obs_names
    ]
    rbc.attach_environment(
        observation_names=obs_names,
        action_names=action_names,
        action_space=list(getattr(env, "flat_action_space", [])),
        observation_space=obs_spaces,
        metadata={
            "building_names": [b.name for b in env.buildings],
            "seconds_per_time_step": env.seconds_per_time_step,
        },
    )
    return rbc


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------


def _extract_kpis(kpi_df, *, level: str, name: Optional[str] = None) -> Dict[str, float]:
    import pandas as pd
    sub = kpi_df[kpi_df["level"] == level]
    if name is not None:
        sub = sub[sub["name"] == name]
    out: Dict[str, float] = {}
    for _, row in sub.iterrows():
        try:
            out[str(row["cost_function"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Entity-interface rollout
# ---------------------------------------------------------------------------


def entity_rollout(
    agent: BaseAgent,
    *,
    env_seed: int,
    label: str,
    max_steps: int = 6000,
) -> Dict[str, Any]:
    """Run one full-episode entity rollout and return KPIs.

    The agent is called with per-agent observation vectors from the
    EntityContractAdapter.  If the agent is not an IQL/CQL entity agent,
    it is called directly (e.g. RBCSmartPolicy which handles its own dispatch).
    """
    env = _make_env(env_seed)
    adapter = _make_adapter(env)

    obs_payload, _ = env.reset()
    obs_list, obs_names, _ = adapter.to_agent_encoded_observations(obs_payload)
    action_names = [list(names) for names in env.action_names]

    # Attach environment (gives obs/action dims, needed by IQLEntityAgent)
    from gymnasium import spaces as gym_spaces
    obs_spaces = [
        gym_spaces.Box(low=-1e6, high=1e6, shape=(len(obs),), dtype=np.float32)
        for obs in obs_names
    ]
    agent.attach_environment(
        observation_names=obs_names,
        action_names=action_names,
        action_space=list(getattr(env, "flat_action_space", [])),
        observation_space=obs_spaces,
        metadata={
            "building_names": [b.name for b in env.buildings],
            "seconds_per_time_step": env.seconds_per_time_step,
        },
    )

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        actions = agent.predict(obs_list, deterministic=True)

        # Convert to entity action format
        env_actions = adapter.to_entity_actions(actions, action_names)
        next_obs_payload, _, terminated, truncated, _ = env.step(env_actions)
        next_obs_list, _, _ = adapter.to_agent_encoded_observations(next_obs_payload)
        obs_list = next_obs_list
        step += 1
        if step % 1000 == 0:
            print(f"  [{label} seed={env_seed}] step {step}", flush=True)
        if step >= max_steps:
            print(f"  [{label} seed={env_seed}] WARN: max_steps={max_steps} reached", flush=True)
            break

    kpi_df = env.evaluate()
    print(f"  [{label} seed={env_seed}] done in {step} steps", flush=True)

    district = _extract_kpis(kpi_df, level="district")
    return {
        "env_seed": env_seed,
        "label": label,
        "district": district,
        "steps": step,
    }


def rbc_rollout(*, env_seed: int) -> Dict[str, Any]:
    """RBCSmartPolicy rollout with the entity adapter."""
    env = _make_env(env_seed)
    adapter = _make_adapter(env)

    obs_payload, _ = env.reset()
    obs_list, obs_names, _ = adapter.to_agent_encoded_observations(obs_payload)
    action_names = [list(names) for names in env.action_names]

    rbc = _make_rbcsmart(obs_names, action_names, env)

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        actions = rbc.predict(obs_list, deterministic=True)
        env_actions = adapter.to_entity_actions(actions, action_names)
        next_obs_payload, _, terminated, truncated, _ = env.step(env_actions)
        next_obs_list, _, _ = adapter.to_agent_encoded_observations(next_obs_payload)
        obs_list = next_obs_list
        step += 1
        if step % 1000 == 0:
            print(f"  [RBC seed={env_seed}] step {step}", flush=True)

    kpi_df = env.evaluate()
    print(f"  [RBC seed={env_seed}] done in {step} steps", flush=True)
    district = _extract_kpis(kpi_df, level="district")
    return {"env_seed": env_seed, "label": "RBCSmart", "district": district, "steps": step}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(runs: List[Dict[str, Any]], kpis: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std for each KPI across runs."""
    result: Dict[str, Dict[str, float]] = {}
    for k in kpis:
        vals = [r["district"].get(k, float("nan")) for r in runs]
        vals_finite = [v for v in vals if np.isfinite(v)]
        if vals_finite:
            result[k] = {
                "mean": float(np.mean(vals_finite)),
                "std": float(np.std(vals_finite, ddof=0)),
                "n": len(vals_finite),
            }
        else:
            result[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return result


def _fmt(val: float, std: float) -> str:
    if not np.isfinite(val):
        return "N/A"
    return f"{val:.4f} ± {std:.4f}"


def _delta(a_mean: float, b_mean: float, b_std: float, higher_better: bool) -> str:
    """Show Δ% from a to b and significance direction."""
    if not (np.isfinite(a_mean) and np.isfinite(b_mean) and b_mean != 0):
        return ""
    delta_pct = (b_mean - a_mean) / abs(a_mean) * 100
    if higher_better:
        sign = "▲" if delta_pct > 0 else "▼"
    else:
        sign = "▼" if delta_pct < 0 else "▲"
    return f"{sign}{abs(delta_pct):.1f}%"


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _print_report(
    rbc_agg: Dict[str, Any],
    iql_agg: Optional[Dict[str, Any]],
    cql_agg: Optional[Dict[str, Any]],
    kpis: Sequence[str],
) -> None:
    header = f"{'KPI':<42} {'RBCSmart':>22}"
    if iql_agg:
        header += f" {'IQL':>22} {'Δ(IQL-RBC)':>12}"
    if cql_agg:
        header += f" {'CQL':>22} {'Δ(CQL-RBC)':>12}"
    print(header)
    print("-" * len(header))

    for k in kpis:
        hib = k in HIGHER_IS_BETTER
        rbc = rbc_agg.get(k, {})
        r_m, r_s = rbc.get("mean", float("nan")), rbc.get("std", float("nan"))
        row = f"{k:<42} {_fmt(r_m, r_s):>22}"
        if iql_agg:
            iql = iql_agg.get(k, {})
            i_m, i_s = iql.get("mean", float("nan")), iql.get("std", float("nan"))
            row += f" {_fmt(i_m, i_s):>22} {_delta(r_m, i_m, i_s, hib):>12}"
        if cql_agg:
            cql = cql_agg.get(k, {})
            c_m, c_s = cql.get("mean", float("nan")), cql.get("std", float("nan"))
            row += f" {_fmt(c_m, c_s):>22} {_delta(r_m, c_m, c_s, hib):>12}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--iql-root", type=Path, default=None,
                   help="Output root from train_iql_entity.py")
    p.add_argument("--cql-root", type=Path, default=None,
                   help="Output root from train_cql_entity.py")
    p.add_argument("--no-cql", action="store_true",
                   help="Skip CQL even if --cql-root is provided")
    p.add_argument("--no-iql", action="store_true",
                   help="Skip IQL even if --iql-root is provided")
    p.add_argument(
        "--eval-seeds", default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS),
        help="Comma-separated eval seeds (default: 200-209)"
    )
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path for results")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    if not eval_seeds:
        print("[benchmark] ERROR: --eval-seeds is empty", file=sys.stderr)
        return 1

    print(f"[benchmark] eval_seeds = {eval_seeds}")
    print(f"[benchmark] iql_root   = {args.iql_root}")
    print(f"[benchmark] cql_root   = {args.cql_root}")
    print()

    # --- RBCSmart rollouts ---
    print("=== RBCSmart rollouts ===")
    rbc_runs: List[Dict[str, Any]] = []
    for seed in eval_seeds:
        rbc_runs.append(rbc_rollout(env_seed=seed))

    # --- IQL rollouts ---
    iql_runs: Optional[List[Dict[str, Any]]] = None
    if args.iql_root and not args.no_iql:
        print("\n=== IQL rollouts ===")
        iql_agent = IQLEntityAgent.from_model_dir(
            args.iql_root, device=args.device
        )
        iql_runs = []
        for seed in eval_seeds:
            iql_runs.append(
                entity_rollout(iql_agent, env_seed=seed, label="IQL")
            )

    # --- CQL rollouts ---
    cql_runs: Optional[List[Dict[str, Any]]] = None
    if args.cql_root and not args.no_cql:
        print("\n=== CQL rollouts ===")
        cql_agent = CQLEntityAgent.from_model_dir(
            args.cql_root, device=args.device
        )
        cql_runs = []
        for seed in eval_seeds:
            cql_runs.append(
                entity_rollout(cql_agent, env_seed=seed, label="CQL")
            )

    # --- Aggregate ---
    rbc_agg = _aggregate(rbc_runs, HEADLINE_KPIs)
    iql_agg = _aggregate(iql_runs, HEADLINE_KPIs) if iql_runs else None
    cql_agg = _aggregate(cql_runs, HEADLINE_KPIs) if cql_runs else None

    # --- Print report ---
    print("\n\n=== Benchmark Results ===")
    print(f"Eval seeds: {eval_seeds}  (n={len(eval_seeds)})\n")
    _print_report(rbc_agg, iql_agg, cql_agg, HEADLINE_KPIs)

    # --- Save results ---
    results = {
        "eval_seeds": eval_seeds,
        "iql_root": str(args.iql_root) if args.iql_root else None,
        "cql_root": str(args.cql_root) if args.cql_root else None,
        "RBCSmart": {"runs": rbc_runs, "aggregate": rbc_agg},
    }
    if iql_runs is not None:
        results["IQL"] = {"runs": iql_runs, "aggregate": iql_agg}  # type: ignore[index]
    if cql_runs is not None:
        results["CQL"] = {"runs": cql_runs, "aggregate": cql_agg}  # type: ignore[index]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Remove kpi_df (not serialisable) before saving
        def _clean(r: Dict) -> Dict:
            return {k: v for k, v in r.items() if k != "kpi_df"}
        for key in ("RBCSmart", "IQL", "CQL"):
            if key in results and "runs" in results[key]:
                results[key]["runs"] = [_clean(r) for r in results[key]["runs"]]
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\n[benchmark] Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
