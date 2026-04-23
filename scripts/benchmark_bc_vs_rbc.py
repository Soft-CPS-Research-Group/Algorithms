"""Benchmark RBC vs trained Behavioral Cloning policy on CityLearn.

Runs both controllers under identical CityLearn conditions (same schema, reward
function, episode length, seed) and writes a side-by-side KPI comparison to
`docs/offline_rl/bc_vs_rbc_benchmark.md`.

Usage
-----

.. code-block:: bash

    python scripts/benchmark_bc_vs_rbc.py \\
        --model runs/offline_bc/bc-v1/model.pth \\
        --stats runs/offline_bc/bc-v1/normalization_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from citylearn.citylearn import CityLearnEnv  # noqa: E402
from citylearn.reward_function import RewardFunction  # noqa: E402

from algorithms.agents.ev_data_collection_agent import EVDataCollectionRBC  # noqa: E402
from algorithms.agents.offline_bc_agent import OfflineBCAgent  # noqa: E402

DATASET_SCHEMA = "./datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
TARGET_BUILDING_INDEX = 4  # Building 5
TARGET_BUILDING_NAME = "Building_5"

# District-level KPIs we surface in the report (CityLearn's official names).
HEADLINE_KPIS = [
    "electricity_consumption_total",
    "carbon_emissions_total",
    "cost_total",
    "all_time_peak_average",
    "daily_peak_average",
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "annual_normalized_unserved_energy_total",
    "zero_net_energy",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env() -> CityLearnEnv:
    return CityLearnEnv(
        schema=DATASET_SCHEMA,
        central_agent=False,
        reward_function=RewardFunction,
    )


def attach_agent(agent, env: CityLearnEnv) -> None:
    """Mimic the wrapper's attach_environment call."""
    agent.attach_environment(
        observation_names=env.observation_names,
        action_names=env.action_names,
        action_space=env.action_space,
        observation_space=env.observation_space,
        metadata={},
    )


def _mask_non_target_actions(actions: List[List[float]]) -> List[List[float]]:
    """Zero out actions for every agent except the target building.

    Both agents in this benchmark only "really" control Building 5; the BC
    agent already returns zeros for the other 16 agents (it was trained on
    Building 5 alone). To get an apples-to-apples comparison we mask the
    RBC's actions the same way — otherwise the district-level KPIs are
    dominated by the RBC driving 16 *other* buildings.
    """
    masked: List[List[float]] = []
    for idx, agent_actions in enumerate(actions):
        if idx == TARGET_BUILDING_INDEX:
            masked.append(list(agent_actions))
        else:
            masked.append([0.0] * len(agent_actions))
    return masked


def rollout(agent, env: CityLearnEnv, *, label: str, mask_non_target: bool = False) -> Dict[str, float]:
    """Run one full episode and return aggregate reward stats for the target building."""
    obs, _info = env.reset(), None
    if isinstance(obs, tuple):
        obs = obs[0]

    total_reward_target = 0.0
    total_reward_district = 0.0
    n_steps = 0
    done = False
    while not done:
        actions = agent.predict(obs, deterministic=True)
        if mask_non_target:
            actions = _mask_non_target_actions(actions)
        next_obs, rewards, terminated, truncated, _info = env.step(actions)
        total_reward_target += float(rewards[TARGET_BUILDING_INDEX])
        total_reward_district += float(np.sum(rewards))
        n_steps += 1
        done = bool(terminated) or bool(truncated)
        obs = next_obs

    print(f"  [{label}] episode finished after {n_steps} steps")
    return {
        "steps": n_steps,
        "reward_sum_target": total_reward_target,
        "reward_mean_target": total_reward_target / max(n_steps, 1),
        "reward_sum_district": total_reward_district,
        "reward_mean_district": total_reward_district / max(n_steps, 1),
    }


def evaluate_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """CityLearn's normalized KPIs (1.0 == no-control baseline; lower is better)."""
    return env.evaluate()


def kpi_table(df: pd.DataFrame, *, level: str, name: str | None = None) -> Dict[str, float]:
    sub = df[df["level"] == level]
    if name is not None:
        sub = sub[sub["name"] == name]
    return {row["cost_function"]: float(row["value"]) for _, row in sub.iterrows()}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if np.isnan(value):
            return "—"
        return f"{value:.4f}"
    return str(value)


def pct_delta(bc: float | None, rbc: float | None) -> str:
    """Relative change of BC vs RBC (lower KPI is better, so negative = BC wins)."""
    if bc is None or rbc is None:
        return "—"
    if isinstance(bc, float) and (np.isnan(bc) or np.isnan(rbc)):
        return "—"
    if rbc == 0:
        return "—"
    delta = (bc - rbc) / abs(rbc) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}%"


def render_kpi_section(
    title: str,
    rbc_kpis: Dict[str, float],
    bc_kpis: Dict[str, float],
    keys: List[str],
) -> str:
    lines = [f"### {title}", "", "| KPI | RBC | BC | Δ (BC − RBC) | Winner |", "|---|---:|---:|---:|:---:|"]
    for k in keys:
        rbc_v = rbc_kpis.get(k)
        bc_v = bc_kpis.get(k)
        if rbc_v is None and bc_v is None:
            continue
        if (rbc_v is None or (isinstance(rbc_v, float) and np.isnan(rbc_v))) or (
            bc_v is None or (isinstance(bc_v, float) and np.isnan(bc_v))
        ):
            winner = "—"
        elif abs(bc_v - rbc_v) < 1e-6:
            winner = "tie"
        elif bc_v < rbc_v:
            winner = "🟢 BC"
        else:
            winner = "🔴 RBC"
        lines.append(
            f"| `{k}` | {fmt(rbc_v)} | {fmt(bc_v)} | {pct_delta(bc_v, rbc_v)} | {winner} |"
        )
    return "\n".join(lines) + "\n"


def render_full_diff_table(
    rbc_kpis: Dict[str, float], bc_kpis: Dict[str, float]
) -> str:
    keys = sorted(set(rbc_kpis) | set(bc_kpis))
    lines = ["| KPI | RBC | BC | Δ (BC − RBC) |", "|---|---:|---:|---:|"]
    for k in keys:
        rbc_v = rbc_kpis.get(k)
        bc_v = bc_kpis.get(k)
        lines.append(f"| `{k}` | {fmt(rbc_v)} | {fmt(bc_v)} | {pct_delta(bc_v, rbc_v)} |")
    return "\n".join(lines) + "\n"


def build_report(
    rbc_district: Dict[str, float],
    bc_district: Dict[str, float],
    rbc_building: Dict[str, float],
    bc_building: Dict[str, float],
    rbc_rollout: Dict[str, float],
    bc_rollout: Dict[str, float],
    *,
    model_path: str,
    stats_path: str,
) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = f"""# BC vs RBC — CityLearn Benchmark

_Generated_: {timestamp}
_Dataset_: `{DATASET_SCHEMA}`
_Reward function_: `RewardFunction` (CityLearn default)
_Episode length_: {rbc_rollout['steps']} steps (full year, single episode)
_BC checkpoint_: `{model_path}`
_BC normalization stats_: `{stats_path}`
_Target building_: **{TARGET_BUILDING_NAME}** (agent index {TARGET_BUILDING_INDEX})

> All KPIs are reported in CityLearn's **normalized form**: a value of `1.0`
> equals the *no-control baseline* (every controllable device idle). Values
> **below 1.0 mean the controller improved over baseline**; values above 1.0
> mean it made things worse. So when comparing two controllers, **lower is
> better**.

---

## 1. Reward summary

| Quantity | RBC | BC |
|---|---:|---:|
| Episode length (steps) | {rbc_rollout['steps']} | {bc_rollout['steps']} |
| Σ reward (target building) | {fmt(rbc_rollout['reward_sum_target'])} | {fmt(bc_rollout['reward_sum_target'])} |
| Mean reward (target building) | {fmt(rbc_rollout['reward_mean_target'])} | {fmt(bc_rollout['reward_mean_target'])} |
| Σ reward (district, all 17 agents) | {fmt(rbc_rollout['reward_sum_district'])} | {fmt(bc_rollout['reward_sum_district'])} |
| Mean reward (district) | {fmt(rbc_rollout['reward_mean_district'])} | {fmt(bc_rollout['reward_mean_district'])} |

> The reward function used here is CityLearn's default `RewardFunction`, which
> returns the **negative net electricity consumption** at each step. So a
> *higher* (less negative) sum means the controller drew less energy from the
> grid overall. Note that the **BC agent only controls Building 5** in this
> setup — the other 16 buildings receive idle (zero) actions in both runs, so
> the district-level reward differences are driven entirely by Building 5's
> behaviour.

---

## 2. Headline KPIs — district level

{render_kpi_section("District", rbc_district, bc_district, HEADLINE_KPIS)}

---

## 3. Headline KPIs — Building 5 (the building we trained on)

{render_kpi_section(TARGET_BUILDING_NAME, rbc_building, bc_building, HEADLINE_KPIS)}

---

## 4. Full KPI dump — district

<details>
<summary>Click to expand full district KPI table</summary>

{render_full_diff_table(rbc_district, bc_district)}

</details>

## 5. Full KPI dump — Building 5

<details>
<summary>Click to expand full Building 5 KPI table</summary>

{render_full_diff_table(rbc_building, bc_building)}

</details>

---

## 6. How to read these numbers

| Metric | What it means | Why we care |
|---|---|---|
| `electricity_consumption_total` | Total grid electricity drawn (normalized vs baseline) | The headline "did the controller cut consumption?" |
| `carbon_emissions_total` | Carbon footprint of grid draw, weighted by hourly emission factor | Captures *when* energy is used, not just how much |
| `cost_total` | Monetary cost of grid energy, weighted by tariff (incl. peak pricing) | Direct economic impact |
| `all_time_peak_average` | Highest single-step grid draw, normalized | Grid-stress proxy; expensive to provision for |
| `daily_peak_average` | Average of each day's peak | Smoothness of daily demand |
| `ramping_average` | Mean step-to-step change in district load | Penalizes "spiky" control |
| `daily_one_minus_load_factor_average` | `1 − (mean / peak)` per day | Lower = flatter, more efficient utilization |
| `annual_normalized_unserved_energy_total` | EV/thermal demand the controller failed to satisfy | **Constraint violation indicator** — should stay near 0 |
| `zero_net_energy` | Imbalance between consumed and produced (PV) energy | Self-sufficiency proxy |

### Verdict heuristic

Since BC was trained to **imitate** the RBC, the expected outcome is that
BC's KPIs are *very close* to RBC's. Large divergences indicate either:
* the policy generalizes (could be good or bad — it depends on which way),
* or it has acquired distribution-shift errors at states the RBC rarely
  visits (a known BC failure mode).

For a successful Milestone 2 we want **|Δ| ≤ a few percent** on the headline
KPIs, with **`annual_normalized_unserved_energy_total` not getting worse**.
"""
    return md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/offline_bc/bc-v1/model.pth",
        help="Path to BC model checkpoint",
    )
    parser.add_argument(
        "--stats",
        default="runs/offline_bc/bc-v1/normalization_stats.json",
        help="Path to BC normalization stats",
    )
    parser.add_argument(
        "--output",
        default="docs/offline_rl/bc_vs_rbc_benchmark.md",
        help="Output markdown report path (relative to repo root)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = str((REPO_ROOT / args.model).resolve()) if not Path(args.model).is_absolute() else args.model
    stats_path = str((REPO_ROOT / args.stats).resolve()) if not Path(args.stats).is_absolute() else args.stats
    output_path = (REPO_ROOT / args.output).resolve()

    # ------------------------------------------------------------------
    # Run RBC
    # ------------------------------------------------------------------
    print("[RBC] building env + agent...")
    rbc_env = make_env()
    rbc_agent = EVDataCollectionRBC(config={"algorithm": {"hyperparameters": {"target_building_index": TARGET_BUILDING_INDEX}}})
    attach_agent(rbc_agent, rbc_env)
    rbc_rollout = rollout(rbc_agent, rbc_env, label="RBC", mask_non_target=True)
    rbc_kpis_df = evaluate_kpis(rbc_env)

    # ------------------------------------------------------------------
    # Run BC
    # ------------------------------------------------------------------
    print("[BC] building env + agent...")
    bc_env = make_env()
    bc_agent = OfflineBCAgent(
        config={
            "algorithm": {
                "hyperparameters": {
                    "model_path": model_path,
                    "stats_path": stats_path,
                    "target_building_index": TARGET_BUILDING_INDEX,
                    "device": "auto",
                }
            }
        }
    )
    attach_agent(bc_agent, bc_env)
    bc_rollout = rollout(bc_agent, bc_env, label="BC", mask_non_target=True)
    bc_kpis_df = evaluate_kpis(bc_env)

    # ------------------------------------------------------------------
    # Extract district + Building 5 KPIs
    # ------------------------------------------------------------------
    rbc_district = kpi_table(rbc_kpis_df, level="district")
    bc_district = kpi_table(bc_kpis_df, level="district")

    # Building names: env exposes them via env.buildings; the order in
    # `name` column is the building name. Fall back to first building name
    # if the canonical "Building_5" string isn't present.
    available_buildings = sorted(set(rbc_kpis_df[rbc_kpis_df["level"] == "building"]["name"]))
    if TARGET_BUILDING_NAME in available_buildings:
        building_name = TARGET_BUILDING_NAME
    else:
        # Resolve by index into the alphabetical building list.
        building_name = available_buildings[TARGET_BUILDING_INDEX] if TARGET_BUILDING_INDEX < len(available_buildings) else available_buildings[0]
        print(f"  (note: '{TARGET_BUILDING_NAME}' not found; using '{building_name}')")

    rbc_building = kpi_table(rbc_kpis_df, level="building", name=building_name)
    bc_building = kpi_table(bc_kpis_df, level="building", name=building_name)

    # ------------------------------------------------------------------
    # Render markdown
    # ------------------------------------------------------------------
    md = build_report(
        rbc_district=rbc_district,
        bc_district=bc_district,
        rbc_building=rbc_building,
        bc_building=bc_building,
        rbc_rollout=rbc_rollout,
        bc_rollout=bc_rollout,
        model_path=str(Path(model_path).relative_to(REPO_ROOT)) if Path(model_path).is_relative_to(REPO_ROOT) else model_path,
        stats_path=str(Path(stats_path).relative_to(REPO_ROOT)) if Path(stats_path).is_relative_to(REPO_ROOT) else stats_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\n✅ Report written to {output_path}")

    # Also dump raw KPI CSVs alongside the report for further analysis.
    raw_dir = output_path.parent / "bc_vs_rbc_raw"
    raw_dir.mkdir(exist_ok=True)
    rbc_kpis_df.to_csv(raw_dir / "rbc_kpis.csv", index=False)
    bc_kpis_df.to_csv(raw_dir / "bc_kpis.csv", index=False)
    with (raw_dir / "rollout.json").open("w", encoding="utf-8") as fh:
        json.dump({"rbc": rbc_rollout, "bc": bc_rollout}, fh, indent=2)
    print(f"   Raw KPIs in   {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
