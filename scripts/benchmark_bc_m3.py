"""M3 Behavioral Cloning benchmark: BC vs RBC vs Random on Building 5.

Runs three controllers under identical CityLearn conditions on the
``citylearn_three_phase_dynamic_topology_demo_v1`` dataset (entity + dynamic
topology interface), aggregates KPIs across multiple env seeds, and writes a
side-by-side markdown report.

Multi-seed schema
-----------------
* **RBC**     : 5 rollouts, env_seeds = [22, 23, 24, 25, 26].
* **Random**  : 5 rollouts, env_seeds = [22, 23, 24, 25, 26], action RNG seeded
                from the same seed.
* **BC**      : ``len(seeds) * eval_rollouts`` rollouts. For each training seed
                under ``--bc-root`` we load ``model.pth`` + ``normalization_stats.json``
                and run ``eval_rollouts`` rollouts under the same env_seeds.

KPIs are extracted directly from ``CityLearnEnv.evaluate()`` (a DataFrame with
the columns ``cost_function``, ``value``, ``name``, ``level``). All comparators
are evaluated under the same ``V2GPenaltyReward`` used during M2 data
collection.

The wrapper (``utils.wrapper_citylearn.Wrapper_CityLearn``) is invoked with
``deterministic=True`` so that ``update()`` is never called. MLflow and progress
tracking are disabled via ``tracking.mlflow_enabled=False`` /
``tracking.progress_updates_enabled=False`` in the synthetic config.

Usage
-----

.. code-block:: bash

    # Smoke (1 BC seed, 1 rollout per controller)
    python scripts/benchmark_bc_m3.py \\
        --bc-root runs/offline_bc_m3/bc-m3-v1 \\
        --output  docs/offline_rl/m2/bc_vs_rbc_vs_random_benchmark_m3.md \\
        --smoke

    # Full
    python scripts/benchmark_bc_m3.py \\
        --bc-root runs/offline_bc_m3/bc-m3-v1 \\
        --output  docs/offline_rl/m2/bc_vs_rbc_vs_random_benchmark_m3.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Quiet the wrapper's per-step DEBUG logs before importing it.
from loguru import logger as _loguru_logger  # noqa: E402

_loguru_logger.remove()
_loguru_logger.add(sys.stderr, level="WARNING")

from citylearn.citylearn import CityLearnEnv  # noqa: E402

from algorithms.agents.base_agent import BaseAgent  # noqa: E402
from algorithms.agents.offline_bc_agent import OfflineBCAgent  # noqa: E402
from algorithms.agents.rbc_agent import RuleBasedPolicy  # noqa: E402
from reward_function.V2G_Reward import V2GPenaltyReward  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_SCHEMA = "./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json"
TARGET_BUILDING_INDEX = 4  # Building_5 in this dataset
TARGET_BUILDING_NAME = "Building_5"

DEFAULT_ENV_SEEDS: Tuple[int, ...] = (22, 23, 24, 25, 26)

# District-level KPIs we surface in the headline section.
HEADLINE_DISTRICT_KPIS: Tuple[str, ...] = (
    "electricity_consumption_total",
    "carbon_emissions_total",
    "cost_total",
    "all_time_peak_average",
    "daily_peak_average",
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "annual_normalized_unserved_energy_total",
    "zero_net_energy",
)

# Building-level KPIs of particular interest for Building 5.
HEADLINE_BUILDING_KPIS: Tuple[str, ...] = (
    "electricity_consumption_total",
    "carbon_emissions_total",
    "cost_total",
    "annual_normalized_unserved_energy_total",
    "ev_departure_success_rate",
    "bess_throughput_total_kwh",
    "bess_equivalent_full_cycles",
    "bess_capacity_fade_ratio",
)


# ---------------------------------------------------------------------------
# Random baseline (no new agent class — local closure)
# ---------------------------------------------------------------------------


class _RandomAgent(BaseAgent):
    """Uniform random sampler over each agent's action space.

    Lives only inside this benchmark to avoid polluting the agent registry.
    """

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.use_raw_observations = True
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._lows: List[np.ndarray] = []
        self._highs: List[np.ndarray] = []

    def attach_environment(  # type: ignore[override]
        self,
        *,
        observation_names,
        action_names,
        action_space,
        observation_space,
        metadata=None,
    ) -> None:
        self._lows = [np.asarray(s.low, dtype=np.float32).flatten() for s in action_space]
        self._highs = [np.asarray(s.high, dtype=np.float32).flatten() for s in action_space]

    def predict(self, observations, deterministic: Optional[bool] = None):  # type: ignore[override]
        actions: List[List[float]] = []
        for low, high in zip(self._lows, self._highs):
            sample = self._rng.uniform(low=low, high=high).astype(np.float32)
            actions.append(sample.tolist())
        return actions

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        return None

    def export_artifacts(  # type: ignore[override]
        self, output_dir, context=None
    ) -> Dict[str, Any]:
        return {"artifact_type": "random_baseline", "seed": self._seed}


# ---------------------------------------------------------------------------
# Env construction & rollout
# ---------------------------------------------------------------------------
#
# We construct the env in the same mode M2 used to collect the offline dataset:
# ``interface='flat', topology_mode='static'``. Although the dataset's schema
# declares ``dynamic`` topology, CityLearn lets you override it explicitly,
# and that's what ``run_experiment.py`` does by default (its
# ``simulator.interface`` defaults to ``'flat'`` and ``topology_mode`` to
# ``'static'``). Under those flags:
#   * ``env.reset()`` / ``env.step()`` return per-agent lists (legacy layout).
#   * ``env.observation_names[i]`` are the 35-name flat per-building schema
#     that BC was trained on (e.g. ``month, day_type, hour, ...``).
#   * ``env.action_space`` is a list of per-agent ``Box`` spaces.
#   * Topology stays static at 17 buildings, matching M2's
#     ``topology_versions_seen=[0]``.
#
# This removes the need to involve ``Wrapper_CityLearn`` or the entity
# adapter at all — agents talk to CityLearn directly with flat per-agent
# observations and per-agent action vectors, exactly as during M2 collection.


def _make_env(seed: int) -> CityLearnEnv:
    return CityLearnEnv(
        schema=DATASET_SCHEMA,
        central_agent=False,
        interface="flat",
        topology_mode="static",
        reward_function=V2GPenaltyReward,
        random_seed=int(seed),
    )


def _clip_actions(
    actions: List[List[float]], action_space: Sequence[Any]
) -> List[List[float]]:
    """Clip per-agent actions to their respective bounds."""
    clipped: List[List[float]] = []
    for vec, space in zip(actions, action_space):
        low = np.asarray(getattr(space, "low", np.full(len(vec), -np.inf)), dtype=np.float64).flatten()
        high = np.asarray(getattr(space, "high", np.full(len(vec), np.inf)), dtype=np.float64).flatten()
        arr = np.asarray(vec, dtype=np.float64).flatten()
        if arr.shape[0] != low.shape[0]:
            if arr.shape[0] < low.shape[0]:
                arr = np.concatenate([arr, np.zeros(low.shape[0] - arr.shape[0])])
            else:
                arr = arr[: low.shape[0]]
        clipped.append(np.clip(arr, low, high).tolist())
    return clipped


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _rollout(agent: BaseAgent, *, env_seed: int, label: str) -> Dict[str, Any]:
    """Run one full-episode rollout under a fresh env and return KPIs.

    Uses the same env mode M2 collected its dataset under
    (``interface='flat', topology_mode='static'``), so the BC agent sees
    exactly the 35-dim per-building observation schema it was trained on.
    """
    env = _make_env(env_seed)
    obs_list, _ = env.reset()

    agent.attach_environment(
        observation_names=env.observation_names,
        action_names=env.action_names,
        action_space=env.action_space,
        observation_space=env.observation_space,
        metadata={
            "seconds_per_time_step": getattr(env, "seconds_per_time_step", None),
            "building_names": [b.name for b in env.buildings],
            "interface": "flat",
            "topology_mode": "static",
            "topology_version": 0,
        },
    )

    n_agents = len(env.action_names)
    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        actions = agent.predict(obs_list, deterministic=True)
        if not isinstance(actions, list) or len(actions) != n_agents:
            raise RuntimeError(
                f"[{label}] agent.predict returned {type(actions).__name__} of "
                f"length {len(actions) if hasattr(actions,'__len__') else '?'}, "
                f"expected list of {n_agents} per-agent vectors."
            )
        actions = _clip_actions(actions, env.action_space)
        obs_list, _rewards, terminated, truncated, _info = env.step(actions)
        step += 1
        if step % 1000 == 0:
            print(f"  [{label} seed={env_seed}] step {step}", flush=True)

    kpi_df = env.evaluate()
    print(f"  [{label} seed={env_seed}] episode done in {step} steps", flush=True)

    district = _extract_kpis(kpi_df, level="district")
    building = _extract_kpis(kpi_df, level="building", name=TARGET_BUILDING_NAME)
    return {
        "env_seed": env_seed,
        "kpi_df": kpi_df,
        "district": district,
        "building": building,
        "steps": step,
    }


def _extract_kpis(
    df: pd.DataFrame,
    *,
    level: str,
    name: Optional[str] = None,
) -> Dict[str, float]:
    sub = df[df["level"] == level]
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
# Multi-seed orchestration
# ---------------------------------------------------------------------------


def _run_rbc_rollouts(env_seeds: Sequence[int]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for env_seed in env_seeds:
        agent = RuleBasedPolicy(config={
            "algorithm": {"hyperparameters": {}},
            "simulator": {"dataset_path": DATASET_SCHEMA},
        })
        runs.append(_rollout(agent, env_seed=env_seed, label="RBC"))
    return runs


def _run_random_rollouts(env_seeds: Sequence[int]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for env_seed in env_seeds:
        agent = _RandomAgent(seed=int(env_seed))
        runs.append(_rollout(agent, env_seed=env_seed, label="Random"))
    return runs


def _discover_bc_seeds(bc_root: Path) -> List[Tuple[int, Path, Path]]:
    """Return (seed, model_path, stats_path) tuples sorted by seed.

    Expects ``<bc_root>/seed_<N>/model.pth`` + ``normalization_stats.json``.
    """
    found: List[Tuple[int, Path, Path]] = []
    for entry in sorted(bc_root.iterdir() if bc_root.is_dir() else []):
        if not entry.is_dir() or not entry.name.startswith("seed_"):
            continue
        try:
            seed = int(entry.name.split("_", 1)[1])
        except ValueError:
            continue
        model = entry / "model.pth"
        stats = entry / "normalization_stats.json"
        if model.is_file() and stats.is_file():
            found.append((seed, model, stats))
    if not found:
        raise FileNotFoundError(
            f"No BC seed dirs with model.pth+normalization_stats.json under {bc_root}"
        )
    return found


def _run_bc_rollouts(
    bc_seeds: Sequence[Tuple[int, Path, Path]],
    env_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for train_seed, model_path, stats_path in bc_seeds:
        for env_seed in env_seeds:
            agent = OfflineBCAgent(config={
                "algorithm": {
                    "hyperparameters": {
                        "model_path": str(model_path),
                        "stats_path": str(stats_path),
                        "target_building_index": TARGET_BUILDING_INDEX,
                        "device": "auto",
                    }
                }
            })
            run = _rollout(agent, env_seed=env_seed, label=f"BC-s{train_seed}")
            run["train_seed"] = train_seed
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.mean(clean)), float(statistics.stdev(clean))


def _aggregate(
    runs: Sequence[Dict[str, Any]], *, scope: str, kpi_keys: Sequence[str]
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return ``{kpi: {'mean': ..., 'std': ..., 'n': ...}}``."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for kpi in kpi_keys:
        vals = [r[scope].get(kpi) for r in runs if scope in r]
        mean, std = _mean_std(vals)
        out[kpi] = {
            "mean": mean,
            "std": std,
            "n": int(sum(1 for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v)))),
        }
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], *, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and np.isnan(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_mean_std(stat: Dict[str, Optional[float]], *, digits: int = 4) -> str:
    mean = stat.get("mean")
    std = stat.get("std")
    if mean is None:
        return "—"
    return f"{_fmt(mean, digits=digits)} ± {_fmt(std, digits=digits)}"


def _delta_significance(
    bc: Dict[str, Optional[float]],
    rbc: Dict[str, Optional[float]],
) -> Tuple[Optional[float], str]:
    """Heuristic: |Δmean| > max(BC_std, RBC_std, ABS_FLOOR) ⇒ 'significant'.

    The ``ABS_FLOOR`` (1e-4) prevents the verdict from flipping based on
    sub-rounding-noise differences when both controllers are
    bit-for-bit identical (e.g. BC perfectly imitating a deterministic
    RBC produces ``std ~ 1e-14`` and ``Δ ~ 1e-15`` — formally
    ``|Δ| > std`` but practically meaningless).
    """
    ABS_FLOOR = 1e-4
    bc_mean, bc_std = bc.get("mean"), bc.get("std")
    rbc_mean, rbc_std = rbc.get("mean"), rbc.get("std")
    if bc_mean is None or rbc_mean is None:
        return None, "—"
    delta = bc_mean - rbc_mean
    threshold = max(bc_std or 0.0, rbc_std or 0.0, ABS_FLOOR)
    if abs(delta) <= threshold:
        marker = "≈ within noise"
    elif delta < 0:
        marker = "🟢 BC better"
    else:
        marker = "🔴 RBC better"
    return delta, marker


def _render_kpi_table(
    title: str,
    rbc_agg: Dict[str, Dict[str, Optional[float]]],
    bc_agg: Dict[str, Dict[str, Optional[float]]],
    random_agg: Dict[str, Dict[str, Optional[float]]],
    keys: Sequence[str],
) -> str:
    lines = [
        f"### {title}",
        "",
        "| KPI | Random (mean ± std) | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |",
        "|---|---:|---:|---:|---:|:---|",
    ]
    for k in keys:
        rnd = random_agg.get(k, {"mean": None, "std": None})
        rbc = rbc_agg.get(k, {"mean": None, "std": None})
        bc = bc_agg.get(k, {"mean": None, "std": None})
        if rnd["mean"] is None and rbc["mean"] is None and bc["mean"] is None:
            continue
        delta, verdict = _delta_significance(bc, rbc)
        lines.append(
            f"| `{k}` | {_fmt_mean_std(rnd)} | {_fmt_mean_std(rbc)} | {_fmt_mean_std(bc)} | {_fmt(delta)} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def _render_full_dump(
    rbc_agg: Dict[str, Dict[str, Optional[float]]],
    bc_agg: Dict[str, Dict[str, Optional[float]]],
    random_agg: Dict[str, Dict[str, Optional[float]]],
) -> str:
    keys = sorted(set(rbc_agg) | set(bc_agg) | set(random_agg))
    lines = [
        "| KPI | Random | RBC | BC | Δ (BC − RBC) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in keys:
        rnd = random_agg.get(k, {"mean": None, "std": None})
        rbc = rbc_agg.get(k, {"mean": None, "std": None})
        bc = bc_agg.get(k, {"mean": None, "std": None})
        delta = (
            (bc["mean"] - rbc["mean"])
            if (bc["mean"] is not None and rbc["mean"] is not None)
            else None
        )
        lines.append(
            f"| `{k}` | {_fmt_mean_std(rnd)} | {_fmt_mean_std(rbc)} | {_fmt_mean_std(bc)} | {_fmt(delta)} |"
        )
    return "\n".join(lines) + "\n"


def _build_report(
    *,
    bc_root: Path,
    bc_seeds: Sequence[Tuple[int, Path, Path]],
    env_seeds: Sequence[int],
    eval_rollouts: int,
    smoke: bool,
    rbc_runs: Sequence[Dict[str, Any]],
    random_runs: Sequence[Dict[str, Any]],
    bc_runs: Sequence[Dict[str, Any]],
    rbc_district_agg: Dict[str, Dict[str, Optional[float]]],
    bc_district_agg: Dict[str, Dict[str, Optional[float]]],
    random_district_agg: Dict[str, Dict[str, Optional[float]]],
    rbc_building_agg: Dict[str, Dict[str, Optional[float]]],
    bc_building_agg: Dict[str, Dict[str, Optional[float]]],
    random_building_agg: Dict[str, Dict[str, Optional[float]]],
) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    train_seed_list = ", ".join(str(s) for s, _, _ in bc_seeds)
    env_seed_list = ", ".join(str(s) for s in env_seeds)
    smoke_note = " **(SMOKE MODE — single rollout per controller, results indicative only)**" if smoke else ""

    return f"""# M3 — BC vs RBC vs Random — CityLearn Benchmark

_Generated_: {timestamp}{smoke_note}
_Dataset_: `{DATASET_SCHEMA}` (interface=`flat`, topology_mode=`static` — same mode used for M2 collection)
_Reward function_: `V2GPenaltyReward` (matches M2 data collection)
_Target building_: **{TARGET_BUILDING_NAME}** (agent index {TARGET_BUILDING_INDEX})
_BC checkpoint root_: `{bc_root}`
_BC training seeds_: [{train_seed_list}]
_Env seeds_ (per controller): [{env_seed_list}]
_Rollouts per BC training seed_: {eval_rollouts}
_Total rollouts_: RBC={len(rbc_runs)}, Random={len(random_runs)}, BC={len(bc_runs)}

> All KPIs are CityLearn's normalized values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across BC training
> seeds for BC). The "Verdict" column flags `|Δmean| > max(BC_std, RBC_std, 1e-4)` as
> significant; otherwise the difference is within noise.

---

## 1. Headline KPIs — district level

{_render_kpi_table("District", rbc_district_agg, bc_district_agg, random_district_agg, HEADLINE_DISTRICT_KPIS)}

---

## 2. Headline KPIs — Building 5 (training target)

{_render_kpi_table(TARGET_BUILDING_NAME, rbc_building_agg, bc_building_agg, random_building_agg, HEADLINE_BUILDING_KPIS)}

---

## 3. Full KPI dump — district

<details>
<summary>Click to expand</summary>

{_render_full_dump(rbc_district_agg, bc_district_agg, random_district_agg)}

</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

{_render_full_dump(rbc_building_agg, bc_building_agg, random_building_agg)}

</details>

---

## 5. How to read these numbers

| Metric | What it means | Why we care |
|---|---|---|
| `electricity_consumption_total` | Total grid electricity drawn (normalized vs baseline) | Headline efficiency |
| `carbon_emissions_total` | Carbon footprint of grid draw | Captures *when* energy is used |
| `cost_total` | Monetary cost (tariff-weighted) | Direct economic impact |
| `all_time_peak_average` | Highest single-step grid draw | Grid-stress proxy |
| `daily_peak_average` | Average of each day's peak | Smoothness of daily demand |
| `ramping_average` | Mean step-to-step change in district load | Penalizes "spiky" control |
| `daily_one_minus_load_factor_average` | `1 − (mean / peak)` per day | Lower = flatter utilization |
| `annual_normalized_unserved_energy_total` | EV/thermal demand the controller failed to satisfy | **Constraint violation indicator** |
| `zero_net_energy` | Imbalance vs PV generation | Self-sufficiency proxy |
| `ev_departure_success_rate` | Fraction of EV departures meeting SoC target | Service-level KPI |
| `bess_throughput_total_kwh` | Total energy cycled through battery | Wear proxy |
| `bess_equivalent_full_cycles` | Equivalent full charge/discharge cycles | Wear proxy |
| `bess_capacity_fade_ratio` | Capacity loss vs nominal | Long-term degradation |

### Verdict heuristic

BC is trained to imitate the RBC's *clean* actions on Building 5. We expect:

* **BC ≈ RBC** on most KPIs (small Δ within noise) ⇒ behaviour cloning succeeded.
* **Random ≫ RBC** on cost / unserved energy ⇒ task is non-trivial; BC's parity
  with RBC is meaningful, not an artefact of a degenerate task.
* **Large adverse Δ** on `annual_normalized_unserved_energy_total` ⇒ BC is
  failing in safety-critical states (a known BC failure mode on out-of-distribution
  observations); this would motivate the M4 IQL/TD3+BC step.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-root",
        required=True,
        help="Root dir produced by train_bc_m3.py (contains seed_<N>/ subdirs).",
    )
    parser.add_argument(
        "--output",
        default="docs/offline_rl/m2/bc_vs_rbc_vs_random_benchmark_m3.md",
        help="Output markdown report path (relative to repo root).",
    )
    parser.add_argument(
        "--env-seeds",
        default=",".join(str(s) for s in DEFAULT_ENV_SEEDS),
        help="Comma-separated env seeds for the per-controller rollouts.",
    )
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=5,
        help="Rollouts per BC training seed AND per RBC/Random run "
             "(equals len(env_seeds) used). Set <5 to subset env_seeds.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 1 BC training seed, 1 env seed, 1 rollout per controller.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    bc_root = Path(args.bc_root).expanduser()
    if not bc_root.is_absolute():
        bc_root = (REPO_ROOT / bc_root).resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    env_seeds: List[int] = [int(s) for s in args.env_seeds.split(",") if s.strip()]
    if args.eval_rollouts > 0:
        env_seeds = env_seeds[: args.eval_rollouts]
    if not env_seeds:
        raise ValueError("No env seeds resolved; check --env-seeds / --eval-rollouts.")

    bc_seeds = _discover_bc_seeds(bc_root)
    if args.smoke:
        bc_seeds = bc_seeds[:1]
        env_seeds = env_seeds[:1]
        print(f"[smoke] using bc_seeds={[s for s,_,_ in bc_seeds]}, env_seeds={env_seeds}")

    print(f"[plan] BC training seeds: {[s for s,_,_ in bc_seeds]}")
    print(f"[plan] env seeds        : {env_seeds}")
    total_bc = len(bc_seeds) * len(env_seeds)
    print(f"[plan] total rollouts   : RBC={len(env_seeds)}, Random={len(env_seeds)}, BC={total_bc}")

    print("[run] RBC...")
    rbc_runs = _run_rbc_rollouts(env_seeds)
    print(f"[run] RBC done ({len(rbc_runs)} rollouts)")

    print("[run] Random...")
    random_runs = _run_random_rollouts(env_seeds)
    print(f"[run] Random done ({len(random_runs)} rollouts)")

    print("[run] BC...")
    bc_runs = _run_bc_rollouts(bc_seeds, env_seeds)
    print(f"[run] BC done ({len(bc_runs)} rollouts)")

    # Aggregate
    rbc_district_agg = _aggregate(rbc_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)
    bc_district_agg = _aggregate(bc_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)
    random_district_agg = _aggregate(random_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)

    rbc_building_agg = _aggregate(rbc_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)
    bc_building_agg = _aggregate(bc_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)
    random_building_agg = _aggregate(random_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)

    # Full dumps use the union of KPIs observed in any run.
    all_district_keys = sorted(
        set().union(*(r["district"].keys() for r in (*rbc_runs, *bc_runs, *random_runs)))
    )
    all_building_keys = sorted(
        set().union(*(r["building"].keys() for r in (*rbc_runs, *bc_runs, *random_runs)))
    )
    rbc_district_full = _aggregate(rbc_runs, scope="district", kpi_keys=all_district_keys)
    bc_district_full = _aggregate(bc_runs, scope="district", kpi_keys=all_district_keys)
    random_district_full = _aggregate(random_runs, scope="district", kpi_keys=all_district_keys)
    rbc_building_full = _aggregate(rbc_runs, scope="building", kpi_keys=all_building_keys)
    bc_building_full = _aggregate(bc_runs, scope="building", kpi_keys=all_building_keys)
    random_building_full = _aggregate(random_runs, scope="building", kpi_keys=all_building_keys)

    md = _build_report(
        bc_root=bc_root,
        bc_seeds=bc_seeds,
        env_seeds=env_seeds,
        eval_rollouts=len(env_seeds),
        smoke=args.smoke,
        rbc_runs=rbc_runs,
        random_runs=random_runs,
        bc_runs=bc_runs,
        rbc_district_agg=rbc_district_agg,
        bc_district_agg=bc_district_agg,
        random_district_agg=random_district_agg,
        rbc_building_agg=rbc_building_agg,
        bc_building_agg=bc_building_agg,
        random_building_agg=random_building_agg,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\n✅ Report written to {output_path}")

    # Persist raw aggregates and per-rollout KPIs alongside the report.
    raw_dir = output_path.parent / "bc_vs_rbc_vs_random_raw_m3"
    raw_dir.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset": DATASET_SCHEMA,
        "reward_function": "V2GPenaltyReward",
        "target_building": {"index": TARGET_BUILDING_INDEX, "name": TARGET_BUILDING_NAME},
        "bc_root": str(bc_root),
        "bc_training_seeds": [s for s, _, _ in bc_seeds],
        "env_seeds": list(env_seeds),
        "smoke": bool(args.smoke),
        "aggregates": {
            "district": {
                "RBC": rbc_district_full,
                "BC": bc_district_full,
                "Random": random_district_full,
            },
            "building_5": {
                "RBC": rbc_building_full,
                "BC": bc_building_full,
                "Random": random_building_full,
            },
        },
        "per_rollout": {
            "RBC": [
                {"env_seed": r["env_seed"], "district": r["district"], "building": r["building"]}
                for r in rbc_runs
            ],
            "Random": [
                {"env_seed": r["env_seed"], "district": r["district"], "building": r["building"]}
                for r in random_runs
            ],
            "BC": [
                {
                    "train_seed": r.get("train_seed"),
                    "env_seed": r["env_seed"],
                    "district": r["district"],
                    "building": r["building"],
                }
                for r in bc_runs
            ],
        },
    }
    (raw_dir / "aggregates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Persist raw KPI DataFrames per rollout for reproducibility.
    for label, runs in (("rbc", rbc_runs), ("random", random_runs), ("bc", bc_runs)):
        for r in runs:
            seed_tag = f"env{r['env_seed']}"
            if "train_seed" in r:
                seed_tag = f"train{r['train_seed']}_{seed_tag}"
            r["kpi_df"].to_csv(raw_dir / f"kpis_{label}_{seed_tag}.csv", index=False)

    print(f"   Raw KPIs in   {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
