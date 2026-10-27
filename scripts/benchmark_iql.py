"""Benchmark IQL vs RBC and BC on Building 5.

Self-contained benchmark mirroring :mod:`scripts.benchmark_bc`. Renders a
3-column report (RBC | BC | IQL) so the reader can see both whether IQL
beats RBC (the primary success criterion) and whether IQL beats BC (i.e.
whether offline RL gained anything over imitation).

Multi-seed schema
-----------------
* **RBC**: ``len(env_seeds)`` rollouts, one per env seed.
* **BC**: ``len(bc_seeds) × len(env_seeds)`` rollouts.
* **IQL**: ``len(iql_seeds) × len(env_seeds)`` rollouts.

Default eval seeds: ``200..209`` (disjoint from RBC dataset seeds 22..31).

Usage
-----
.. code-block:: bash

    # Smoke (1 IQL seed, 1 BC seed, 1 env seed)
    .venv/bin/python -m scripts.benchmark_iql \\
        --iql-root runs/offline_iql/run-001 \\
        --bc-root runs/offline_bc/run-001 \\
        --run-id iql_run001 \\
        --output docs/offline_rl/benchmarks.md \\
        --smoke

    # Full
    .venv/bin/python -m scripts.benchmark_iql \\
        --iql-root runs/offline_iql/run-001 \\
        --bc-root runs/offline_bc/run-001 \\
        --run-id iql_run001 \\
        --output docs/offline_rl/benchmarks.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._benchmark_common import (  # noqa: E402
    DATASET_SCHEMA,
    HEADLINE_BUILDING_KPIS,
    HEADLINE_DISTRICT_KPIS,
    REPO_ROOT,
    TARGET_BUILDING_INDEX,
    TARGET_BUILDING_NAME,
    aggregate,
    delta_significance,
    fmt,
    fmt_mean_std,
    rollout,
    union_keys,
)

from algorithms.offline_rl.bc_agent import BCAgent  # noqa: E402
from algorithms.offline_rl.iql_agent import IQLAgent  # noqa: E402
from algorithms.offline_rl.rbc import OfflineRBC  # noqa: E402


DEFAULT_EVAL_SEEDS: Tuple[int, ...] = tuple(range(200, 210))


# ---------------------------------------------------------------------------
# Discovery & rollouts
# ---------------------------------------------------------------------------


def _make_rbc() -> OfflineRBC:
    return OfflineRBC(
        config={
            "algorithm": {"hyperparameters": {}},
            "simulator": {"dataset_path": DATASET_SCHEMA},
        }
    )


def _run_rbc_rollouts(env_seeds: Sequence[int]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for env_seed in env_seeds:
        runs.append(rollout(_make_rbc(), env_seed=env_seed, label="RBC"))
    return runs


def _discover_seed_dirs(root: Path) -> List[Tuple[int, Path]]:
    """Return ``(seed, seed_dir)`` for every ``seed_<N>/`` with policy.pt."""
    if not root.is_dir():
        raise FileNotFoundError(f"root not a directory: {root}")
    found: List[Tuple[int, Path]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("seed_"):
            continue
        try:
            seed = int(entry.name.split("_", 1)[1])
        except ValueError:
            continue
        if (entry / "policy.pt").is_file() and (entry / "architecture.json").is_file():
            found.append((seed, entry))
    if not found:
        raise FileNotFoundError(f"No seed dirs (policy.pt + architecture.json) under {root}")
    return found


def _run_bc_rollouts(
    bc_seeds: Sequence[Tuple[int, Path]], env_seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for train_seed, seed_dir in bc_seeds:
        for env_seed in env_seeds:
            agent = BCAgent.from_seed_dir(seed_dir)
            run = rollout(agent, env_seed=env_seed, label=f"BC-s{train_seed}")
            run["train_seed"] = train_seed
            runs.append(run)
    return runs


def _run_iql_rollouts(
    iql_seeds: Sequence[Tuple[int, Path]], env_seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for train_seed, seed_dir in iql_seeds:
        for env_seed in env_seeds:
            agent = IQLAgent.from_seed_dir(seed_dir)
            run = rollout(agent, env_seed=env_seed, label=f"IQL-s{train_seed}")
            run["train_seed"] = train_seed
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Report (3-column: RBC | BC | IQL)
# ---------------------------------------------------------------------------


def _render_kpi_table(
    title: str,
    *,
    rbc_agg: Dict[str, Dict[str, Any]],
    bc_agg: Dict[str, Dict[str, Any]],
    iql_agg: Dict[str, Dict[str, Any]],
    keys: Sequence[str],
) -> str:
    lines = [
        f"### {title}",
        "",
        "| KPI | RBC (mean ± std) | BC (mean ± std) | IQL (mean ± std) | "
        "Δ (IQL − RBC) | Verdict (IQL vs RBC) |",
        "|---|---:|---:|---:|---:|:---|",
    ]
    for k in keys:
        ra = rbc_agg.get(k, {"mean": None, "std": None})
        ba = bc_agg.get(k, {"mean": None, "std": None})
        ia = iql_agg.get(k, {"mean": None, "std": None})
        if ra["mean"] is None and ba["mean"] is None and ia["mean"] is None:
            continue
        delta, verdict = delta_significance(ia, ra, cand_label="IQL", base_label="RBC")
        lines.append(
            f"| `{k}` | {fmt_mean_std(ra)} | {fmt_mean_std(ba)} | "
            f"{fmt_mean_std(ia)} | {fmt(delta)} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def _full_agg(runs: Sequence[Dict[str, Any]], scope: str) -> Dict[str, Dict[str, Any]]:
    return aggregate(runs, scope=scope, kpi_keys=union_keys(runs, scope))


def _render_full_dump(
    *,
    rbc_agg: Dict[str, Dict[str, Any]],
    bc_agg: Dict[str, Dict[str, Any]],
    iql_agg: Dict[str, Dict[str, Any]],
) -> str:
    keys = sorted(set(rbc_agg) | set(bc_agg) | set(iql_agg))
    lines = [
        "| KPI | RBC | BC | IQL | Δ (IQL − RBC) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in keys:
        ra = rbc_agg.get(k, {"mean": None, "std": None})
        ba = bc_agg.get(k, {"mean": None, "std": None})
        ia = iql_agg.get(k, {"mean": None, "std": None})
        delta = (
            (ia["mean"] - ra["mean"])
            if (ia["mean"] is not None and ra["mean"] is not None)
            else None
        )
        lines.append(
            f"| `{k}` | {fmt_mean_std(ra)} | {fmt_mean_std(ba)} | "
            f"{fmt_mean_std(ia)} | {fmt(delta)} |"
        )
    return "\n".join(lines) + "\n"


def _build_report(
    *,
    iql_root: Path,
    bc_root: Path,
    iql_seeds: Sequence[Tuple[int, Path]],
    bc_seeds: Sequence[Tuple[int, Path]],
    env_seeds: Sequence[int],
    smoke: bool,
    rbc_runs: Sequence[Dict[str, Any]],
    bc_runs: Sequence[Dict[str, Any]],
    iql_runs: Sequence[Dict[str, Any]],
    rbc_district_agg: Dict[str, Dict[str, Any]],
    bc_district_agg: Dict[str, Dict[str, Any]],
    iql_district_agg: Dict[str, Dict[str, Any]],
    rbc_building_agg: Dict[str, Dict[str, Any]],
    bc_building_agg: Dict[str, Dict[str, Any]],
    iql_building_agg: Dict[str, Dict[str, Any]],
) -> str:
    iql_seed_list = ", ".join(str(s) for s, _ in iql_seeds)
    bc_seed_list = ", ".join(str(s) for s, _ in bc_seeds)
    env_seed_list = ", ".join(str(s) for s in env_seeds)
    smoke_note = (
        " **(SMOKE MODE — single rollout per controller, results indicative only)**"
        if smoke else ""
    )

    headline_district = _render_kpi_table(
        "District",
        rbc_agg=rbc_district_agg,
        bc_agg=bc_district_agg,
        iql_agg=iql_district_agg,
        keys=HEADLINE_DISTRICT_KPIS,
    )
    headline_building = _render_kpi_table(
        TARGET_BUILDING_NAME,
        rbc_agg=rbc_building_agg,
        bc_agg=bc_building_agg,
        iql_agg=iql_building_agg,
        keys=HEADLINE_BUILDING_KPIS,
    )

    return f"""# IQL vs RBC vs BC — CityLearn Benchmark
{smoke_note}
_Dataset_: `{DATASET_SCHEMA}` (interface=`flat`, topology_mode=`static`)
_Reward function (env)_: `V2GPenaltyReward`
_Target building_: **{TARGET_BUILDING_NAME}** (agent index {TARGET_BUILDING_INDEX})
_IQL-checkpoint root_: `{iql_root}`
_BC-checkpoint root_:  `{bc_root}`
_IQL training seeds_: [{iql_seed_list}]
_BC training seeds_:  [{bc_seed_list}]
_Env seeds_ (per controller): [{env_seed_list}] ← disjoint from RBC dataset seeds 22..31
_Total rollouts_: RBC={len(rbc_runs)}, BC={len(bc_runs)}, IQL={len(iql_runs)}

> All KPIs are CityLearn's normalised values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across training
> seeds where applicable). The "Verdict" column flags
> `|Δmean| > max(IQL_std, RBC_std, 1e-4)` as significant; otherwise the
> difference is within noise.
>
> All three controllers operate on the same env in identical conditions.
> The RBC controls every building. BC controls Building 5 with the BC
> policy and defers the other 16 buildings to a fresh RBC. IQL does the
> same with the IQL policy. Off-target buildings should therefore be
> bit-identical across the three columns; any deltas there are pure
> downstream coupling effects.

---

## 1. Headline KPIs — district level

{headline_district}

---

## 2. Headline KPIs — Building 5 (training target)

{headline_building}

---

## 3. Full KPI dump — district

<details>
<summary>Click to expand</summary>

{_render_full_dump(
    rbc_agg=_full_agg(rbc_runs, 'district'),
    bc_agg=_full_agg(bc_runs, 'district'),
    iql_agg=_full_agg(iql_runs, 'district'),
)}

</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

{_render_full_dump(
    rbc_agg=_full_agg(rbc_runs, 'building'),
    bc_agg=_full_agg(bc_runs, 'building'),
    iql_agg=_full_agg(iql_runs, 'building'),
)}

</details>

---

## 5. Success criterion

> IQL succeeds if it beats RBC by **more than 1σ** on at least one of
> {{`cost_total`, `all_time_peak_average`, `ramping_average`}} at the
> district level **with `annual_normalized_unserved_energy_total` = 0**.
>
> A favourable comparison against BC is informative but not part of the
> success contract — the contract is "offline RL beats the data-collection
> policy on its own success metric".
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iql-root", required=True,
        help="Root dir produced by scripts/train_iql.py (contains seed_<N>/).",
    )
    parser.add_argument(
        "--bc-root", required=True,
        help="Root dir produced by scripts/train_bc.py (contains seed_<N>/).",
    )
    parser.add_argument(
        "--output",
        default="docs/offline_rl/benchmarks.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--run-id",
        default="iql_run001",
        help=(
            "Identifier for this IQL run (e.g. iql_run001). Raw KPI CSVs are "
            "written to datasets/offline_rl/benchmarks/<run-id>/."
        ),
    )
    parser.add_argument(
        "--env-seeds", default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS),
        help="Comma-separated env seeds for evaluation (default 200..209).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: 1 train seed each, 1 env seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    iql_root = Path(args.iql_root).expanduser()
    if not iql_root.is_absolute():
        iql_root = (REPO_ROOT / iql_root).resolve()
    bc_root = Path(args.bc_root).expanduser()
    if not bc_root.is_absolute():
        bc_root = (REPO_ROOT / bc_root).resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()

    env_seeds: List[int] = [int(s) for s in args.env_seeds.split(",") if s.strip()]
    if not env_seeds:
        raise ValueError("No env seeds resolved; check --env-seeds.")

    iql_seeds = _discover_seed_dirs(iql_root)
    bc_seeds = _discover_seed_dirs(bc_root)
    if args.smoke:
        iql_seeds = iql_seeds[:1]
        bc_seeds = bc_seeds[:1]
        env_seeds = env_seeds[:1]
        print(
            f"[smoke] iql_seeds={[s for s,_ in iql_seeds]} "
            f"bc_seeds={[s for s,_ in bc_seeds]} env_seeds={env_seeds}"
        )

    print(f"[plan] IQL training seeds: {[s for s,_ in iql_seeds]}")
    print(f"[plan] BC training seeds : {[s for s,_ in bc_seeds]}")
    print(f"[plan] env seeds         : {env_seeds}")
    n_iql = len(iql_seeds) * len(env_seeds)
    n_bc = len(bc_seeds) * len(env_seeds)
    print(f"[plan] total rollouts    : RBC={len(env_seeds)}, BC={n_bc}, IQL={n_iql}")

    print("[run] RBC ...")
    rbc_runs = _run_rbc_rollouts(env_seeds)
    print(f"[run] RBC done ({len(rbc_runs)} rollouts)")

    print("[run] BC ...")
    bc_runs = _run_bc_rollouts(bc_seeds, env_seeds)
    print(f"[run] BC done ({len(bc_runs)} rollouts)")

    print("[run] IQL ...")
    iql_runs = _run_iql_rollouts(iql_seeds, env_seeds)
    print(f"[run] IQL done ({len(iql_runs)} rollouts)")

    rbc_district_agg = aggregate(rbc_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)
    bc_district_agg  = aggregate(bc_runs,  scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)
    iql_district_agg = aggregate(iql_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS)
    rbc_building_agg = aggregate(rbc_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)
    bc_building_agg  = aggregate(bc_runs,  scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)
    iql_building_agg = aggregate(iql_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS)

    md = _build_report(
        iql_root=iql_root, bc_root=bc_root,
        iql_seeds=iql_seeds, bc_seeds=bc_seeds, env_seeds=env_seeds,
        smoke=args.smoke,
        rbc_runs=rbc_runs, bc_runs=bc_runs, iql_runs=iql_runs,
        rbc_district_agg=rbc_district_agg, bc_district_agg=bc_district_agg,
        iql_district_agg=iql_district_agg,
        rbc_building_agg=rbc_building_agg, bc_building_agg=bc_building_agg,
        iql_building_agg=iql_building_agg,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\n[ok] Report written to {output_path}")

    raw_dir = REPO_ROOT / "datasets" / "offline_rl" / "benchmarks" / args.run_id
    raw_dir.mkdir(exist_ok=True)
    payload = {
        "dataset": DATASET_SCHEMA,
        "reward_function": "V2GPenaltyReward",
        "target_building": {"index": TARGET_BUILDING_INDEX, "name": TARGET_BUILDING_NAME},
        "iql_root": str(iql_root),
        "bc_root": str(bc_root),
        "iql_training_seeds": [s for s, _ in iql_seeds],
        "bc_training_seeds": [s for s, _ in bc_seeds],
        "env_seeds": list(env_seeds),
        "smoke": bool(args.smoke),
        "aggregates": {
            "district": {
                "RBC": _full_agg(rbc_runs, "district"),
                "BC":  _full_agg(bc_runs,  "district"),
                "IQL": _full_agg(iql_runs, "district"),
            },
            "building_5": {
                "RBC": _full_agg(rbc_runs, "building"),
                "BC":  _full_agg(bc_runs,  "building"),
                "IQL": _full_agg(iql_runs, "building"),
            },
        },
        "per_rollout": {
            "RBC": [
                {"env_seed": r["env_seed"], "district": r["district"], "building": r["building"]}
                for r in rbc_runs
            ],
            "BC": [
                {"train_seed": r.get("train_seed"), "env_seed": r["env_seed"],
                 "district": r["district"], "building": r["building"]}
                for r in bc_runs
            ],
            "IQL": [
                {"train_seed": r.get("train_seed"), "env_seed": r["env_seed"],
                 "district": r["district"], "building": r["building"]}
                for r in iql_runs
            ],
        },
    }
    (raw_dir / "aggregates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for label, runs in (("rbc", rbc_runs), ("bc", bc_runs), ("iql", iql_runs)):
        for r in runs:
            seed_tag = f"env{r['env_seed']}"
            if "train_seed" in r:
                seed_tag = f"train{r['train_seed']}_{seed_tag}"
            r["kpi_df"].to_csv(raw_dir / f"kpis_{label}_{seed_tag}.csv", index=False)
    print(f"  Raw KPIs in {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
