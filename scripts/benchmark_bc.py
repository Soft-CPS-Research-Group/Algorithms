"""Benchmark BC vs RBC on Building 5.

Self-contained benchmark. Mirrors :mod:`scripts.benchmark_bc_m3` in
structure but:

* Uses **RBC** (``OfflineRBC``) as the baseline — the same
 controller that produced the offline dataset, so BC's parity is measured
 against its own behaviour policy.
* Loads BC checkpoints via ``BCAgent.from_seed_dir``.
* Evaluates on env seeds **disjoint** from the RBC dataset seeds (22..31):
 defaults to ``200..209`` (10 seeds).
* No Random comparator — the v1 benchmark already established Random ≫ RBC
 on this task; re-running it adds noise without information for.

Multi-seed schema
-----------------
* **RBC** : ``len(env_seeds)`` rollouts, one per env seed.
* **BC** : ``len(bc_seeds) × len(env_seeds)`` rollouts. For each BC
 training seed under ``--bc-root`` we load the seed dir and run one
 rollout per env seed.

Usage
-----
.. code-block:: bash

 # Smoke (1 BC training seed, 1 env seed, 1 rollout per controller)
 .venv/bin/python -m scripts.benchmark_bc \\
 --bc-root runs/offline_bc/run-001 \\
 --output docs/offline_rl/bc_vs_rbc_benchmark.md \\
 --smoke

 # Full (5 BC seeds × 10 eval seeds = 50 BC rollouts; 10 RBC rollouts)
 .venv/bin/python -m scripts.benchmark_bc \\
 --bc-root runs/offline_bc/run-001 \\
 --output docs/offline_rl/bc_vs_rbc_benchmark.md
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

from scripts._benchmark_common import (
 DATASET_SCHEMA,
 HEADLINE_BUILDING_KPIS,
 HEADLINE_DISTRICT_KPIS,
 REPO_ROOT,
 TARGET_BUILDING_INDEX,
 TARGET_BUILDING_NAME,
 aggregate,
 fmt,
 fmt_mean_std,
 delta_significance,
 rollout,
 union_keys,
)

from algorithms.offline_rl.bc_agent import BCAgent # noqa: E402
from algorithms.offline_rl.rbc import OfflineRBC # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Disjoint from RBC dataset seeds 22..31. 10 eval seeds.
DEFAULT_EVAL_SEEDS: Tuple[int, ...] = tuple(range(200, 210))


# ---------------------------------------------------------------------------
# controller rollouts
# ---------------------------------------------------------------------------


def _make_rbc() -> OfflineRBC:
 """RBC with the standard schema config."""
 return OfflineRBC(
 config={
 "algorithm": {"hyperparameters": {}},
 "simulator": {"dataset_path": DATASET_SCHEMA},
 }
 )


def _run_rbc_rollouts(env_seeds: Sequence[int]) -> List[Dict[str, Any]]:
 runs: List[Dict[str, Any]] = []
 for env_seed in env_seeds:
 agent = _make_rbc()
 runs.append(rollout(agent, env_seed=env_seed, label="RBC"))
 return runs


def _discover_bc_seed_dirs(root: Path) -> List[Tuple[int, Path]]:
 """Return ``(seed, seed_dir)`` for every ``seed_<N>/`` under ``root`` with
 a ``policy.pt`` and ``architecture.json`` (BC layout).
 """
 if not root.is_dir():
 raise FileNotFoundError(f"BC root not a directory: {root}")
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
 raise FileNotFoundError(
 f"No BC seed dirs (policy.pt + architecture.json) under {root}"
 )
 return found


def _run_bc_rollouts(
 bc_seeds: Sequence[Tuple[int, Path]],
 env_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
 runs: List[Dict[str, Any]] = []
 for train_seed, seed_dir in bc_seeds:
 for env_seed in env_seeds:
 agent = BCAgent.from_seed_dir(seed_dir)
 run = rollout(agent, env_seed=env_seed, label=f"BC-s{train_seed}")
 run["train_seed"] = train_seed
 runs.append(run)
 return runs


# ---------------------------------------------------------------------------
# Report (2-column: RBC vs BC)
# ---------------------------------------------------------------------------


def _render_kpi_table(
 title: str,
 *,
 rbc_agg: Dict[str, Dict[str, Any]],
 bc_agg: Dict[str, Dict[str, Any]],
 keys: Sequence[str],
) -> str:
 lines = [
 f"### {title}",
 "",
 "| KPI | RBC (mean ± std) | BC (mean ± std) | "
 "Δ (BC − RBC) | Verdict |",
 "|---|---:|---:|---:|:---|",
 ]
 for k in keys:
 ba = rbc_agg.get(k, {"mean": None, "std": None})
 ca = bc_agg.get(k, {"mean": None, "std": None})
 if ba["mean"] is None and ca["mean"] is None:
 continue
 delta, verdict = delta_significance(
 ca, ba, cand_label="BC", base_label="RBC"
 )
 lines.append(
 f"| `{k}` | {fmt_mean_std(ba)} | {fmt_mean_std(ca)} | "
 f"{fmt(delta)} | {verdict} |"
 )
 return "\n".join(lines) + "\n"


def _full_agg(runs: Sequence[Dict[str, Any]], scope: str) -> Dict[str, Dict[str, Any]]:
 keys = union_keys(runs, scope)
 return aggregate(runs, scope=scope, kpi_keys=keys)


def _render_full_dump(
 *, rbc_agg: Dict[str, Dict[str, Any]], bc_agg: Dict[str, Dict[str, Any]]
) -> str:
 keys = sorted(set(rbc_agg) | set(bc_agg))
 lines = [
 "| KPI | RBC | BC | Δ (BC − RBC) |",
 "|---|---:|---:|---:|",
 ]
 for k in keys:
 ba = rbc_agg.get(k, {"mean": None, "std": None})
 ca = bc_agg.get(k, {"mean": None, "std": None})
 delta = (
 (ca["mean"] - ba["mean"])
 if (ca["mean"] is not None and ba["mean"] is not None)
 else None
 )
 lines.append(
 f"| `{k}` | {fmt_mean_std(ba)} | {fmt_mean_std(ca)} | {fmt(delta)} |"
 )
 return "\n".join(lines) + "\n"


def _build_report(
 *,
 bc_root: Path,
 bc_seeds: Sequence[Tuple[int, Path]],
 env_seeds: Sequence[int],
 smoke: bool,
 rbc_runs: Sequence[Dict[str, Any]],
 bc_runs: Sequence[Dict[str, Any]],
 rbc_district_agg: Dict[str, Dict[str, Any]],
 bc_district_agg: Dict[str, Dict[str, Any]],
 rbc_building_agg: Dict[str, Dict[str, Any]],
 bc_building_agg: Dict[str, Dict[str, Any]],
) -> str:
 timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
 train_seed_list = ", ".join(str(s) for s, _ in bc_seeds)
 env_seed_list = ", ".join(str(s) for s in env_seeds)
 smoke_note = (
 " **(SMOKE MODE — single rollout per controller, results indicative only)**"
 if smoke else ""
 )

 headline_district = _render_kpi_table(
 "District",
 rbc_agg=rbc_district_agg,
 bc_agg=bc_district_agg,
 keys=HEADLINE_DISTRICT_KPIS,
 )
 headline_building = _render_kpi_table(
 TARGET_BUILDING_NAME,
 rbc_agg=rbc_building_agg,
 bc_agg=bc_building_agg,
 keys=HEADLINE_BUILDING_KPIS,
 )

 return f"""# BC vs RBC — CityLearn Benchmark

_Generated_: {timestamp}{smoke_note}
_Dataset_: `{DATASET_SCHEMA}` (interface=`flat`, topology_mode=`static`)
_Reward function_: `V2GPenaltyReward` (env reward; reward is a separate scalar used for IQL)
_Target building_: **{TARGET_BUILDING_NAME}** (agent index {TARGET_BUILDING_INDEX})
_BC-checkpoint root_: `{bc_root}`
_BC-training seeds_: [{train_seed_list}]
_Env seeds_ (per controller): [{env_seed_list}] ← disjoint from RBC dataset seeds 22..31
_Total rollouts_: RBC={len(rbc_runs)}, BC={len(bc_runs)}

> All KPIs are CityLearn's normalized values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across BC training
> seeds for BC). The "Verdict" column flags `|Δmean| > max(BC_std, RBC_std, 1e-4)` as
> significant; otherwise the difference is within noise.
>
> The BC agent controls Building 5 only; the other 16 agents are driven by
> a fresh RBC instance (so off-target buildings are bit-identical between
> the two columns and Δ there should be ≈ 0 — they only differ via downstream
> coupling effects).

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
)}

</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

{_render_full_dump(
 rbc_agg=_full_agg(rbc_runs, 'building'),
 bc_agg=_full_agg(bc_runs, 'building'),
)}

</details>

---

## 5. Success criterion 

> BC succeeds if its district headline KPIs are **within 1σ of RBC** on
> all of cost / peak / ramping / unserved-energy. That demonstrates the
> pipeline reproduces its behaviour policy faithfully — a precondition for
> the IQL step that follows.
>
> A *strict improvement* over RBC is not expected from BC alone (BC is
> imitation, not optimisation); that's IQL's job in the next step.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument(
 "--bc-root", required=True,
 help="Root dir produced by scripts/train_bc.py (contains seed_<N>/).",
 )
 parser.add_argument(
 "--output",
 default="docs/offline_rl/bc_vs_rbc_benchmark.md",
 help="Output markdown report path (relative to repo root).",
 )
 parser.add_argument(
 "--env-seeds", default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS),
 help="Comma-separated env seeds for evaluation rollouts (default 200..209).",
 )
 parser.add_argument(
 "--smoke", action="store_true",
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
 if not env_seeds:
 raise ValueError("No env seeds resolved; check --env-seeds.")

 bc_seeds = _discover_bc_seed_dirs(bc_root)
 if args.smoke:
 bc_seeds = bc_seeds[:1]
 env_seeds = env_seeds[:1]
 print(f"[smoke] using bc_seeds={[s for s,_ in bc_seeds]}, env_seeds={env_seeds}")

 print(f"[plan] BC training seeds: {[s for s,_ in bc_seeds]}")
 print(f"[plan] env seeds : {env_seeds}")
 total_bc = len(bc_seeds) * len(env_seeds)
 print(f"[plan] total rollouts : RBC={len(env_seeds)}, BC={total_bc}")

 print("[run] RBC...")
 rbc_runs = _run_rbc_rollouts(env_seeds)
 print(f"[run] RBC done ({len(rbc_runs)} rollouts)")

 print("[run] BC...")
 bc_runs = _run_bc_rollouts(bc_seeds, env_seeds)
 print(f"[run] BC done ({len(bc_runs)} rollouts)")

 rbc_district_agg = aggregate(
 rbc_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS
 )
 bc_district_agg = aggregate(
 bc_runs, scope="district", kpi_keys=HEADLINE_DISTRICT_KPIS
 )
 rbc_building_agg = aggregate(
 rbc_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS
 )
 bc_building_agg = aggregate(
 bc_runs, scope="building", kpi_keys=HEADLINE_BUILDING_KPIS
 )

 md = _build_report(
 bc_root=bc_root, bc_seeds=bc_seeds, env_seeds=env_seeds,
 smoke=args.smoke,
 rbc_runs=rbc_runs, bc_runs=bc_runs,
 rbc_district_agg=rbc_district_agg, bc_district_agg=bc_district_agg,
 rbc_building_agg=rbc_building_agg, bc_building_agg=bc_building_agg,
 )

 output_path.parent.mkdir(parents=True, exist_ok=True)
 output_path.write_text(md, encoding="utf-8")
 print(f"\n[ok] Report written to {output_path}")

 # Persist raw aggregates and per-rollout KPIs alongside the report.
 raw_dir = output_path.parent / "bc_vs_rbc_raw"
 raw_dir.mkdir(exist_ok=True)
 payload = {
 "generated_at": datetime.utcnow().isoformat() + "Z",
 "dataset": DATASET_SCHEMA,
 "reward_function": "V2GPenaltyReward",
 "target_building": {
 "index": TARGET_BUILDING_INDEX,
 "name": TARGET_BUILDING_NAME,
 },
 "bc_root": str(bc_root),
 "bc_training_seeds": [s for s, _ in bc_seeds],
 "env_seeds": list(env_seeds),
 "smoke": bool(args.smoke),
 "aggregates": {
 "district": {
 "RBC": _full_agg(rbc_runs, "district"),
 "BC": _full_agg(bc_runs, "district"),
 },
 "building_5": {
 "RBC": _full_agg(rbc_runs, "building"),
 "BC": _full_agg(bc_runs, "building"),
 },
 },
 "per_rollout": {
 "RBC": [
 {
 "env_seed": r["env_seed"],
 "district": r["district"],
 "building": r["building"],
 }
 for r in rbc_runs
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
 (raw_dir / "aggregates.json").write_text(
 json.dumps(payload, indent=2), encoding="utf-8"
 )

 for label, runs in (("rbc", rbc_runs), ("bc", bc_runs)):
 for r in runs:
 seed_tag = f"env{r['env_seed']}"
 if "train_seed" in r:
 seed_tag = f"train{r['train_seed']}_{seed_tag}"
 r["kpi_df"].to_csv(raw_dir / f"kpis_{label}_{seed_tag}.csv", index=False)

 print(f" Raw KPIs in {raw_dir}")
 return 0


if __name__ == "__main__":
 raise SystemExit(main())
