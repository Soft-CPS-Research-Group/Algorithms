"""Phase 10 — Curate offline-RL initiative figures.

Given a completed pipeline output directory at
``runs/offline_iql_cql_initiative_15min/`` (or any compatible run dir),
produce 11 thesis-grade PNGs at ``docs/offline_rl/iql_cql_figures/``:

  01_pipeline_overview.png            — pipeline architecture diagram
  02_dataset_stats.png                — dataset summary table
  03_action_coverage_group_a.png      — action distribution (showcase group)
  04_reward_by_regime.png             — reward by RBC regime
  05_correlations_group_a.png         — feature x reward correlations
  06_temporal_patterns.png            — temporal patterns
  07_training_loss_group_a.png        — IQL vs CQL training loss
  08_training_valmse_all.png          — validation MSE across all groups
  09_training_cql_penalty.png         — CQL conservative penalty curve
  10_benchmark_kpi_bars.png           — KPI comparison bars
  11_iql_vs_cql_scatter.png           — per-seed cost scatter

Idempotent: re-running with the same inputs overwrites the same outputs.
Writes a ``.curation.done`` sentinel on success.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure ``scripts.*`` imports resolve when this file is executed directly
# (e.g. ``python scripts/curate_initiative_figures.py``).  Without this guard
# the script silently drops figs 01/07/08/09 because the helper imports raise
# ModuleNotFoundError, which the script swallows as warnings.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "offline_rl" / "iql_cql_figures"
DEFAULT_SHOWCASE_GROUP = "obs627_act1"
DEFAULT_GROUPS = ["obs627_act1", "obs706_act2", "obs749_act3", "obs785_act3"]

logger = logging.getLogger("curate_initiative_figures")

# (source filename template, destination filename) -- `{group}` is replaced with showcase group.
FEATURE_ANALYSIS_COPY_MAP = [
    ("01_dataset_stats_table.png",     "02_dataset_stats.png"),
    ("03_action_coverage_{group}.png", "03_action_coverage_group_a.png"),
    ("04_reward_by_regime.png",        "04_reward_by_regime.png"),
    ("05_correlations_{group}.png",    "05_correlations_group_a.png"),
    ("07_temporal_patterns.png",       "06_temporal_patterns.png"),
]

# Benchmark fig 10 -- KPI bar layout.
BENCHMARK_KPIS = [
    # (key, panel title)
    ("cost_total",             "Total cost"),
    ("carbon_emissions_total", "Carbon emissions"),
    ("daily_peak_average",     "Daily peak avg"),
    ("ramping_average",        "Ramping avg"),
]
BENCHMARK_ALGORITHMS = ["RBCSmart", "IQL", "CQL"]
ALGO_COLORS = {"RBCSmart": "#9E9E9E", "IQL": "#2196F3", "CQL": "#F44336"}


def _copy_feature_analysis_figures(
    *,
    run_dir: Path,
    showcase_group: str,
    output_dir: Path,
) -> List[Path]:
    """Copy 5 figures from <run_dir>/feature_analysis/figures/ with renames.

    Returns the list of destination paths that were successfully created.
    Missing sources are logged as warnings and skipped.
    """
    src_dir = run_dir / "feature_analysis" / "figures"
    if not src_dir.exists():
        logger.warning(
            "[curate] feature_analysis/figures not found at %s -- skipping figs 02-06",
            src_dir,
        )
        return []

    produced: List[Path] = []
    for src_template, dst_name in FEATURE_ANALYSIS_COPY_MAP:
        src_name = src_template.format(group=showcase_group)
        src = src_dir / src_name
        if not src.exists():
            logger.warning("[curate] source figure missing: %s -- skipping", src)
            continue
        dst = output_dir / dst_name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        produced.append(dst)
        logger.info("[curate] copied %s -> %s", src.name, dst.name)
    return produced


def _render_pipeline_diagram(*, output_dir: Path) -> Path | None:
    """Render the pipeline architecture diagram.

    Imports ``fig_pipeline`` from ``scripts.generate_architecture_figures``
    (renders to ``fig_arch_pipeline.png`` in a tmp dir) and copies the result
    to ``01_pipeline_overview.png`` in ``output_dir``.

    Returns the destination path, or ``None`` if the import or render fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "01_pipeline_overview.png"

    try:
        from scripts.generate_architecture_figures import fig_pipeline
    except ImportError as exc:
        logger.warning(
            "[curate] could not import fig_pipeline from generate_architecture_figures: %s",
            exc,
        )
        return None

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            produced = fig_pipeline(tmp_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[curate] fig_pipeline raised %s -- skipping fig 01", exc)
            return None
        if produced is None or not Path(produced).exists():
            logger.warning("[curate] fig_pipeline produced no PNG -- skipping fig 01")
            return None
        shutil.copy2(produced, dst)

    logger.info("[curate] rendered %s", dst.name)
    return dst


def _render_training_curves(
    *,
    run_dir: Path,
    showcase_group: str,
    groups: List[str],
    output_dir: Path,
) -> List[Path]:
    """Render training-curve figures using generate_training_figures helpers.

    Produces:
    * 07_training_loss_group_a.png   -- IQL vs CQL loss curves for showcase group.
    * 08_training_valmse_all.png     -- val MSE for IQL vs CQL across all groups.
    * 09_training_cql_penalty.png    -- CQL conservative penalty curve.

    Returns the list of destination paths that were successfully produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    iql_root = run_dir / "models-iql"
    cql_root = run_dir / "models-cql"
    iql_arg = iql_root if iql_root.exists() else None
    cql_arg = cql_root if cql_root.exists() else None

    if iql_arg is None and cql_arg is None:
        logger.warning(
            "[curate] no models-iql or models-cql under %s -- skipping figs 07-09",
            run_dir,
        )
        return []

    try:
        from scripts.generate_training_figures import (
            _plot_cql_penalty,
            _plot_loss_curves,
            _plot_val_mse,
        )
    except ImportError as exc:
        logger.warning("[curate] could not import training-curve helpers: %s", exc)
        return []

    import tempfile

    produced: List[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # fig 07 -- loss curves for showcase group
        try:
            loss = _plot_loss_curves(iql_arg, cql_arg, showcase_group, tmp_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[curate] _plot_loss_curves raised %s -- skipping fig 07", exc)
            loss = None
        if loss and Path(loss).exists():
            dst = output_dir / "07_training_loss_group_a.png"
            shutil.copy2(loss, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

        # fig 08 -- val MSE all groups
        try:
            val = _plot_val_mse(iql_arg, cql_arg, groups, tmp_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[curate] _plot_val_mse raised %s -- skipping fig 08", exc)
            val = None
        if val and Path(val).exists():
            dst = output_dir / "08_training_valmse_all.png"
            shutil.copy2(val, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

        # fig 09 -- CQL penalty
        try:
            pen = _plot_cql_penalty(cql_arg, groups, tmp_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[curate] _plot_cql_penalty raised %s -- skipping fig 09", exc)
            pen = None
        if pen and Path(pen).exists():
            dst = output_dir / "09_training_cql_penalty.png"
            shutil.copy2(pen, dst)
            produced.append(dst)
            logger.info("[curate] rendered %s", dst.name)

    return produced


def _render_benchmark_kpi_bars(*, results_json: Path, output_dir: Path) -> Path | None:
    """Render 2x2 bar chart of KPIs comparing RBCSmart vs IQL vs CQL.

    Reads ``results_json`` with the canonical benchmark schema (each algo has an
    ``aggregate`` map of ``{kpi: {mean, std, n}}``). Returns the path to the
    produced PNG, or None on missing input or no KPIs at all.
    """
    if not results_json.exists():
        logger.warning(
            "[curate] benchmark results.json not found at %s -- skipping fig 10",
            results_json,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    with results_json.open() as f:
        data = json.load(f)

    # Phase 13: emit wilcoxon_report.json alongside the PNGs so section 7
    # p-values in docs/offline_rl/iql_cql_initiative.md are reproducible
    # from a single curate call. Log-only side effect if scipy is absent
    # (individual pvalues will be None with an explanatory note).
    report = _compute_wilcoxon_report(data)
    _persist_wilcoxon_report(report, output_dir)
    _log_wilcoxon_summary(report)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Benchmark KPIs -- RBCSmart vs IQL vs CQL", fontsize=13)

    any_drawn = False
    for ax, (key, title) in zip(axes.flat, BENCHMARK_KPIS):
        means: List[float] = []
        stds: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        for algo in BENCHMARK_ALGORITHMS:
            agg = data.get(algo, {}).get("aggregate", {}).get(key)
            if agg is None:
                continue
            means.append(float(agg["mean"]))
            stds.append(float(agg.get("std", 0.0) or 0.0))
            labels.append(algo)
            colors.append(ALGO_COLORS.get(algo, "#666666"))
        if not means:
            ax.set_visible(False)
            continue
        any_drawn = True
        x = list(range(len(means)))
        ax.bar(x, means, yerr=stds, color=colors, capsize=4,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if not any_drawn:
        plt.close(fig)
        logger.warning(
            "[curate] no benchmark KPIs found in %s -- skipping fig 10",
            results_json,
        )
        return None

    fig.tight_layout()
    dst = output_dir / "10_benchmark_kpi_bars.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[curate] rendered %s", dst.name)
    return dst


def _render_iql_vs_cql_scatter(*, results_json: Path, output_dir: Path) -> Path | None:
    """Render per-eval-seed scatter of IQL ``cost_total`` vs CQL ``cost_total``.

    Pairs runs by ``env_seed``. Adds a y=x reference line and, when n>=2 seeds,
    annotates the paired Wilcoxon p-value (else ``n=1, p=N/A``).

    Returns the destination path or None.
    """
    if not results_json.exists():
        logger.warning(
            "[curate] benchmark results.json not found at %s -- skipping fig 11",
            results_json,
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    with results_json.open() as f:
        data = json.load(f)

    def _by_seed(algo: str) -> dict:
        return {
            r["env_seed"]: r["district"]["cost_total"]
            for r in data.get(algo, {}).get("runs", [])
            if isinstance(r, dict) and "env_seed" in r and "district" in r
            and "cost_total" in r["district"]
        }

    iql_runs = _by_seed("IQL")
    cql_runs = _by_seed("CQL")
    seeds = sorted(set(iql_runs) & set(cql_runs))
    if not seeds:
        logger.warning(
            "[curate] no paired IQL/CQL eval seeds in %s -- skipping fig 11",
            results_json,
        )
        return None

    iql_costs = [iql_runs[s] for s in seeds]
    cql_costs = [cql_runs[s] for s in seeds]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(iql_costs, cql_costs, s=60, alpha=0.75,
               edgecolor="black", linewidth=0.6, color="#673AB7")

    # y=x reference line
    lo = min(min(iql_costs), min(cql_costs))
    hi = max(max(iql_costs), max(cql_costs))
    margin = (hi - lo) * 0.05 if hi > lo else 0.1
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, alpha=0.5, label="y = x")

    # Annotation
    if len(seeds) >= 2:
        try:
            from scipy.stats import wilcoxon
            _stat, pval = wilcoxon(iql_costs, cql_costs)
            annotation = f"n={len(seeds)}, Wilcoxon p={pval:.3f}"
        except Exception as exc:  # pragma: no cover - defensive
            annotation = f"n={len(seeds)}, p=N/A ({exc})"
    else:
        annotation = f"n={len(seeds)}, p=N/A"
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("IQL cost_total")
    ax.set_ylabel("CQL cost_total")
    ax.set_title("Per-seed cost comparison -- IQL vs CQL")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    dst = output_dir / "11_iql_vs_cql_scatter.png"
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[curate] rendered %s", dst.name)
    return dst


# -----------------------------------------------------------------------------
# Wilcoxon report helpers (Phase 13)
# -----------------------------------------------------------------------------
#
# Emit a machine-readable ``wilcoxon_report.json`` alongside the curated PNGs
# so every p-value cited in section 7 of docs/offline_rl/iql_cql_initiative.md
# is regeneratable from a single ``curate_initiative_figures`` invocation
# (and, transitively, from ``run_entity_pipeline`` when Phase 10 fires).

_WILCOXON_KPIS = (
    "cost_total",
    "carbon_emissions_total",
    "daily_peak_average",
    "ramping_average",
    "annual_normalized_unserved_energy_total",
    "zero_net_energy",
)
_WILCOXON_ALGOS = ("RBCSmart", "IQL", "CQL")
_WILCOXON_PAIRS = (
    ("IQL", "RBCSmart"),
    ("CQL", "RBCSmart"),
    ("IQL", "CQL"),
)


def _extract_per_seed_kpi(runs, kpi):
    """Return ``{env_seed: value}`` for one KPI across ``runs``."""
    out = {}
    for r in runs or []:
        if not isinstance(r, dict):
            continue
        seed = r.get("env_seed")
        district = r.get("district") or {}
        if seed is None or kpi not in district:
            continue
        try:
            out[int(seed)] = float(district[kpi])
        except (TypeError, ValueError):
            continue
    return out


def _paired_wilcoxon(a, b):
    """Return ``(pvalue, note)``; ``pvalue=None`` when the test cannot apply.

    * lengths differ -> ``(None, ...)``
    * n < 2          -> ``(None, ...)``
    * a == b         -> ``(1.0, "identical series ...")`` shortcut
    * scipy missing or raises -> ``(None, "scipy ...")``
    """
    if len(a) != len(b):
        return None, f"length mismatch a={len(a)} b={len(b)}"
    if len(a) < 2:
        return None, f"n={len(a)} < 2"
    if list(a) == list(b):
        return 1.0, "identical series (all pairwise differences = 0)"
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:  # pragma: no cover - scipy assumed available
        return None, f"scipy not available: {exc}"
    try:
        _stat, p = wilcoxon(a, b)
        return float(p), ""
    except ValueError as exc:
        return None, f"scipy error: {exc}"


def _compute_wilcoxon_report(data, kpis=_WILCOXON_KPIS):
    """Structured per-KPI p-value + delta report over algos.

    ``data`` is the parsed benchmark ``results.json`` dict; the schema
    matches what :mod:`scripts.benchmark_entity_agents` writes.

    Returned dict structure::

        {
            "eval_seeds": [...],
            "kpis": {
                kpi: {
                    "algos": {algo: {"mean", "std", "n"}, ...},
                    "deltas_pct_vs_rbc": {"IQL": pct, "CQL": pct},
                    "pvalues": {
                        "IQL_vs_RBCSmart": {"p", "note", "n"},
                        "CQL_vs_RBCSmart": {...},
                        "IQL_vs_CQL": {...},
                    },
                }
            }
        }
    """
    eval_seeds = list(data.get("eval_seeds", []))
    out = {"eval_seeds": eval_seeds, "kpis": {}}
    for kpi in kpis:
        entry = {"algos": {}, "deltas_pct_vs_rbc": {}, "pvalues": {}}
        per_seed = {}
        for algo in _WILCOXON_ALGOS:
            runs = data.get(algo, {}).get("runs", [])
            per_seed[algo] = _extract_per_seed_kpi(runs, kpi)
            agg = data.get(algo, {}).get("aggregate", {}).get(kpi)
            if agg:
                entry["algos"][algo] = {
                    "mean": float(agg["mean"]),
                    "std": float(agg.get("std", 0.0) or 0.0),
                    "n": int(agg.get("n", len(per_seed[algo]))),
                }
            elif per_seed[algo]:
                vals = list(per_seed[algo].values())
                n = len(vals)
                mean = sum(vals) / n
                var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
                entry["algos"][algo] = {
                    "mean": mean,
                    "std": var ** 0.5,
                    "n": n,
                }
        rbc_mean = entry["algos"].get("RBCSmart", {}).get("mean")
        if rbc_mean not in (None, 0):
            for algo in ("IQL", "CQL"):
                m = entry["algos"].get(algo, {}).get("mean")
                if m is not None:
                    entry["deltas_pct_vs_rbc"][algo] = 100.0 * (m - rbc_mean) / rbc_mean
        for a, b in _WILCOXON_PAIRS:
            key = f"{a}_vs_{b}"
            sa, sb = per_seed.get(a, {}), per_seed.get(b, {})
            common = sorted(set(sa) & set(sb))
            if len(common) < 2:
                entry["pvalues"][key] = {
                    "p": None,
                    "note": f"n={len(common)} paired seeds",
                    "n": len(common),
                }
                continue
            va = [sa[s] for s in common]
            vb = [sb[s] for s in common]
            p, note = _paired_wilcoxon(va, vb)
            entry["pvalues"][key] = {"p": p, "note": note, "n": len(common)}
        out["kpis"][kpi] = entry
    return out


def _persist_wilcoxon_report(report, output_dir):
    """Atomically write ``report`` to ``<output_dir>/wilcoxon_report.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "wilcoxon_report.json"
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    os.replace(tmp, dst)
    return dst


def _log_wilcoxon_summary(report):
    """Log a compact per-KPI Wilcoxon summary at INFO level."""
    for kpi, entry in report["kpis"].items():
        pv = entry.get("pvalues", {})
        parts = []
        for pair in ("IQL_vs_RBCSmart", "CQL_vs_RBCSmart", "IQL_vs_CQL"):
            info = pv.get(pair, {})
            p = info.get("p")
            parts.append(f"{pair}={p:.3f}" if isinstance(p, float) else f"{pair}=N/A")
        logger.info("[curate] wilcoxon %s | %s", kpi, " | ".join(parts))


def _write_sentinel(*, output_dir: Path, run_dir: Path, produced: List[Path]) -> Path:
    """Atomically write ``.curation.done`` JSON sentinel describing this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / ".curation.done"
    tmp = sentinel.with_suffix(".done.tmp")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "n_figures": len(produced),
        "figures": sorted(p.name for p in produced),
    }
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, sentinel)
    return sentinel


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Pipeline output directory, e.g. runs/offline_iql_cql_initiative_15min",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for curated PNGs.",
    )
    p.add_argument(
        "--showcase-group",
        type=str,
        default=DEFAULT_SHOWCASE_GROUP,
        help="Agent group key used for per-group figures (default obs627_act1).",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Agent group keys to include in aggregate figures.",
    )
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _build_parser().parse_args(argv)
    logger.info("[curate] run_dir=%s", args.run_dir)
    logger.info("[curate] output_dir=%s", args.output_dir)
    logger.info("[curate] showcase_group=%s", args.showcase_group)
    logger.info("[curate] groups=%s", args.groups)

    produced: List[Path] = []

    # fig 01 -- pipeline diagram (no run-dir dependency)
    pipeline = _render_pipeline_diagram(output_dir=args.output_dir)
    if pipeline:
        produced.append(pipeline)

    # figs 02-06 -- feature-analysis copies
    produced.extend(_copy_feature_analysis_figures(
        run_dir=args.run_dir,
        showcase_group=args.showcase_group,
        output_dir=args.output_dir,
    ))

    # figs 07-09 -- training curves
    produced.extend(_render_training_curves(
        run_dir=args.run_dir,
        showcase_group=args.showcase_group,
        groups=args.groups,
        output_dir=args.output_dir,
    ))

    # figs 10-11 -- benchmark
    bench_json = args.run_dir / "benchmark" / "results.json"
    bars = _render_benchmark_kpi_bars(results_json=bench_json, output_dir=args.output_dir)
    if bars:
        produced.append(bars)
    scatter = _render_iql_vs_cql_scatter(results_json=bench_json, output_dir=args.output_dir)
    if scatter:
        produced.append(scatter)

    if not produced:
        logger.error("[curate] no figures produced -- aborting without sentinel")
        return 1

    sentinel = _write_sentinel(
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        produced=produced,
    )
    logger.info("[curate] wrote sentinel %s (n_figures=%d)", sentinel, len(produced))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
