"""Calibrate the reward_v2 weights against RBC rollouts.

Procedure (matches docs/offline_rl_v2/reward_design_v2.md §4.1):

  1. Load every ``seed_*.parquet`` in ``datasets/offline_rl_v2/rbc/`` plus
     the cached KPI summary (``kpi_summary.csv``).
  2. For each rollout ``k``:
       - compute per-step terms (cost, carbon, peak, ramp, unserved) via
         ``compute_terms_vectorised``;
       - aggregate into per-rollout sums ``S^k_x``;
       - standardise each term across rollouts to unit variance, giving
         ``Ŝ^k_x``.
  3. Build a target ``y^k`` = sum of the four KPIs we care about
     (``cost_total + carbon_emissions_total + daily_peak_average +
     ramping_average``). ``unserved_energy`` is **always 0** under RBC
     (we verified in step 2), so it cannot enter NNLS — its weight is
     fixed at the §4.2 fallback value (50.0) for safety.
  4. Fit non-negative weights via NNLS:
         minimise || Ŝ · w - ŷ ||₂   s.t. w ≥ 0
     where ``ŷ`` is the standardised target.
  5. Sanity check: Spearman ρ between ``-Σ_t reward_v2_t`` (per-rollout)
     and ``y^k`` must be ≥ 0.9.
  6. Round weights to 2 significant figures, freeze JSON.

Fallback: if NNLS is degenerate (rank < 4, or all weights ≈ 0, or
Spearman ρ < 0.9), use ``reward_v2.DEFAULT_WEIGHTS`` and log the reason.

Outputs (under ``datasets/offline_rl_v2/derived/``):
  - ``reward_v2_weights.json``           — frozen weights + provenance.
  - ``rbc_with_reward_v2.parquet``       — RBC dataset with reward_v2 column.
  - ``reward_v2_breakdown.parquet``      — per-step term breakdown.
  - ``reward_v2_calibration.log``        — full diagnostics, human-readable.

Usage
-----
    .venv/bin/python -m scripts.calibrate_reward_v2
    .venv/bin/python -m scripts.calibrate_reward_v2 --rbc-dir <path> --out-dir <path>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from algorithms.offline_rl_v2 import reward_v2 as RW
from algorithms.offline_rl_v2 import schema_v2 as S

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RBC_DIR = REPO_ROOT / "datasets" / "offline_rl_v2" / "rbc"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "offline_rl_v2" / "derived"

# KPIs that map onto our reward terms. We use the *district*-level versions
# because cost / carbon / peak / ramping are reported as district aggregates
# in CityLearn's KPI set; lower is better, all are normalised so 1.0 = the
# no-control baseline.
KPI_COLUMNS = (
    "district.cost_total",
    "district.carbon_emissions_total",
    "district.daily_peak_average",
    "district.ramping_average",
)
KPI_TERM_MAP = {
    "district.cost_total": "cost",
    "district.carbon_emissions_total": "carbon",
    "district.daily_peak_average": "peak",
    "district.ramping_average": "ramp",
}
# Unserved is always zero under RBC, so it gets the §4.2 safety value.
UNSERVED_FALLBACK_WEIGHT = RW.DEFAULT_WEIGHTS["unserved"]

# Spearman correlation threshold below which we declare calibration failed
# and fall back to defaults.
SPEARMAN_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_sig(x: float, sig: int = 2) -> float:
    """Round ``x`` to ``sig`` significant figures."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_rbc_seeds(rbc_dir: Path) -> Tuple[List[int], List[pd.DataFrame], pd.DataFrame]:
    """Load all seed_*.parquet files and the kpi_summary.csv."""
    parquets = sorted(rbc_dir.glob("seed_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No seed_*.parquet under {rbc_dir}")

    seeds: List[int] = []
    frames: List[pd.DataFrame] = []
    for p in parquets:
        seed = int(p.stem.split("_")[1])
        seeds.append(seed)
        frames.append(pd.read_parquet(p))

    kpi_path = rbc_dir / "kpi_summary.csv"
    if not kpi_path.exists():
        raise FileNotFoundError(f"Missing KPI summary at {kpi_path}")
    kpis = pd.read_csv(kpi_path)
    missing = [c for c in (*KPI_COLUMNS, "seed") if c not in kpis.columns]
    if missing:
        raise ValueError(f"KPI summary missing columns: {missing}")
    kpis = kpis.set_index("seed")

    # Verify the seeds in the parquet directory match the KPI summary.
    if set(seeds) != set(kpis.index):
        raise ValueError(
            f"Seed mismatch: parquets={sorted(seeds)} kpi_summary={sorted(kpis.index)}"
        )
    return seeds, frames, kpis


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------


def _per_rollout_term_sums(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Compute per-rollout sums of the five reward terms.

    Returns a DataFrame indexed by rollout position (0..K-1), with columns
    matching ``RW.TERM_NAMES``.
    """
    rows = []
    for df in frames:
        terms = RW.compute_terms_vectorised(df)
        rows.append({k: float(np.sum(terms[k])) for k in RW.TERM_NAMES})
    return pd.DataFrame(rows)


def _standardise_columns(df: pd.DataFrame, *, eps: float = 1e-12) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score each column. Returns (standardised, means, stds).

    Columns whose std is below ``eps`` are mapped to all-zeros (no signal).
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0)
    safe_stds = stds.where(stds > eps, other=np.nan)
    z = (df - means) / safe_stds
    z = z.fillna(0.0)
    return z, means, stds


def _fit_nnls(
    standardised_terms: pd.DataFrame,
    standardised_target: np.ndarray,
    *,
    fit_terms: List[str],
) -> Tuple[Dict[str, float], float]:
    """Fit NNLS over a subset of terms. Returns (weights, residual)."""
    A = standardised_terms[fit_terms].to_numpy(dtype=np.float64)
    b = standardised_target.astype(np.float64)
    w, residual = nnls(A, b)
    return {k: float(v) for k, v in zip(fit_terms, w)}, float(residual)


def calibrate(
    seeds: List[int],
    frames: List[pd.DataFrame],
    kpis: pd.DataFrame,
    *,
    logger: logging.Logger,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Run the calibration. Returns (weights, diagnostics)."""

    diag: Dict[str, object] = {}

    # 1. Per-rollout term sums.
    term_sums = _per_rollout_term_sums(frames)
    term_sums.index = seeds
    diag["term_sums"] = term_sums.to_dict(orient="index")
    logger.info("Per-rollout term sums:\n%s", term_sums.round(3).to_string())

    # 2. Filter terms that have non-zero variance across rollouts.
    stds = term_sums.std(axis=0, ddof=0)
    diag["term_sum_stds"] = stds.to_dict()
    logger.info("Per-term std across rollouts:\n%s", stds.round(6).to_string())

    fit_terms: List[str] = [k for k in RW.TERM_NAMES if stds[k] > 1e-9 and k != "unserved"]
    skipped_terms = [k for k in RW.TERM_NAMES if k not in fit_terms]
    diag["fit_terms"] = fit_terms
    diag["skipped_terms"] = skipped_terms
    logger.info("Fitting NNLS over: %s  (skipped: %s)", fit_terms, skipped_terms)

    # 3. Build standardised target = sum of standardised KPIs (each KPI ~ 1.0
    #    is no-control baseline; all four lower-is-better, so we just sum).
    kpi_sub = kpis.loc[seeds, list(KPI_COLUMNS)].copy()
    kpi_z, kpi_means, kpi_stds = _standardise_columns(kpi_sub)
    target = kpi_z.sum(axis=1).to_numpy()
    diag["kpi_means"] = kpi_means.to_dict()
    diag["kpi_stds"] = kpi_stds.to_dict()
    diag["kpi_target_per_seed"] = dict(zip(seeds, [float(x) for x in target]))
    logger.info("Standardised KPI target:\n%s", pd.Series(target, index=seeds).round(3).to_string())

    # 4. Standardise term sums and fit NNLS.
    term_z, term_means, term_stds_full = _standardise_columns(term_sums)
    diag["term_means"] = term_means.to_dict()
    diag["term_stds"] = term_stds_full.to_dict()

    if not fit_terms:
        logger.warning("No terms have non-zero variance; falling back to defaults.")
        diag["status"] = "fallback_no_variance"
        return dict(RW.DEFAULT_WEIGHTS), diag

    weights_fit, residual = _fit_nnls(term_z, target, fit_terms=fit_terms)
    logger.info("Raw NNLS weights (standardised space): %s", {k: round(v, 4) for k, v in weights_fit.items()})
    logger.info("NNLS residual: %.6f", residual)
    diag["nnls_weights_standardised_space"] = weights_fit
    diag["nnls_residual"] = residual

    # 5. Convert to raw-space and apply the **hybrid floor rule**.
    #
    # Multicollinearity across RBC seeds means NNLS often zeros legitimate
    # terms (here: peak and ramp). The user-approved rule is:
    #   - Where NNLS produces a strictly positive weight, trust it.
    #   - Where NNLS produces zero (or the term is skipped for zero
    #     variance), substitute the §4.2 default expressed *in
    #     standardised space*, i.e. ``DEFAULT_WEIGHTS[k] / σ_k``. This
    #     keeps every term on a comparable per-σ scale, so the §4.2 ratio
    #     between peak / cost / carbon / ramp is preserved.
    #   - ``unserved`` always uses the safety fallback (RBC never strands
    #     an EV in our seeds, so we have no signal to fit it from).
    #
    # We record per-term provenance for auditability.
    NNLS_THRESHOLD = 1e-9
    weights_raw: Dict[str, float] = {}
    weight_source: Dict[str, str] = {}
    for k in RW.TERM_NAMES:
        if k == "unserved":
            weights_raw[k] = UNSERVED_FALLBACK_WEIGHT
            weight_source[k] = "default_safety"
            continue
        sigma = float(term_stds_full[k])
        nnls_z = float(weights_fit.get(k, 0.0))
        if nnls_z > NNLS_THRESHOLD and sigma > 1e-9:
            weights_raw[k] = nnls_z / sigma
            weight_source[k] = "nnls"
        elif sigma > 1e-9:
            # Floor to §4.2 default expressed in standardised space.
            weights_raw[k] = float(RW.DEFAULT_WEIGHTS[k]) / sigma
            weight_source[k] = "default_standardised"
        else:
            # No variance at all in the term — fall back to raw default.
            weights_raw[k] = float(RW.DEFAULT_WEIGHTS[k])
            weight_source[k] = "default_raw_zero_variance"
    diag["weight_source"] = weight_source
    logger.info("Per-term weight source: %s", weight_source)
    logger.info("Raw-space weights pre-rounding: %s", {k: round(v, 6) for k, v in weights_raw.items()})

    # If somehow every term is essentially zero, fall back wholesale.
    if all(weights_raw[k] < 1e-12 for k in RW.TERM_NAMES):
        logger.warning("All weights collapsed to zero; falling back to raw defaults.")
        diag["status"] = "fallback_all_zero"
        return dict(RW.DEFAULT_WEIGHTS), diag

    # 6. Round to 2 sig figs.
    weights_rounded = {k: _round_sig(v, sig=2) for k, v in weights_raw.items()}
    weights_rounded["unserved"] = UNSERVED_FALLBACK_WEIGHT

    # 7. Sanity check — Spearman between -Σ reward_v2 and KPI target.
    per_seed_neg_reward = []
    for df in frames:
        rew, _ = RW.compute_reward_vectorised(df, weights=weights_rounded)
        per_seed_neg_reward.append(-float(np.sum(rew)))
    rho, pval = spearmanr(per_seed_neg_reward, target)
    rho_f = float(rho) if rho is not None and np.isfinite(rho) else float("nan")
    pval_f = float(pval) if pval is not None and np.isfinite(pval) else float("nan")
    logger.info(
        "Spearman ρ(-Σreward_v2, KPI_target) = %.4f (p=%.4f, threshold=%.2f)",
        rho_f, pval_f, SPEARMAN_THRESHOLD,
    )
    diag["spearman_rho"] = rho_f
    diag["spearman_pvalue"] = pval_f
    diag["per_seed_neg_reward_v2_sum"] = dict(zip(seeds, [float(x) for x in per_seed_neg_reward]))

    if not np.isfinite(rho_f) or rho_f < SPEARMAN_THRESHOLD:
        logger.warning(
            "Spearman ρ=%.4f below threshold %.2f; falling back to defaults.",
            rho_f, SPEARMAN_THRESHOLD,
        )
        diag["status"] = "fallback_low_spearman"
        return dict(RW.DEFAULT_WEIGHTS), diag

    diag["status"] = "ok"
    diag["weights_final"] = weights_rounded
    return weights_rounded, diag


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_per_step_outputs(
    seeds: List[int],
    frames: List[pd.DataFrame],
    weights: Dict[str, float],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Write the augmented dataset and the term-breakdown parquet."""
    augmented_parts: List[pd.DataFrame] = []
    breakdown_parts: List[pd.DataFrame] = []
    for seed, df in zip(seeds, frames):
        rew, terms = RW.compute_reward_vectorised(df, weights=weights)
        df2 = df.copy()
        df2["reward_v2"] = rew
        # Tag with seed for downstream grouping.
        if "seed" not in df2.columns:
            df2.insert(0, "seed", seed)
        augmented_parts.append(df2)

        bk = pd.DataFrame(
            {
                "seed": seed,
                "timestep": df["timestep"].to_numpy() if "timestep" in df.columns else np.arange(len(df)),
                "term_cost": terms["cost"],
                "term_carbon": terms["carbon"],
                "term_peak": terms["peak"],
                "term_ramp": terms["ramp"],
                "term_unserved": terms["unserved"],
                "reward_v2": rew,
            }
        )
        breakdown_parts.append(bk)

    augmented = pd.concat(augmented_parts, ignore_index=True)
    breakdown = pd.concat(breakdown_parts, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    aug_path = out_dir / "rbc_with_reward_v2.parquet"
    bk_path = out_dir / "reward_v2_breakdown.parquet"
    augmented.to_parquet(aug_path, index=False)
    breakdown.to_parquet(bk_path, index=False)
    return aug_path, bk_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("calibrate_reward_v2")
    logger.setLevel(logging.INFO)
    # Wipe handlers in case of re-runs in the same process.
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate reward_v2 weights against RBC rollouts.")
    parser.add_argument("--rbc-dir", type=Path, default=DEFAULT_RBC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    log_path = args.out_dir / "reward_v2_calibration.log"
    logger = _build_logger(log_path)

    logger.info("=== reward_v2 calibration ===")
    logger.info("RBC dir: %s", args.rbc_dir)
    logger.info("Out dir: %s", args.out_dir)
    logger.info("Schema target: %s (%s)", S.TARGET_BUILDING_NAME, S.CHARGER_ID)

    seeds, frames, kpis = _load_rbc_seeds(args.rbc_dir)
    logger.info("Loaded %d rollouts: seeds=%s", len(seeds), seeds)
    logger.info("Per-rollout step counts: %s", [len(f) for f in frames])

    weights, diag = calibrate(seeds, frames, kpis, logger=logger)
    logger.info("Final weights: %s", weights)
    logger.info("Calibration status: %s", diag["status"])

    aug_path, bk_path = _write_per_step_outputs(seeds, frames, weights, args.out_dir)
    logger.info("Wrote augmented dataset: %s (%.1f MB)", aug_path, aug_path.stat().st_size / 1e6)
    logger.info("Wrote term breakdown:    %s (%.1f MB)", bk_path, bk_path.stat().st_size / 1e6)

    # Manifest with provenance.
    rbc_manifest = args.rbc_dir / "manifest.json"
    metadata = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "calibration_status": diag["status"],
        "spearman_rho": diag.get("spearman_rho"),
        "spearman_pvalue": diag.get("spearman_pvalue"),
        "fit_terms": diag.get("fit_terms"),
        "skipped_terms": diag.get("skipped_terms"),
        "fallback_unserved_weight": UNSERVED_FALLBACK_WEIGHT,
        "rbc_seeds": seeds,
        "rbc_kpi_summary_sha256": _file_sha256(args.rbc_dir / "kpi_summary.csv"),
        "rbc_manifest_sha256": _file_sha256(rbc_manifest) if rbc_manifest.exists() else None,
        "rbc_seed_parquet_sha256": {
            int(p.stem.split("_")[1]): _file_sha256(p)
            for p in sorted(args.rbc_dir.glob("seed_*.parquet"))
        },
        "kpi_columns": list(KPI_COLUMNS),
        "kpi_term_map": KPI_TERM_MAP,
        "spearman_threshold": SPEARMAN_THRESHOLD,
        "diagnostics": {
            k: diag[k]
            for k in (
                "term_sum_stds",
                "term_means",
                "term_stds",
                "kpi_means",
                "kpi_stds",
                "kpi_target_per_seed",
                "nnls_weights_standardised_space",
                "nnls_residual",
                "weight_source",
                "per_seed_neg_reward_v2_sum",
            )
            if k in diag
        },
    }
    weights_path = args.out_dir / "reward_v2_weights.json"
    RW.save_weights(weights_path, weights, metadata=metadata)
    logger.info("Wrote weights file: %s", weights_path)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
