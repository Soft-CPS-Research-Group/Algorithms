# v2 Step 3 — Reward Calibration (Status)

> Step 3 of the v2 sequencing in `OFFLINE_RL_AGENTS.md`. Done.
> Next: Step 4 (BC-v2) → Step 5 (IQL-v2).

---

## What was done

- Implemented the reward function `algorithms/offline_rl_v2/reward_v2.py`
  (5-term weighted sum: cost, carbon, peak, ramp, unserved). Pure-Python +
  numpy, with both a per-step and a vectorised variant.
- Wrote 12 unit tests in `tests/offline_rl_v2/test_reward_v2.py` covering
  finite output, sign of each term, peak-only-above-mean semantics, ramp
  on constant load, unserved gating at EV departure, vectorised vs
  loop equivalence, monotonicity, weights I/O, and v2-vs-v1 peak
  penalty. **12/12 green.**
- Implemented `scripts/calibrate_reward_v2.py`. Reads all 10 RBC seeds
  + cached KPIs, runs NNLS in standardised space, applies the
  hybrid floor rule (see below), checks Spearman ρ, freezes weights.
- Ran calibration → wrote weights, augmented dataset, term breakdown,
  and run log under `datasets/offline_rl_v2/derived/`.

---

## Final frozen weights

```
cost     = 0.050    (NNLS-fit)
carbon   = 0.056    (NNLS-fit)
peak     = 0.025    (default_standardised — §4.2 ratio applied per-σ)
ramp     = 0.0017   (default_standardised — §4.2 ratio applied per-σ)
unserved = 50.0     (default_safety — RBC produced no signal to fit)
```

These weights operate on **raw** term values (cost in $, carbon in kg
CO₂, peak/ramp in kWh, unserved in kWh). Per-rollout contribution
magnitudes (mean across the 10 RBC seeds, in dimensionless reward
units):

| Term | Per-rollout contribution | Comment |
|---|---:|---|
| cost | ≈ 100 | NNLS-anchored |
| carbon | ≈ 100 | NNLS-anchored |
| peak | ≈ 270 | ~3× cost — matches §4.2 `peak:cost = 2:1` ratio (slightly amplified by 2-sig-fig rounding) |
| ramp | ≈ 18 | small but non-zero gradient toward smoother profiles |
| unserved | 0 | RBC never strands an EV — safety net for future iterations |

**Sanity check passed:** Spearman ρ between per-seed `−Σreward_v2_t`
and per-seed standardised KPI sum = **0.927** (p=0.0001), above the
0.90 threshold from `reward_design_v2.md` §4.1.

---

## Procedure (what the calibration script actually does)

1. Load `seed_22..31.parquet` and `kpi_summary.csv` from
   `datasets/offline_rl_v2/rbc/`.
2. For each rollout, compute per-step terms via
   `compute_terms_vectorised`; aggregate to per-rollout sums
   `S^k_x`.
3. Standardise `S^k_x` across rollouts (z-score per term).
4. Build the calibration target `y^k` = sum over the four matching
   district KPIs (`cost_total`, `carbon_emissions_total`,
   `daily_peak_average`, `ramping_average`), each standardised across
   rollouts. Lower KPI = better; we sum the standardised values so
   the target is a single scalar per rollout.
5. Fit non-negative weights via `scipy.optimize.nnls` over the four
   non-zero-variance terms. (`unserved` has no variance — RBC never
   strands an EV — so it cannot enter the regression.)
6. Apply the **hybrid floor rule** (see §"Deviation from the doc"
   below) to convert NNLS weights to raw-space and patch any zeros.
7. Round final weights to 2 significant figures.
8. Recompute `reward_v2` per rollout with the rounded weights, check
   Spearman ρ against the KPI target, fail-fast if below 0.90.
9. Persist outputs.

---

## Deviation from the doc — hybrid floor rule

`docs/offline_rl_v2/reward_design_v2.md` §4.1 specified raw NNLS in
standardised space, with §4.2 as a *failure* fallback. In practice
NNLS produced a result that passed the Spearman threshold but **zeroed
both `peak` and `ramp` weights**, because:

- All 10 rollouts use the same deterministic RBC policy; cross-seed
  variance is small (~3% RSD on cost) and is driven by env
  stochasticity.
- Across seeds, `cost`, `carbon`, `peak` and `ramp` term sums move
  together (when one rollout is "unlucky", all four go up). NNLS
  collapsed the explanatory weight onto `cost` and `carbon` and zeroed
  the rest — textbook collinear-design pathology.

A reward that ignores peak and ramping is exactly what v1's M4 reward
did, and it's the failure mode v2 was created to avoid. So we
deviated from the spec with explicit user approval:

- **Where NNLS produced a strictly positive weight** (here: cost,
  carbon): use the NNLS raw-space weight.
- **Where NNLS produced zero or skipped a term** (here: peak, ramp):
  use the §4.2 default expressed in *standardised space*, i.e.
  `w_raw[k] = DEFAULT_WEIGHTS[k] / σ_k`. This preserves the §4.2
  *ratios* (peak:cost = 2:1, ramp:cost = 1:1) but on a per-σ scale
  rather than raw-units, so the four terms contribute on comparable
  magnitudes regardless of unit.
- **`unserved`:** fixed at the §4.2 safety value `50.0` (RBC has no
  unserved events to fit).

The chosen rule is recorded per-term in
`reward_v2_weights.json :: metadata.diagnostics.weight_source`:

```json
"weight_source": {
  "cost":     "nnls",
  "carbon":   "nnls",
  "peak":     "default_standardised",
  "ramp":     "default_standardised",
  "unserved": "default_safety"
}
```

If a later iteration's behaviour policy produces enough peak/ramp
variance to make NNLS separate them from cost/carbon, those entries
will flip to `"nnls"` automatically — but per the working agreement
the calibration is **not** re-run when the behaviour policy is swapped:
the same frozen yardstick must be used across iterations.

---

## Outputs

Under `datasets/offline_rl_v2/derived/`:

| File | Size | Purpose |
|---|---:|---|
| `reward_v2_weights.json` | 4 KB | Frozen weights + provenance (NNLS diagnostics, weight_source per term, Spearman, KPI hashes, RBC parquet sha256s). |
| `rbc_with_reward_v2.parquet` | 5.3 MB | RBC dataset (10 seeds, 87 590 rows) with a populated `reward_v2` column. Will be the BC-v2 / IQL-v2 input. |
| `reward_v2_breakdown.parquet` | 1.3 MB | Per-step term-level breakdown (`term_cost`, `term_carbon`, `term_peak`, `term_ramp`, `term_unserved`, `reward_v2`) for analysis. |
| `reward_v2_calibration.log` | ~3 KB | Full run log: NNLS weights, residual, Spearman, weight sources. |

The weights JSON is **the** artefact downstream code reads via
`reward_v2.load_weights(path)`. Everything else is for inspection /
debugging.

---

## Test coverage

`tests/offline_rl_v2/test_reward_v2.py` (12 tests, 1.9 s):

| Test | What it asserts |
|---|---|
| `test_finite_on_zero_action` | Reward is finite when action = 0 and obs = zeros. |
| `test_sign_cost_negative` | Positive grid import + positive price → reward < 0. |
| `test_sign_carbon_negative` | Positive grid import + positive carbon → reward < 0. |
| `test_peak_only_above_mean` | `peak` term is 0 below the rolling 24h mean. |
| `test_ramp_zero_on_constant_load` | `ramp` term is 0 when consumption is constant. |
| `test_unserved_only_at_departure` | `unserved` is 0 when EV is connected this step and next. |
| `test_unserved_fires_at_departure` | `unserved` > 0 when EV is connected now and disconnected next. |
| `test_monotonic_in_each_term` | Reward is monotone non-increasing in each individual term. |
| `test_vectorised_matches_loop` | `compute_reward_vectorised` matches per-step loop on `seed_22.parquet` (rtol=1e-12). |
| `test_reward_v2_outperforms_v1_on_peaks` | A constructed peak-heavy trajectory gets a worse v2 reward than a smooth one (v1 reward proxy did not). |
| `test_save_and_load_weights` | Weight roundtrip preserves values and term order. |
| `test_load_weights_rejects_missing_terms` | Missing terms in JSON → ValueError. |

---

## Caveats and known limitations

| Caveat | Why it matters | Plan |
|---|---|---|
| `peak` and `ramp` weights are floored to defaults, not NNLS-fit. | Means the cross-seed signal didn't separate these terms from cost/carbon. The reward still penalises peaks and ramping correctly per the §4.2 ratios, but the magnitudes are *prescribed*, not *learned*. | Acceptable for v2 step 3. If we revisit, the doc's escape hatch is K=30 seeds — but that won't help if the RBC genuinely couples them. The real fix is a more diverse behaviour policy (i.e. step 7's iteration). |
| `unserved` weight is the §4.2 fallback (50.0) with zero empirical fit. | RBC never strands an EV; we have no data to calibrate this term. | Re-examine if a future behaviour policy *does* produce unserved events. The frozen 50.0 keeps the safety pressure on BC/IQL regardless. |
| Cross-seed RSD ~3% is low. | The Spearman threshold passes (0.927) but with only 10 points and 4 fit terms, NNLS is sensitive. | The hybrid floor explicitly bakes in robustness against this. K=30 is available if needed. |
| Calibration is **not** re-run when the behaviour policy is swapped. | Per `reward_design_v2.md`, the yardstick must stay fixed for cross-iteration comparability. | This is by design. The `metadata.rbc_seed_parquet_sha256` map in the weights JSON pins exactly which RBC dataset produced these weights; if we ever do re-fit, it's a deliberate v2.x revision, not a silent change. |

---

## Files added in step 3

- `algorithms/offline_rl_v2/reward_v2.py`
- `tests/offline_rl_v2/__init__.py`
- `tests/offline_rl_v2/test_reward_v2.py`
- `scripts/calibrate_reward_v2.py`
- `datasets/offline_rl_v2/derived/reward_v2_weights.json`
- `datasets/offline_rl_v2/derived/rbc_with_reward_v2.parquet`
- `datasets/offline_rl_v2/derived/reward_v2_breakdown.parquet`
- `datasets/offline_rl_v2/derived/reward_v2_calibration.log`

## Files updated in step 3

- `OFFLINE_RL_AGENTS.md` — retrospective gains the hybrid-floor
  deviation; quick facts table updated with frozen weights.

## Files **not** touched in step 3

- All v1 code, datasets, runs, docs.
- `algorithms/offline_rl_v2/schema_v2.py`, `rbc_v2.py`,
  `scripts/collect_rbc_dataset_v2.py`, the RBC dataset Parquet files,
  `manifest.json`, `kpi_summary.csv`.
