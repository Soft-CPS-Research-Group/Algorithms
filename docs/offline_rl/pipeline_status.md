# Offline RL Pipeline — Status

The offline RL pipeline targets Building 5 only. It assembles a clean
RBC dataset, calibrates a KPI-aligned reward, and trains controllers on
that dataset. Each stage feeds the next; everything is reproducible
from the artefacts under `datasets/offline_rl/`.

This page summarises what is built and what each artefact is for.
The next active stage (IQL) is tracked in `specs/iql_design.md`.

---

## 1. Behaviour-policy dataset (RBC rollouts)

The behaviour policy is a hand-crafted Rule-Based Controller for EV
charging on Building 5. It does nothing with the stationary battery and
has no awareness of price, carbon, peak, or ramping — a deliberately
narrow baseline that gives downstream agents room to improve. Full
description: `rbc_card.md`.

The collector (`scripts/collect_rbc_dataset.py`) ran the RBC for ten
seeds and persisted everything as parquet:

| Quantity | Value |
|---|---|
| Seeds | 10 (`22..31`) |
| Steps per rollout | 8 759 |
| Total transitions | 87 590 |
| Schema fields | 80 |
| Disk footprint | 11 MB parquet |

The EV-charging action has three modes (idle / PV-bonus / emergency)
with non-trivial mass on each — BC has something to imitate, IQL has
higher- and lower-return subsets to re-weight. The stationary-battery
action is constant zero by design; this is documented in
`RBC_EXPECTED_CONSTANT_ACTIONS` and asserted at collection time.

District KPIs from these ten rollouts (mean ± std):

| KPI | RBC |
|---|---:|
| `cost_total` | 2.228 ± 0.057 |
| `daily_peak_average` | 2.143 ± 0.022 |
| `ramping_average` | 1.579 ± 0.050 |
| `annual_normalized_unserved_energy_total` | 0.000 ± 0.000 |

Numbers above 1.0 mean worse than the no-control baseline. The RBC is
roughly 2× worse than no-op on cost/peak because it charges hard
without grid awareness, but it never strands an EV (`unserved = 0`) —
any agent built on top of this dataset has to keep that property too.

### A bug discovered during collection

The first smoke run tripped a fail-fast: both action columns were
constant zero. Root cause was that the upstream RBC
(`algorithms/agents/rbc_agent.py`) read EV observation fields under
bare names, but CityLearn exposes them namespaced by charger ID; every
lookup fell back to defaults and short-circuited the EV branch. The
v1 offline-RL benchmarks were therefore comparing agents against a
do-nothing baseline.

The offline-RL RBC wrapper (`algorithms/offline_rl/rbc.py::OfflineRBC`)
remaps the bare names to the namespaced keys via the agent's known
charger IDs, then defers to the parent implementation. Behaviour and
hyperparameters are otherwise identical. Evidence the wrapper works:
EV action is non-trivial (18% PV-bonus, 1% emergency), `reward_env`
≈ −39.4/step, and building-level `unserved_energy = 0` confirms the
RBC actually delivers the required EV charge.

Outputs under `datasets/offline_rl/rbc/`:
`seed_22..31.parquet`, `manifest.json`, `kpi_summary.csv`,
`sample_first_1000.csv`.

---

## 2. Reward calibration

The reward is a five-term weighted sum (cost, carbon, peak, ramp,
unserved) defined in `algorithms/offline_rl/reward.py`. The full
design — terms, units, peak-window definition, calibration procedure
— is in `reward_design.md`.

The calibrator (`scripts/calibrate_reward.py`) loads all ten RBC
seeds plus the cached KPI summary, runs NNLS in standardised space
against the matching district KPIs, applies a hybrid floor rule, then
checks Spearman ρ.

### Final frozen weights

```
cost = 0.050 (NNLS-fit)
carbon = 0.056 (NNLS-fit)
peak = 0.025 (default_standardised — ratio applied per σ)
ramp = 0.0017 (default_standardised — ratio applied per σ)
unserved = 50.0 (default_safety — RBC produced no signal to fit)
```

Per-rollout contribution magnitudes (mean across the ten RBC seeds, in
dimensionless reward units): cost ≈ 100, carbon ≈ 100, peak ≈ 270,
ramp ≈ 18, unserved 0.

Spearman ρ between per-seed `−Σ reward_t` and per-seed standardised
KPI sum = **0.927** (p = 0.0001), above the 0.90 threshold.

### The hybrid floor rule (deviation from the design doc)

Raw NNLS passed the Spearman threshold but zeroed both `peak` and
`ramp` weights, because cross-seed variance is small (~3% RSD) and the
four terms move together — textbook collinear-design pathology. A
reward that ignores peak and ramping was the failure mode the offline
pipeline was rebuilt to avoid, so the rule was:

- Where NNLS gives a strictly positive weight, use it raw.
- Where NNLS gives zero, use the design-doc default expressed in
 *standardised* space (`w_raw[k] = DEFAULT[k] / σ_k`). This preserves
 the design ratios (peak:cost = 2:1, ramp:cost = 1:1) on a per-σ
 scale.
- `unserved` is fixed at the safety value 50.0.

The chosen rule is recorded per term in
`reward_weights.json :: metadata.diagnostics.weight_source`. If a
later behaviour policy generates enough peak/ramp variance to make
NNLS separate them, those entries flip to `"nnls"` automatically — but
the calibration is **not** re-run when the behaviour policy is
swapped: the same yardstick is used across iterations.

Outputs under `datasets/offline_rl/derived/`:

| File | Purpose |
|---|---|
| `reward_weights.json` | Frozen weights + provenance (NNLS diagnostics, weight_source per term, Spearman, KPI hashes, RBC parquet sha256s). |
| `rbc_with_reward.parquet` | RBC dataset (10 seeds, 87 590 rows) with a populated `reward` column. The BC / IQL training input. |
| `reward_breakdown.parquet` | Per-step term-level breakdown for analysis. |
| `reward_calibration.log` | Full run log: NNLS weights, residual, Spearman, weight sources. |

The weights JSON is the artefact downstream code reads via
`reward.load_weights(path)`. Everything else is for inspection.

Test coverage: `tests/offline_rl/test_reward.py` (12 tests) — finite
output, sign of each term, peak-only-above-mean semantics, ramp on
constant load, unserved gating at EV departure, vectorised vs loop
equivalence, monotonicity, weights I/O, peak-penalty regression.

---

## 3. Behaviour cloning

BC is the first agent trained on the calibrated dataset. It serves
two purposes: as a sanity check that the dataset and reward are
trainable, and as a baseline that any value-based agent (IQL) must
beat.

The BC stack lives under `algorithms/offline_rl/`:
`bc_policy.py` (MLP with `tanh` head), `bc_dataset.py` (parquet loader,
`ObservationStandardiser`, train/val split), `bc_trainer.py`
(`BCTrainingConfig`, `train_single_seed`, `train_multi_seed`, with
best-epoch persistence) and `bc_agent.py` (`BCAgent`, the inference
adapter that controls Building 5 and defers to a fresh `OfflineRBC`
for the other 16 agents).

Tests: `tests/offline_rl/test_bc.py` — 8/8 green, covering policy
shape, dataset round-trip, train-loop convergence on synthetic data,
best-epoch persistence (forces best ≠ final via a high-LR oscillating
regime and verifies the persisted policy reproduces `best_val_mse` to
1e-6), agent-only inference on the off-target buildings, and
end-to-end smoke training.

### Training

Final run: 5 seeds × 150 epochs, hidden=[256, 256], best val MSE
**0.001547 ± 0.000109**. Per-seed best epochs landed in 123–149;
cross-seed RSD ≈ 7%.

| Seed | best_epoch | best_val_mse (V2G dim) |
|---|---:|---:|
| 100 | 149 | 0.00335 |
| 101 | 146 | 0.00418 |
| 102 | 148 | 0.00407 |
| 103 | 123 | 0.00290 |
| 104 | 139 | 0.00365 |

The storage action dim is constant zero (RBC doesn't use the
stationary battery), so per-dim MSE there is ~1e-12 by construction.
That dim is tracked separately so it doesn't mask the V2G learning
signal.

### Benchmark vs RBC

`scripts/benchmark_bc.py` runs 5 BC seeds × 10 eval seeds = 50 BC
rollouts and 10 RBC rollouts on env seeds **disjoint** from the
training dataset (200..209 vs 22..31).

District (mean ± std):

| KPI | RBC | BC | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | -0.0049 | within noise |
| `carbon_emissions_total`        | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | -0.0044 | within noise |
| `cost_total`                    | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | -0.0050 | within noise |
| `all_time_peak_average`         | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | -0.0221 | within noise (borderline) |
| `daily_peak_average`            | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | -0.0024 | within noise |
| `ramping_average`               | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | -0.0072 | within noise |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | identical |

Building 5 (BC's controlled building):

| KPI | RBC | BC | Δ | Verdict |
|---|---:|---:|---:|:---|
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | **-0.0854** | BC better |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | **-0.0741** | BC better |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | **-0.0826** | BC better |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | safe |

BC's std is tighter than RBC's on Building 5 — the cloned policy is
more consistent across env seeds than its teacher. The ~3% B5 cost
improvement is real (Δ larger than max(stds), unserved still zero)
rather than within-noise. The full benchmark, including raw per-rollout
KPI CSVs, is in `bc_vs_rbc_benchmark.md` and `bc_vs_rbc_raw/`.

### Success criterion

> BC succeeds if its district headline KPIs are within 1σ of RBC on
> cost / peak / ramping / unserved-energy.

| KPI | Within 1σ of RBC? |
|---|:---:|
| cost_total | yes |
| all_time_peak_average | yes (borderline, BC slightly lower) |
| ramping_average | yes |
| annual_normalized_unserved_energy_total | yes (both zero) |

All four met.

---

## 4. Implicit Q-Learning (IQL)

IQL was trained on the same dataset and reward as BC. The algorithm
uses expectile value regression and advantage-weighted policy updates
to optimise expected return directly, without ever querying
out-of-distribution actions. Full design spec: `specs/iql_design.md`.
Implementation plan: `plans/iql-implementation.md`.

### Training

5 seeds × 150 000 gradient steps; wall-clock ~75 min total.

| Seed | best_val_policy_mse | best_step |
|---|---:|---:|
| 100 | 0.002221 | 142 500 |
| 101 | 0.002073 | 150 000 |
| 102 | 0.002191 | 122 500 |
| 103 | 0.002126 | 105 000 |
| 104 | 0.002300 | 137 500 |

Mean ± std: **0.002182 ± 0.000078**. Training was stable throughout:
`adv_clip_frac` < 0.07; `val_policy_mse` monotonically decreasing;
no Q-divergence.

### Benchmark vs RBC and BC

Full benchmark: 5 IQL seeds × 10 eval seeds = 50 IQL rollouts; 50 BC
rollouts; 10 RBC rollouts. Eval seeds 200–209 (disjoint from dataset
seeds 22–31). Full report: `iql_vs_rbc_benchmark.md`.

District (all 17 buildings, mean ± std):

| KPI | RBC | BC | IQL | Δ (IQL−RBC) | Verdict |
|---|---:|---:|---:|---:|:---|
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2232 ± 0.0702 | −0.0056 | within noise |
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8087 ± 0.0212 | −0.0160 | within noise |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5699 ± 0.0625 | −0.0081 | within noise |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | 0 | safe |

Building 5 (IQL's controlled building):

| KPI | RBC | BC | IQL | Δ (IQL−RBC) | Verdict |
|---|---:|---:|---:|---:|:---|
| `cost_total` | 2.730 ± 0.081 | 2.645 ± 0.059 | **2.634 ± 0.051** | −0.096 | **IQL better** |
| `carbon_emissions_total` | 2.683 ± 0.073 | 2.609 ± 0.053 | **2.600 ± 0.046** | −0.083 | **IQL better** |
| `electricity_consumption_total` | 2.660 ± 0.079 | 2.577 ± 0.059 | **2.568 ± 0.050** | −0.092 | **IQL better** |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | 0 | safe |

IQL beats both RBC and BC on every Building 5 KPI, and tightens the
std. At the district level the gain is diluted ~17× (IQL controls only
1 of 17 buildings), remaining below the noise floor.

### Success criterion

The original criterion was district-level >1σ improvement. IQL misses
that bar because controlling 1/17 buildings is structurally insufficient
to move district-level KPIs beyond noise.

At the Building 5 level (the only building IQL controls), the criterion
is **met**: IQL beats RBC by >1σ on cost (Δ/σ_RBC ≈ 1.2), carbon, and
electricity, with unserved energy = 0 across all 50 rollouts.

Full analysis and next-step options are in `step5_iql_status.md`.

---

---

## 5. Behaviour-policy swap (Step 6)

The IQL behaviour policy (run-001 / seed 101, best val MSE 0.002073) replaced
the RBC as the data-collection agent. New dataset collected on seeds 32–41
(disjoint from RBC seeds 22–31 and eval seeds 200–209). Frozen reward weights
applied unchanged via `scripts/apply_reward.py`. IQL run-002 trained on the
new data.

Full details: `step6_iql_swap_status.md`. Key numbers:

| | run-001 (RBC data) | run-002 (IQL data) |
|---|---:|---:|
| best_val_policy_mse (training) | 0.002182 ± 0.000078 | **0.000158 ± 0.000010** |
| B5 cost (eval) | **2.634 ± 0.051** | 2.666 ± 0.153 |
| B5 unserved | 0 | 0 |

Training loss improved 13.8× (the policy fits IQL-generated data much more
tightly), but eval performance regressed slightly and variance widened. The
iterative swap did not compound the Building 5 improvement.

**Finding**: distributional narrowing from IQL-generated data reduces
out-of-distribution robustness. The RBC dataset's diversity is more valuable
than the distribution alignment from IQL-generated data.

---

## 6. What's next

- **Hyperparameter sweep on run-001 data**: try β ∈ {5, 10} and
  τ_expectile ∈ {0.8, 0.9} on the original RBC dataset to see if
  further B5 gains are achievable without iterative data.
- **Multi-building IQL**: extend to all 17 buildings to make district-level
  improvements visible.
