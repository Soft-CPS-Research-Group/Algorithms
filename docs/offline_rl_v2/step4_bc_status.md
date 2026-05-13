# v2 Step 4 — Behaviour Cloning (Status)

> Step 4 of the v2 sequencing in `OFFLINE_RL_AGENTS.md`. **Done.**
> Success criterion met. Next: Step 5 (IQL-v2).

---

## Headline result (plain English)

BC-v2 was trained on the 10-seed RBC-v2 dataset and evaluated on env seeds
**disjoint** from the dataset (200..209 vs RBC dataset's 22..31). Across 50
BC rollouts (5 training seeds × 10 eval seeds) compared to 10 RBC rollouts
on the same eval seeds:

* **District-level KPIs:** BC-v2 is statistically indistinguishable from
  RBC-v2 on every headline KPI (all Δ within 1σ). No regression anywhere.
* **Building 5 (the training target):** BC-v2 is **better** than RBC-v2 on
  cost / carbon / consumption by ~3% — a real, significant improvement,
  not a within-noise tie.
* **Safety:** zero unserved energy on both controllers — BC has not
  learned to skip EV charging.

This is the v2 success story we've been chasing since the v1
post-mortem: BC reproduces its behaviour policy faithfully on a clean
dataset and even improves slightly where it controls. v1 BC, by
contrast, learned to output `tanh(0)≈0` because the dataset's storage
column was all zeros from the wrapper bug.

---

## What was done

### 4A — Modules + tests

Self-contained BC-v2 implementation under `algorithms/offline_rl_v2/`,
no imports from the v1 `algorithms/offline/` tree:

* `bc_policy_v2.py` — MLP with `tanh` head and per-dim action range.
* `bc_dataset_v2.py` — parquet loader, `ObservationStandardiser`
  (fit on train slice only), train/val split.
* `bc_trainer_v2.py` — `BCTrainingConfigV2`, `train_single_seed`,
  `train_multi_seed`. Tracks best-epoch val MSE and **persists
  best-epoch weights** (added in 4C — see below).
* `bc_agent_v2.py` — `BCAgentV2`, inference-only `BaseAgent`. Controls
  Building 5 only; the other 16 agents come from a fresh
  `RuleBasedPolicyV2`. `update()` is a no-op.

Tests: `tests/offline_rl_v2/test_bc_v2.py`, **8/8 green** (was 7/7
before adding the best-epoch persistence test in 4C). Full v2 suite:
**20/20 green** (`pytest tests/offline_rl_v2/`).

### 4B — Smoke run (1 seed × 5 epochs, hidden=[64,64])

Smoke verified end-to-end wiring; final val MSE 0.0081, per-dim
[2.8e-6, 0.0163]. Storage dim collapsed to ~zero (RBC always emits 0
there), V2G dim carried the signal — exactly the diagnostic v2 was
designed to expose.

### 4C — Full training (5 seeds × 50 epochs, then 150)

First pass: 5 seeds × 50 epochs, hidden=[256,256]. Best epochs were
48–49 (still descending at the budget cap). Persisted best val MSE
0.003184 ± 0.000264.

Discovered the trainer was saving final-epoch weights, not best-epoch.
Added best-epoch checkpointing (snapshot best state on CPU each time
val MSE improves; save as `policy.pt` at end). Added a unit test that
forces best ≠ final via a high-LR oscillating regime and verifies the
persisted policy reproduces `best_val_mse` to 1e-6.

Re-ran 5 × 150 epochs: best val MSE **0.001547 ± 0.000109** (≈2×
better than the 50-epoch run). Best epochs 123–149. Per-dim breakdown
(V2G dim only — storage is trivially zero):

| Seed | best_epoch | best_val_mse (V2G dim) |
|---|---:|---:|
| 100 | 149 | 0.00335 |
| 101 | 146 | 0.00418 |
| 102 | 148 | 0.00407 |
| 103 | 123 | 0.00290 |
| 104 | 139 | 0.00365 |

Cross-seed RSD ≈ 7%; tight.

### 4D — Benchmark (5 BC × 10 eval = 50 rollouts; 10 RBC rollouts)

`scripts/benchmark_bc_v2.py` — 2-column report (RBC-v2 vs BC-v2),
no Random comparator (v1 already established Random ≫ RBC on this
task; re-running adds noise without information).

**District (mean ± std across runs):**

| KPI | RBC-v2 | BC-v2 | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | -0.0049 | within noise |
| `carbon_emissions_total`       | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | -0.0044 | within noise |
| `cost_total`                   | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | -0.0050 | within noise |
| `all_time_peak_average`        | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | -0.0221 | within noise (borderline) |
| `daily_peak_average`           | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | -0.0024 | within noise |
| `ramping_average`              | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | -0.0072 | within noise |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | identical |

**Building 5 (BC's controlled building):**

| KPI | RBC-v2 | BC-v2 | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `cost_total`           | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | **-0.0854** | 🟢 BC-v2 better |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | **-0.0741** | 🟢 BC-v2 better |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | **-0.0826** | 🟢 BC-v2 better |
| `annual_normalized_unserved_energy_total` | 0 | 0 | 0 | safe |

Notes:
* BC's std is **tighter** than RBC's on B5 (e.g. cost: 0.059 vs 0.081)
  → BC-v2 is more consistent across env seeds than its teacher.
* District `all_time_peak`: Δ = -0.022, max(stds) = 0.023 → just
  inside the noise threshold; reported as "within noise" but it's a
  borderline real reduction.
* `zero_net_energy` rounded to within-noise on a small std → ignore.

---

## Success criterion (from Step 4 design)

> BC-v2 succeeds if its district headline KPIs are within 1σ of RBC-v2
> on cost / peak / ramping / unserved-energy.

| KPI | Within 1σ of RBC? |
|---|:---:|
| cost_total | ✓ |
| all_time_peak_average | ✓ (borderline, BC slightly lower) |
| ramping_average | ✓ |
| annual_normalized_unserved_energy_total | ✓ (both zero) |

**All four met. Step 4 complete.**

---

## Outputs

* Code:
  * `algorithms/offline_rl_v2/bc_policy_v2.py`
  * `algorithms/offline_rl_v2/bc_dataset_v2.py`
  * `algorithms/offline_rl_v2/bc_trainer_v2.py`
  * `algorithms/offline_rl_v2/bc_agent_v2.py`
  * `scripts/train_bc_v2.py`
  * `scripts/benchmark_bc_v2.py`
* Tests: `tests/offline_rl_v2/test_bc_v2.py` — 8/8 green.
* Trained models: `runs/offline_bc_v2/run-001/seed_{100..104}/`.
* Benchmark: `docs/offline_rl_v2/bc_v2_vs_rbc_v2_benchmark.md`,
  raw aggregates + per-rollout KPI CSVs in `docs/offline_rl_v2/bc_v2_vs_rbc_v2_raw/`.

---

## Decisions and surprises

* **Best-epoch persistence added mid-step.** The original trainer saved
  final-epoch weights only; with epochs=150 some seeds' final epoch was
  not the best epoch, which would have silently degraded the persisted
  model. Caught while planning the 150-epoch retrain. Fix is one tensor
  snapshot per improvement + a contract test.
* **No need to add LR scheduler or early stopping.** Loss curves
  flattened cleanly by epoch ~140; the marginal gain from 50 → 150
  epochs (val MSE 0.0032 → 0.0015) was material, but pushing further
  hits diminishing returns.
* **B5 cost improvement is real, not measurement noise.** ~3.1%
  reduction with BC std *smaller* than RBC std — BC-v2 has learned a
  slightly smoother V2G policy than the deterministic RBC.
* **Storage action dim is constant zero.** Per design — RBC-v2 doesn't
  use the stationary battery. BC trivially learns this (per-dim MSE
  ~1e-12). Tracked separately so it doesn't mask V2G learning signal.

---

## Next step

Step 5 (IQL-v2): build an offline IQL trainer on the same dataset
using `reward_v2`. Same eval protocol (seeds 200..209, 10 RBC
rollouts, 5 IQL training seeds × 10 eval rollouts). Success bar is
strict improvement vs RBC on at least one of {cost, peak, ramping}
without violating unserved energy.
