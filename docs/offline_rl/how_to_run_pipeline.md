# How to Run the Offline RL Pipeline

End-to-end instructions for reproducing the full offline RL pipeline:
data collection → reward calibration → BC training → IQL training →
benchmark. All commands run from the repo root with the project virtualenv
active (`.venv/bin/python`).

The pipeline is **sequential**: each step produces artefacts the next step
reads. Skip any step only if its output artefacts already exist.

---

## Prerequisites

```bash
# Activate the virtualenv
source .venv/bin/activate

# Verify environment
python --version          # 3.10.x
python -c "import torch; print(torch.__version__)"    # 2.5.x
python -c "import citylearn; print(citylearn.__version__)"  # 0.3.1

# Run the full test suite to confirm baseline health
pytest                    # should be 121/121 green
```

The CityLearn dataset must be present at:
```
datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json
```

---

## Step 1 — Collect RBC dataset

Runs the Rule-Based Controller for Building 5 across ten seeds and persists
each rollout as a parquet file.

```bash
python -m scripts.collect_rbc_dataset \
    --output datasets/offline_rl/rbc \
    --seeds 22,23,24,25,26,27,28,29,30,31
```

**Outputs** under `datasets/offline_rl/rbc/`:

| File | Description |
|---|---|
| `seed_22.parquet` … `seed_31.parquet` | Per-seed transition tables (8 759 rows each) |
| `manifest.json` | Schema hash, seed list, behaviour policy class |
| `kpi_summary.csv` | Per-seed district KPIs |
| `sample_first_1000.csv` | First 1 000 rows for quick inspection |

**Expected**: 87 590 total rows; `action_electrical_storage` constant 0;
`action_electric_vehicle_storage_charger_5_1` non-trivial (std > 0.05);
`annual_normalized_unserved_energy_total = 0` for every seed.

**Full run**: ~8 min on CPU.

---

## Step 2 — Calibrate the reward

Fits a five-term weighted-sum reward against the RBC KPI signals and writes
the frozen weight file that all downstream agents read.

```bash
python -m scripts.calibrate_reward \
    --rbc-dir datasets/offline_rl/rbc \
    --output-dir datasets/offline_rl/derived
```

**Outputs** under `datasets/offline_rl/derived/`:

| File | Description |
|---|---|
| `reward_weights.json` | Frozen weights + calibration provenance |
| `rbc_with_reward.parquet` | Full 87 590-row dataset with populated `reward` column |
| `reward_breakdown.parquet` | Per-step term-level breakdown |
| `reward_calibration.log` | Full run log |

**Expected**: Spearman ρ ≥ 0.90 between `−Σ reward_t` and the KPI sum.
Actual achieved: **ρ = 0.927**.

**Full run**: ~30 s.

---

## Step 3 — Train BC (Behaviour Cloning)

Trains a five-seed MLP policy on the RBC dataset by imitation.

```bash
# Full run (5 seeds × 50 epochs, ~20 min)
python -m scripts.train_bc \
    --output runs/offline_bc/run-001

# Smoke run (1 seed, 5 epochs, tiny net)
python -m scripts.train_bc \
    --output runs/offline_bc/smoke \
    --seeds 100 \
    --epochs 5 \
    --hidden-layers 64,64
```

**Outputs** under `runs/offline_bc/run-001/seed_<N>/` for each seed:

| File | Description |
|---|---|
| `policy.pt` | Best-val-MSE policy weights |
| `obs_standardiser.npz` | Observation mean/std fitted on train split |
| `architecture.json` | Network shape metadata |
| `metrics.jsonl` | One line per epoch: train/val MSE |
| `seed_summary.json` | Final and best-epoch stats |

Plus `multi_seed_summary.json` and `seeds_index.json` at the run root.

**Expected**: best val MSE ≈ 0.0015 ± 0.0001 (run-001 achieved
0.001547 ± 0.000109).

**Full run**: ~20 min on CPU.

---

## Step 4 — Benchmark BC vs RBC

Evaluates the trained BC checkpoints on ten held-out env seeds and writes a
markdown report.

```bash
# Full benchmark (5 BC training seeds × 10 eval seeds = 50 BC rollouts; 10 RBC rollouts)
python -m scripts.benchmark_bc \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/bc_vs_rbc_benchmark.md

# Smoke (1 BC seed × 1 eval seed, fast check)
python -m scripts.benchmark_bc \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/bc_vs_rbc_benchmark.md \
    --smoke
```

**Outputs**:

- `docs/offline_rl/bc_vs_rbc_benchmark.md` — markdown report with 2-column
  KPI tables (RBC | BC, Δ, verdict)
- `docs/offline_rl/bc_vs_rbc_raw/aggregates.json` — raw aggregates JSON
- `docs/offline_rl/bc_vs_rbc_raw/kpis_*.csv` — per-rollout KPI CSVs

**Eval seeds**: 200–209 (disjoint from dataset seeds 22–31).

**Full run**: ~65 min on CPU (60 total rollouts × ~65 s/rollout).

---

## Step 5 — Train IQL (Implicit Q-Learning)

Trains a five-seed IQL policy on the same dataset using the calibrated
reward.

```bash
# Full run (5 seeds × 150 000 gradient steps, ~75 min)
nohup python -m scripts.train_iql \
    --output runs/offline_iql/run-001 \
    --seeds 100,101,102,103,104 \
    --gradient-steps 150000 \
    > runs/offline_iql/run-001/train.log 2>&1 &
echo $!   # note the PID to monitor progress

# Monitor
tail -f runs/offline_iql/run-001/train.log

# Smoke run (1 seed, 200 steps, tiny net — verifies wiring, ~2 s)
python -m scripts.train_iql \
    --output runs/offline_iql/smoke \
    --seeds 100 \
    --gradient-steps 200 \
    --hidden-layers 64,64 \
    --eval-every 50
```

**Outputs** under `runs/offline_iql/run-001/seed_<N>/` for each seed:

| File | Description |
|---|---|
| `policy.pt` | Best-val-MSE Gaussian policy weights |
| `q1.pt`, `q2.pt` | Final twin-Q weights |
| `value.pt` | Final value network weights |
| `obs_standardiser.npz` | Same standardiser format as BC |
| `architecture.json` | Network shape metadata |
| `metrics.jsonl` | One line per eval step: V/Q/policy losses, val MSE, advantage stats |
| `seed_summary.json` | Best checkpoint step, best val MSE, final metrics |

Plus `multi_seed_summary.json` and `seeds_index.json` at the run root.

**Expected**: best val policy MSE ≈ 0.002 ± 0.0003.

**Key metrics to eyeball** in `metrics.jsonl`:
- `val_policy_mse` should decrease over training.
- `adv_clip_frac` should stay below ~0.1 (if > 0.3, lower `--beta-advantage`).
- `q_loss` may spike briefly but should recover; if it diverges permanently,
  halve `--learning-rate`.

**Full run**: ~75 min on CPU.

---

## Step 6 — Benchmark IQL vs RBC vs BC

Evaluates IQL, BC and RBC on the same ten held-out env seeds.

```bash
# Full benchmark
python -m scripts.benchmark_iql \
    --iql-root runs/offline_iql/run-001 \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/iql_vs_rbc_benchmark.md

# Smoke (1 IQL seed × 1 eval seed)
python -m scripts.benchmark_iql \
    --iql-root runs/offline_iql/run-001 \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/iql_vs_rbc_benchmark.md \
    --smoke
```

**Outputs**:

- `docs/offline_rl/iql_vs_rbc_benchmark.md` — 3-column report (RBC | BC | IQL)
- `docs/offline_rl/iql_vs_rbc_raw/aggregates.json`
- `docs/offline_rl/iql_vs_rbc_raw/kpis_*.csv`

**Success criterion**: IQL beats RBC by > 1σ on at least one of
`{cost_total, all_time_peak_average, ramping_average}` at the district level
with `annual_normalized_unserved_energy_total = 0`.

**Full run**: ~100 min on CPU (70 total rollouts).

---

## All-at-once reference (sequential, no-smoke)

```bash
# Step 1 – collect
python -m scripts.collect_rbc_dataset \
    --output datasets/offline_rl/rbc

# Step 2 – calibrate reward
python -m scripts.calibrate_reward \
    --rbc-dir datasets/offline_rl/rbc \
    --output-dir datasets/offline_rl/derived

# Step 3 – train BC
python -m scripts.train_bc \
    --output runs/offline_bc/run-001

# Step 4 – benchmark BC
python -m scripts.benchmark_bc \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/bc_vs_rbc_benchmark.md

# Step 5 – train IQL
nohup python -m scripts.train_iql \
    --output runs/offline_iql/run-001 \
    --seeds 100,101,102,103,104 \
    --gradient-steps 150000 \
    > runs/offline_iql/run-001/train.log 2>&1 &

# Step 6 – benchmark IQL (after training completes)
python -m scripts.benchmark_iql \
    --iql-root runs/offline_iql/run-001 \
    --bc-root runs/offline_bc/run-001 \
    --output docs/offline_rl/iql_vs_rbc_benchmark.md
```

---

## Artefact map

```
datasets/offline_rl/
├── rbc/
│   ├── seed_22..31.parquet       ← Step 1 output
│   ├── manifest.json
│   ├── kpi_summary.csv
│   └── sample_first_1000.csv
└── derived/
    ├── reward_weights.json       ← Step 2 output (frozen)
    ├── rbc_with_reward.parquet   ← Step 2 output (training input)
    ├── reward_breakdown.parquet
    └── reward_calibration.log

runs/
├── offline_bc/run-001/
│   ├── seed_100..104/            ← Step 3 output
│   │   ├── policy.pt
│   │   ├── obs_standardiser.npz
│   │   ├── architecture.json
│   │   ├── metrics.jsonl
│   │   └── seed_summary.json
│   ├── multi_seed_summary.json
│   └── seeds_index.json
└── offline_iql/run-001/
    ├── seed_100..104/            ← Step 5 output
    │   ├── policy.pt
    │   ├── q1.pt, q2.pt, value.pt
    │   ├── obs_standardiser.npz
    │   ├── architecture.json
    │   ├── metrics.jsonl
    │   └── seed_summary.json
    ├── multi_seed_summary.json
    └── seeds_index.json

docs/offline_rl/
├── bc_vs_rbc_benchmark.md        ← Step 4 output
├── bc_vs_rbc_raw/
├── iql_vs_rbc_benchmark.md       ← Step 6 output
└── iql_vs_rbc_raw/
```

> `runs/` is gitignored (`.pt` blobs are large). Parquet artefacts under
> `datasets/` are committed. Reports under `docs/` are committed.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `collect_rbc_dataset` — both action columns are zero | Observation name mismatch in the RBC | Use `OfflineRBC` (not the raw `RuleBasedPolicy`) — it remaps charger-namespaced obs keys |
| `calibrate_reward` — Spearman ρ < 0.90 | Insufficient reward signal / wrong seeds | Check `reward_calibration.log`; verify parquet seeds match `--seeds` |
| `train_bc` — val MSE not decreasing | LR too high or too low | Default 3e-4 is stable; inspect `metrics.jsonl` per epoch |
| `train_iql` — `q_loss` diverges permanently | LR issue or batch too small | Halve `--learning-rate`; bump `--batch-size` to 512 |
| `train_iql` — `adv_clip_frac` > 0.3 | Advantage temperature too high | Lower `--beta-advantage` from 3.0 to 1.0 |
| `benchmark_*` — agent.predict wrong length | Old checkpoint from mismatched env | Retrain with correct dataset schema |
| `pytest` failures on `test_iql` | Imports of non-existent modules | Ensure all four `iql_*.py` modules are present in `algorithms/offline_rl/` |
