# Offline RL — Building 5

Single-building offline reinforcement learning pipeline. Builds an
RBC dataset, calibrates a KPI-aligned reward, then trains controllers
on that dataset and benchmarks them against the behaviour policy.

This page is the entry point. For each stage the linked doc has the
detail; this README only sets the scope and points at the right file.

---

## Scope

- **Building:** Building 5 only (the EV-equipped one).
- **Behaviour policy:** the in-house RBC, wrapped to fix an
  observation-key bug that silently disabled EV charging in the
  upstream implementation. The wrapper lives at
  `algorithms/offline_rl/rbc.py::OfflineRBC`.
- **Dataset:** ten RBC rollouts, parquet, ~88 k transitions.
- **Reward:** five-term weighted sum (cost, carbon, peak, ramp,
  unserved), calibrated against district KPIs and frozen.
- **Eval protocol:** rollouts on env seeds 200..209, disjoint from
  the dataset's collection seeds 22..31. Five training seeds × ten
  eval seeds = 50 rollouts per learned agent; ten rollouts for the
  RBC baseline.

---

## Map of the package

### Code

```
algorithms/offline_rl/
├── schema.py         # column names, dtypes, validation, hash
├── rbc.py            # OfflineRBC — behaviour-policy wrapper
├── reward.py         # 5-term weighted reward + I/O helpers
├── bc_policy.py      # MLP with tanh head
├── bc_dataset.py     # parquet loader, ObservationStandardiser, splits
├── bc_trainer.py     # BCTrainingConfig, train_single_seed/_multi_seed
└── bc_agent.py       # BCAgent — inference adapter (B5 + 16× RBC)
```

### Scripts

```
scripts/collect_rbc_dataset.py   # generate the RBC parquet dataset
scripts/calibrate_reward.py      # NNLS + hybrid floor → frozen weights
scripts/train_bc.py              # multi-seed BC training
scripts/benchmark_bc.py          # BC vs RBC on disjoint eval seeds
scripts/_benchmark_common.py     # shared env builder + KPI extractor
```

### Tests

```
tests/offline_rl/
├── test_reward.py    # 12 tests — terms, monotonicity, vectorised vs loop, weights I/O
└── test_bc.py        # 8 tests — policy/dataset/trainer + best-epoch persistence + agent
```

### Data artefacts

```
datasets/offline_rl/
├── rbc/
│   ├── seed_22..31.parquet          # ten RBC rollouts (87 590 rows total)
│   ├── manifest.json                # provenance + column hashes
│   ├── kpi_summary.csv              # per-seed district KPIs
│   └── sample_first_1000.csv        # eyeball-friendly first rows
└── derived/
    ├── reward_weights.json          # frozen weights + provenance
    ├── rbc_with_reward.parquet      # RBC dataset + populated reward column
    ├── reward_breakdown.parquet     # per-step term-level breakdown
    └── reward_calibration.log       # NNLS run log
```

### Docs

```
docs/offline_rl/
├── README.md                # this file
├── pipeline_status.md       # what's built, current numbers, success criteria met
├── rbc_card.md              # what the RBC does, where it falls short
├── dataset_schema.md        # exact column layout + provenance contract
├── reward_design.md         # term definitions, calibration procedure, weights
├── kpi_reference.md         # CityLearn KPIs and how reward terms target them
├── bc_vs_rbc_benchmark.md   # final BC benchmark report
├── bc_vs_rbc_raw/           # per-rollout KPI CSVs from that report
├── specs/
│   └── iql_design.md        # frozen IQL design (next stage)
└── plans/                   # implementation plans (per stage)
```

---

## Where things stand

The RBC dataset, the reward, and BC are all in. BC matches RBC at
district level on every headline KPI and beats it by ~3% on Building 5
cost / carbon / consumption with zero unserved energy on both sides —
the dataset and reward are trainable, and BC is a credible baseline
for value-based methods.

Current numbers and the full trail are in `pipeline_status.md`.

The next stage is IQL on the same dataset and reward; the design is
frozen in `specs/iql_design.md` and the implementation plan goes
under `plans/`.
