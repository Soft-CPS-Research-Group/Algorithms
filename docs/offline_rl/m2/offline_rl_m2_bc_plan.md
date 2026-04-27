# Milestone 3 (BC) — Implementation Plan

> Status: **finalized, in execution.**
> Predecessors: `m1/offline_m1_conclusions.md`, `m2/offline_rl_m2_plan.md`.

---

## 1. Goal

Train an Offline RL **Behaviour Cloning** policy from the M2 dataset for
**Building 5** of the dynamic-topology demo dataset, addressing the
methodological gaps identified in `docs/offline_rl/m1/offline_m1_conclusions.md`,
and benchmark it head-to-head against the **RBC behaviour policy** and a
**random policy** using CityLearn's official KPIs reported as **mean ± std
across multiple seeds and rollouts**.

---

## 2. Scope (locked in)

| Decision | Choice |
|---|---|
| Number of BC models | **1** — Building 5 only (= `agent_4` CSV from M2) |
| Train/val split | Hold out **2 random clean episodes** for validation; train on remaining 5 clean + 3 noisy = 8 episodes |
| Action target | `action_clean_*` (RBC's intended action) — noisy rows contribute novel **states**, not noisy targets |
| Multi-seed | **3 training seeds × 5 evaluation rollouts each** = 15 BC rollouts; 5 RBC and 5 Random rollouts |
| Comparators | RBC behaviour policy + Trained BC + Random policy |
| Eval reward function | `V2GPenaltyReward` (matches data collection) |
| Regularization | `weight_decay=1e-5` + `dropout(p=0.1)` between hidden layers (default-on) + `gradient_clip_norm=1.0` |
| Eval dataset | `citylearn_three_phase_dynamic_topology_demo_v1` (same as M2 collection) |
| Obs/action alignment | Reorder env observations by **name** to match training schema; fail-fast on missing features |
| Random policy | Uniform over each agent's `action_space.low/high`; per-rollout RNG seeded |
| Eval-time topology change | Fail-fast in `OfflineBCAgent` with explicit re-train remediation message |
| Bundle export | Best-seed model + `multi_seed_summary.json` + `seeds_index.json` |
| MLflow | Enabled for training (1 parent + 3 nested seed runs under `offline_bc_m3`); benchmark stays JSON-only |
| `--smoke` mode | 1 seed × 1 rollout per controller for fast report iteration |

---

## 3. Dataset characterization

- File: `datasets/offline_rl/m2/offline_dataset_agent_4.csv`
- 10 episodes × 8,759 timesteps = **87,590 rows**
- Schema: `episode, timestep, topology_version, policy_mode, noise_sigma_applied, obs_*(35), action_*(2), action_clean_*(2), reward, next_obs_*(35), terminated, truncated`
- `policy_mode`: episodes 0–6 = `clean`; episodes 7–9 = `noisy` (σ=0.1)
- Building 5 actions: `electrical_storage`, `electric_vehicle_storage_charger_5_1`
  (note: 2D in M2, vs 3D in M1's legacy dataset — **not directly comparable to M1 numbers**)

---

## 4. Components to build / modify

### 4.1 Files

| File | Change |
|---|---|
| `algorithms/offline/data_loader.py` | Add `action_target_column` (`"action"` vs `"action_clean"`); add `val_episodes_mode = "random:N"` (clean-only pool); expose `policy_mode` + `topology_version` summary in returned bundle for diagnostics; fix `obs_next` regex bug (M1 polish #9) |
| `algorithms/offline/bc_policy.py` | Add `dropout: float` ctor param; insert `nn.Dropout(p)` between hidden layers when `p>0`; record in `architecture_summary()` |
| `algorithms/offline/bc_trainer.py` | Add `weight_decay`, `dropout`, `gradient_clip_norm` to `BCTrainingConfig`; thread to `Adam(weight_decay=…)`, `BCPolicy(dropout=…)`, `torch.nn.utils.clip_grad_norm_`; new `train_bc_multi_seed(seeds, …)` driver producing `seed_<n>/` subdirs and aggregated `multi_seed_summary.json` + `seeds_index.json`; MLflow parent + nested runs under `offline_bc_m3` |
| `algorithms/agents/offline_bc_agent.py` | Replace positional obs feed with **name-aligned reordering**; fail-fast on missing feature names or action-dim mismatch with explicit remediation |
| `scripts/train_bc_m3.py` *(new)* | CLI driver wrapping `train_bc_multi_seed` |
| `scripts/benchmark_bc_m3.py` *(new)* | Multi-controller (RBC, BC, Random) × multi-seed × multi-rollout benchmark on dynamic-topology dataset with `V2GPenaltyReward`; mean ± std reporting; `--smoke` mode |

### 4.2 Random-policy baseline

50-line callable in `benchmark_bc_m3.py`. Closure over `env.action_space[i].low/high`,
draws from `np.random.default_rng(base_seed + rollout_idx).uniform(low, high)` per
step. No new agent class.

### 4.3 No changes to

- `algorithms/agents/rbc_agent.py` — used as-is
- `algorithms/agents/district_data_collection_agent.py` — frozen M2 deliverable
- `utils/wrapper_citylearn.py` — already exposes `topology_version`
- The CSV itself — no re-collection needed

---

## 5. Training procedure

### 5.1 Hyperparameters (defaults)

```yaml
hidden_layers: [256, 256]
dropout: 0.1
learning_rate: 3.0e-4
weight_decay: 1.0e-5
gradient_clip_norm: 1.0
batch_size: 256
epochs: 50
val_episodes_mode: "random:2"   # 2 random clean episodes; noisy stay in train
action_target: "action_clean"
seeds: [22, 23, 24]
device: "auto"
mlflow_enabled: true
mlflow_experiment: "offline_bc_m3"
```

### 5.2 Per-seed flow

1. Set torch + numpy + random seeds.
2. `load_dataset(csv, seed=…, val_episodes_mode="random:2", action_target="action_clean")`:
   - Choose 2 random episodes from `{ep | policy_mode==clean}` deterministically from `seed`.
   - Train set = remaining 5 clean + 3 noisy episodes.
   - Standardize obs using **train-only** mean/std.
   - Targets = `action_clean_*` columns.
3. Train 50 epochs, record best val loss.
4. Persist `seed_<n>/{model.pth, normalization_stats.json, training_metadata.json, loss_history.json, loss_curve.png, val_episodes.json}`.

### 5.3 Across seeds

Write `multi_seed_summary.json`:
```json
{
  "seeds": [22, 23, 24],
  "best_val_loss":  {"mean": ..., "std": ..., "per_seed": [...]},
  "best_epoch":     {"mean": ..., "std": ..., "per_seed": [...]},
  "training_seconds_total": ...,
  "val_episodes_per_seed": {"22": [...], "23": [...], "24": [...]}
}
```
Plus `seeds_index.json`: `{seed -> {val_loss, model_path, stats_path}}`.

---

## 6. Evaluation procedure

### 6.1 Configuration

| Parameter | Value |
|---|---|
| Dataset | `citylearn_three_phase_dynamic_topology_demo_v1` |
| Reward | `V2GPenaltyReward` |
| Episode length | 8,759 steps (full year, single episode per rollout) |
| Rollouts per agent | 5 |
| Total BC rollouts | 3 seeds × 5 rollouts = 15 |
| Total RBC rollouts | 5 |
| Total Random rollouts | 5 |
| Building 5 control | All 3 controllers; other 16 buildings receive **idle (zero) actions** |

### 6.2 Stochasticity sources

- BC: model-init seed (3) + CityLearn env seed per rollout (5).
- Random: 5 distinct sampler seeds, paired with the same 5 env seeds.
- RBC: deterministic policy, but 5 env seeds → 5 distinct EV-arrival schedules → 5 KPI samples.
- Same 5 env seeds reused across all controllers.

### 6.3 KPI aggregation

For every `(controller, kpi)` pair: mean and std across rollouts. Relative delta on means: `(BC_mean - RBC_mean) / |RBC_mean| × 100%`. Mark "significant" only if `|BC_mean - RBC_mean| > max(BC_std, RBC_std)`.

### 6.4 Headline KPIs reported

District + Building 5 levels for: `electricity_consumption_total`, `carbon_emissions_total`, `cost_total`, `all_time_peak_average`, `daily_peak_average`, `ramping_average`, `daily_one_minus_load_factor_average`, `annual_normalized_unserved_energy_total`, `zero_net_energy`. Plus EV/battery: `ev_departure_success_rate`, `bess_throughput_total_kwh`, `bess_equivalent_full_cycles`, `bess_capacity_fade_ratio`. Full KPI dump in collapsed `<details>`.

### 6.5 Report

`docs/offline_rl/m2/bc_vs_rbc_vs_random_benchmark_m3.md`. Cells render `mean ± std`. Includes per-seed BC KPIs collapsed for transparency, and loss curves.

---

## 7. Output layout

```
runs/offline_bc_m3/<run_id>/
├── seed_22/
│   ├── model.pth
│   ├── normalization_stats.json
│   ├── training_metadata.json
│   ├── loss_history.json
│   ├── loss_curve.png
│   └── val_episodes.json
├── seed_23/ ...
├── seed_24/ ...
├── multi_seed_summary.json
├── seeds_index.json
└── benchmark/
    ├── kpis_raw.csv
    ├── kpis_aggregated.csv
    └── rollout_logs/

docs/offline_rl/m2/
├── offline_rl_m2_bc_plan.md
├── bc_vs_rbc_vs_random_benchmark_m3.md
└── bc_vs_rbc_vs_random_raw/
```

---

## 8. Validation checklist

| Check | How |
|---|---|
| Dataset loads, 35 obs / 2 actions | trainer assertion |
| Val episodes ⊂ clean and disjoint from train | trainer assertion |
| Train loss strictly decreases over first 5 epochs | smoke trace |
| Per-seed best epochs differ | confirms multi-seed actually varies |
| BC inference produces 17 action vectors of correct lengths | benchmark assertion before rollout |
| Idle actions for non-target buildings sampled-checked | benchmark sanity print first 100 steps |
| `unserved_energy_total` ≤ 1.0 for all controllers | report renders ⚠️ if violated |
| No NaN in any KPI cell | exit non-zero if any |
| Random KPIs strictly worse than RBC on most headlines | sanity check |
| Reproducibility: same seed → same `best_val_loss` | rerun seed 22 |
| `--smoke` completes in <2 min | manual |
| MLflow shows 1 parent + 3 seed runs | manual UI check |
| Best-seed bundle bit-identical to `seed_<best>/model.pth` | sha256 |

---

## 9. M1 gaps closed by this plan

| M1 issue | Status in M3 |
|---|---|
| #1 Dataset too narrow | ✅ M2 dataset (10 ep + noisy) |
| #2 Reward never used | ⏸️ Out of scope for BC; deferred to M4 (IQL / TD3+BC) |
| #3 Single-building specialist | ⏳ Deferred per user (single-building scope; multi-building-type model is future work) |
| #4 No regularization | ✅ weight_decay, dropout, grad clip default-on |
| #5 done collapses term/trunc | ✅ Already separate in M2 dataset |
| #6 Validation = last episode only | ✅ Random-2-clean hold-out, varies per seed |
| #7 No multi-seed evaluation | ✅ 3 train seeds × 5 eval rollouts |
| #8 No random baseline | ✅ Random policy added |
| #9 Fragile column regex | ✅ Fixed |
| #10 deterministic flag dead | 🟡 Documented; no code change |
| #11 No feature analysis | ⏳ Future thesis-track |
| #12 Reward mismatch with KPIs | ✅ V2GPenaltyReward everywhere |
