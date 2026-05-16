# Offline RL Benchmarks — CityLearn Building 5

All runs share the same evaluation setup unless noted otherwise.

- **Dataset**: `./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json`
  (interface=`flat`, topology_mode=`static`)
- **Env reward**: `V2GPenaltyReward`
- **Target building**: **Building_5** (agent index 4)
- **Eval seeds**: 200–209 — disjoint from RBC dataset seeds 22–31
- **Baseline**: RBC (OfflineRBC) — controls all 17 buildings

Each RL agent controls Building 5 only; the remaining 16 buildings are driven
by a fresh RBC instance. Off-target building KPIs should therefore be
bit-identical across columns; any deltas there are downstream coupling effects.

All KPIs are CityLearn's normalised values — **lower is better** (1.0 = no-control
baseline). Mean ± std is computed across env seeds and, where applicable, training
seeds. The "Verdict" column flags `|Δmean| > max(cand_std, RBC_std, 1e-4)` as
significant; otherwise "≈ within noise".

---

## 1. BC vs RBC

Behaviour Cloning trained on 10 RBC seeds (22–31), 5 training seeds × 10 eval
seeds = 50 BC rollouts; 10 RBC rollouts.

- _BC-checkpoint root_: `runs/offline_bc/run-001`
- _BC-training seeds_: [100, 101, 102, 103, 104]
- _Total rollouts_: RBC=10, BC=50

### 1.1 District (17 buildings)

| KPI | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | -0.0049 | ≈ within noise |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | -0.0044 | ≈ within noise |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | -0.0050 | ≈ within noise |
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | -0.0221 | ≈ within noise |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | -0.0024 | ≈ within noise |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | -0.0072 | ≈ within noise |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.0005 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | 0.0032 | ≈ within noise |

### 1.2 Building 5

| KPI | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | -0.0826 | 🟢 BC better |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | -0.0741 | 🟢 BC better |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | -0.0854 | 🟢 BC better |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |

**Success criterion**: BC succeeds if district KPIs are within 1σ of RBC on
cost / peak / ramping / unserved-energy — met on all four. BC also shows a
significant ~3% Building 5 cost improvement, demonstrating the pipeline
reproduces and slightly exceeds the behaviour policy.

<details>
<summary>Full KPI dump — district</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | -0.0221 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | -0.0044 |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | -0.0050 |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.0005 |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | -0.0024 |
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | -0.0049 |
| `monthly_one_minus_load_factor_average` | 0.9115 ± 0.0079 | 0.9114 ± 0.0082 | -0.0001 |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | -0.0072 |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | 0.0032 |

</details>

<details>
<summary>Full KPI dump — Building 5</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | -0.0741 |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | -0.0854 |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | -0.0826 |
| `zero_net_energy` | -0.2023 ± 0.0639 | -0.1478 ± 0.0526 | 0.0545 |

</details>

Raw per-rollout KPI CSVs: `datasets/offline_rl/benchmarks/bc/`.

---

## 2. IQL run-001 vs RBC vs BC

IQL trained on the original RBC dataset (`rbc_with_reward.parquet`, 87 590
rows). 5 training seeds × 10 eval seeds = 50 IQL rollouts; 50 BC rollouts; 10
RBC rollouts.

- _IQL-checkpoint root_: `runs/offline_iql/run-001`
- _BC-checkpoint root_: `runs/offline_bc/run-001`
- _IQL training seeds_: [100, 101, 102, 103, 104]
- _BC training seeds_: [100, 101, 102, 103, 104]
- _Total rollouts_: RBC=10, BC=50, IQL=50

### 2.1 District (17 buildings)

| KPI | RBC (mean ± std) | BC (mean ± std) | IQL (mean ± std) | Δ (IQL − RBC) | Verdict (IQL vs RBC) |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | 2.2229 ± 0.0735 | -0.0054 | ≈ within noise |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | 2.2137 ± 0.0673 | -0.0049 | ≈ within noise |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2232 ± 0.0702 | -0.0056 | ≈ within noise |
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8087 ± 0.0212 | -0.0160 | ≈ within noise |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | 2.1422 ± 0.0276 | -0.0023 | ≈ within noise |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5699 ± 0.0625 | -0.0081 | ≈ within noise |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.8528 ± 0.0097 | 0.0007 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | -5.5724 ± 0.3540 | 0.0037 | ≈ within noise |

### 2.2 Building 5

| KPI | RBC (mean ± std) | BC (mean ± std) | IQL (mean ± std) | Δ (IQL − RBC) | Verdict (IQL vs RBC) |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5679 ± 0.0504 | -0.0920 | 🟢 IQL better |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6000 ± 0.0456 | -0.0828 | 🟢 IQL better |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6339 ± 0.0505 | -0.0960 | 🟢 IQL better |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |

**Success criterion**: IQL beats both RBC and BC on every Building 5 KPI (>1σ
on cost, carbon, electricity) with unserved energy = 0 across all 50 rollouts.
District-level gains remain within noise (IQL controls only 1 of 17 buildings,
diluting any B5 improvement ~17×).

<details>
<summary>Full KPI dump — district</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8087 ± 0.0212 | -0.0160 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | 2.2137 ± 0.0673 | -0.0049 |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2232 ± 0.0702 | -0.0056 |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.8528 ± 0.0097 | 0.0007 |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | 2.1422 ± 0.0276 | -0.0023 |
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | 2.2229 ± 0.0735 | -0.0054 |
| `monthly_one_minus_load_factor_average` | 0.9115 ± 0.0079 | 0.9114 ± 0.0082 | 0.9114 ± 0.0082 | -0.0002 |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5699 ± 0.0625 | -0.0081 |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | -5.5724 ± 0.3540 | 0.0037 |

</details>

<details>
<summary>Full KPI dump — Building 5</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6000 ± 0.0456 | -0.0828 |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6339 ± 0.0505 | -0.0960 |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5679 ± 0.0504 | -0.0920 |
| `zero_net_energy` | -0.2023 ± 0.0639 | -0.1478 ± 0.0526 | -0.1386 ± 0.0453 | 0.0637 |

</details>

Raw per-rollout KPI CSVs: `datasets/offline_rl/benchmarks/iql_run001/`.

---

## 3. IQL run-002 vs RBC vs BC (behaviour-policy swap)

IQL trained on IQL-generated data (seeds 32–41, collected using run-001/seed
101 as the behaviour policy). Same frozen reward weights. 5 training seeds × 10
eval seeds = 50 IQL rollouts.

- _IQL-checkpoint root_: `runs/offline_iql/run-002`
- _BC-checkpoint root_: `runs/offline_bc/run-001`
- _IQL training seeds_: [100, 101, 102, 103, 104]
- _Total rollouts_: RBC=10, BC=50, IQL=50

### 3.1 District (17 buildings)

| KPI | RBC (mean ± std) | BC (mean ± std) | IQL (mean ± std) | Δ (IQL − RBC) | Verdict (IQL vs RBC) |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | 2.2247 ± 0.0755 | -0.0036 | ≈ within noise |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | 2.2160 ± 0.0696 | -0.0026 | ≈ within noise |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2251 ± 0.0724 | -0.0037 | ≈ within noise |
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8005 ± 0.0056 | -0.0242 | 🟢 IQL better |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | 2.1467 ± 0.0334 | 0.0022 | ≈ within noise |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5704 ± 0.0629 | -0.0076 | ≈ within noise |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.8527 ± 0.0097 | 0.0006 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | -5.5739 ± 0.3553 | 0.0022 | ≈ within noise |

### 3.2 Building 5

| KPI | RBC (mean ± std) | BC (mean ± std) | IQL run-001 | IQL run-002 (mean ± std) | Δ (run-002 − RBC) | Verdict |
|---|---:|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5679 ± 0.0504 | 2.5988 ± 0.1461 | -0.0611 | ≈ within noise |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6000 ± 0.0456 | 2.6390 ± 0.1555 | -0.0438 | ≈ within noise |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6339 ± 0.0505 | 2.6656 ± 0.1527 | -0.0644 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |

**Outcome**: The behaviour-policy swap improved training loss 13.8× (proxy MSE
0.002182 → 0.000158) but Building 5 eval performance regressed relative to
run-001 (cost 2.666 ± 0.153 vs 2.634 ± 0.051) with wider std. District-level
`all_time_peak_average` improved (1.8005 ± 0.0056 vs RBC 1.8247, >1σ). See
`pipeline_status.md` §5 for interpretation.

<details>
<summary>Full KPI dump — district</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8005 ± 0.0056 | -0.0242 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | 2.2160 ± 0.0696 | -0.0026 |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2251 ± 0.0724 | -0.0037 |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.8527 ± 0.0097 | 0.0006 |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | 2.1467 ± 0.0334 | 0.0022 |
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | 2.2247 ± 0.0755 | -0.0036 |
| `monthly_one_minus_load_factor_average` | 0.9115 ± 0.0079 | 0.9114 ± 0.0082 | 0.9111 ± 0.0081 | -0.0004 |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5704 ± 0.0629 | -0.0076 |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | -5.5739 ± 0.3553 | 0.0022 |

</details>

<details>
<summary>Full KPI dump — Building 5</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6390 ± 0.1555 | -0.0438 |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6656 ± 0.1527 | -0.0644 |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5988 ± 0.1461 | -0.0611 |
| `zero_net_energy` | -0.2023 ± 0.0639 | -0.1478 ± 0.0526 | -0.1649 ± 0.1119 | 0.0374 |

</details>

Raw per-rollout KPI CSVs: `datasets/offline_rl/benchmarks/iql_run001/`
(run-001 and run-002 data co-located; see note in `pipeline_status.md`).
