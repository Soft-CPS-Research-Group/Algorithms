# IQL vs RBC vs BC — CityLearn Benchmark

_Dataset_: `./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json` (interface=`flat`, topology_mode=`static`)
_Reward function (env)_: `V2GPenaltyReward`
_Target building_: **Building_5** (agent index 4)
_IQL-checkpoint root_: `/Users/guilherme.desousa/MEIA/Thesis/Project/repos/Algorithms/runs/offline_iql/run-002`
_BC-checkpoint root_:  `/Users/guilherme.desousa/MEIA/Thesis/Project/repos/Algorithms/runs/offline_bc_v2/run-001`
_IQL training seeds_: [100, 101, 102, 103, 104]
_BC training seeds_:  [100, 101, 102, 103, 104]
_Env seeds_ (per controller): [200, 201, 202, 203, 204, 205, 206, 207, 208, 209] ← disjoint from RBC dataset seeds 22..31
_Total rollouts_: RBC=10, BC=50, IQL=50

> All KPIs are CityLearn's normalised values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across training
> seeds where applicable). The "Verdict" column flags
> `|Δmean| > max(IQL_std, RBC_std, 1e-4)` as significant; otherwise the
> difference is within noise.
>
> All three controllers operate on the same env in identical conditions.
> The RBC controls every building. BC controls Building 5 with the BC
> policy and defers the other 16 buildings to a fresh RBC. IQL does the
> same with the IQL policy. Off-target buildings should therefore be
> bit-identical across the three columns; any deltas there are pure
> downstream coupling effects.

---

## 1. Headline KPIs — district level

### District

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


---

## 2. Headline KPIs — Building 5 (training target)

### Building_5

| KPI | RBC (mean ± std) | BC (mean ± std) | IQL (mean ± std) | Δ (IQL − RBC) | Verdict (IQL vs RBC) |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5988 ± 0.1461 | -0.0611 | ≈ within noise |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6390 ± 0.1555 | -0.0438 | ≈ within noise |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6656 ± 0.1527 | -0.0644 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |


---

## 3. Full KPI dump — district

<details>
<summary>Click to expand</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | 1.8005 ± 0.0056 | -0.0242 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | 2.2160 ± 0.0696 | -0.0026 |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | 2.2251 ± 0.0724 | -0.0037 |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.8527 ± 0.0097 | 0.0006 |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | 2.1467 ± 0.0334 | 0.0022 |
| `discomfort_cold_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_proportion` | — | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_proportion` | — | — | — | — |
| `discomfort_proportion` | — | — | — | — |
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | 2.2247 ± 0.0755 | -0.0036 |
| `monthly_one_minus_load_factor_average` | 0.9115 ± 0.0079 | 0.9114 ± 0.0082 | 0.9111 ± 0.0081 | -0.0004 |
| `one_minus_thermal_resilience_proportion` | — | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — | — |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | 1.5704 ± 0.0629 | -0.0076 |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | -5.5739 ± 0.3553 | 0.0022 |


</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

| KPI | RBC | BC | IQL | Δ (IQL − RBC) |
|---|---:|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | 2.6390 ± 0.1555 | -0.0438 |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | 2.6656 ± 0.1527 | -0.0644 |
| `discomfort_cold_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_proportion` | — | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_proportion` | — | — | — | — |
| `discomfort_proportion` | — | — | — | — |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | 2.5988 ± 0.1461 | -0.0611 |
| `one_minus_thermal_resilience_proportion` | — | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — | — |
| `zero_net_energy` | -0.2023 ± 0.0639 | -0.1478 ± 0.0526 | -0.1649 ± 0.1119 | 0.0374 |


</details>

---

## 5. Success criterion

> IQL succeeds if it beats RBC by **more than 1σ** on at least one of
> {`cost_total`, `all_time_peak_average`, `ramping_average`} at the
> district level **with `annual_normalized_unserved_energy_total` = 0**.
>
> A favourable comparison against BC is informative but not part of the
> success contract — the contract is "offline RL beats the data-collection
> policy on its own success metric".
