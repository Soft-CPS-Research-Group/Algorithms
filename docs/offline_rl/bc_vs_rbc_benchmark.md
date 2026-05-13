# BC vs RBC — CityLearn Benchmark

_Dataset_: `./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json` (interface=`flat`, topology_mode=`static`)
_Reward function_: `V2GPenaltyReward` (env reward; reward is a separate scalar used for IQL)
_Target building_: **Building_5** (agent index 4)
_Env seeds_ (per controller): [200, 201, 202, 203, 204, 205, 206, 207, 208, 209] ← disjoint from RBC dataset seeds 22..31

> All KPIs are CityLearn's normalized values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across BC training
> seeds for BC). The "Verdict" column flags `|Δmean| > max(BC_std, RBC_std, 1e-4)` as
> significant; otherwise the difference is within noise.
>
> The BC agent controls Building 5 only; the other 16 agents are driven by
> a fresh RBC instance (so off-target buildings are bit-identical between
> the two columns and Δ there should be ≈ 0 — they only differ via downstream
> coupling effects).

---

## 1. Headline KPIs — district level

### District

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

---

## 2. Headline KPIs — Building 5 (training target)

### Building_5

| KPI | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|:---|
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | -0.0826 | 🟢 BC better |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | -0.0741 | 🟢 BC better |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | -0.0854 | 🟢 BC better |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |

---

## 3. Full KPI dump — district

<details>
<summary>Click to expand</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `all_time_peak_average` | 1.8247 ± 0.0228 | 1.8026 ± 0.0095 | -0.0221 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.2186 ± 0.0717 | 2.2142 ± 0.0677 | -0.0044 |
| `cost_total` | 2.2289 ± 0.0750 | 2.2239 ± 0.0707 | -0.0050 |
| `daily_one_minus_load_factor_average` | 0.8521 ± 0.0103 | 0.8527 ± 0.0098 | 0.0005 |
| `daily_peak_average` | 2.1445 ± 0.0296 | 2.1421 ± 0.0278 | -0.0024 |
| `discomfort_cold_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_proportion` | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_proportion` | — | — | — |
| `discomfort_proportion` | — | — | — |
| `electricity_consumption_total` | 2.2283 ± 0.0783 | 2.2234 ± 0.0740 | -0.0049 |
| `monthly_one_minus_load_factor_average` | 0.9115 ± 0.0079 | 0.9114 ± 0.0082 | -0.0001 |
| `one_minus_thermal_resilience_proportion` | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — |
| `ramping_average` | 1.5780 ± 0.0673 | 1.5707 ± 0.0631 | -0.0072 |
| `zero_net_energy` | -5.5761 ± 0.3704 | -5.5729 ± 0.3544 | 0.0032 |

</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.6828 ± 0.0732 | 2.6087 ± 0.0533 | -0.0741 |
| `cost_total` | 2.7300 ± 0.0813 | 2.6446 ± 0.0589 | -0.0854 |
| `discomfort_cold_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_cold_proportion` | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_maximum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_delta_minimum` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `discomfort_hot_proportion` | — | — | — |
| `discomfort_proportion` | — | — | — |
| `electricity_consumption_total` | 2.6599 ± 0.0794 | 2.5773 ± 0.0585 | -0.0826 |
| `one_minus_thermal_resilience_proportion` | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — |
| `zero_net_energy` | -0.2023 ± 0.0639 | -0.1478 ± 0.0526 | 0.0545 |

</details>

---

## 5. Success criterion ()

> BC succeeds if its district headline KPIs are **within 1σ of RBC** on
> all of cost / peak / ramping / unserved-energy. That demonstrates the 
> pipeline reproduces its behaviour policy faithfully — a precondition for
> the IQL step that follows.
>
> A *strict improvement* over RBC is not expected from BC alone (BC is
> imitation, not optimisation); that's IQL's job in the next step.
