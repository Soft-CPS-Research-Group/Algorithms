# M3 — BC vs RBC vs Random — CityLearn Benchmark

_Generated_: 2026-04-27 22:08:59 UTC
_Dataset_: `./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json` (interface=`flat`, topology_mode=`static` — same mode used for M2 collection)
_Reward function_: `V2GPenaltyReward` (matches M2 data collection)
_Target building_: **Building_5** (agent index 4)
_BC checkpoint root_: `/Users/guilherme.desousa/MEIA/Thesis/Project/repos/Algorithms/runs/offline_bc_m3/bc-m3-v1`
_BC training seeds_: [22, 23, 24]
_Env seeds_ (per controller): [22, 23, 24, 25, 26]
_Rollouts per BC training seed_: 5
_Total rollouts_: RBC=5, Random=5, BC=15

> All KPIs are CityLearn's normalized values; **lower is better** (1.0 = no-control
> baseline). Mean ± std is computed across env seeds (and across BC training
> seeds for BC). The "Verdict" column flags `|Δmean| > max(BC_std, RBC_std)` as
> significant; otherwise the difference is within noise.

---

## 1. Headline KPIs — district level

### District

| KPI | Random (mean ± std) | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 3.0446 ± 0.0118 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `carbon_emissions_total` | 2.9503 ± 0.0138 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `cost_total` | 2.7678 ± 0.0111 | 0.9603 ± 0.0000 | 0.9603 ± 0.0000 | 0.0000 | ≈ within noise |
| `all_time_peak_average` | 1.3849 ± 0.0759 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `daily_peak_average` | 1.7809 ± 0.0215 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `ramping_average` | 3.0042 ± 0.0317 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0000 | ≈ within noise |
| `daily_one_minus_load_factor_average` | 0.9662 ± 0.0016 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `zero_net_energy` | -1.0482 ± 0.0664 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0000 | ≈ within noise |


---

## 2. Headline KPIs — Building 5 (training target)

### Building_5

| KPI | Random (mean ± std) | RBC (mean ± std) | BC (mean ± std) | Δ (BC − RBC) | Verdict |
|---|---:|---:|---:|---:|:---|
| `electricity_consumption_total` | 3.2401 ± 0.0257 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `carbon_emissions_total` | 3.1409 ± 0.0233 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 | ≈ within noise |
| `cost_total` | 3.0494 ± 0.0279 | 0.9936 ± 0.0000 | 0.9936 ± 0.0000 | 0.0000 | ≈ within noise |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ≈ within noise |


---

## 3. Full KPI dump — district

<details>
<summary>Click to expand</summary>

| KPI | Random | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|---:|
| `all_time_peak_average` | 1.3849 ± 0.0759 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `carbon_emissions_total` | 2.9503 ± 0.0138 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `cost_total` | 2.7678 ± 0.0111 | 0.9603 ± 0.0000 | 0.9603 ± 0.0000 | 0.0000 |
| `daily_one_minus_load_factor_average` | 0.9662 ± 0.0016 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `daily_peak_average` | 1.7809 ± 0.0215 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `electricity_consumption_total` | 3.0446 ± 0.0118 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `ramping_average` | 3.0042 ± 0.0317 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0000 |
| `zero_net_energy` | -1.0482 ± 0.0664 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0000 |


</details>

## 4. Full KPI dump — Building 5

<details>
<summary>Click to expand</summary>

| KPI | Random | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| `bess_capacity_fade_ratio` | — | — | — | — |
| `bess_equivalent_full_cycles` | — | — | — | — |
| `bess_throughput_total_kwh` | — | — | — | — |
| `carbon_emissions_total` | 3.1409 ± 0.0233 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `cost_total` | 3.0494 ± 0.0279 | 0.9936 ± 0.0000 | 0.9936 ± 0.0000 | 0.0000 |
| `electricity_consumption_total` | 3.2401 ± 0.0257 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 |
| `ev_departure_success_rate` | — | — | — | — |


</details>

---

## 5. How to read these numbers

| Metric | What it means | Why we care |
|---|---|---|
| `electricity_consumption_total` | Total grid electricity drawn (normalized vs baseline) | Headline efficiency |
| `carbon_emissions_total` | Carbon footprint of grid draw | Captures *when* energy is used |
| `cost_total` | Monetary cost (tariff-weighted) | Direct economic impact |
| `all_time_peak_average` | Highest single-step grid draw | Grid-stress proxy |
| `daily_peak_average` | Average of each day's peak | Smoothness of daily demand |
| `ramping_average` | Mean step-to-step change in district load | Penalizes "spiky" control |
| `daily_one_minus_load_factor_average` | `1 − (mean / peak)` per day | Lower = flatter utilization |
| `annual_normalized_unserved_energy_total` | EV/thermal demand the controller failed to satisfy | **Constraint violation indicator** |
| `zero_net_energy` | Imbalance vs PV generation | Self-sufficiency proxy |
| `ev_departure_success_rate` | Fraction of EV departures meeting SoC target | Service-level KPI |
| `bess_throughput_total_kwh` | Total energy cycled through battery | Wear proxy |
| `bess_equivalent_full_cycles` | Equivalent full charge/discharge cycles | Wear proxy |
| `bess_capacity_fade_ratio` | Capacity loss vs nominal | Long-term degradation |

### Verdict heuristic

BC is trained to imitate the RBC's *clean* actions on Building 5. We expect:

* **BC ≈ RBC** on most KPIs (small Δ within noise) ⇒ behaviour cloning succeeded.
* **Random ≫ RBC** on cost / unserved energy ⇒ task is non-trivial; BC's parity
  with RBC is meaningful, not an artefact of a degenerate task.
* **Large adverse Δ** on `annual_normalized_unserved_energy_total` ⇒ BC is
  failing in safety-critical states (a known BC failure mode on out-of-distribution
  observations); this would motivate the M4 IQL/TD3+BC step.
