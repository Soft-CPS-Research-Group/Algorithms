# BC vs RBC — CityLearn Benchmark

_Generated_: 2026-04-22 23:02:34 UTC
_Dataset_: `./datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json`
_Reward function_: `RewardFunction` (CityLearn default)
_Episode length_: 8759 steps (full year, single episode)
_BC checkpoint_: `runs/offline_bc/bc-v1/model.pth`
_BC normalization stats_: `runs/offline_bc/bc-v1/normalization_stats.json`
_Target building_: **Building_5** (agent index 4)

> All KPIs are reported in CityLearn's **normalized form**: a value of `1.0`
> equals the *no-control baseline* (every controllable device idle). Values
> **below 1.0 mean the controller improved over baseline**; values above 1.0
> mean it made things worse. So when comparing two controllers, **lower is
> better**.

---

## 1. Reward summary

| Quantity | RBC | BC |
|---|---:|---:|
| Episode length (steps) | 8759 | 8759 |
| Σ reward (target building) | -17377.8280 | -17232.1307 |
| Mean reward (target building) | -1.9840 | -1.9674 |
| Σ reward (district, all 17 agents) | -117932.8198 | -117787.1225 |
| Mean reward (district) | -13.4642 | -13.4476 |

> The reward function used here is CityLearn's default `RewardFunction`, which
> returns the **negative net electricity consumption** at each step. So a
> *higher* (less negative) sum means the controller drew less energy from the
> grid overall. Note that the **BC agent only controls Building 5** in this
> setup — the other 16 buildings receive idle (zero) actions in both runs, so
> the district-level reward differences are driven entirely by Building 5's
> behaviour.

---

## 2. Headline KPIs — district level

### District

| KPI | RBC | BC | Δ (BC − RBC) | Winner |
|---|---:|---:|---:|:---:|
| `electricity_consumption_total` | 1.1777 | 1.1757 | -0.17% | 🟢 BC |
| `carbon_emissions_total` | 1.1666 | 1.1650 | -0.14% | 🟢 BC |
| `cost_total` | 1.2050 | 1.2026 | -0.20% | 🟢 BC |
| `all_time_peak_average` | 1.1207 | 1.1102 | -0.93% | 🟢 BC |
| `daily_peak_average` | 1.1559 | 1.1548 | -0.10% | 🟢 BC |
| `ramping_average` | 1.0808 | 1.0809 | +0.01% | 🔴 RBC |
| `daily_one_minus_load_factor_average` | 0.9613 | 0.9613 | -0.00% | 🟢 BC |
| `annual_normalized_unserved_energy_total` | 0.0000 | 0.0000 | — | tie |
| `zero_net_energy` | 0.9189 | 0.9192 | +0.03% | 🔴 RBC |


---

## 3. Headline KPIs — Building 5 (the building we trained on)

### Building_5

| KPI | RBC | BC | Δ (BC − RBC) | Winner |
|---|---:|---:|---:|:---:|
| `electricity_consumption_total` | 4.0203 | 3.9866 | -0.84% | 🟢 BC |
| `carbon_emissions_total` | 3.8319 | 3.8051 | -0.70% | 🟢 BC |
| `cost_total` | 4.4853 | 4.4445 | -0.91% | 🟢 BC |
| `annual_normalized_unserved_energy_total` | 0.0000 | 0.0000 | — | tie |
| `zero_net_energy` | -0.3785 | -0.3735 | +1.33% | 🔴 RBC |


---

## 4. Full KPI dump — district

<details>
<summary>Click to expand full district KPI table</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `all_time_peak_average` | 1.1207 | 1.1102 | -0.93% |
| `annual_normalized_unserved_energy_total` | 0.0000 | 0.0000 | — |
| `bess_capacity_fade_ratio` | 0.0002 | 0.0001 | -9.34% |
| `bess_charge_total_kwh` | 1826.3701 | 1656.3450 | -9.31% |
| `bess_discharge_total_kwh` | 1474.5318 | 1336.4667 | -9.36% |
| `bess_equivalent_full_cycles` | 15.1696 | 13.7537 | -9.33% |
| `bess_throughput_total_kwh` | 3300.9018 | 2992.8116 | -9.33% |
| `carbon_emissions_baseline_daily_average_kgco2` | 45.5843 | 45.5843 | -0.00% |
| `carbon_emissions_baseline_total_kgco2` | 16636.3798 | 16636.3798 | -0.00% |
| `carbon_emissions_control_daily_average_kgco2` | 50.8451 | 50.7953 | -0.10% |
| `carbon_emissions_control_total_kgco2` | 18556.3590 | 18538.1814 | -0.10% |
| `carbon_emissions_delta_daily_average_kgco2` | 5.2608 | 5.2110 | -0.95% |
| `carbon_emissions_delta_total_kgco2` | 1919.9792 | 1901.8016 | -0.95% |
| `carbon_emissions_total` | 1.1666 | 1.1650 | -0.14% |
| `community_counterfactual_cost_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_counterfactual_cost_total_eur` | 0.0000 | 0.0000 | — |
| `community_grid_export_after_local_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_export_after_local_total_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_import_after_local_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_import_after_local_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_export_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_local_export_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_import_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_local_import_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_share_of_demand` | — | — | — |
| `community_local_share_of_export` | — | — | — |
| `community_market_savings_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_market_savings_total_eur` | 0.0000 | 0.0000 | — |
| `community_settled_cost_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_settled_cost_total_eur` | 0.0000 | 0.0000 | — |
| `cost_baseline_daily_average_eur` | 10.5682 | 10.5682 | -0.00% |
| `cost_baseline_total_eur` | 3856.9628 | 3856.9627 | -0.00% |
| `cost_control_daily_average_eur` | 15.4353 | 15.4162 | -0.12% |
| `cost_control_total_eur` | 5633.2315 | 5626.2832 | -0.12% |
| `cost_delta_daily_average_eur` | 4.8670 | 4.8480 | -0.39% |
| `cost_delta_total_eur` | 1776.2687 | 1769.3205 | -0.39% |
| `cost_total` | 1.2050 | 1.2026 | -0.20% |
| `daily_one_minus_load_factor_average` | 0.9613 | 0.9613 | -0.00% |
| `daily_peak_average` | 1.1559 | 1.1548 | -0.10% |
| `discomfort_cold_delta_average` | 0.0000 | 0.0000 | — |
| `discomfort_cold_delta_maximum` | 0.0000 | 0.0000 | — |
| `discomfort_cold_delta_minimum` | 0.0000 | 0.0000 | — |
| `discomfort_cold_proportion` | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 | 0.0000 | — |
| `discomfort_hot_delta_maximum` | 0.0000 | 0.0000 | — |
| `discomfort_hot_delta_minimum` | 0.0000 | 0.0000 | — |
| `discomfort_hot_proportion` | — | — | — |
| `discomfort_proportion` | — | — | — |
| `electrical_service_violation_time_step_count` | 0.0000 | 0.0000 | — |
| `electrical_service_violation_total_kwh` | 0.0000 | 0.0000 | — |
| `electricity_consumption_baseline_daily_average_kwh` | 242.9562 | 242.9562 | -0.00% |
| `electricity_consumption_baseline_total_kwh` | 88668.8974 | 88668.8973 | -0.00% |
| `electricity_consumption_control_daily_average_kwh` | 275.6324 | 275.3505 | -0.10% |
| `electricity_consumption_control_total_kwh` | 100594.3527 | 100491.4674 | -0.10% |
| `electricity_consumption_delta_daily_average_kwh` | 32.6762 | 32.3943 | -0.86% |
| `electricity_consumption_delta_total_kwh` | 11925.4553 | 11822.5701 | -0.86% |
| `electricity_consumption_total` | 1.1777 | 1.1757 | -0.17% |
| `equity_bpr_asset_poor_over_rich` | — | — | — |
| `equity_cr20_benefit` | 0.7893 | 0.7893 | +0.00% |
| `equity_gini_benefit` | 0.7383 | 0.7383 | +0.00% |
| `equity_losers_percent` | 35.7143 | 35.7143 | +0.00% |
| `ev_charge_total_kwh` | 12109.5612 | 12103.6185 | -0.05% |
| `ev_departure_soc_deficit_mean` | 0.2695 | 0.2695 | -0.02% |
| `ev_departure_success_rate` | 0.1184 | 0.1184 | +0.00% |
| `ev_v2g_export_total_kwh` | 3690.4049 | 3684.5991 | -0.16% |
| `monthly_one_minus_load_factor_average` | 0.9770 | 0.9768 | -0.02% |
| `one_minus_thermal_resilience_proportion` | — | — | — |
| `phase_imbalance_ratio_average` | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — |
| `pv_export_daily_average_kwh` | 224.1841 | 223.9872 | -0.09% |
| `pv_export_total_kwh` | 81817.8598 | 81745.9969 | -0.09% |
| `pv_generation_daily_average_kwh` | 398.4627 | 398.4627 | +0.00% |
| `pv_generation_total_kwh` | 145422.3011 | 145422.3011 | +0.00% |
| `pv_self_consumption_ratio` | 0.4374 | 0.4379 | +0.11% |
| `ramping_average` | 1.0808 | 1.0809 | +0.01% |
| `zero_net_energy` | 0.9189 | 0.9192 | +0.03% |
| `zero_net_energy_baseline_daily_average_kwh` | 66.3040 | 66.3040 | -0.00% |
| `zero_net_energy_baseline_total_kwh` | 24198.2107 | 24198.2104 | -0.00% |
| `zero_net_energy_control_daily_average_kwh` | 90.3369 | 90.2490 | -0.10% |
| `zero_net_energy_control_total_kwh` | 32969.2049 | 32937.1082 | -0.10% |
| `zero_net_energy_delta_daily_average_kwh` | 24.0329 | 23.9449 | -0.37% |
| `zero_net_energy_delta_total_kwh` | 8770.9942 | 8738.8978 | -0.37% |


</details>

## 5. Full KPI dump — Building 5

<details>
<summary>Click to expand full Building 5 KPI table</summary>

| KPI | RBC | BC | Δ (BC − RBC) |
|---|---:|---:|---:|
| `annual_normalized_unserved_energy_total` | 0.0000 | 0.0000 | — |
| `bess_capacity_fade_ratio` | 0.0026 | 0.0023 | -9.34% |
| `bess_charge_total_kwh` | 1826.3701 | 1656.3450 | -9.31% |
| `bess_discharge_total_kwh` | 1474.5318 | 1336.4667 | -9.36% |
| `bess_equivalent_full_cycles` | 257.8830 | 233.8134 | -9.33% |
| `bess_throughput_total_kwh` | 3300.9018 | 2992.8116 | -9.33% |
| `carbon_emissions_baseline_daily_average_kgco2` | 1.8577 | 1.8577 | -0.00% |
| `carbon_emissions_baseline_total_kgco2` | 677.9750 | 677.9750 | -0.00% |
| `carbon_emissions_control_daily_average_kgco2` | 7.1185 | 7.0687 | -0.70% |
| `carbon_emissions_control_total_kgco2` | 2597.9542 | 2579.7766 | -0.70% |
| `carbon_emissions_delta_daily_average_kgco2` | 5.2608 | 5.2110 | -0.95% |
| `carbon_emissions_delta_total_kgco2` | 1919.9792 | 1901.8016 | -0.95% |
| `carbon_emissions_total` | 3.8319 | 3.8051 | -0.70% |
| `community_counterfactual_cost_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_counterfactual_cost_total_eur` | 0.0000 | 0.0000 | — |
| `community_grid_export_after_local_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_export_after_local_total_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_import_after_local_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_grid_import_after_local_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_export_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_local_export_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_import_daily_average_kwh` | 0.0000 | 0.0000 | — |
| `community_local_import_total_kwh` | 0.0000 | 0.0000 | — |
| `community_local_share_of_demand` | — | — | — |
| `community_local_share_of_export` | — | — | — |
| `community_market_savings_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_market_savings_total_eur` | 0.0000 | 0.0000 | — |
| `community_settled_cost_daily_average_eur` | 0.0000 | 0.0000 | — |
| `community_settled_cost_total_eur` | 0.0000 | 0.0000 | — |
| `cost_baseline_daily_average_eur` | -3.0459 | -3.0459 | -0.00% |
| `cost_baseline_total_eur` | -1111.6154 | -1111.6154 | -0.00% |
| `cost_control_daily_average_eur` | 1.8212 | 1.8021 | -1.05% |
| `cost_control_total_eur` | 664.6533 | 657.7051 | -1.05% |
| `cost_delta_daily_average_eur` | 4.8670 | 4.8480 | -0.39% |
| `cost_delta_total_eur` | 1776.2687 | 1769.3205 | -0.39% |
| `cost_total` | 4.4853 | 4.4445 | -0.91% |
| `discomfort_cold_delta_average` | 0.0000 | 0.0000 | — |
| `discomfort_cold_delta_maximum` | 0.0000 | 0.0000 | — |
| `discomfort_cold_delta_minimum` | 0.0000 | 0.0000 | — |
| `discomfort_cold_proportion` | — | — | — |
| `discomfort_hot_delta_average` | 0.0000 | 0.0000 | — |
| `discomfort_hot_delta_maximum` | 0.0000 | 0.0000 | — |
| `discomfort_hot_delta_minimum` | 0.0000 | 0.0000 | — |
| `discomfort_hot_proportion` | — | — | — |
| `discomfort_proportion` | — | — | — |
| `electrical_service_violation_time_step_count` | 0.0000 | 0.0000 | — |
| `electrical_service_violation_total_kwh` | 0.0000 | 0.0000 | — |
| `electricity_consumption_baseline_daily_average_kwh` | 11.8438 | 11.8438 | -0.00% |
| `electricity_consumption_baseline_total_kwh` | 4322.4857 | 4322.4855 | -0.00% |
| `electricity_consumption_control_daily_average_kwh` | 47.6159 | 47.2167 | -0.84% |
| `electricity_consumption_control_total_kwh` | 17377.8280 | 17232.1307 | -0.84% |
| `electricity_consumption_delta_daily_average_kwh` | 35.7721 | 35.3729 | -1.12% |
| `electricity_consumption_delta_total_kwh` | 13055.3424 | 12909.6452 | -1.12% |
| `electricity_consumption_total` | 4.0203 | 3.9866 | -0.84% |
| `equity_relative_benefit_percent` | — | — | — |
| `ev_charge_total_kwh` | 12109.5612 | 12103.6185 | -0.05% |
| `ev_departure_soc_deficit_mean` | 0.0798 | 0.0794 | -0.54% |
| `ev_departure_success_rate` | 0.7813 | 0.7813 | +0.00% |
| `ev_v2g_export_total_kwh` | 3690.4049 | 3684.5991 | -0.16% |
| `one_minus_thermal_resilience_proportion` | — | — | — |
| `phase_imbalance_ratio_average` | — | — | — |
| `power_outage_normalized_unserved_energy_total` | — | — | — |
| `pv_export_daily_average_kwh` | 32.3975 | 32.2006 | -0.61% |
| `pv_export_total_kwh` | 11823.7507 | 11751.8878 | -0.61% |
| `pv_generation_daily_average_kwh` | 41.5639 | 41.5639 | +0.00% |
| `pv_generation_total_kwh` | 15169.1030 | 15169.1030 | +0.00% |
| `pv_self_consumption_ratio` | 0.2205 | 0.2253 | +2.15% |
| `zero_net_energy` | -0.3785 | -0.3735 | +1.33% |
| `zero_net_energy_baseline_daily_average_kwh` | -17.4340 | -17.4340 | -0.00% |
| `zero_net_energy_baseline_total_kwh` | -6362.6720 | -6362.6723 | -0.00% |
| `zero_net_energy_control_daily_average_kwh` | 6.5989 | 6.5110 | -1.33% |
| `zero_net_energy_control_total_kwh` | 2408.3222 | 2376.2255 | -1.33% |
| `zero_net_energy_delta_daily_average_kwh` | 24.0329 | 23.9449 | -0.37% |
| `zero_net_energy_delta_total_kwh` | 8770.9942 | 8738.8978 | -0.37% |


</details>

---

## 6. How to read these numbers

| Metric | What it means | Why we care |
|---|---|---|
| `electricity_consumption_total` | Total grid electricity drawn (normalized vs baseline) | The headline "did the controller cut consumption?" |
| `carbon_emissions_total` | Carbon footprint of grid draw, weighted by hourly emission factor | Captures *when* energy is used, not just how much |
| `cost_total` | Monetary cost of grid energy, weighted by tariff (incl. peak pricing) | Direct economic impact |
| `all_time_peak_average` | Highest single-step grid draw, normalized | Grid-stress proxy; expensive to provision for |
| `daily_peak_average` | Average of each day's peak | Smoothness of daily demand |
| `ramping_average` | Mean step-to-step change in district load | Penalizes "spiky" control |
| `daily_one_minus_load_factor_average` | `1 − (mean / peak)` per day | Lower = flatter, more efficient utilization |
| `annual_normalized_unserved_energy_total` | EV/thermal demand the controller failed to satisfy | **Constraint violation indicator** — should stay near 0 |
| `zero_net_energy` | Imbalance between consumed and produced (PV) energy | Self-sufficiency proxy |

### Verdict heuristic

Since BC was trained to **imitate** the RBC, the expected outcome is that
BC's KPIs are *very close* to RBC's. Large divergences indicate either:
* the policy generalizes (could be good or bad — it depends on which way),
* or it has acquired distribution-shift errors at states the RBC rarely
  visits (a known BC failure mode).

For a successful Milestone 2 we want **|Δ| ≤ a few percent** on the headline
KPIs, with **`annual_normalized_unserved_energy_total` not getting worse**.
