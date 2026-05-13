# CityLearn KPI Reference

> CityLearn evaluates a controller against a **no-control baseline**:
> the same environment is replayed with the agent doing nothing
> (charging actions = 0). Each KPI is then reported as a *ratio*
>
> KPI = controller_value / baseline_value
>
> So **`1.0` means "tied with doing nothing"** and **lower is better**
> for almost all KPIs (one exception below). A value of `0.92` means
> "9 % improvement over no-control".

This file explains, for each KPI:

1. **What it measures** (plain English).
2. **How it is computed** (math).
3. **Which actions move it** (lever).
4. **How 's reward function targets it** (link to
 `reward_design.md`).

---

## 1. The headline KPIs

CityLearn reports many KPIs; for the milestone we focus on these.

| # | KPI | Better when … | reward term |
|---|-----|---------------|----------------|
| 1 | `cost_total` | Buy energy cheap, sell back expensive | `−price · net_consumption` |
| 2 | `carbon_emissions_total` | Consume when grid is green | `−carbon · max(0, net)` |
| 3 | `electricity_consumption_total` | Use less, self-consume more | `−max(0, net)` |
| 4 | `daily_peak_average` | Avoid spikes | `−max(0, net − rolling_mean)` |
| 5 | `all_time_peak_average` | Don't hit a new yearly peak | implicit via daily_peak term |
| 6 | `ramping_average` | Smooth load | `−|net_t − net_{t−1}|` |
| 7 | `daily_one_minus_load_factor_average` | Flatter daily profile | implicit via peak + ramping |
| 8 | `annual_normalized_unserved_energy_total` | Meet EV / building demand | `−unserved_energy` |
| 9 | `zero_net_energy` | Match consumption to local PV | embedded in cost + consumption terms |

A `1.0` baseline ratio + the calibration step (see `reward_design.md`)
ensures that the per-step reward, summed over an episode, **correlates
positively with the sum of (1 − KPI)** for KPIs 1–8 — i.e. minimising
negative reward at each step pushes episode-end KPIs down.

---

## 2. KPI by KPI

### 2.1 `cost_total`

- **What.** Total electricity bill over the year, divided by the
 baseline bill.
- **Math.** With `p_t` the price ($/kWh) and `e_t` the building's net
 consumption (kWh, positive = import),
 `cost = Σ_t p_t · max(0, e_t)`. The baseline replays the same series
 with the controller doing nothing.
- **Lever.** Shift loads to cheap hours (charge EV/battery off-peak,
 discharge on-peak); export when prices are high.
- **RBC blind spot.** RBC ignores price entirely.

### 2.2 `carbon_emissions_total`

- **What.** Total CO₂ emitted by the building's grid imports.
- **Math.** With `c_t` the grid carbon intensity (kg CO₂ / kWh),
 `emissions = Σ_t c_t · max(0, e_t)`. Exports to the grid count as 0
 (no carbon credit).
- **Lever.** Same as cost, but using `c_t` instead of `p_t`. Prices and
 carbon are correlated but not identical, so a separate term matters.
- **RBC blind spot.** RBC ignores carbon entirely.

### 2.3 `electricity_consumption_total`

- **What.** Total energy drawn from the grid over the year.
- **Math.** `E = Σ_t max(0, e_t)`. Exports do not reduce this number
 (they are tracked separately via `zero_net_energy`).
- **Lever.** Increase self-consumption: store PV in battery / EV when
 it would otherwise be exported, use it later when the building would
 otherwise import.
- **RBC blind spot.** Stationary battery is unused → most PV that
 exceeds instantaneous load is exported, then bought back later.

### 2.4 `daily_peak_average`

- **What.** Average over the year of the *daily maximum* net
 consumption, divided by the baseline equivalent.
- **Math.** `peak_d = max_{t ∈ day d} e_t`; KPI averages `peak_d` over
 the 365 days, then normalises.
- **Lever.** Anticipate high-demand hours and pre-charge / discharge to
 shave the peak.
- **RBC blind spot.** Per-charger, no peak awareness; emergency mode +
 PV bonus can both contribute to peaks.

### 2.5 `all_time_peak_average`

- **What.** The single highest hour of the year.
- **Math.** `max_t e_t`, normalised. (CityLearn averages this across
 buildings, hence "average" in the name.)
- **Lever.** Same as daily peak, but one big mistake costs the whole
 year. A useful tail-risk metric.

### 2.6 `ramping_average`

- **What.** Hour-to-hour change in net consumption — penalises jumpy
 controllers.
- **Math.** `Σ_t |e_t − e_{t−1}|`, normalised.
- **Lever.** Smooth transitions: feather charging start/stop, avoid
 binary on/off behaviour.
- **RBC blind spot.** Emergency cliff (rate jumps to 1.0 within 1 h of
 departure) and PV bonus (rate jumps to 0.6 when sun appears) both
 create ramps.

### 2.7 `daily_one_minus_load_factor_average`

- **What.** *Load factor* on day `d` = mean(load) / max(load). KPI uses
 `1 − load_factor` so **lower is better**, then averages over days.
- **Math.** `lf_d = mean_t e_t / max_t e_t`; KPI = mean over `d` of
 `(1 − lf_d)`.
- **Lever.** Spread load across the day. Equivalent to "raise the
 average without raising the peak" — pushes for flat profiles.

### 2.8 `annual_normalized_unserved_energy_total`

- **What.** Energy demand the controller failed to meet (e.g. EV not
 charged to required SoC by departure).
- **Math.** Sum of unmet energy across the year, normalised by total
 required energy.
- **Lever.** Don't strand the EV. Don't refuse to import when the
 building needs energy.
- **RBC behaviour.** RBC is conservative (emergency rate = 1.0) so this
 is typically near 0. Any agent must keep it near 0 too — this is a
 *safety* KPI, not an *efficiency* one.

### 2.9 `zero_net_energy`

- **What.** How close the building is to net-zero over the year.
- **Math.** `|Σ_t e_t| / Σ_t |e_t|` (or similar — CityLearn variants
 exist). Lower means the year-totals of import and export roughly
 cancel.
- **Lever.** Use storage to keep PV energy on-site instead of exporting
 it and re-importing later.

---

## 3. Composite reward strategy

The per-step reward must **decrease (become more positive)** when the
agent's action makes any of KPIs 1–8 better at the end of the episode.
Naively summing the individual physical quantities would give wildly
different scales (cost in $, ramping in kWh, unserved in kWh-equivalent
…), so the reward applies **per-term weights calibrated against
RBC rollouts**.

The calibration is a one-time linear regression:

> Run RBC for N rollouts. For each rollout, compute the per-step terms
> `(c_t, g_t, p_t, r_t, u_t)` (cost, carbon, peak excursion, ramp,
> unserved). For each rollout, compute the episode-end KPI ratios
> `(K_cost, K_carbon, K_peak, K_ramp, K_unserved)`. Solve for weights
> `w` such that `Σ_t w · (c_t, g_t, p_t, r_t, u_t)` correlates with
> `−Σ KPI_ratios` across rollouts. Freeze `w` into a manifest.

Full details, code references and unit tests live in
`reward_design.md`.

---

## 4. Sources

- CityLearn KPI implementation: upstream `citylearn` package
 (`citylearn/cost_function.py`).
- v1 evidence that RBC ≈ no-op tied on KPIs:
 `docs/offline_rl/m4/iql_vs_rbc_vs_bc_vs_random_benchmark_m4.md`.
- Per-building reward proxy used in v1 (M4):
 `datasets/offline_rl/m2/derived/b5_reward_manifest.json` — kept for
 historical comparison; reward replaces it.
