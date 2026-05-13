# Dataset Schema — v2

> The v2 dataset is a flat table of `(obs, action, reward, next_obs)`
> transitions, one row per `(seed, timestep)`. **The schema is fixed**:
> when we swap the behaviour policy (RBC → BC → IQL → …), only the
> values in `action_*` and `reward` change. Column names, dtypes,
> ordering and semantics stay constant.

---

## 1. Storage layout

```
datasets/offline_rl_v2/
├── rbc/
│   ├── seed_0.parquet
│   ├── seed_1.parquet
│   ├── ...
│   ├── seed_9.parquet
│   ├── manifest.json          # collection metadata (see §6)
│   ├── kpi_summary.csv        # per-seed end-of-episode KPIs
│   └── sample_first_1000.csv  # human-inspection sample (committed)
├── derived/
│   └── rbc_with_reward_v2.parquet   # one row per transition, all seeds, with reward_v2 column
└── README.md                  # quick pointer to this doc
```

**File format.** Apache Parquet, Snappy compression, one row group per
file. Reasons: typed columns (no float-vs-string confusion), ~10×
smaller than CSV, fast columnar reads, native pandas / pyarrow support.

**Naming convention.**
`<behaviour_policy>/seed_<n>.parquet` — `<behaviour_policy>` ∈ {`rbc`,
`bc_v2`, `iql_v2`, …}.

---

## 2. One-page column cheat sheet

> Total: ~80 columns. Grouped here by role. See §3 for the full table.

| Group | Columns | Count |
|------|--------|------:|
| Bookkeeping | `episode`, `timestep`, `seed`, `policy_mode` | 4 |
| Time | `obs_month`, `obs_day_type`, `obs_hour` | 3 |
| Weather (current + 3-step forecast) | `obs_outdoor_dry_bulb_temperature*`, `obs_outdoor_relative_humidity*`, `obs_diffuse_solar_irradiance*`, `obs_direct_solar_irradiance*` | 16 |
| Carbon | `obs_carbon_intensity` | 1 |
| Building electrical | `obs_non_shiftable_load`, `obs_solar_generation`, `obs_electrical_storage_soc`, `obs_net_electricity_consumption` | 4 |
| Pricing (current + 3-step forecast) | `obs_electricity_pricing*` | 4 |
| EV / charger | `obs_electric_vehicle_charger_*`, `obs_connected_electric_vehicle_*`, `obs_incoming_electric_vehicle_*` | 7 |
| **Action** | `action_electrical_storage`, `action_electric_vehicle_storage_charger_5_1` | **2** |
| **Reward** | `reward_env`, `reward_v2` | **2** |
| Next-obs mirror | `next_obs_*` (same as obs group) | 35 |
| Termination | `terminated`, `truncated` | 2 |

Totals: **~80 columns**, **2 action columns** (this is the supervision
target — verify variance!), **2 reward columns** (env + v2).

---

## 3. Full column reference

> All columns are `float32` unless noted. Booleans stored as `uint8`
> (0/1). Missing values are stored as NaN, not as zeros.

### 3.1 Bookkeeping

| Column | Dtype | Units | Meaning |
|--------|-------|-------|---------|
| `episode` | `int32` | — | Episode index within this file (always 0 for now). |
| `timestep` | `int32` | hour-of-year | 0…8759. |
| `seed` | `int32` | — | RNG seed used for this rollout. Matches filename. |
| `policy_mode` | `category` | — | `"behaviour"` always (this is a behaviour-policy rollout). Future-proofing for mixed datasets. |

### 3.2 Observation columns (prefixed `obs_`)

| Column | Dtype | Units | Source |
|--------|-------|-------|--------|
| `obs_month` | `int8` | 1–12 | calendar |
| `obs_day_type` | `int8` | 1–8 | CityLearn day-of-week + holiday code |
| `obs_hour` | `int8` | 1–24 | calendar |
| `obs_outdoor_dry_bulb_temperature` | float | °C | weather |
| `obs_outdoor_dry_bulb_temperature_predicted_{1,2,3}` | float | °C | 1/2/3-step-ahead forecast |
| `obs_outdoor_relative_humidity` | float | % | weather |
| `obs_outdoor_relative_humidity_predicted_{1,2,3}` | float | % | forecast |
| `obs_diffuse_solar_irradiance` | float | W/m² | weather |
| `obs_diffuse_solar_irradiance_predicted_{1,2,3}` | float | W/m² | forecast |
| `obs_direct_solar_irradiance` | float | W/m² | weather |
| `obs_direct_solar_irradiance_predicted_{1,2,3}` | float | W/m² | forecast |
| `obs_carbon_intensity` | float | kg CO₂ / kWh | grid carbon signal |
| `obs_non_shiftable_load` | float | kWh | building inflexible load |
| `obs_solar_generation` | float | kWh | building PV output |
| `obs_electrical_storage_soc` | float | 0–1 | stationary battery SoC |
| `obs_net_electricity_consumption` | float | kWh | building net draw from grid (negative = export) |
| `obs_electricity_pricing` | float | $/kWh | grid price |
| `obs_electricity_pricing_predicted_{1,2,3}` | float | $/kWh | forecast |
| `obs_electric_vehicle_charger_charger_5_1_connected_state` | uint8 | 0/1 | car plugged in? |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_departure_time` | float | hour-of-day | when the connected car leaves |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_required_soc_departure` | float | 0–1 | target SoC at departure |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_soc` | float | 0–1 | EV current SoC |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_battery_capacity` | float | kWh | EV pack size |
| `obs_electric_vehicle_charger_charger_5_1_incoming_state` | uint8 | 0/1 | car arriving soon? |
| `obs_incoming_electric_vehicle_at_charger_charger_5_1_estimated_arrival_time` | float | hour-of-day | next arrival |

### 3.3 Action columns

| Column | Dtype | Range | Meaning |
|--------|-------|-------|---------|
| `action_electrical_storage` | float | [−1, +1] | normalised charge (+) / discharge (−) of stationary battery |
| `action_electric_vehicle_storage_charger_5_1` | float | [−1, +1] | normalised charge (+) / discharge (−) of EV via charger 5_1 |

> ⚠️ **Schema-stability invariant.** These are the *only* action columns.
> They store the action the **behaviour policy actually took** — not a
> "clean" or post-processed version. The v1 bug (`action_clean_*` all
> zeros) is impossible by construction in v2 because there is no
> `_clean` column at all. The collector additionally writes
> `manifest.json` with per-column action statistics and **fails fast**
> if any action column has zero variance across the rollout.

### 3.4 Reward columns

| Column | Dtype | Meaning |
|--------|-------|---------|
| `reward_env` | float | The reward returned by the environment's `V2GPenaltyReward` (per-building component, before community mixing). Kept for traceability. |
| `reward_v2` | float | The **KPI-aligned reward** computed by `algorithms/offline_rl_v2/reward_v2.py`. **This is the training signal for BC/IQL-v2.** See `reward_design_v2.md`. |

### 3.5 Next-observation columns

For every `obs_X` above, there is a `next_obs_X` with identical dtype
and units, holding the value at `t+1`.

### 3.6 Termination

| Column | Dtype | Meaning |
|--------|-------|---------|
| `terminated` | uint8 | Episode ended naturally (always 0 for fixed-horizon). |
| `truncated` | uint8 | Episode ended by time limit (1 only on the last row). |

---

## 4. Schema-stability guarantee

When we replace the behaviour policy:

- **Same**: every `obs_*`, every `next_obs_*`, both action *names*,
  `reward_env`, `terminated`, `truncated`, `episode`, `timestep`,
  `policy_mode`.
- **Different values, same columns**: `action_electrical_storage`,
  `action_electric_vehicle_storage_charger_5_1`, `reward_env`,
  `reward_v2` (because the actions differ → state evolves differently
  → next_obs differs → all derived signals differ).
- **Different**: `seed` set may change.

This is enforced by a single `SCHEMA = pa.schema([...])` in
`algorithms/offline_rl_v2/data_loader_v2.py`. The collector validates
its output against this schema before writing.

---

## 5. Volume

| Quantity | Value |
|----------|-------|
| Steps per rollout | 8 760 (1 year, hourly) |
| Seeds (RBC v2) | 10 |
| Total transitions | 87 600 |
| Approx Parquet size per file | ~2–3 MB |
| Approx total on disk | ~25–30 MB (vs ~250 MB for equivalent CSV) |

---

## 6. `manifest.json` contents

Written once per behaviour-policy directory. Used for reproducibility
and for fail-fast checks.

```json
{
  "behaviour_policy": "rbc",
  "behaviour_policy_class": "algorithms.agents.rbc_agent.RuleBasedPolicy",
  "behaviour_policy_hyperparameters": { ... },
  "schema_hash": "sha256:...",
  "env_config_hash": "sha256:...",
  "citylearn_dataset": "datasets/citylearn_three_phase_dynamic_topology_demo_v1",
  "building": "Building_5",
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "steps_per_rollout": 8760,
  "n_transitions": 87600,
  "action_stats": {
    "action_electrical_storage":               { "mean": ..., "std": ..., "min": ..., "max": ... },
    "action_electric_vehicle_storage_charger_5_1": { "mean": ..., "std": ..., "min": ..., "max": ... }
  },
  "reward_stats": {
    "reward_env": { ... },
    "reward_v2":  { ... }
  },
  "kpi_summary_path": "kpi_summary.csv",
  "collected_at": "<ISO timestamp>",
  "code_git_sha": "<short sha>"
}
```

The collector **fails** if `action_stats[*]["std"] < 1e-6` for any
action column — the M2 failure mode is a hard error in v2.
