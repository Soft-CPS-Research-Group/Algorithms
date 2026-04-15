# Offline RL Milestone 1 — Implementation Notes

This document describes every file created or modified to build the **offline RL data-collection pipeline** and explains the resulting dataset.

---

## Overview

The goal is to run a deterministic Rule-Based Controller (RBC) through the full training platform (`run_experiment.py` → `Wrapper_CityLearn` → `CityLearnEnv`) and capture every transition for **Building 5** (agent index 4) into a single CSV. This CSV is the offline RL dataset: a record of what the behaviour policy observed, what action it chose, what reward it received, and what happened next.

The pipeline is invoked with:

```bash
python run_experiment.py \
  --config configs/templates/ev_data_collection_local.yaml \
  --job_id offline-data-collection
```

Output lands at `runs/jobs/offline-data-collection/offline_dataset.csv`.

---

## Files Created

### 1. `algorithms/agents/ev_data_collection_agent.py`

**Purpose:** New `BaseAgent` subclass that serves two roles simultaneously:

1. **Behaviour policy** — ports the `BasicElectricVehicleRBC_ReferenceController` hour→action map so every hour of the day maps to a fixed action value per controllable device (battery storage, EV charger, cooling, heating, DHW, washing machine).
2. **Data collector** — records every transition tuple `(obs, action, reward, next_obs, done)` for the target building into an in-memory buffer.

**Why it was needed:** The platform's agent contract (`BaseAgent`) is different from CityLearn's native `Agent` class. The existing `RuleBasedPolicy` in `rbc_agent.py` implements a more sophisticated EV-aware heuristic but does not collect transitions. A new agent was needed to combine the reference RBC logic with data logging.

**Key design choices:**

| Method | What it does |
|--------|-------------|
| `__init__` | Sets `use_raw_observations = True` so the wrapper passes unprocessed CityLearn observations (all 35 features for Building 5). Reads `target_building_index` from config hyperparameters. |
| `attach_environment` | Called by the wrapper after env creation. Caches `observation_names` and `action_names` for all 17 agents (needed for multi-agent `predict()`), pre-computes the `hour` index position per agent, and builds the hour→action map for each device. |
| `predict` | For every agent, looks up the current `hour` from raw observations and returns the mapped action value per device. Returns `List[List[float]]` — one sub-list per agent, as CityLearn requires actions for all 17 buildings. |
| `update` | Extracts agent 4's slice from the multi-agent arrays, converts to Python lists, and appends to `self._transitions` with episode/timestep metadata. This is the data-collection hook — `update()` receives all five tuple elements naturally from the wrapper. |
| `export_artifacts` | Writes the accumulated buffer to `<output_dir>/offline_dataset.csv` with named columns and returns manifest-compatible metadata. |

### 2. `configs/templates/ev_data_collection_local.yaml`

**Purpose:** Config template that wires everything together for a local data-collection run.

**Why it was needed:** The platform requires a YAML config for every run. This template specifies:

- `algorithm.name: EVDataCollectionRBC` → routes to the new agent via the registry
- `algorithm.hyperparameters.target_building_index: 4` → Building 5 (0-indexed)
- `simulator.dataset_path` → points to `citylearn_challenge_2022_phase_all_plus_evs/schema.json`
- `simulator.episodes: 3` → three passes over the year for dataset diversity
- `simulator.reward_function: RewardFunction` → CityLearn's base reward class
- `tracking.mlflow_enabled: false` → lightweight JSONL-only logging
- `checkpointing.checkpoint_interval: null` → no checkpoints (stateless RBC)

---

## Files Modified

### 3. `algorithms/registry.py`

**Change:** Added import and registration entry:

```python
from algorithms.agents.ev_data_collection_agent import EVDataCollectionRBC

ALGORITHM_REGISTRY = {
    ...
    "EVDataCollectionRBC": EVDataCollectionRBC,
}
```

**Why:** Without registration, `create_agent(config)` cannot instantiate the agent — the runner would fail with an "unsupported algorithm" error.

### 4. `utils/config_schema.py`

**Change:** Added two new Pydantic models and extended the algorithm discriminated union:

```python
class EVDataCollectionRBCHyperparameters(BaseModel):
    target_building_index: int = Field(default=4, ge=0)

class EVDataCollectionRBCAlgorithmConfig(BaseModel):
    name: Literal["EVDataCollectionRBC"]
    hyperparameters: EVDataCollectionRBCHyperparameters = ...
    networks: Optional[AlgorithmNetworks] = None
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None
```

The `ProjectConfig.algorithm` union was extended to include `EVDataCollectionRBCAlgorithmConfig`.

**Why:** The platform validates every config through Pydantic before running. Without a schema model, the config would fail validation. The `Literal["EVDataCollectionRBC"]` discriminator routes Pydantic to the correct model based on `algorithm.name`.

### 5. `utils/bundle_validator.py`

**Changes:**

1. Added `"offline_dataset"` to `SUPPORTED_ARTIFACT_FORMATS`.
2. Added a format-specific check: `offline_dataset` artifacts must end in `.csv`.
3. Relaxed the artifact-count constraint: the existing rule `len(artifacts) == num_agents` was gated so it only applies to non-`offline_dataset` formats. Our agent exports 1 CSV for 1 building out of 17, so the strict count check would fail.

**Why:** The bundle validator runs after every `export_artifacts()` call. Without these changes, it would reject the CSV artifact as an unknown format.

---

## How Everything Combines

```
ev_data_collection_local.yaml
        │
        ▼
run_experiment.py
  ├─ validate_config() ──► config_schema.py (EVDataCollectionRBCAlgorithmConfig)
  ├─ CityLearnEnv(schema=schema.json) ──► loads 17 buildings + chargers + EVs
  ├─ Wrapper_CityLearn(env, config)
  │    └─ set_model(agent) ──► agent.attach_environment(obs_names, action_names, ...)
  ├─ create_agent(config) ──► registry.py ──► EVDataCollectionRBC(config)
  ├─ wrapper.learn(episodes=3)
  │    └─ for each of 3 episodes × 8759 steps:
  │         agent.predict(observations)  → hour-mapped actions for all 17 agents
  │         env.step(actions)            → next_obs, rewards, terminated, truncated
  │         agent.update(...)            → appends Building 5 transition to buffer
  ├─ agent.export_artifacts()  → writes offline_dataset.csv (26,277 rows)
  ├─ build_manifest()         → artifact_manifest.json
  └─ validate_bundle_contract() ──► bundle_validator.py (accepts "offline_dataset")
```

---

## The Dataset

### Location

```
runs/jobs/offline-data-collection/offline_dataset.csv
```

### Shape

| Metric | Value |
|--------|-------|
| Rows | 26,277 (3 episodes × 8,759 steps) |
| Columns | 76 |
| File size | ~18 MB |
| Episodes | 3 (indices 0, 1, 2) |
| Steps per episode | 8,759 |

### Column Layout

The 76 columns are structured as follows:

| Group | Prefix | Count | Description |
|-------|--------|-------|-------------|
| Metadata | `episode`, `timestep` | 2 | Episode index (0–2) and within-episode step counter |
| Observations | `obs_*` | 35 | Raw (unencoded) observation values at time $t$ |
| Actions | `action_*` | 2 | Actions taken by the RBC at time $t$ |
| Reward | `reward` | 1 | Scalar reward from CityLearn's `RewardFunction` |
| Next observations | `next_obs_*` | 35 | Raw observation values at time $t+1$ |
| Done flag | `done` | 1 | 1 if episode ended, 0 otherwise |

### Observation Columns (35)

These are the raw CityLearn observations for Building 5, unmodified by encoders:

| # | Column | Type |
|---|--------|------|
| 1–4 | `month`, `day_type`, `hour`, (weather group below) | Temporal |
| 5–20 | `outdoor_dry_bulb_temperature` (×4), `outdoor_relative_humidity` (×4), `diffuse_solar_irradiance` (×4), `direct_solar_irradiance` (×4) | Weather (each with 3 predictions) |
| 21–25 | `carbon_intensity`, `non_shiftable_load`, `solar_generation`, `electrical_storage_soc`, `net_electricity_consumption` | Energy |
| 26–29 | `electricity_pricing`, `electricity_pricing_predicted_1/2/3` | Pricing |
| 30–35 | `charger_5_1_connected_state`, `departure_time`, `required_soc_departure`, `soc`, `battery_capacity`, `incoming_state`, `estimated_arrival_time` | EV charger |

### Action Columns (2)

| Column | Device | Range | RBC behaviour |
|--------|--------|-------|---------------|
| `action_electrical_storage` | Battery (6.4 kWh, 5 kW) | Continuous | Charge at night (0.091), discharge during day (−0.08) |
| `action_electric_vehicle_storage_charger_5_1` | EV charger (7.4 kW, V2G) | Continuous | Hour-dependent: charge morning (1.0), V2G discharge midday (−1.0), moderate evening (0.8) |

### Reward

CityLearn's base `RewardFunction` returns a per-building scalar reward at each step. Values are typically negative (penalties for energy consumption), with 0.0 indicating no net consumption.

### Done Flag

`done = 1` only at the last step of each episode (step 8,758). All other steps have `done = 0`.

### How to Use This Dataset

This CSV is a standard offline RL dataset in the `(s, a, r, s', done)` format. To train an offline RL model:

```python
import pandas as pd

df = pd.read_csv("runs/jobs/offline-data-collection/offline_dataset.csv")

obs_cols = [c for c in df.columns if c.startswith("obs_")]
action_cols = [c for c in df.columns if c.startswith("action_")]
next_obs_cols = [c for c in df.columns if c.startswith("next_obs_")]

observations = df[obs_cols].values        # (26277, 35)
actions = df[action_cols].values          # (26277, 2)
rewards = df["reward"].values             # (26277,)
next_observations = df[next_obs_cols].values  # (26277, 35)
terminals = df["done"].values             # (26277,)
```

Libraries like `d3rlpy` can consume these arrays directly for algorithms such as BC, TD3+BC, CQL, or IQL.
