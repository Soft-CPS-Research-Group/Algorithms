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

### Observation Columns (35) — detailed

These are the raw CityLearn observations for Building 5, unmodified by encoders. We pass them through verbatim because the RBC sets `use_raw_observations = True`. Below is each column with its meaning, units, and why it matters for an EV-charging policy.

#### Temporal (3)

| Column | Values | Description |
|--------|--------|-------------|
| `obs_month` | 1–12 | Month of the year. Drives seasonal patterns (winter vs summer demand). |
| `obs_hour` | 1–24 | Hour of the day. CityLearn uses 1–24 (not 0–23). The single most important feature for the RBC, since its action map is hour-keyed. |
| `obs_day_type` | 1–8 | 1=Mon … 7=Sun, 8=holiday. Weekday vs weekend changes load patterns. |

#### Weather (16)

Each of four base measurements appears 4× (current + 3 predictions). The `_predicted_1/2/3` columns are CityLearn's built-in 6h / 12h / 24h forecasts.

| Column family | Units | Description |
|---------------|-------|-------------|
| `obs_outdoor_dry_bulb_temperature[_predicted_1/2/3]` | °C | Current and forecasted outdoor temperature. Affects heating/cooling load. |
| `obs_outdoor_relative_humidity[_predicted_1/2/3]` | % | Humidity, affects perceived comfort and HVAC load. |
| `obs_diffuse_solar_irradiance[_predicted_1/2/3]` | W/m² | Scattered sunlight (cloudy-sky component). |
| `obs_direct_solar_irradiance[_predicted_1/2/3]` | W/m² | Direct beam sunlight. **Key signal**: high direct irradiance → lots of free PV energy available. |

> **Why are weather observations relevant?** Solar irradiance directly determines how much free PV energy is available for charging. An intelligent agent could learn to pre-charge the EV before clouds arrive, or to discharge to the grid during peak solar to avoid PV curtailment.

#### Energy (5)

| Column | Units | Description |
|--------|-------|-------------|
| `obs_carbon_intensity` | kgCO₂/kWh | Carbon intensity of grid electricity right now. Lower = greener. Useful for carbon-aware charging. |
| `obs_non_shiftable_load` | kWh | Building's baseline electricity demand this hour (lights, appliances). The agent **cannot** shift this load. |
| `obs_solar_generation` | kWh | PV panel output this hour. Free energy if used; wasted if not. |
| `obs_electrical_storage_soc` | 0–1 | Home battery state of charge (0 = empty, 1 = full). |
| `obs_net_electricity_consumption` | kWh | Net grid draw: positive = importing from grid, negative = exporting back. **This is what the reward function penalizes.** |

#### Pricing (4)

| Column | Units | Description |
|--------|-------|-------------|
| `obs_electricity_pricing` | $/kWh | Current grid electricity price. |
| `obs_electricity_pricing_predicted_1` | $/kWh | Forecast: next hour. |
| `obs_electricity_pricing_predicted_2` | $/kWh | Forecast: 2 hours ahead. |
| `obs_electricity_pricing_predicted_3` | $/kWh | Forecast: 3 hours ahead. |

> **Why pricing matters:** Time-of-use pricing makes V2G profitable. An agent can learn to charge when prices are low and discharge during peak-price hours.

#### EV Charger — `charger_5_1` (7) — the most domain-specific group

These are the columns most likely to confuse newcomers, because CityLearn uses **sentinel values** (−0.1, −1) to indicate "no car connected".

| Column | Values | Description |
|--------|--------|-------------|
| `obs_electric_vehicle_charger_charger_5_1_connected_state` | 0 or 1 | **Is an EV currently plugged in?** 0 = no car (any charging action is wasted), 1 = car connected. |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_departure_time` | hour (1–24) or −1 | **When will the connected EV leave?** −1 = no car. E.g., 8 means departure at 8 AM. Critical for knowing how much time remains to charge. |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_required_soc_departure` | 0–1 or −0.1 | **What SOC does the driver need at departure?** −0.1 = no car. E.g., 0.8 means the EV must be at 80% when it leaves. Failing this is a service violation. |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_soc` | 0–1 or −0.1 | **Current EV battery level.** −0.1 = no car. E.g., 0.35 = 35% charged. |
| `obs_connected_electric_vehicle_at_charger_charger_5_1_battery_capacity` | kWh or −0.1 | **EV battery size.** −0.1 = no car. In our dataset always 40 kWh when a car is connected (Electric_Vehicle_1). |
| `obs_electric_vehicle_charger_charger_5_1_incoming_state` | 0 or 1 | **Is a new EV about to arrive this timestep?** Provides one-step lookahead so the agent can prepare. |
| `obs_incoming_electric_vehicle_at_charger_charger_5_1_estimated_arrival_time` | hour (1–24) or −1 | **When will the incoming EV arrive?** −1 = no incoming car. |

> **Why sentinel values?** When no EV is plugged in, fields like SOC and departure time are undefined. CityLearn could have used `NaN`, but it chose −0.1 (for [0,1] fields) and −1 (for hour fields) so the values stay numeric and don't break vector operations. Encoder rules in [`configs/encoders/default.json`](configs/encoders/default.json) handle these specially (`NormalizeWithMissing` maps −0.1 → 0.0, `OnehotEncoding` gives −1 its own class). Since our RBC uses **raw** observations, the sentinels appear as-is in the CSV — any model trained on this data must handle them.

### Action Columns (2)

| Column | Device | Range | RBC behaviour |
|--------|--------|-------|---------------|
| `action_electrical_storage` | Home battery (6.4 kWh, 5 kW nominal) | [−1, 1] | Slow night charge (+0.091), slow daytime discharge (−0.08) |
| `action_electric_vehicle_storage_charger_5_1` | EV charger (7.4 kW, V2G-capable) | [−1, 1] | Hour-dependent: morning charge (+1.0), midday V2G discharge (−1.0), evening discharge (−0.6), nighttime charge (+0.8) |

The actual power delivered is `action × max_power`. Sign convention: **positive = charge** (energy into storage), **negative = discharge** (energy out, possibly back to grid for V2G).

### Reward

| Column | Type | Description |
|--------|------|-------------|
| `reward` | float | CityLearn's base `RewardFunction` returns approximately **−1 × net_electricity_consumption**. Negative reward = building consumed grid electricity this hour. Positive reward = building exported electricity. The future ORL model will try to maximize this (i.e., minimize grid consumption). |

> **Important:** the agent does **not** compute the reward — CityLearn does. The reward function class is set in the config (`simulator.reward_function: RewardFunction`), wired via [`reward_function/__init__.py`](reward_function/__init__.py)'s `REWARD_FUNCTION_MAP`, and called automatically inside `env.step()`.

### Next-Observation Columns (35) — prefixed `next_obs_`

Identical structure to `obs_*` but representing the state **after** the action was applied and one hour passed. This is what makes the dataset usable for offline RL algorithms like CQL, IQL, or TD3+BC, which need full $(s, a, r, s')$ tuples to estimate value functions.

For pure Behavioral Cloning (Milestone 2), only the `obs_*` and `action_*` columns are used.

### Done Flag

| Column | Values | Description |
|--------|--------|-------------|
| `done` | 0 or 1 | 1 only at the last step of each episode (timestep 8,758). All other steps are 0. Marks episode boundaries. |

### How to Load the Dataset

```python
import pandas as pd

df = pd.read_csv("datasets/offline_rl/offline_dataset.csv")

obs_cols = [c for c in df.columns if c.startswith("obs_")]
action_cols = [c for c in df.columns if c.startswith("action_")]
next_obs_cols = [c for c in df.columns if c.startswith("next_obs_")]

observations = df[obs_cols].values        # (26277, 35)
actions = df[action_cols].values          # (26277, 2)
rewards = df["reward"].values             # (26277,)
next_observations = df[next_obs_cols].values  # (26277, 35)
terminals = df["done"].values             # (26277,)
```

---

## Milestone 2 (Upcoming) — Behavioral Cloning Model

The dataset above is the input for the next milestone: train an Offline RL model that imitates the RBC's behaviour. We start with **Behavioral Cloning (BC)** — the simplest possible offline algorithm — to validate the full training & evaluation pipeline before moving on to more advanced methods (IQL, CQL).

### What is Behavioral Cloning?

BC treats offline RL as **supervised regression**: given a dataset of $(s, a)$ pairs, train a neural network $\pi_\theta$ to predict $a$ from $s$ by minimizing
$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \|\pi_\theta(s_i) - a_i\|^2.$$

It does **not** use the reward, the next state, or the done flag — only observations and actions. This means BC can never beat the behaviour policy (it just imitates), but it's the right baseline: if BC can't even reproduce the RBC, no fancier algorithm will work either.

### Library choice — PyTorch

We'll use **PyTorch** (already in [`requirements.txt`](requirements.txt#L94) at version `2.5.1+cu121`) instead of a specialized RL library like `d3rlpy`, because:

- It's already used everywhere in the repo (MADDPG agent, all networks).
- BC is just supervised regression — no need for a heavy RL framework.
- Full control over the training loop = transparency for learning.
- The trained model integrates naturally with `BaseAgent`, which already inherits from `torch.nn.Module`.

### Architecture choices and the reasoning behind them

A small Multi-Layer Perceptron (MLP): 35 inputs → 256 hidden → 256 hidden → 2 outputs.

| Choice | Reasoning |
|--------|-----------|
| **2 hidden layers** | One layer is too shallow to capture interactions between features (e.g., "EV connected AND solar high AND morning"). Three+ layers risk overfitting on 26K samples. Two is the sweet spot for this problem size. |
| **256 neurons per layer** | 35 inputs × 256 ≈ 9K parameters per layer. With 26,277 training samples, we have ~10× more samples than parameters per layer — a safe ratio. Smaller (64–128) might underfit; larger (512+) wastes capacity and slows training. |
| **ReLU activation** (hidden) | The default for hidden layers. Computationally cheap, doesn't suffer from vanishing gradients like sigmoid/tanh, and works empirically across nearly every architecture. |
| **Tanh activation** (output) | Tanh outputs values in [−1, 1], which **exactly matches our action space** (electrical_storage and electric_vehicle_storage are both in [−1, 1]). Without it, the network could output e.g. 47.3 and CityLearn would clip — wasting learning signal. Tanh enforces the constraint as part of the model. |

### Training hyperparameters and the reasoning behind them

| Hyperparameter | Value | Reasoning |
|----------------|-------|-----------|
| **Learning rate** | `3e-4` | The "[Karpathy constant](https://twitter.com/karpathy/status/801621764144971776)" — the safest default for Adam across most deep learning tasks. High enough to train fast, low enough to avoid divergence. |
| **Batch size** | `256` | Big enough that gradient estimates are stable (low noise), small enough to fit easily in memory and to give many updates per epoch (26,277 ÷ 256 ≈ 103 updates/epoch). Smaller = noisier but more frequent updates; larger = smoother but fewer updates. 256 balances both. |
| **Epochs** | `50` | Each epoch sees all samples once. With our small model and plenty of data, the loss typically plateaus by epoch 30–50. Adding early stopping later is easy if needed. |
| **Optimizer** | Adam | Adapts the learning rate per parameter automatically. Works out-of-the-box for ~95% of deep learning problems. SGD with momentum can sometimes generalize better but requires careful tuning. |
| **Loss** | MSE | $\frac{1}{N}\sum (a_{pred} - a_{true})^2$. Standard for continuous regression. Penalizes big errors more than small ones — being off by 1.0 is much worse than being off by 0.1, which matches our intuition. |
| **Train / val split** | 80 / 20 **by episode** | Episodes 0 & 1 → training, episode 2 → validation. We do **not** shuffle rows: hour $t$ and $t+1$ are nearly identical, so a random row split would let the model "cheat" by memorizing instead of generalizing. Episode-level split tests true generalization to a fresh year. |

### Input standardization — why it matters

The 35 input features have wildly different scales:

| Feature | Typical range |
|---------|---------------|
| `electrical_storage_soc` | 0 to 1 |
| `outdoor_dry_bulb_temperature` | −10 to 40 (°C) |
| `solar_generation` | 0 to ~5000 (W/m²) |
| `electricity_pricing` | 0.05 to 0.35 ($/kWh) |
| `month` | 1 to 12 |

If we feed raw values to a neural network, two bad things happen:

1. **Domination** — At initialization, the network treats inputs equally, so a feature with values around 5000 produces activations 5000× larger than one around 0.5. Gradient updates are dominated by large-scale features, and small-scale ones (like SOC, **which is critical for EV charging**) are essentially ignored for many epochs.

2. **Slow convergence** — Gradient descent works best when the loss landscape is roughly spherical. Wildly different scales create elongated, narrow ravines in the loss surface. The optimizer bounces back and forth in the steep direction while making slow progress in the shallow direction.

**The fix — standardization:**

For each feature, compute mean $\mu$ and standard deviation $\sigma$ from the **training set only**, then transform every value:
$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

After this, every feature has mean 0 and standard deviation 1. The network sees consistent scales, gradients are balanced, and training becomes fast and stable.

> **Why "training set only"?** Computing stats over the entire dataset (including validation) leaks information from the validation set into training — we'd subtly cheat on our generalization test. Compute $\mu, \sigma$ from training, then **apply the same fixed values** to validation and inference. This is standard ML hygiene.
>
> **Why save the stats with the model?** At inference time the BC agent gets raw observations. To reproduce training behavior, those observations must be standardized using the **exact same** $\mu$ and $\sigma$. The stats become part of the model artifact — without them, the model is useless.

### Planned file structure (Milestone 2)

```
algorithms/
├── agents/
│   └── offline_bc_agent.py            ← BaseAgent that loads & runs trained BC
└── offline/                           ← Offline RL infrastructure
    ├── __init__.py
    ├── bc_policy.py                   ← MLP architecture (shared by trainer & agent)
    ├── bc_trainer.py                  ← Training loop logic
    └── data_loader.py                 ← CSV loading, episode split, standardization

scripts/
└── train_offline_bc.py                ← CLI entrypoint for training

configs/templates/
├── train_offline_bc.yaml              ← Training hyperparameters
└── eval_offline_bc_local.yaml         ← Agent config for in-simulator evaluation

runs/offline_bc/<run_id>/              ← Training output
├── model.pth                          ← Trained weights
├── normalization_stats.json           ← μ, σ per feature + action info
├── training_metadata.json             ← Hyperparameters, final losses, dataset path
├── loss_history.json                  ← Per-epoch train/val loss
└── loss_curve.png                     ← Plot of training & validation loss
```

After training, the BC agent can be evaluated under identical CityLearn conditions as the RBC by running:
```bash
python run_experiment.py --config configs/templates/eval_offline_bc_local.yaml --job_id bc-v1-eval
```

This makes **direct comparison** (RBC vs BC under the same dataset, same reward, same buildings) possible — the central reason for going through the full platform integration in Milestone 1.
