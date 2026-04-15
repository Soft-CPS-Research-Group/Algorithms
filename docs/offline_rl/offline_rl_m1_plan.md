## Plan: Offline RL Data-Collection Pipeline (Milestone 1)

Implement a new `BaseAgent`-compatible RBC that ports `BasicElectricVehicleRBC_ReferenceController`'s hour-based action map, runs it through the full `run_experiment.py` → `Wrapper_CityLearn` → `CityLearnEnv` integration targeting **Building 5** (agent index 4, 0-indexed), and records every transition `(obs, action, reward, next_obs, done)` into a single CSV for offline RL training. The agent sets `use_raw_observations = True` so all 35 raw observations are preserved in the dataset. Data is accumulated in `update()` and flushed in `export_artifacts()`.

### Dataset & Environment Context

The CityLearn environment is defined by `datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json`, which references 17 buildings. Building 5's data comes from:

- **`Building_5.csv`** — 8760 hourly rows of energy simulation data (loads, solar generation, temperatures, etc.)
- **`charger_5_1.csv`** — EV charger time series (connection state, departure/arrival events, SOC)

**Charger 5_1 specs**: 7.4 kW max charge, 0 kW min charge, 7.4 kW max discharge (V2G-capable), 0.95 efficiency.

**Building 5 raw observations (35 total)**:

| Group | Observations | Count |
|-------|-------------|-------|
| Temporal | `month`, `day_type`, `hour`, `daylight_savings_status` | 4 |
| Weather | `outdoor_dry_bulb_temperature` (+predicted 6h/12h/24h), `outdoor_relative_humidity` (+predicted 6h/12h/24h), `diffuse_solar_irradiance` (+predicted 6h/12h/24h), `direct_solar_irradiance` (+predicted 6h/12h/24h) | 16 |
| Energy | `carbon_intensity`, `non_shiftable_load`, `solar_generation`, `electrical_storage_soc`, `net_electricity_consumption` | 5 |
| Pricing | `electricity_pricing`, `electricity_pricing_predicted_1`, `electricity_pricing_predicted_2`, `electricity_pricing_predicted_3` | 4 |
| **Subtotal: base** | | **28** |
| Charger (`charger_5_1`) | `electric_vehicle_charger_charger_5_1_connected_state`, `connected_electric_vehicle_at_charger_charger_5_1_departure_time`, `connected_electric_vehicle_at_charger_charger_5_1_required_soc_departure`, `connected_electric_vehicle_at_charger_charger_5_1_soc`, `connected_electric_vehicle_at_charger_charger_5_1_battery_capacity`, `electric_vehicle_charger_charger_5_1_incoming_state`, `incoming_electric_vehicle_at_charger_charger_5_1_estimated_arrival_time` | 7 |

> Note: although `offline_rl_info.md` originally listed 19 observation slots (12 common + 7 charger), that count excluded weather/irradiance features. CityLearn's `schema.json` marks all 28 base observations as active for every building. The encoder rules in `configs/encoders/default.json` would remove 12 weather features — but since the RBC sets `use_raw_observations = True`, it receives all 35 values unmodified. The CSV stores these raw values so the future offline RL model can choose its own encoding.

**Building 5 actions (2)**:
- `electrical_storage` — battery charge/discharge (continuous, bounded by ±5 kW nominal)
- `electric_vehicle_storage` — EV charge/discharge via `charger_5_1` (continuous, bounded by ±7.4 kW)

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **D1 — Policy** | Port `BasicElectricVehicleRBC_ReferenceController`'s hour→action map into a `BaseAgent` subclass; extract `hour` index from raw observation names via `attach_environment()` | Reuses the coordinator's proven EV reference heuristic while fitting the platform's `BaseAgent` contract |
| **D2 — Reward** | Use CityLearn's base reward (`reward_function: RewardFunction` in config) | Reward is computed inside `env.step()` automatically. `RewardFunction` is CityLearn's base class — simple, interpretable, and already registered in `run_experiment.py`'s `REWARD_FUNCTION_MAP`. Avoids `V2GPenaltyReward`'s complex observation-dict dependencies for this first milestone |
| **D3 — Episodes** | **3 episodes × 8760 steps = 26,280 transitions** | CityLearn's EV arrivals/departures have stochasticity across episodes; 3 episodes provide diversity for offline RL while keeping the dataset small (~5 MB CSV). The offline RL Q&A doc recommends multiple episodes for better coverage |
| **D4 — CSV schema** | `episode, timestep, obs_<name>_0 … obs_<name>_34, action_0, action_1, reward, next_obs_<name>_0 … next_obs_<name>_34, done` — raw values for agent 4 (Building 5) only | Raw format preserves all 35 observations; named columns (from `observation_names[4]`) enable the future offline RL model to select its own feature subset and encoding |
| **D5 — Collection point** | Inside the new agent's `update()` method | `update()` already receives `observations, actions, rewards, next_observations, terminated, truncated` — the exact five-tuple needed. No wrapper modification required; `export_artifacts()` writes the accumulated buffer to disk |
| **D6 — Config** | New template + new `AlgorithmConfig` variant discriminated on `algorithm.name = "EVDataCollectionRBC"` | Follows the existing discriminated-union pattern in `config_schema.py`; keeps all changes additive |

### Steps

1. **Create the agent** in `algorithms/agents/ev_data_collection_agent.py` — a `BaseAgent` subclass with `use_raw_observations = True`. In `attach_environment()`, store all agents' `observation_names` and `action_names` (needed for multi-agent `predict()`), and specifically cache agent 4's observation names for CSV column headers plus the `hour` index position for action-map lookup. In `predict()`, replicate `BasicElectricVehicleRBC_ReferenceController`'s hour→action map: for each agent, look up `hour` from its raw observation array, resolve hour candidates (handling 0–23 vs 1–24 encoding), and return the mapped action value per device (`electrical_storage`, `electric_vehicle_storage`, `cooling_device`, etc.) — the map adapts automatically per agent's `action_names`. In `update()`, extract agent 4's slice — `observations[4]`, `actions[4]`, `rewards[4]`, `next_observations[4]`, `terminated or truncated` — and append it to an in-memory list along with episode and timestep counters. In `export_artifacts()`, write the accumulated list to `<output_dir>/offline_dataset.csv` with named columns and return manifest-compatible metadata.

2. **Register the agent** in `algorithms/registry.py` — import `EVDataCollectionRBC` from the new module and add `"EVDataCollectionRBC": EVDataCollectionRBC` to `ALGORITHM_REGISTRY`.

3. **Add schema validation** in `utils/config_schema.py` — create `EVDataCollectionRBCHyperparameters` with `target_building_index: int = 4` and `EVDataCollectionRBCAlgorithmConfig` (with `name: Literal["EVDataCollectionRBC"]` + the hyperparameters model). Add it to the `Union` type in `ProjectConfig.algorithm`.

4. **Update bundle validator** in `utils/bundle_validator.py` — add `"offline_dataset"` to `SUPPORTED_ARTIFACT_FORMATS` so `validate_bundle_contract()` accepts the CSV artifact. Add a format-specific check: for `"offline_dataset"` format, the artifact path must end in `.csv`.

5. **Create the config template** at `configs/templates/ev_data_collection_local.yaml` — set `simulator.dataset_path` to `datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json`, `simulator.episodes: 3`, `simulator.central_agent: false`, `algorithm.name: EVDataCollectionRBC`, `simulator.reward_function: RewardFunction`, disable checkpointing, use JSONL tracking (no MLflow dependency).

6. **Validate end-to-end** — run `python run_experiment.py --config configs/templates/ev_data_collection_local.yaml --job_id offline-data-collection` and verify the CSV appears at `runs/jobs/offline-data-collection/offline_dataset.csv` with ≈26,280 rows, 35 obs columns + 2 action columns + reward + 35 next_obs columns + done + episode/timestep metadata.

### Further Considerations

1. **Multi-agent actions for non-target buildings**: the RBC must return valid actions for all 17 agents (CityLearn requires it), but only Building 5 transitions are logged. The hour-based action map adapts per agent's `action_names` — buildings without chargers simply won't have `electric_vehicle_storage` in their action list and will get only `electrical_storage` actions.
2. **Noise injection for dataset diversity**: the offline RL Q&A recommends 70% clean + 30% noisy policy for richer state-action coverage. This can be added as a follow-up (e.g., a `noise_fraction` hyperparameter that applies Gaussian noise to actions in a random subset of episodes) after confirming the deterministic pipeline works.
3. **V2GPenaltyReward upgrade**: once the basic pipeline is validated with `RewardFunction`, a follow-up milestone can switch to `V2GPenaltyReward` for more nuanced EV-aware reward shaping — the CSV would then capture richer reward signals for offline RL training.
