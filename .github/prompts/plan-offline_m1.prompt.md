## Plan: Offline RL Data-Collection Pipeline (Milestone 1)

Implement a new `BaseAgent`-compatible RBC (porting `BasicElectricVehicleRBC_ReferenceController`'s hour-based action map) that records all transitions for Building 5 into a CSV during the standard `run_experiment.py` flow. The agent uses `use_raw_observations = True`, captures `(obs, action, reward, next_obs, done)` in `update()`, and writes the CSV in `export_artifacts()`. A new config template, registry entry, and schema model complete the integration.

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **D1 — Policy** | Port `BasicElectricVehicleRBC_ReferenceController`'s hour→action map into a `BaseAgent` subclass; extract `hour` index from raw observation names in `attach_environment()` | Reuses your coordinator's proven EV reference heuristic while fitting the platform contract |
| **D2 — Reward** | Use CityLearn's default reward (`reward_function: default` in config) | Reward is injected at the CityLearn env level, not the agent level — `env.step()` returns it automatically. Avoids `V2GPenaltyReward`'s complex observation-dict dependencies for this first milestone |
| **D3 — Episodes** | **3 episodes × 8760 steps = 26,280 transitions** | CityLearn's EV arrivals/departures may have stochasticity across episodes; 3 episodes provide enough data for offline RL while keeping the dataset small (~2 MB CSV). The Q&A doc recommends multiple episodes |
| **D4 — CSV schema** | `episode, timestep, obs_<name>_0 … obs_<name>_N, action_0 … action_K, reward, next_obs_<name>_0 … next_obs_<name>_N, done` — raw observation values for agent 4 only | Raw format preserves maximum information; named columns enable the future offline RL model to select its own encoding |
| **D5 — Collection point** | Inside the new agent's `update()` — it already receives all five tuple elements | Natural hook; no wrapper modification needed; `export_artifacts()` writes the accumulated buffer to CSV |
| **D6 — Config** | New template + new `AlgorithmConfig` variant discriminated on `algorithm.name = "EVDataCollectionRBC"` | Follows the existing pattern (`RuleBasedAlgorithmConfig`, `MADDPGAlgorithmConfig`); keeps all changes additive |

### Steps

1. **Create the agent** in `algorithms/agents/ev_data_collection_agent.py` — a `BaseAgent` subclass with `use_raw_observations = True`. In `attach_environment()`, store `observation_names[4]` and `action_names[4]` (agent index 4 = Building 5) plus derive the `hour` index for action-map lookup. In `predict()`, replicate the `BasicElectricVehicleRBC_ReferenceController` hour→action map logic (look up hour from raw obs, return mapped values for every agent — only agent 4's are meaningful). In `update()`, append the Building 5 slice `(obs[4], actions[4], rewards[4], next_obs[4], terminated or truncated)` to an in-memory list. In `export_artifacts()`, write the list to `<output_dir>/offline_dataset.csv` with named columns and return `{"format": "offline_dataset", "artifacts": [{"agent_index": 4, "path": "offline_dataset.csv", "format": "csv"}]}`.

2. **Register the agent** in `algorithms/registry.py` — add `"EVDataCollectionRBC": EVDataCollectionRBC` to `ALGORITHM_REGISTRY` and import the class.

3. **Add schema validation** in `utils/config_schema.py` — create `EVDataCollectionRBCAlgorithmConfig` (minimal: just `name` literal + optional `hyperparameters` with `target_building_index: int = 4`). Add it to the `AlgorithmConfig` discriminated union on `ProjectConfig`.

4. **Update bundle validator** in `utils/bundle_validator.py` — add `"offline_dataset"` and `"csv"` to the set of accepted artifact formats so `validate_bundle_contract()` doesn't reject the CSV artifact.

5. **Create the config template** at `configs/templates/ev_data_collection_local.yaml` — point `simulator.dataset_path` to `datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json`, set `simulator.episodes: 3`, `algorithm.name: EVDataCollectionRBC`, `reward_function: default`, disable checkpointing and MLflow, keep tracking as JSONL.

6. **Validate end-to-end** — run `python run_experiment.py --config configs/templates/ev_data_collection_local.yaml --job_id offline-data-collection` and verify the CSV appears at `runs/jobs/offline-data-collection/offline_dataset.csv` with ≈26,280 rows and 19 obs columns + action columns + reward + next_obs + done + metadata.

### Further Considerations

1. **Multi-agent actions for non-target buildings**: the RBC must return valid actions for all 17 agents (CityLearn requires it), but only Building 5 transitions are logged. Use the same hour-based map for all buildings — `attach_environment()` provides per-agent action names, so the map adapts per building's available devices automatically.
2. **Noise injection for dataset diversity**: the offline RL Q&A recommends 70% clean + 30% noisy policy. This can be a follow-up enhancement (add Gaussian noise to actions with a config flag) after confirming the deterministic pipeline works.
3. **File destination**: this plan should be saved to `docs/offline_rl/offline_rl_m1_plan.md` once approved.
