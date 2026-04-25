# Offline RL Milestone 2 — Plan

## Goal

Collect a richer offline-RL dataset by running the existing **RBC behaviour
policy** (`RuleBasedPolicy`) over **all buildings** in
`citylearn_three_phase_dynamic_topology_demo_v1`, across **10 episodes** with
**noise injection on 3 of them**, capturing **every observation field** plus
per-building reward and properly-separated `terminated`/`truncated` flags.

Output: one CSV per building under the runner's job dir.

This addresses the M1 conclusions (`docs/offline_rl/offline_m1_conclusions.md`):
- A.1: dataset narrowness → noise + more episodes.
- A.3: single-building specialist → district-wide collection.
- B.5: collapsed `done` flag → separate `terminated` and `truncated`.
- C.12: weak reward → use `V2GPenaltyReward`.

---

## Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | CSV layout | One CSV per building |
| 2 | Noise strategy | Per-episode mode mix: 7 clean + 3 noisy |
| 3 | Noise sigma | 0.1 (Gaussian, clipped to `[-1, 1]`) |
| 4 | Episodes | 10 |
| 5 | Reward | `V2GPenaltyReward` |
| 6 | Termination flags | Separate `terminated` and `truncated` columns |
| 7 | Per-building scope | Strict — no district aggregates inside per-building CSVs |
| 8 | Output location | `runs/jobs/<job_id>/` via `export_artifacts()` |
| 9 | Agent class | New: `DistrictDataCollectionRBC` (composes `RuleBasedPolicy`) |
| 10 | Reproducibility | Single base seed in config, per-episode RNG = `seed + episode_index` |
| 11 | `topology_version` column | Included in every row |
| 12 | Reward fallback | **Fail-fast.** If `V2GPenaltyReward` fails to initialise/probe, abort with a clear message instructing the user to re-run with `simulator.reward_function: RewardFunction`. **No mid-run reward swap; the same reward function is used for every step of every episode.** |

---

## Environment Notes

- Dataset: `datasets/citylearn_three_phase_dynamic_topology_demo_v1/`
  - `interface: "entity"`, `topology_mode: "dynamic"`.
  - `simulation_end_time_step: 8759` → 8,759 steps per episode.
- Wrapper auto-detects entity mode and rebuilds layout on topology changes
  (`utils/wrapper_citylearn.py:148-172, 309-333`).
- `MADDPG` is forbidden in dynamic mode; `RuleBasedPolicy` works.
- Total transitions ≈ 10 × 8,759 × N_buildings (one row per building per step in
  that building's file).

---

## Per-Building CSV Schema

For a building `b` with observation names `O_b` and action names `A_b`:

| Group | Columns | Count |
|---|---|---|
| Metadata | `episode`, `timestep`, `topology_version`, `policy_mode`, `noise_sigma_applied` | 5 |
| Observations | `obs_<name>` for every name in `O_b` | \|O_b\| |
| Actions (executed by env) | `action_<name>` for every name in `A_b` | \|A_b\| |
| Actions (clean RBC reference) | `action_clean_<name>` for every name in `A_b` | \|A_b\| |
| Reward | `reward` | 1 |
| Next observations | `next_obs_<name>` for every name in `O_b` | \|O_b\| |
| Termination | `terminated`, `truncated` | 2 |

`policy_mode` ∈ {`clean`, `noisy`}. `noise_sigma_applied` is `0.0` on clean
episodes and equals `noise_sigma` on noisy episodes.

Logging both `action` (executed, possibly noisy) and `action_clean` (RBC's
intended action) gives downstream BC the choice: imitate executed actions or
imitate the clean reference.

---

## Noise Mechanics (precise)

```
rng_episode = np.random.default_rng(seed + episode_index)

for each step:
    a_clean = rbc.predict(obs)                         # list[list[float]] over agents
    if episode_index in noisy_episode_indices:
        for each agent i:
            noise_i = rng_episode.normal(0, sigma, size=len(a_clean[i]))
            a_executed[i] = clip(a_clean[i] + noise_i, -1.0, 1.0)
        policy_mode = "noisy"; noise_sigma_applied = sigma
    else:
        a_executed = a_clean
        policy_mode = "clean"; noise_sigma_applied = 0.0
    env.step(a_executed)
    log per-building row with both a_executed and a_clean
```

---

## Output Layout

```
runs/jobs/<job_id>/
├── offline_dataset_building_<id>.csv  (one per active building)
├── dataset_metadata.json              ← seed, sigma, noisy_episodes,
│                                        episodes, schema, reward_fn,
│                                        per-building row counts,
│                                        obs/action names, topology versions seen
├── config.resolved.yaml
└── artifact_manifest.json             ← N entries, one per CSV, format=offline_dataset
```

---

## Files to Create / Modify

**New:**
- `algorithms/agents/district_data_collection_agent.py`
- `configs/templates/district_data_collection_local.yaml`
- `docs/offline_rl/m2/offline_rl_m2_plan.md` (this file)

**Modified:**
- `algorithms/registry.py` — register `DistrictDataCollectionRBC`.
- `utils/config_schema.py` — Pydantic model + Union extension.
- `utils/bundle_validator.py` — only if the N-CSV vs M-agent count gate trips.

---

## Validation Steps (post-Build)

1. Run:
   ```bash
   python run_experiment.py \
     --config configs/templates/district_data_collection_local.yaml \
     --job_id offline-data-collection-v2
   ```
2. Assert one CSV per active building exists in `runs/jobs/offline-data-collection-v2/`.
3. For one CSV: row count ≈ `episodes * 8759`, `episode` ranges 0-9,
   `policy_mode` is "clean" for episodes {0..6} and "noisy" for {7,8,9},
   `noise_sigma_applied` matches `policy_mode`.
4. `terminated`/`truncated` present; `terminated.sum() + truncated.sum() == episodes`.
5. `action_clean_*` ≠ `action_*` (within tolerance) only on noisy-episode rows;
   equal on clean rows.
6. Noisy-episode action distribution shifts visibly vs clean, all within `[-1, 1]`.
7. `dataset_metadata.json` is well-formed and reproducible (same seed → identical
   noisy actions).
8. Bundle validator passes.
9. `topology_version` column exists and is monotonically non-decreasing per episode.

---

## Out of Scope

- Feature engineering / column pruning (post-collection step).
- Training BC / IQL / TD3+BC.
- District-level reward column.
- ONNX / inference bundle export.
- Multi-seed evaluation / random-policy baseline.
