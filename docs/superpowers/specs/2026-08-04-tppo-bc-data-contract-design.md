# TPPO BC Data Contract And Setup Visibility Design

## Purpose

Fix two defects that stopped the Wave A `tppo-recovery-wa-tppo-bc-pretrain-s7`
run:

1. Behavior-cloning demonstrations were recorded in one representation and
   validated against another. Every current-topology sample was discarded and
   pretraining trained on nothing.
2. After episode 1 completed, the job stopped progressing before episode 2
   began. The stall watchdog was inactive during setup, so no stack trace
   located the block.

Add a small local validation harness that reproduces the full pipeline before
any server run.

## Non-Goals

- No PPO or critic changes.
- No new BC algorithm.
- No dataset changes.
- No new export or KPI logic.

## Observed Evidence

- `runs/Results/tppo-recovery-wa-tppo-bc-pretrain-s7.log` prints one warning
  per building at `19:09:26-28`. Every building reports the current topology
  as incompatible.
- The same file prints `Completed episode 1/3` at `19:09:28.987`.
- No `Episode: 2/3` line, no traceback, and no exit follow.
- Configured pretraining is 35,040 samples per building, 4 epochs, batch 64.
- Pretraining actually executed zero optimizer batches because no sample
  passed the shape check.

## Root Cause

### Data Contract Defect

`AgentTransformerPPO.update()` records demonstrations with the wrapper-
supplied encoded observation. `_run_bc_pretraining()` derives the expected
observation width from `state.layout` at the current topology using
`_infer_obs_dim`. The stored vector is the encoded model input; the expected
width is the raw entity-adapter layout width. These differ when the wrapper
uses `minmax_space` normalization or when topology has changed since the
sample was recorded. Every sample fails the check.

Additionally, pretraining rejects every historical topology group even when
its stored layout is internally consistent. This discards otherwise usable
demonstrations after a topology change.

### Visibility Defect

The wrapper arms the stall watchdog on any `*_start` phase and cancels on
any `*_end` phase. Episode reset, entity layout adaptation, and manual KPI
export are already bracketed by phase progress writes, so a hang inside them
does produce a watchdog trace when the watchdog is enabled.

The remaining uncovered code is:

- `model.on_episode_start(...)`;
- `model.on_episode_end(...)`;
- `_attach_model_environment_metadata(...)` calls that run outside a bracketed
  phase (initial reset in `__init__`, mid-step topology reattach).

The failing run also lacks proof that `stall_watchdog_enabled: true` and a
finite `stall_watchdog_timeout_seconds` were set. A watchdog stack trace was
not captured. Server configuration must set both explicitly, and the
callbacks must be bracketed for coverage.

### Silent Empty Pretraining

`_run_bc_pretraining` completes without error when every group is rejected.
PPO then trains from an uninitialized actor with no supervised warm-up. The
run continues without the intended teacher influence.

## Corrective Contract

### Demonstration Record

`Demonstration` gains one field:

- `encoded_length: int`

`record_demonstration` derives `encoded_length` from the stored vector.
`record_demonstration` refuses to store a sample whose observation length
differs from the current layout's encoded model input length. Rejected
samples increment a `rejected_at_record` counter and log once per building.

### Pretraining Grouping

`_run_bc_pretraining` iterates over every group in
`demonstrations_for_building_by_signature(building_idx)`. For each group it:

- reads `demo.layout` (immutable per demonstration);
- validates group internal consistency: every sample has the same
  `encoded_length` and the same layout signature;
- trains against that stored layout, not the current `state.layout`.

The historical-topology warning is removed. A per-group summary log takes
its place:

```
BC pretraining: building=<id> group=<signature> samples=<n>
    batches=<n> epochs=<n>
```

`_infer_obs_dim` is not used for validation. Layouts control their own
encoded width.

### Empty-Sample Failure

After collection ends and before pretraining starts,
`_run_bc_pretraining` counts usable samples per building. If any building
has zero usable samples, the agent raises:

```
BC pretraining has zero compatible demonstrations for Building_<id>.
Check demonstration_episodes, teacher policy, and record_demonstration.
```

The run fails fast. It does not enter PPO with an untrained actor.

### Pretraining Progress Logs

`_run_bc_pretraining` logs at these points:

- start: total buildings, total usable samples, total planned batches;
- per building at start: usable samples, planned batches, planned epochs;
- per epoch at end: building id, epoch index, mean loss, batches trained;
- end: total batches trained, total buildings trained.

Progress log lines use `INFO` level.

### Watchdog Coverage For Setup

Bracket the two agent lifecycle callbacks and any out-of-loop metadata
attachment with `_write_phase_progress` pairs:

- `episode_start_callback_start` / `episode_start_callback_end` around
  `model.on_episode_start(...)`;
- `episode_end_callback_start` / `episode_end_callback_end` around
  `model.on_episode_end(...)`;
- `model_attach_start` / `model_attach_end` around every explicit
  `_attach_model_environment_metadata()` call, including the one in
  `__init__` after the initial `env.reset()`.

Wave A server configurations set `tracking.stall_watchdog_enabled: true`
and a finite `tracking.stall_watchdog_timeout_seconds`. The canary and smoke
configurations set the same.

## Local Validation Harness

Two configuration files under `configs/recovery/tppo/wave_a/local/`:

### Canary Configuration

`tppo_bc_pretrain_canary.yaml`:

- 2 buildings;
- 3 episodes (1 demo, 1 PPO, 1 deterministic);
- 16 simulator steps per episode;
- CPU only, `require_cuda: false`;
- `max_samples_per_building: 16`;
- `pretraining_epochs: 1`;
- `batch_size: 4`;
- one intentional topology change between step 8 and step 9.

The canary uses a synthetic entity payload fixture so the check runs without
the full CityLearn dataset.

### Smoke Configuration

`tppo_bc_pretrain_smoke.yaml`:

- Full CityLearn dynamic 15-minute dataset;
- first 192 steps per episode;
- 3 episodes;
- `max_samples_per_building: 128`;
- `pretraining_epochs: 1`;
- `batch_size: 16`;
- CPU or a single local CUDA device.

### Pass Criteria

Both configurations must produce:

- `Completed episode 3/3`;
- pretraining log lines with nonzero usable samples;
- pretraining log lines with nonzero trained batches per building;
- no `Skipping behavior-cloning demonstrations` warning;
- no stall watchdog activation.

An intentional test stall (unit test only) must produce a watchdog stack
trace and identify the stuck phase.

## Configuration Reduction For Wave A

`configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml` and
`tppo_bc_auxiliary.yaml` change:

- `max_samples_per_building: 35040` becomes `max_samples_per_building: 4096`;
- `pretraining_epochs: 4` becomes `pretraining_epochs: 2`;
- `batch_size: 64` remains.

This bounds pretraining at a maximum of 128 optimizer batches per building.
Wave A is a screening run. Full-sample pretraining, if justified, follows
after the canary and smoke pass.

## Success Criteria

- All new unit and integration tests pass under `pytest -q`.
- The canary run completes locally with the pass criteria above.
- The smoke run completes locally with the pass criteria above.
- A deliberate stall in a unit test produces a watchdog stack trace.
- The server BC run reaches `Completed episode 3/3` and exports final KPIs.

## Test Coverage

- `test_behavior_cloning_regularizer.py`:
  - `record_demonstration` refuses shape-mismatched samples;
  - `Demonstration` stores `encoded_length`;
  - grouping returns every stored signature.
- `test_agent_transformer_ppo_behavior_cloning.py`:
  - stored encoded vector equals the vector `predict` consumes;
  - pretraining trains every stored signature group;
  - pretraining raises on zero usable samples for any building;
  - pretraining logs planned and completed batch counts.
- `test_agent_transformer_ppo_wrapper_integration.py`:
  - watchdog arms during `on_episode_start` callback;
  - watchdog arms during `on_episode_end` callback;
  - watchdog arms during out-of-loop `_attach_model_environment_metadata`;
  - watchdog stack trace names the stuck phase for a deliberate hang.
- `test_tppo_recovery_wave_a_configs.py`:
  - Wave A BC configs have `max_samples_per_building: 4096` and
    `pretraining_epochs: 2`;
  - canary and smoke configs exist and pass schema validation.

## Migration And Backward Compatibility

Existing checkpoints without `encoded_length` are unsupported. BC checkpoint
resume raises a clear error:

```
Checkpoint predates BC data contract. Re-collect demonstrations under the
current representation before resuming.
```

Non-BC TPPO checkpoints are unaffected.

## Rollout Order

1. Land data contract, empty-sample failure, and unit tests.
2. Land pretraining group logs and watchdog coverage.
3. Land canary and smoke configs and their tests.
4. Run `pytest -q`.
5. Run canary locally.
6. Run smoke locally.
7. Commit and push for the server run.
