# Simulator Limits (Phase 1)

As of March 10, 2026, this repository treats the simulator as an external dependency and does **not** modify simulator internals in this phase.

## Current limits (relevant to product)

1. No full snapshot/restore API for mid-episode resume.
- Checkpoint resume currently restores **agent state only** (weights, optimizers, replay buffer when available).
- Environment state is re-created from the beginning of the episode.

2. `time_step_ratio` is part of the wrapper-agent contract.
- CityLearn wrappers and scheduler logic assume `time_step_ratio` is available and coherent with environment stepping.
- Test fixtures and mock environments must define `time_step_ratio` to match runtime behavior.

3. Runtime contract is bundle-first, not environment-state-first.
- Reproducibility in this phase is achieved through configs + artifacts + manifest, not through simulator state snapshots.

## Phase 2 backlog (simulator-facing)

1. True resume support
- Add simulator APIs to capture/restore full environment state (including all dynamic entities and internal counters).
- Add versioned snapshot format and compatibility checks.

2. Deterministic replay guarantees
- Formalize random seed handling across simulator + wrappers + algorithms.
- Add replay validation tests for resumed runs.

3. Time-step contract hardening
- Formalize `time_step_ratio` and related cadence fields in simulator docs and schema.
- Add runtime assertions for incompatible configs.

4. Operational tooling
- Add snapshot integrity checks and migration tooling between simulator versions.
