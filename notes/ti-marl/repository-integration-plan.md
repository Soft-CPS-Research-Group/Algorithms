# Repository integration plan

## Simulator

Add `runtime_status_v1` and `entity_action_execution_v1` as additive entity
contracts, with tests, documentation, release notes and package version 1.7.0.
Algorithms changes its pin only after clean installation validation.

Status (2026-08-17): Simulator 1.7.0 is published, installed from its PyPI
wheel, and validated locally. Algorithms now pins that release in its standard
and Jetson dependency paths.

## Algorithms

- New `algorithms/ti_marl` package and registry entry `TIMARL`.
- Typed config model requiring entity interface and decentralised agents.
- Optional structured observation/transition hooks on `ExecutionUnit`.
- Wrapper retains entity payload and `info` only for units requesting it.
- Existing vector agents and topology behaviour remain unchanged.
- Bundle validator accepts non-deployable `ti_marl_torch` artefacts without a
  fixed artefact-per-current-agent requirement.
- Checkpoints and manifests record all semantic contract versions and hashes.

## Version control and privacy

Design documents, source, tests and generic templates are versioned. Campaign
configs, checkpoints, traces and numerical results remain in ignored local
paths until publication is explicitly authorised. Changes are delivered in
small subsystem-focused commits on the user-selected branch.
