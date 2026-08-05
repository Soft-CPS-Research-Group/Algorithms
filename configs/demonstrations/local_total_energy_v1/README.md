# Local total-energy demonstrations v1

Portable, replay-validated schedules used only as offline/warm-start teachers
for independent building-local agents. They never enable a runtime service
teacher and they do not add community observations to a local actor.

Each directory contains:

- `replay_schedule.json`: semantic actions with repository-relative dataset
  provenance;
- `manifest.json`: hashes, source solve/replay identifiers, scope and accepted
  claims.

`week1_corrected_soc_diagnostic` is deliberately marked `diagnostic_only`:
the window `[0, 672)` is truncated at the right boundary for eight EV sessions.
It is suitable for smoke tests and implementation diagnostics, not for model
selection or generalization claims. Promotion schedules must have
`boundary_service_exact: true` and pass their matching CityLearn replay before
packaging.
