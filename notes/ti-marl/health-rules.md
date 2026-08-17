# TI-MARL health derivation and closure

## Principle

The Simulator reports evidence. The TIC derives health and consequences.
`fault_mode`, availability, connection and channel state remain separately
queryable in every snapshot and trace.

## Derivation inputs

- event domain and raw fault mode;
- active physical duration;
- last update and last fresh sample age;
- semantic observation/action type;
- source/target channel;
- criticality and dependency type;
- availability and connection evidence;
- configured recovery hysteresis in physical time/fresh sample counts.

## Initial derivation policy

- A frozen/stuck sample starts `DEGRADED`; it becomes `STALE` only after the
  semantic freshness threshold.
- An unavailable sensor channel with no admissible cached value is `MISSING`.
- An unavailable sensor channel with an admissible cache is `STALE`, with its
  true age.
- Impaired value quality is `DEGRADED` unless a semantic rule escalates it.
- An intermittent actuator channel may be `DEGRADED`; an unavailable channel
  invalidates dependent ports without declaring the owning module failed.
- Community-link loss affects only community/external observations.
- Normal asset disconnection changes relations/group membership and does not
  create a health failure.
- Explicit asset unavailability may derive module `FAILED` and disable only its
  scoped groups.

Thresholds are keyed by semantic type and criticality and expressed in
seconds, so 15-second, 15-minute and hourly environments share meaning.
Recovery requires the configured duration/consecutive fresh samples and cannot
be inferred from the end of a fault event alone.

## Closure requirements

Each effect names a source, condition, target, operation, priority and rule ID.
Supported operations include port invalidation, bound contraction, observation
removal/substitution, group suspension, constraint modification, operating-mode
change and fallback activation. Conflicting equal-priority effects fail
compilation; dependency cycles fail validation.

Rules bind exact observation paths to exact action ports. A degraded EV SoC may
disable discharge while an explicitly configured service fallback charges at
the maximum value remaining after BMS, site, phase and feasibility bounds. A
missing main/grid meter instead isolates all learned local action groups. The
health state name alone never chooses between these consequences.
