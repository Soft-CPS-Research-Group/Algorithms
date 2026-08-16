# TI-MARL test plan

## Simulator gates

- Preserve fault mode without emitting TI health.
- Keep connection, availability, sensor, actuator and communication status
  independent.
- Report freshness and active duration deterministically.
- Distinguish requested, post-channel, limited and applied commands.
- Preserve `entity_v1` consumers and flat-interface behaviour.
- Build/install the release artefact in a clean environment.

## Compiler gates

- Deterministic snapshot/hash and compatibility signature.
- Unit, scope, provenance and entity-binding validation.
- Session replacement without stale identity leakage.
- Health derivation across duration/criticality thresholds and recovery.
- Scoped closure, conflict/cycle rejection and unknown-type fallback.

## Policy/critic gates

- Instance permutation equivariance and fixed parameter count.
- Several simultaneous groups share one latent context.
- Correct categorical plus Beta log probability.
- Variable-population critic and correct join/leave/bootstrap semantics.
- Checkpoint restore under a different compatible composition.

## End-to-end slice

Use real CityLearn entity payloads with multiple buildings, battery, charger/EV
and deferrable groups. Include module addition/removal, a long stuck event that
crosses degraded-to-stale, sensor loss, actuator loss, community-link loss and
recovery. Verify trace-to-execution correspondence, local feasibility,
parameter-count invariance and baseline regressions.

