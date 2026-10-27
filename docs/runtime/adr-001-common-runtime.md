# ADR: one job runtime with registered agent capabilities

Accepted for stage 2, 2026-10-27.

The entry script and simulator wrapper previously combined job setup, configuration,
tracking, model construction, episode iteration, observations, diagnostics and
output publication. The registry eagerly imported every numerical implementation.
Configuration mixed common fields and agent fields, while some permissive sections
accepted unused parameters. This made agent development depend on unrelated services.

We keep one job lifecycle in `algorithms.runtime`, move lightweight schemas into
`algorithms.configuration`, and register each implementation with its schema,
capabilities and optional environment validator. The catalogue can be inspected
without loading numerical libraries. Agent implementations retain their numerical
interfaces through an adapter to typed contexts, transitions and schedules.

The stateful simulation controller remains a facade. Its episode loop, resource
lifecycle and diagnostics have explicit module owners and share controller state.
Compatibility modules delegate to these owners rather than maintaining copied
implementations. Offline optimizers remain agent-owned hooks and share their
parameter definitions with configuration validation.

Installation distinguishes configuration, training, export, tracking, offline data
and solvers. Existing platform image dependency selections remain unchanged.
Configuration v1 and migration expose previously ignored values without activating
them accidentally. Explicit output requests and evaluate/smoke profiles define
common job intent.

Consequences: extension work needs a schema and a capability declaration; accepted
configuration errors fail earlier. A successful smoke job proves the common
lifecycle, not model quality, exact reproducibility or inference readiness. Full
common-contract certification is stage 3; global final-episode export semantics,
tracking performance and final deployment manifests remain later stages.
