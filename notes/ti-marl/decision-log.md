# TI-MARL decision log

This log incorporates the frozen v16 decisions and the approved implementation
clarifications. A decision can only be changed by an explicit later entry; it
must not be silently reinterpreted in code.

## Frozen decisions

1. One building controller is one local agent in the principal formulation.
2. Chargers, EV sessions, batteries and other controlled elements are
   entity/module instances inside that agent.
3. Observations and actions are entity-bound through explicit identifiers and
   typed relations.
4. A known new asset instance changes the interface graph, not network
   parameter dimensions.
5. There is no permanent neural head per asset instance; shared
   type-conditioned decoders process active action-group instances.
6. All active groups are contextualised by one local latent and cross-group
   interaction before bundle decoding.
7. Community data are authorised aggregate typed observations; the core method
   has no Community Coordinator agent.
8. There is no mandatory offer, negotiation or flexibility-envelope phase.
9. Coordination is learned through aggregate observations, coupled rewards,
   joint physical transitions and training-only central set/graph critics.
10. Execution is decentralised; actors do not receive private observations
    from other agents.
11. Local non-negotiable constraints are incorporated into bundle construction.
    The shared monitor is exceptional and not a normal optimiser.
12. Health and dependencies causally recompile observations, ports, bounds,
    groups, constraints or operating mode.
13. TI-RL is the one-agent reduction; TI-MARL is the multi-agent method.
14. TI-MARL is optimiser-agnostic. TI-MAPPO is implemented first; TI-MATD3 is
    a later comparison.
15. The same semantics support learning, compatibility, safety and traces.
16. `fault_mode` is observed causal evidence and is independent of
    `HealthState`. There is no universal mapping such as `stuck -> STALE`.
17. The TIC derives `HealthState` using fault evidence, event duration,
    last-fresh age, channel semantics, criticality, availability and versioned
    rules, including recovery hysteresis.
18. Asset disconnection, asset unavailability, sensor-channel loss,
    actuator-channel loss and community/cloud communication loss are distinct
    facts and cannot be collapsed into one failure flag.
19. The Simulator exposes runtime facts, quality, availability, relations and
    real execution. The TIC owns dependencies, port validity, bound
    contraction, degraded modes and fallback.
20. A public Simulator contract change requires tests, documentation, release
    notes and a published version before Algorithms updates its dependency pin.

## Initial implementation choices

- Registry algorithm: `TIMARL`; first backbone: TI-MAPPO.
- Actor sharing: one actor family per compatible logical role.
- Encoder: pure-PyTorch relation-aware entity/group encoder with attention
  pooling; no PyG dependency in the first slice.
- Continuous port distribution: Beta on `[0, 1]`; categorical port selection.
- Critic: permutation-invariant `CentralSetCritic` first.
- Local feasibility: analytic typed projection first; no local QP initially.
- Shared hard monitor: disabled in the first slice.
- Unknown optional type: observed-only/uncontrolled; unknown safety-critical
  semantics: compatibility rejection and approved safe fallback.
- First artefact: non-deployable versioned PyTorch research bundle.

## Open experimental decisions

- Whether a resource graph critic materially improves the set critic.
- Whether TI-MATD3 is justified after the TI-MAPPO evidence gates.
- Final reward coefficients, frozen on a development split before confirmation.
- Whether a shared hard monitor is required and demonstrably exceptional.
- Dynamic deployment format after the research bundle is validated.

None of these choices blocks the first vertical slice.

