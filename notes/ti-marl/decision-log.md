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
21. The public typed interface is one versioned, human-editable YAML document.
    It may be written manually or enriched from Simulator `entity_specs`; its
    fixed semantics, ordered observation view, action groups and health rules
    remain reviewable. Schema/type/health splits are compiler internals, not
    separate user configuration surfaces.
22. Decision 21 is superseded in composition, not purpose: the public contract
    is one `typed_agent_interface_v1` YAML per registered agent, loaded from a
    directory. The prior global `typed_interface_v1` prototype is rejected.
23. Public agent interfaces are technology-neutral. Simulator, MQTT, Modbus
    and endpoint bindings belong to adapters and never to the public YAML.
24. Sensors contain channels and channels contain independently typed and
    health-assessed observations. `community` is a normal aggregate sensor
    with community scope, not a privileged special section.
25. All available fields are typed and classified, while only explicit
    `policy_input` observations reach the actor. Trace-only or excluded fields
    remain auditable and require a reason.
26. Health thresholds use physical duration. Each action dependency declares
    the effects of non-nominal observation/channel states; no universal health
    consequence is inferred from the state name alone.
27. Registration is persistent and distinct from runtime activity. Registry
    reload is atomic; new/removed agents and known asset instances preserve
    stable-ID transition semantics and do not resize networks.
28. The same actor, TIC, profiles, normalisation, feasibility and compatibility
    semantics are deployed. Central critics and privileged training context are
    not part of decentralised deployment.
29. The canonical validation surfaces are
    `citylearn_three_phase_electrical_service_demo_15min_parquet`,
    `citylearn_three_phase_dynamic_topology_demo` and the 15-second dynamic
    asset stress fixture.
30. The TIC is deterministic and contains no learned parameters. It compiles
    typed interfaces and runtime facts into health, validity, bounds,
    constraints and snapshots. Neural processing begins in the hierarchical
    snapshot encoder; action selection is performed by the grouped actor, and
    the central set critic exists only during training.
31. Synthetic electrical-service contracts used for development remain
    external experiment inputs. They are applied to an in-memory dataset copy,
    may fill only buildings without an existing conflicting service fact, and
    never rewrite or silently override the canonical dataset schema.

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
- First research artefact remains versioned PyTorch; a deployment bundle must
  explicitly exclude the central critic and include the actor/TIC contract.

## Open experimental decisions

- Whether a resource graph critic materially improves the set critic.
- Whether TI-MATD3 is justified after the TI-MAPPO evidence gates.
- Final reward coefficients, frozen on a development split before confirmation.
- Whether a shared hard monitor is required and demonstrably exceptional.
- Dynamic deployment format after the research bundle is validated.

None of these choices blocks the first vertical slice.
