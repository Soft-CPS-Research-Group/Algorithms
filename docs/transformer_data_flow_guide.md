# TransformerPPO Data Flow Guide

This document explains the current data flow for the Transformer-based pipeline, including what happens when topology changes.

## TL;DR

- Main runtime path:
  - `run_experiment.run_experiment()` -> `Wrapper_CityLearn.set_model(...)` -> `Wrapper_CityLearn.learn()`
  - `Wrapper_CityLearn.predict(...)` -> `AgentTransformerPPO.predict(...)` -> `ObservationTokenizer.forward(...)` -> `TransformerBackbone.forward(...)` -> `ActorHead.forward(...)`
  - `Wrapper_CityLearn.update(...)` -> `AgentTransformerPPO.update(...)` -> optional `AgentTransformerPPO._ppo_update(...)`
- Marker/type logic:
  - `ObservationEnricher.enrich_names(...)` defines marker positions and `marker_to_type`.
  - Wrapper builds per-building marker registry in `Wrapper_CityLearn._enrich_observation_names(...)`.
  - Agent receives registry through `AgentTransformerPPO.update_marker_registry(...)`.
  - Tokenizer uses explicit marker routing in `ObservationTokenizer.forward(..., marker_registry=...)`.
- On topology changes:
  - `Wrapper_CityLearn._refresh_runtime_topology_views()` gets live names/spaces.
  - `Wrapper_CityLearn._check_topology_change(...)` detects changes.
  - `Wrapper_CityLearn._handle_topology_change(...)` re-enriches names, rebuilds encoders, updates marker registry, and calls `AgentTransformerPPO.on_topology_change(...)`.
- Why this matters:
  - Marker IDs can map to different asset types after topology updates.
  - Explicit marker registry prevents wrong projection routing and avoids unknown-size fallback spam.

## Mermaid flowchart (visual)

Mermaid is a text-based diagram format. Many Markdown renderers can display it as a graph.

```mermaid
flowchart TD
    A[run_experiment.run_experiment()] --> B[Wrapper_CityLearn.__init__()]
    B --> C[Wrapper_CityLearn.set_model(model)]
    C --> D[Wrapper_CityLearn._configure_model_transformer_state()]
    D --> E[Wrapper_CityLearn._setup_transformer_enrichers(tokenizer_config)]
    E --> F[Wrapper_CityLearn._enrich_observation_names(building_idx)]
    F --> G[ObservationEnricher.enrich_names(observation_names, action_names)]
    G --> H[AgentTransformerPPO.update_marker_registry(building_idx, marker_registry)]
    H --> I[Wrapper_CityLearn._rebuild_encoders_for_enriched_names(building_idx)]
    I --> J[AgentTransformerPPO.attach_environment(...)]

    J --> K[Wrapper_CityLearn.learn()]
    K --> L[env.reset()]
    L --> M[Wrapper_CityLearn._refresh_runtime_topology_views()]
    M --> N{Wrapper_CityLearn._check_topology_change(building_idx)}

    N -- yes --> O[Wrapper_CityLearn._handle_topology_change(building_idx)]
    O --> O1[Wrapper_CityLearn._enrich_observation_names(building_idx)]
    O1 --> O2[Wrapper_CityLearn._rebuild_encoders_for_enriched_names(building_idx)]
    O2 --> O3[AgentTransformerPPO.on_topology_change(building_idx)]

    N -- no --> P[Wrapper_CityLearn.predict(observations)]
    O3 --> P

    P --> Q[Wrapper_CityLearn.get_all_encoded_observations()]
    Q --> R[Wrapper_CityLearn.get_encoded_observations(building_idx, raw_obs)]
    R --> S[Wrapper_CityLearn._enrich_observation_values(building_idx, raw_values)]
    S --> T[ObservationEnricher.enrich_values(observation_values)]
    T --> U[encoder.transform(...) for each feature]

    U --> V[AgentTransformerPPO.predict(encoded_observations)]
    V --> W[ObservationTokenizer.forward(encoded_obs, marker_registry)]
    W --> W1[_find_marker_positions()]
    W1 --> W2[_extract_groups()]
    W2 --> W3[_lookup_marker_type()]
    W3 --> X[TransformerBackbone.forward(ca_tokens, sro_tokens, nfc_token)]
    X --> Y[ActorHead.forward(ca_embeddings, deterministic)]
    X --> Y2[CriticHead.forward(pooled)]
    Y --> Z[env.step(actions)]

    Z --> AA[Wrapper_CityLearn.update(...)]
    AA --> AB[AgentTransformerPPO.update(...)]
    AB --> AC[AgentTransformerPPO._as_done_flags(...)]
    AC --> AD[RolloutBuffer.add(...)]
    AD --> AE{update_step and enough samples}
    AE -- yes --> AF[AgentTransformerPPO._ppo_update(building_idx, last_obs)]
    AF --> AG[ObservationTokenizer.forward(..., marker_registry)]
    AG --> AH[compute_ppo_loss(...)]
    AE -- no --> AI[Continue next timestep]
```

## Step-by-step guide

1. `run_experiment.run_experiment()` creates environment, wrapper, and agent.
2. `Wrapper_CityLearn.set_model(model)` configures Transformer-specific state.
3. `Wrapper_CityLearn._configure_model_transformer_state()` initializes enrichers and enriched encoders per building.
4. `Wrapper_CityLearn._enrich_observation_names(building_idx)` calls `ObservationEnricher.enrich_names(...)`.
5. `Wrapper_CityLearn._enrich_observation_names(building_idx)` converts marker names to numeric marker values and builds `marker_registry`.
6. `Wrapper_CityLearn._enrich_observation_names(building_idx)` calls `AgentTransformerPPO.update_marker_registry(building_idx, marker_registry)`.
7. `AgentTransformerPPO.attach_environment(...)` creates per-building rollout buffers and internal tracking arrays.
8. `Wrapper_CityLearn.learn()` starts episode loop.
9. At episode start and after each `env.step(...)`, `Wrapper_CityLearn._refresh_runtime_topology_views()` refreshes live env metadata.
10. `Wrapper_CityLearn._check_topology_change(building_idx)` calls `ObservationEnricher.topology_changed(...)`.
11. If changed, `Wrapper_CityLearn._handle_topology_change(building_idx)` re-enriches names, rebuilds encoders, and calls `AgentTransformerPPO.on_topology_change(building_idx)`.
12. For action selection, `Wrapper_CityLearn.predict(...)` encodes observations with `Wrapper_CityLearn.get_all_encoded_observations()`.
13. Per building, `Wrapper_CityLearn.get_encoded_observations(...)` calls `Wrapper_CityLearn._enrich_observation_values(...)`, which calls `ObservationEnricher.enrich_values(...)`.
14. Encoded observations are sent to `AgentTransformerPPO.predict(...)`.
15. `AgentTransformerPPO.predict(...)` calls `ObservationTokenizer.forward(encoded_obs, marker_registry=...)`.
16. `ObservationTokenizer.forward(...)` scans markers, extracts groups, resolves type routing via marker registry (`_lookup_marker_type(...)`), projects CA/SRO/NFC groups, and returns `TokenizedObservation`.
17. `TransformerBackbone.forward(...)` contextualizes tokens.
18. `ActorHead.forward(...)` produces actions; `CriticHead.forward(...)` computes values.
19. Wrapper applies actions with `env.step(actions)`.
20. Training path calls `Wrapper_CityLearn.update(...)`, then `AgentTransformerPPO.update(...)`.
21. `AgentTransformerPPO.update(...)` stores transitions (`RolloutBuffer.add(...)`) and conditionally calls `AgentTransformerPPO._ppo_update(...)`.
22. `AgentTransformerPPO._ppo_update(...)` reuses tokenizer/backbone with marker registry and optimizes via `compute_ppo_loss(...)`.

## Topology-change path (what happens on changes)

When assets connect/disconnect or observation/action topology changes:

1. `Wrapper_CityLearn._refresh_runtime_topology_views()` picks up new `observation_names`, `action_names`, and spaces from live env.
2. `Wrapper_CityLearn._check_topology_change(building_idx)` detects mismatch via `ObservationEnricher.topology_changed(...)`.
3. `Wrapper_CityLearn._handle_topology_change(building_idx)` runs:
   - `Wrapper_CityLearn._enrich_observation_names(building_idx)`
   - `ObservationEnricher.enrich_names(...)`
   - `AgentTransformerPPO.update_marker_registry(...)`
   - `Wrapper_CityLearn._rebuild_encoders_for_enriched_names(building_idx)`
   - `AgentTransformerPPO.on_topology_change(building_idx)`
4. `AgentTransformerPPO.on_topology_change(building_idx)` flushes/updates rollout buffer safely and continues with new topology.

This keeps marker-to-type routing consistent even when marker indices shift (for example, CA marker `1002` can represent a different asset type after a topology change).

## Dummy data walkthrough (illustrative)

This example is simplified to show flow, not exact production values for every encoded slot.

### 1) Raw inputs (one building)

`action_names`

```python
[
  "electrical_storage",
  "electric_vehicle_storage_charger_1_1",
  "washing_machine_1",
]
```

`observation_names` (subset)

```python
[
  "month", "day_type", "hour",
  "electricity_pricing", "electricity_pricing_predicted_1",
  "electricity_pricing_predicted_2", "electricity_pricing_predicted_3",
  "carbon_intensity",
  "electrical_storage_soc",
  "electric_vehicle_charger_charger_1_1_connected_state",
  "connected_electric_vehicle_at_charger_charger_1_1_departure_time",
  "connected_electric_vehicle_at_charger_charger_1_1_required_soc_departure",
  "connected_electric_vehicle_at_charger_charger_1_1_soc",
  "connected_electric_vehicle_at_charger_charger_1_1_battery_capacity",
  "electric_vehicle_charger_charger_1_1_incoming_state",
  "incoming_electric_vehicle_at_charger_charger_1_1_estimated_arrival_time",
  "washing_machine_1_start_time_step",
  "washing_machine_1_end_time_step",
  "non_shiftable_load", "solar_generation", "net_electricity_consumption",
]
```

`observation_values` (subset, same order as names)

```python
[
  6.0, 2.0, 14.0,
  0.21, 0.22, 0.24, 0.25,
  0.43,
  0.58,
  1.0, 5.0, 0.8, 0.6, 45.0, 0.0, 2.0,
  17.0, 20.0,
  120.0, 15.0, 105.0,
]
```

### 2) Name enrichment + marker registry

`ObservationEnricher.enrich_names(...)` inserts markers by CA/SRO/NFC order and outputs `marker_to_type`.

Example marker registry produced in `Wrapper_CityLearn._enrich_observation_names(...)`:

```python
{
  1001.0: ("ca", "battery", None),
  1002.0: ("ca", "ev_charger", "charger_1_1"),
  1003.0: ("ca", "washing_machine", "1"),
  2001.0: ("sro", "temporal", None),
  2002.0: ("sro", "pricing", None),
  2003.0: ("sro", "carbon", None),
  3001.0: ("nfc", "nfc", None),
}
```

### 3) Value enrichment

`ObservationEnricher.enrich_values(...)` inserts marker values into the value vector at cached positions.

Illustrative shape:

```python
[
  # (optional unclassified values first)
  1001.0, <battery features...>,
  1002.0, <ev features...>,
  1003.0, <washing_machine features...>,
  2001.0, <temporal features...>,
  2002.0, <pricing features...>,
  2003.0, <carbon features...>,
  3001.0, <nfc features...>,
]
```

### 4) Encoding in wrapper

`Wrapper_CityLearn.get_encoded_observations(...)` applies configured encoders.

Typical encoded group sizes seen in runtime:

- CA groups: `[1, 61, 2]` (battery, EV, washing machine)
- SRO groups: `[12, 4, 1]` (temporal, pricing, carbon)
- NFC group: `3`

### 5) Tokenization with explicit type routing

`AgentTransformerPPO.predict(...)` calls:

```python
ObservationTokenizer.forward(encoded_obs, marker_registry=marker_registry)
```

Tokenizer behavior:

- Finds marker positions via `_find_marker_positions(...)`.
- Splits groups via `_extract_groups(...)`.
- Resolves marker type via `_lookup_marker_type(...)`.
- Routes to per-type projection layers.

For the washing machine group above, the registry routes marker `1003.0` to `washing_machine` projection. If the current encoded width is 2 but config expects 3, tokenizer pads one zero and logs one deduplicated warning.

### 6) Backbone, heads, and actions

- `TransformerBackbone.forward(...)` returns contextual embeddings.
- `ActorHead.forward(...)` returns one action per CA token.
- `CriticHead.forward(...)` returns value estimate.

Output action shape for this example building:

```python
np.ndarray(shape=(3,))
```

### 7) Topology change example

Suppose EV disconnects and new `action_names` becomes:

```python
["electrical_storage", "washing_machine_1"]
```

Then:

1. `Wrapper_CityLearn._check_topology_change(...)` detects the change.
2. `Wrapper_CityLearn._handle_topology_change(...)` rebuilds enrichment and encoders.
3. New marker registry may become:

```python
{
  1001.0: ("ca", "battery", None),
  1002.0: ("ca", "washing_machine", "1"),
  2001.0: ("sro", "temporal", None),
  2002.0: ("sro", "pricing", None),
  2003.0: ("sro", "carbon", None),
  3001.0: ("nfc", "nfc", None),
}
```

So marker `1002.0` now maps to washing machine (not EV), and tokenizer follows that mapping automatically.
