# ADR-0012 — Schema surface, registry, wrapper capability

Status: accepted
Date: 2026-08-18
Depends on: ADR-0007, ADR-0008, ADR-0009, ADR-0010, ADR-0011

## Context

Three integration points:

1. Config schema (`utils/config_schema.py`).
2. Algorithm registry (`algorithms/registry.py`).
3. Wrapper and config capability guards.

## Decisions

### 12a — dedicated schema class

Use `TransformerMATD3StageConfig` in `utils/config_schema.py`,
mirroring `TransformerPPOStageConfig`. It provides explicit typed
validation and better error messages than dict passthrough.

### 12b — field categories

`TransformerMATD3StageConfig` owns stage identity, tokenizer path,
Transformer shape, MATD3 hyperparameters, and the two behavior-cloning blocks.
Nested models forbid unknown fields and validate cross-field relationships.

The stage schema covers training, exploration, residual control, local action
safety, local price conditioning, replay BC, and demonstration BC. The complete
field list and defaults live in these authoritative models:

- `TransformerMATD3TransformerConfig`
- `TransformerMATD3Hyperparameters`
- `TransformerMATD3ReplayBehaviorCloningConfig`
- `TransformerMATD3DemonstrationBehaviorCloningConfig`

Generic checkpoint and resume settings remain in the top-level
`CheckpointingConfig`. They are not fields of `TransformerMATD3StageConfig`.

### 12c — registry entry

`ALGORITHM_REGISTRY` contains:

```python
"AgentTransformerMATD3": AgentTransformerMATD3,
```

### 12d — dynamic-topology capability

`AgentTransformerMATD3.supports_dynamic_topology: bool = True`
class variable. The controller provides:

- `snapshot_topology_state() -> dict`
- `restore_topology_state(state: dict) -> None`

The contract mirrors TPPO and is consumed by the wrapper's topology
transaction.

### 12e — wrapper guard

The wrapper and config validation use the capability class variable. They accept
any registered agent with `supports_dynamic_topology = True`; no hard-coded
Transformer MATD3 allowlist is required.

### 12f — dropout

`dropout in [0.0, 1.0)`. Off-policy has no PPO ratio constraint;
dropout is a legitimate regularizer.

## Consequences

- Explicit schema class replaces `exploration.params` dict passthrough
  for MATD3-specific fields.
- Schema-time errors surface misconfiguration before runtime.
- Registry integration is a one-line addition.
- Dynamic-topology capability declaration unlocks entity+dynamic
  mode.

## Evidence

- `TransformerPPOStageConfig` provides the typed-schema pattern.
- `ALGORITHM_REGISTRY` provides name-to-class construction.
- `Wrapper_CityLearn` and `validate_config` both check
  `supports_dynamic_topology`.

## Future improvements

- Add `critic_action_input_mode` values `final_base_delta` and
  `final_base_delta_normalized` (see ADR-0008 future improvements).
- Consider a single `behavior_cloning.mode` enum selector if two
  independent enabled flags prove operationally confusing (see
  ADR-0007).
