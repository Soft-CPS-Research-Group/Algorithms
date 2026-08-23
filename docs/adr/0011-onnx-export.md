# ADR-0011 — ONNX export contract

Status: accepted
Date: 2026-08-18
Depends on: ADR-0002, ADR-0008, ADR-0009
Related: ADR-0010, ADR-0012

## Context

Existing export patterns:

- Legacy MATD3: per-agent ONNX with fixed observation width, opset 13,
  `ActionScaledActor`, `tanh`, and affine bounds.
- TPPO: per-building deterministic actor ONNX with tokenizer, backbone, actor
  head, `tanh`, and affine bounds baked in at opset 17.

## Plain-language

Analogy: exporting a REST contract. Inputs, outputs, and internal
logic are frozen at export time. Any external runtime dependency
(residual base policy, safety projector, price adapter) breaks
inference parity when the graph is deployed alone.

## Decisions

### 11a — per-building export

One ONNX per building, per topology. Filename:

```
onnx_models/agent_<building_index>__topology_v<version>.onnx
```

Matches the TPPO filename contract.

### 11b — opset 17

Modern operators required by the shared Transformer backbone.

### 11c — baked graph contents

Baked into every exported graph:

- Layout indices for SRO/NFC/CA segments and NFC subtract expression.
- Per-type tokenizer projections (name-keyed dispatch).
- Transformer backbone (type embedding + encoder).
- Actor MLP (LayerNorm + Linear + GELU + Linear → per-CA scalar).
- `tanh` squash.
- Affine action bounds via low/high buffers.

Not baked (external runtime dependency):

- Residual base policy composition.
- Local action safety projector.
- Local price observation adapter.
- Exploration noise.
- Target networks and twin critics (never exported).

### 11d — export guards

Export raises before writing files when any external dependency is
enabled without an explicit runtime-only opt-in:

- `residual_policy_enabled=true` requires
  `residual_policy_runtime_only_export=true`.
- `local_action_safety_enabled=true` requires
  `local_action_safety_runtime_only_export=true`.
- `local_price_conditioning_enabled=true` requires
  `local_price_conditioning_runtime_only_export=true`. New guard;
  same pattern as above.

Runtime-only exports produce artifacts with
`deployable: false` and `requires_runtime_*` flags in per-artifact
metadata.

### 11e — manifest keys

Manifest mirrors the TPPO export contract:

- Top level: `format: onnx`, `artifacts: [...]`,
  `tokenizer_config_path`, `supports_dynamic_topology: true`,
  `agent_models: [...]`.
- Per `agent_models` entry: `model_path`, `building_index`,
  `building_id`, `topology_version`, `obs_dim`, `n_sro`, `n_ca`,
  `sro_types`, `ca_types`, `ca_action_names`.
- Per artifact `config`: `deployable: bool`,
  `requires_runtime_residual: bool`,
  `requires_runtime_local_action_safety: bool`,
  `requires_runtime_local_price_conditioning: bool`.

### 11f — dynamic axes

`encoded_obs: [batch, obs_dim]` with `batch` as dynamic axis.
`actions: [batch, n_ca]` with `batch` as dynamic axis. `obs_dim` and
`n_ca` are fixed for the exported topology.

## Consequences

- Deployment cost: one ONNX per building per topology. Dynamic
  deployment requires either an external model-routing policy or a
  re-export after topology change (matches TPPO).
- New `local_price_conditioning_runtime_only_export` field in
  `TransformerMATD3StageConfig` (see ADR-0012).

## Evidence

- Legacy MATD3 provides the per-agent affine-action export pattern and runtime
  safety guard.
- `AgentTransformerPPO.export_artifacts` provides the per-building Transformer
  export and manifest pattern.
- `DEFAULT_ONNX_OPSET` is 13 for legacy exports; Transformer actors use 17.

## Future improvements

- Bake local action safety inside the ONNX graph when a static-form
  safety projection is available.
- Bake residual composition when the base policy has a
  representable ONNX form.
- Multi-topology ONNX (polymorphic graph) — requires the shared
  actor architecture from ADR-0002 B/C plus ADR-0004 backbone
  upgrades.
