# TransformerPPO Spec Completion Checklist

Reference spec: `docs/spec.md`

Status legend:
- `[x]` Completed
- `[~]` Partially completed
- `[ ]` Not completed

## P0 - Runtime and integration blockers

- [x] Wrapper -> agent done-flag compatibility (`terminated`/`truncated` scalar handling) (`algorithms/agents/transformer_ppo_agent.py`)
- [x] Topology change handling path is wired and observable in logs (`utils/wrapper_citylearn.py`, `algorithms/agents/transformer_ppo_agent.py`)

## P1 - Core spec coverage

- [x] Marker-based observation enrichment with CA/SRO/NFC grouping (`algorithms/utils/observation_enricher.py`)
- [x] Marker-based token scanning + per-type projection to `d_model` (`algorithms/utils/observation_tokenizer.py`)
- [x] Explicit wrapper->agent marker-registry propagation for type selection at runtime/export (`utils/wrapper_citylearn.py`, `algorithms/agents/transformer_ppo_agent.py`, `algorithms/utils/observation_tokenizer.py`)
- [x] Transformer backbone with type embeddings and pooled output (`algorithms/utils/transformer_backbone.py`)
- [x] PPO actor/critic/buffer/loss components (`algorithms/utils/ppo_components.py`)
- [x] Wrapper integration for transformer agents (`utils/wrapper_citylearn.py`)
- [x] Registry wiring for `AgentTransformerPPO` (`algorithms/registry.py`)
- [x] Tokenizer config + transformer template + schema models (`configs/tokenizers/default.json`, `configs/templates/transformer_ppo.yaml`, `utils/config_schema.py`)

## P1 - Spec gaps that still remain

- [x] `AgentTransformerPPO.export_artifacts()` exports end-to-end deterministic ONNX artifacts per agent (with actor-only fallback if backend export support is limited) (`algorithms/agents/transformer_ppo_agent.py`)
- [ ] Per-building independent model instances (spec asks for separate weight instances per building; implementation currently shares tokenizer/backbone/heads and keeps per-building rollout buffers)
- [ ] Per-CA-type exploration scale/log-std behavior in actor head (current implementation uses one shared `log_std`)
- [~] `EnrichmentResult` does not include `marker_encoder_specs` field from the written interface in spec (wrapper currently rebuilds encoders from marker names/ranges instead)

## P2 - Validation and test-plan coverage

- [x] Unit tests exist for enricher/tokenizer/backbone/PPO components/agent/wrapper integration
- [x] End-to-end tests exist for single-building flow and variable CA count
- [~] Not all tests listed in `docs/spec.md` are present one-to-one by exact name/scope (coverage is strong but not exhaustive vs. checklist wording)

## Logging coverage (TransformerPPO scope)

- [x] Agent lifecycle + update + checkpoint/export/topology logs (`algorithms/agents/transformer_ppo_agent.py`)
- [x] Enricher logs (cache hits, unclassified features, topology changes) using stdlib logging for portability (`algorithms/utils/observation_enricher.py`)
- [x] Tokenizer fallback/unsupported-shape warnings (`algorithms/utils/observation_tokenizer.py`)
- [x] Runtime warning deduplication to avoid step-level log spam for repeated shape mismatches (`algorithms/utils/observation_tokenizer.py`)
- [x] Backbone initialization logs (`algorithms/utils/transformer_backbone.py`)
- [x] PPO component initialization/buffer/loss debug logs (`algorithms/utils/ppo_components.py`)
- [x] Wrapper enrichment/topology handling logs (`utils/wrapper_citylearn.py`)

## Verification

- Full project tests pass after updates.
- Smoke run via `run_experiment.py` completes with artifact manifest and without CA/SRO unknown-size fallback spam.
