# TransformerPPO Implementation Summary

Reference spec: `docs/spec.md`

---

## What Was Built

A Transformer-based PPO agent (`AgentTransformerPPO`) that treats building energy assets as tokens. Variable numbers and types of assets are handled at runtime without retraining, by using marker values embedded in the observation vector as token boundary delimiters.

---

## Architecture Overview

```
Raw Observations (from CityLearn)
  -> ObservationEnricher    — classify features, inject marker values
  -> Encoders               — normalize/encode (markers pass through)
  -> ObservationTokenizer   — scan markers, split into token groups, project to d_model
  -> TransformerBackbone    — self-attention with type embeddings
  -> ActorHead / CriticHead — per-CA actions + state value
  -> PPO update             — GAE + clipped surrogate loss
```

Three token families:
- **CA (Controllable Assets)** — devices the agent controls (battery, EV charger, washing machine). Each CA produces one action.
- **SRO (Shared Read-Only)** — external context (temporal, pricing, carbon).
- **NFC (Non-Flexible Context)** — building's residual load (demand, generation, net consumption).

---

## Key Components

### 1. ObservationEnricher (portable)

**File:** `algorithms/utils/observation_enricher.py`

Classifies raw observation features into CA/SRO/NFC groups using pattern matching against a tokenizer config. Injects numeric marker values (1001+, 2001+, 3001) at token boundaries. Caches enrichment results per topology and detects topology changes.

Portable: pure Python with no training dependencies, so it can be reused in the production inference repo.

### 2. ObservationTokenizer (generic)

**File:** `algorithms/utils/observation_tokenizer.py`

Neural module (`nn.Module`) that scans the encoded observation tensor for marker values, splits it into token groups, and projects each group to `d_model` via per-type `Linear` layers. Supports a marker registry for explicit marker-to-type resolution.

### 3. TransformerBackbone

**File:** `algorithms/utils/transformer_backbone.py`

Standard Transformer encoder with learned type embeddings (CA=0, SRO=1, NFC=2) instead of positional embeddings. Produces contextual embeddings via self-attention. Uses Pre-LN (`norm_first=True`) and GELU activation.

### 4. PPO Components

**File:** `algorithms/utils/ppo_components.py`

- **ActorHead** — MLP per CA embedding; tanh-squashed Gaussian with learnable shared `log_std`.
- **CriticHead** — MLP from mean-pooled embedding to scalar V(s).
- **RolloutBuffer** — On-policy buffer with GAE advantage computation and minibatch iteration.
- **`compute_ppo_loss()`** — Clipped surrogate objective + value loss + entropy bonus.

### 5. AgentTransformerPPO (main agent)

**File:** `algorithms/agents/transformer_ppo_agent.py`

Implements the `BaseAgent` contract. Orchestrates tokenizer, backbone, actor, critic, and per-building rollout buffers. Delegates internals to three helper modules:

- `export_helper.py` — ONNX export (end-to-end deterministic policy per building).
- `state_helper.py` — Per-building state initialization and marker registry management.
- `update_helper.py` — PPO update loop (GAE, minibatch, gradient clipping).

Helpers live in `algorithms/agents/transformer_ppo/`.

### 6. TransformerObservationCoordinator (wrapper integration)

**File:** `utils/wrapper_transformer/transformer_observation_coordinator.py`

Static helper class isolating all Transformer-specific orchestration from the generic CityLearn wrapper. Manages enricher initialization, marker injection during observation encoding, topology change detection/handling, and encoder rebuilding.

### 7. Wrapper Modifications

**File:** `utils/wrapper_citylearn.py`

The existing wrapper was extended to support Transformer agents:
- Detects `is_transformer_agent` flag on the model.
- Delegates to `TransformerObservationCoordinator` for enrichment and topology changes.
- Checks for topology changes at episode start and after each step.
- Enriches observation values with markers before encoding.

Non-Transformer agents (MADDPG, RBC) are unaffected.

---

## Configuration

### Tokenizer Config

**File:** `configs/tokenizers/default.json`

Defines the observation classification schema: which features belong to each CA/SRO/NFC type, their marker value bases, and post-encoding `input_dim` per type.

### Algorithm Template

**File:** `configs/templates/transformer_ppo.yaml`

Default experiment config: Transformer architecture (d_model=64, 4 heads, 2 layers, feedforward=128), PPO hyperparameters (lr=3e-4, gamma=0.99, clip_eps=0.2, 4 epochs).

### Config Schema

**File:** `utils/config_schema.py`

New Pydantic models: `TokenizerConfig`, `TransformerConfig`, `TransformerPPOHyperparameters`, `TransformerPPOAlgorithmConfig`. Added to the `ProjectConfig.algorithm` discriminated union.

### Registry

**File:** `algorithms/registry.py`

`AgentTransformerPPO` registered under key `"AgentTransformerPPO"`.

---

## Test Coverage

| Test File | Scope |
|-----------|-------|
| `test_observation_enricher.py` | Feature classification, marker injection, topology detection |
| `test_observation_tokenizer.py` | Marker scanning, group extraction, projection shapes, variable CA counts |
| `test_transformer_backbone.py` | Forward shapes, type embeddings, gradient flow |
| `test_ppo_components.py` | Actor/critic output shapes, rollout buffer GAE, PPO loss clipping |
| `test_tokenizer_config_schema.py` | Pydantic schema validation for tokenizer config |
| `test_agent_transformer_ppo.py` | Agent instantiation, predict/update, topology changes, checkpoints |
| `test_wrapper_transformer.py` | Enricher setup, topology change handling, marker presence in encoded output |
| `test_wrapper_integration_e2e.py` | Full wrapper + agent integration |
| `test_e2e_transformer_ppo.py` | Complete training loop, valid action ranges, KPI generation |
| `test_transformer_refactor_helpers.py` | Helper-level coverage for state/update/export modules |

---

## Key Design Decisions

1. **Marker-based tokenization** — Sentinel values (1000s/2000s/3000s) in the observation vector serve as token boundaries, enabling variable-length token sequences without metadata side-channels.
2. **Per-type projections** — Each asset type has its own `Linear(input_dim, d_model)`, with `input_dim` explicitly defined in config.
3. **Type embeddings only** — No positional embeddings; pure set semantics within token types.
4. **Topology change handling** — Mid-episode topology changes trigger a PPO update (if buffer has data), then flush the buffer and re-enrich.
5. **Portable enricher** — No ML/training dependencies, enabling identical logic in training and production inference.
6. **Shared backbone, per-building instances** — Architecture is shared but weight instances are independent per building.
