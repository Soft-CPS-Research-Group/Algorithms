# TransformerPPO Agent — Iterative Development Plan

## Scope Summary

Build `AgentTransformerPPO`: a PPO-based agent with a Transformer backbone that tokenizes observations by asset type, enabling variable numbers/types of controllable assets per building. Each building gets its own agent instance (same class, separate weights), mirroring the existing MADDPG pattern.

**Key constraint:** Zero changes to wrapper, runner, or existing agents.

---

## Phase 0 — Project Scaffolding & Encoder Index Map Utility

**Goal:** Build the foundational utility that maps raw feature names to post-encoding slice indices. This is the most critical piece — everything downstream depends on it.

### 0.1 — Encoder Index Map Builder

Create a utility function (likely in `algorithms/utils/encoder_index_map.py`) that:

1. Takes `observation_names: List[str]` (for one building) and the loaded encoder config (`configs/encoders/default.json`)
2. Iterates each raw observation name, matches it against the encoder rules (same matching logic as `wrapper_citylearn.py:592-626` — `equals`, `contains`, `prefixes`, `suffixes`, `default`)
3. Computes how many post-encoding dimensions each raw feature produces:
   - `PeriodicNormalization` → **2** (sin, cos)
   - `OnehotEncoding(classes)` → **len(classes)**
   - `NormalizeWithMissing` → **1**
   - `NoNormalization` → **1**
   - `RemoveFeature` → **0**
4. Returns an ordered dict: `{raw_name: (start_idx, end_idx, n_dims)}` representing the slice in the post-encoded flat vector

**Why first:** The tokenizer's classification + grouping step (Phase 1) depends entirely on knowing which post-encoded indices correspond to which raw feature. Getting this wrong breaks everything downstream.

### 0.2 — Verification Checkpoint

Write `tests/test_encoder_index_map.py`:

- Test with a known building's observation names from the `citylearn_challenge_2022` dataset schema
- Verify that `RemoveFeature` features (weather) produce 0 dims
- Verify `PeriodicNormalization` features produce 2 dims
- Verify `OnehotEncoding` features produce the correct class count (e.g., `day_type` → 8, `departure_time` → 27, `connected_state` → 2)
- Verify the total number of post-encoding dims matches what the wrapper actually produces (cross-check with `wrapper.observation_space[i].shape[0]`)

```bash
pytest tests/test_encoder_index_map.py -v
```

---

## Phase 1 — Observation Tokenizer

**Goal:** Implement `ObservationTokenizer(nn.Module)` that converts a flat post-encoded vector into typed token embeddings.

### 1.1 — Feature Classification Engine

Implement the classification algorithm in the tokenizer's `__init__`:

1. Load tokenizer config (the `algorithm.tokenizer` YAML section)
2. For each raw observation name:
   - Try substring match against `ca_types[*].features` → assign to CA type
   - If CA type has multiple instances (suffix detection, e.g., `_charger_15_1`), group by suffix
   - Else try substring match against `sro_types[*].features` → assign to SRO group
   - Else check `rl.demand_feature` and `rl.generation_features`
   - Else log warning for unmatched
3. Use the encoder index map (Phase 0) to resolve each group's post-encoding slice indices
4. Skip any group with 0 total post-encoding dims (e.g., weather SRO)

**SRO weather handling:** The tokenizer config will still list the weather SRO type for completeness, but since all weather features are `RemoveFeature` → 0 dims, the tokenizer skips it at runtime. No special-casing needed — it's handled by the "skip 0-dim groups" rule.

### 1.2 — Multi-Instance CA Detection

Implement suffix-based grouping:

- CityLearn appends device ID suffixes for multi-instance CAs (e.g., `electric_vehicle_charger_connected_state_charger_15_1`)
- Strategy: for each CA type, collect all matching raw names, extract the suffix (portion after the base feature name), group by suffix
- Features with no suffix → single-instance CA (one token)
- Features sharing a suffix → one CA token per unique suffix

### 1.3 — Per-Type Linear Projections

Create `nn.ModuleDict` of projection layers:

- One `Linear(n_encoded_features, d_model)` per CA type (shared across instances of that type)
- One `Linear(n_encoded_features, d_model)` per SRO group (that has >0 dims)
- One `Linear(n_encoded_features, d_model)` for the RL token

### 1.4 — Forward Pass

`forward(encoded_obs: Tensor[batch, obs_dim]) → TokenizedObservation`:

1. For each CA instance: gather its slice from the flat vector, project through the type's linear layer
2. For each SRO group: gather its slice, project
3. For RL: gather demand and generation slices, compute residual (demand - generation), project
4. Return `TokenizedObservation` dataclass with `ca_tokens`, `sro_tokens`, `rl_token`, metadata

### 1.5 — TokenizedObservation Dataclass

```python
@dataclass
class TokenizedObservation:
    ca_tokens: Tensor    # [batch, N_ca, d_model]
    sro_tokens: Tensor   # [batch, N_sro, d_model]
    rl_token: Tensor     # [batch, 1, d_model]
    ca_types: List[str]  # type name per CA token (for action ordering)
    n_ca: int            # number of CA tokens
```

### 1.6 — Action-CA Mapping

Build a mapping from CA token index → action index using `action_names` and `ca_types[*].action_name` substring matching. This ensures the output action vector is ordered correctly.

### 1.7 — Verification Checkpoint

Write `tests/test_observation_tokenizer.py`:

- **Test classification**: Given known observation names, verify correct CA/SRO/RL assignment
- **Test multi-instance**: With Building_15 names (2 chargers), verify 2 separate CA tokens
- **Test projection shapes**: Verify `ca_tokens.shape == [batch, expected_n_ca, d_model]`
- **Test weather skip**: Verify weather SRO group is absent from output tokens
- **Test RL computation**: Verify RL token uses (demand - generation) values
- **Test single-instance building**: Building with only 1 charger → 1 CA token
- **Test no-CA building**: Building with no controllable assets → 0 CA tokens (edge case)
- **Test action mapping**: Verify action index matches expected CA ordering

```bash
pytest tests/test_observation_tokenizer.py -v
```

---

## Phase 2 — Transformer Backbone

**Goal:** Implement the shared Transformer encoder that processes all token types through self-attention.

### 2.1 — Type Embeddings

Learnable embeddings for the 3 token types (CA=0, SRO=1, RL=2):

- `nn.Embedding(3, d_model)` — added to token embeddings before attention
- No positional embeddings (pure set semantics, matching the PoC design)

### 2.2 — Transformer Encoder

Standard `nn.TransformerEncoder` with configurable:

- `d_model` (embedding dimension)
- `nhead` (attention heads)
- `num_layers` (encoder layers)
- `dim_feedforward` (FFN width)
- `dropout`

Input: concatenated `[CA_tokens; SRO_tokens; RL_token]` + type embeddings → `[batch, N_total, d_model]`
Output: contextual embeddings at each position → `[batch, N_total, d_model]`

### 2.3 — Token Assembly & Output Slicing

Helper that:

1. Concatenates tokens in deterministic order: CAs first, then SROs, then RL
2. Builds the type ID tensor for embedding lookup
3. After Transformer, slices output back into CA positions (first `N_ca`) for the actor head

### 2.4 — Verification Checkpoint

Write `tests/test_transformer_backbone.py`:

- **Test forward shape**: Given N_ca=3, N_sro=2 tokens → output shape `[batch, 6, d_model]`
- **Test variable cardinality**: Same model, different N_ca → output adapts
- **Test CA output extraction**: Verify first N_ca positions are correctly sliced
- **Test gradient flow**: Ensure gradients propagate through attention to all token types

```bash
pytest tests/test_transformer_backbone.py -v
```

---

## Phase 3 — PPO Components

**Goal:** Implement the PPO-specific components: actor head, critic head, rollout buffer, and PPO loss computation.

### 3.1 — Actor Head

Applied to CA token positions only:

- `LayerNorm → Linear(d_model, d_ff) → GELU → Linear(d_ff, 1) → tanh`
- Outputs one scalar action per CA in `[-1, 1]`
- Also outputs log-probability (via squashed Gaussian — see `docs/decisions.md` Decision #1)

**Distribution:** Squashed Gaussian (Normal + tanh) with a learnable log-std parameter per CA type. This gives differentiable log-probs needed for PPO's clipped objective.

### 3.2 — Critic Head

Applied to pooled representation (mean-pool over all token positions):

- `LayerNorm → Linear(d_model, d_ff) → GELU → Linear(d_ff, 1)`
- Outputs scalar V(s)

### 3.3 — PPO Rollout Buffer

On-policy buffer (not a replay buffer). Stores complete trajectories for the current rollout:

- `observations`, `actions`, `log_probs`, `rewards`, `values`, `dones`
- `compute_returns_and_advantages()` using GAE (Generalized Advantage Estimation) with configurable `gamma` and `gae_lambda`
- `get_batches(batch_size)` — yields minibatches for K epochs of PPO updates
- `clear()` — reset after each policy update

### 3.4 — PPO Loss Functions

Implement the three PPO losses:

1. **Clipped surrogate objective** — `min(r * A, clip(r, 1-eps, 1+eps) * A)` where `r = exp(log_pi_new - log_pi_old)`
2. **Value function loss** — MSE between predicted V(s) and returns (optionally clipped)
3. **Entropy bonus** — encourage exploration

Combined: `L = -L_clip + c1 * L_value - c2 * entropy`

### 3.5 — Verification Checkpoint

Write `tests/test_ppo_components.py`:

- **Test actor output**: Given CA embeddings, verify action shape `[batch, N_ca, 1]` and range `[-1, 1]`
- **Test critic output**: Given pooled embedding, verify scalar output
- **Test rollout buffer**: Push transitions, compute GAE, verify advantage normalization
- **Test PPO loss**: With synthetic old/new log-probs, verify clipping behavior
- **Test log-prob computation**: Verify log-probs are correct for the chosen distribution

```bash
pytest tests/test_ppo_components.py -v
```

---

## Phase 4 — AgentTransformerPPO Integration

**Goal:** Wire everything together into the full agent class that satisfies the `BaseAgent` contract.

### 4.1 — `__init__(self, config)`

Read from config:

- `algorithm.hyperparameters`: PPO params (lr, gamma, gae_lambda, clip_eps, epochs, batch_size, entropy_coeff, value_coeff)
- `algorithm.tokenizer`: Token type registry (ca_types, sro_types, rl)
- `algorithm.transformer`: Architecture params (d_model, nhead, num_layers, dim_feedforward, dropout)
- `topology`: num_agents (number of buildings)

Initialize per-building:

- `ObservationTokenizer[i]` (created later in `attach_environment`)
- `TransformerEncoder` (shared architecture, separate instances per building)
- `ActorHead`, `CriticHead`
- `RolloutBuffer`
- `Adam` optimizer

### 4.2 — `attach_environment(...)`

**This is where the tokenizer is built** (unlike MADDPG which ignores this hook):

1. Receive `observation_names: List[List[str]]`, `action_names: List[List[str]]`, spaces
2. Load encoder config from `configs/encoders/default.json`
3. For each building `i`:
   - Build encoder index map for `observation_names[i]`
   - Create `ObservationTokenizer(observation_names[i], action_names[i], encoder_config, tokenizer_config, d_model)`
   - Determine `N_ca` for this building → set action dimension
4. Store the tokenizers as a `nn.ModuleList`

### 4.3 — `predict(observations, deterministic)`

`observations` is `List[np.ndarray]` (one per building, post-encoded).

For each building `i`:

1. Convert `observations[i]` to tensor
2. `tokenized = self.tokenizers[i](obs_tensor)`
3. Assemble tokens → Transformer → contextual embeddings
4. Actor head on CA positions → actions `[N_ca_i]`
5. If not deterministic: sample from distribution, store log-probs and values in rollout buffer
6. If deterministic: use mean action

Return `List[np.ndarray]` — one action array per building.

### 4.4 — `update(...)`

PPO is **on-policy**, so the update pattern differs from MADDPG:

- The rollout buffer accumulates transitions during `predict` calls
- When `update_step=True` and the buffer has enough data:
  1. Compute GAE advantages
  2. For K epochs over minibatches:
     - Recompute log-probs and values with current policy
     - Compute PPO clipped loss + value loss + entropy
     - Backprop and step optimizer
  3. Clear rollout buffer
- Return loss metrics dict (for MLflow logging)

**Scheduling flags:** Respect `update_step` and `initial_exploration_done`. For PPO, `initial_exploration_done` can be gated on a minimum number of steps before the first update.

### 4.5 — `export_artifacts(output_dir, context)`

Export each building's model to ONNX:

- The ONNX export needs special handling since the Transformer has variable-length input
- Export with dynamic axes for the token sequence dimension
- Return metadata dict matching the manifest contract

### 4.6 — `save_checkpoint / load_checkpoint`

Serialize:

- All network state dicts (tokenizers, transformer, actor, critic)
- Optimizer state
- Training step counter
- Rollout buffer state (optional — can be discarded on resume since PPO is on-policy)

### 4.7 — `is_initial_exploration_done(global_learning_step)`

Return `global_learning_step >= min_steps_before_update` (configurable, default could be 1 episode worth of steps).

### 4.8 — Verification Checkpoint

Write `tests/test_agent_transformer_ppo.py`:

- **Test instantiation**: Create agent with valid config, verify all components initialized
- **Test attach_environment**: With mock observation/action names, verify tokenizers are built
- **Test predict shape**: Verify output action count matches N_ca per building
- **Test predict stochastic vs deterministic**: Verify stochastic stores log-probs, deterministic doesn't
- **Test update mechanics**: Push synthetic transitions, trigger update, verify loss decreases
- **Test checkpoint round-trip**: save → load → verify weights match
- **Test multi-building**: 2 buildings with different CA counts → independent tokenizers and outputs

```bash
pytest tests/test_agent_transformer_ppo.py -v
```

---

## Phase 5 — Config Schema & Registry

**Goal:** Integrate into the config validation system and registry.

### 5.1 — Config Schema

Add to `utils/config_schema.py`:

- `TransformerPPOHyperparameters` — PPO-specific params (lr, gamma, gae_lambda, clip_eps, ppo_epochs, minibatch_size, entropy_coeff, value_coeff, max_grad_norm)
- `TokenizerConfig` — ca_types, sro_types, rl section (mirrors the YAML structure from `docs/base.md`)
- `TransformerConfig` — d_model, nhead, num_layers, dim_feedforward, dropout
- `TransformerPPOAlgorithmConfig` — integrates into the discriminated union alongside MADDPG/RBC/SingleAgentRL

### 5.2 — Config Template

Create `configs/templates/transformer_ppo_local.yaml` with sensible defaults:

```yaml
algorithm:
  name: AgentTransformerPPO
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 128
    dropout: 0.1
  hyperparameters:
    learning_rate: 3e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    ppo_epochs: 4
    minibatch_size: 64
    entropy_coeff: 0.01
    value_coeff: 0.5
    max_grad_norm: 0.5
  tokenizer:
    # ... (full token type registry)
```

### 5.3 — Registry

In `algorithms/registry.py`:

```python
from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO

ALGORITHM_REGISTRY["AgentTransformerPPO"] = AgentTransformerPPO
```

### 5.4 — Verification Checkpoint

```bash
pytest tests/test_config_validation.py -v   # new schema validates
pytest tests/test_registry.py -v             # new agent discoverable
pytest -v                                    # full suite — nothing broken
```

---

## Phase 6 — End-to-End Integration Test

**Goal:** Verify the agent works within the full training loop.

### 6.1 — Smoke Test

Run a short training session (2-3 episodes, small time window) with the challenge dataset:

- Verify `attach_environment` → tokenizer builds without error
- Verify `predict` → returns correctly shaped actions
- Verify `update` → PPO loss is computed and parameters change
- Verify `export_artifacts` → ONNX files are written
- Verify `save_checkpoint` / `load_checkpoint` → round-trip works

### 6.2 — Multi-Architecture Test

Test across building archetypes (from evaluation plan):

- **No CAs** → 0 CA tokens, no actions (if applicable)
- **EV only** → 1 CA token
- **Battery only** → 1 CA token
- **EV + Battery** → 2 CA tokens
- **Multi-charger (Building_15)** → 2 CA tokens
- **21 chargers (i-charging)** → 21 CA tokens (stress test)

### 6.3 — Verification Checkpoint

```bash
pytest -v                                                                          # full suite
python run_experiment.py --config configs/templates/transformer_ppo_local.yaml     # smoke run
```

---

## File Layout (New Files)

```
algorithms/
├── agents/
│   └── transformer_ppo_agent.py       # AgentTransformerPPO (Phase 4)
├── utils/
│   ├── encoder_index_map.py           # Encoder → index map utility (Phase 0)
│   ├── observation_tokenizer.py       # ObservationTokenizer (Phase 1)
│   ├── transformer_backbone.py        # Transformer encoder + assembly (Phase 2)
│   └── ppo_components.py             # Actor/Critic heads, rollout buffer, losses (Phase 3)
configs/
├── templates/
│   └── transformer_ppo_local.yaml    # Config template (Phase 5)
tests/
├── test_encoder_index_map.py         # Phase 0 tests
├── test_observation_tokenizer.py     # Phase 1 tests
├── test_transformer_backbone.py      # Phase 2 tests
├── test_ppo_components.py            # Phase 3 tests
└── test_agent_transformer_ppo.py     # Phase 4 tests
```

---

## Tokenizer Config Reference (SROs without weather)

Weather features are effectively discarded. The config lists the weather SRO for completeness, but since all weather features have `RemoveFeature` encoding → 0 post-encoding dims, the tokenizer skips it automatically. Active SRO types:

| SRO Group | Features | Post-Encoding Dims |
|-----------|----------|:---:|
| `temporal` | month, hour, day_type | 12 (2+2+8) |
| `pricing` | electricity_pricing + 3 predicted | 4 |
| `carbon` | carbon_intensity | 1 |
| ~~`weather`~~ | ~~all weather features~~ | **0 (skipped)** |

---

## Dependency Graph

```
Phase 0 (Encoder Index Map)
    ↓
Phase 1 (Tokenizer) ← depends on Phase 0
    ↓
Phase 2 (Transformer) ← independent of Phase 1, but tested together
    ↓
Phase 3 (PPO Components) ← independent of Phase 1-2, can be parallelized
    ↓
Phase 4 (Agent Integration) ← depends on Phase 1 + 2 + 3
    ↓
Phase 5 (Config/Registry) ← depends on Phase 4
    ↓
Phase 6 (E2E Testing) ← depends on Phase 5
```

> **Note:** Phases 2 and 3 are independent of each other and could be developed in parallel.
