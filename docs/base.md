# Universal Local Controller — Context Base

## Objective

Build a **Universal Local Controller (ULC)** that manages energy assets within a building using a Transformer-based architecture where **assets are represented as tokens**. The controller must handle variable numbers and types of assets at runtime without retraining — buildings with/without EVs, batteries, different charger counts, assets that connect/disconnect dynamically.

The ULC receives tokenized observations and returns **one action per controllable asset**, with variable output structure matching the input set.

---

## Data Pipeline

Understanding how observations flow is essential before implementing the tokenizer.

```
CityLearn env.step()
    │  raw observations: List[List[float]]  (one list per building)
    ▼
Wrapper (utils/wrapper_citylearn.py)
    │  Applies encoder rules from configs/encoders/default.json
    │  (PeriodicNorm, OneHot, NormalizeWithMissing, RemoveFeature, etc.)
    │  Output: List[np.ndarray] — flat encoded arrays per building
    ▼
Agent.predict(encoded_observations)
    │  Tokenizer slices encoded vector, groups by asset, projects to d_model
    │  Output: ca_tokens [N_ca, d_model], sro_tokens [N_sro, d_model], rl_token [1, d_model]
    ▼
Transformer Encoder (shared backbone)
    │  Concatenates all tokens, adds type embeddings, self-attention
    │  Output: contextual embeddings for all token positions
    ▼
Actor Head (applied to CA positions only)       Critic Head (pooled over all positions)
    │  Output: actions [N_ca, 1]                    │  Output: V(s) scalar
```

### Key Facts

- Tokenization happens **inside the agent**, not in the wrapper. The wrapper delivers `List[np.ndarray]` as it does today.
- Zero changes to wrapper or existing agents (MADDPG, RBC).
- `attach_environment()` provides **raw feature names** (e.g., `["month", "electric_vehicle_soc", ...]`), but `predict()` receives **post-encoded** vectors with expanded dimensions.
- Only `RuleBasedPolicy` bypasses encoding (`use_raw_observations = True`).

### Post-Encoding Dimension Expansion

The tokenizer must map raw feature names → post-encoding indices. The encoder rules expand dimensions:

| Feature | Raw → Encoded | Encoder |
|---------|:---:|---------|
| `month` | 1 → **2** | PeriodicNormalization (sin, cos) |
| `hour` | 1 → **2** | PeriodicNormalization (sin, cos) |
| `day_type` | 1 → **8** | OnehotEncoding (8 classes) |
| `connected_state` | 1 → **2** | OnehotEncoding ([0, 1]) |
| `departure_time` | 1 → **26** | OnehotEncoding (26 classes) |
| `arrival_time` | 1 → **26** | OnehotEncoding (26 classes) |
| `electrical_storage_soc` | 1 → **1** | NormalizeWithMissing |
| `electricity_pricing` | 1 → **1** | NoNormalization |
| Weather features | 1 → **0** | RemoveFeature (dropped entirely) |

**Implication:** The tokenizer needs both `observation_names` (raw, from `attach_environment()`) AND the encoder config (`configs/encoders/default.json`) to build the correct index map at init time.

---

## Token Taxonomy

| Type ID | Type | Definition | Has Action? | Examples |
|:---:|------|------------|:---:|---------|
| 0 | **CA** | Controllable Asset — device the agent controls | Yes (1 per CA) | EV charger, battery, washing machine |
| 1 | **SRO** | Shared Read-Only — external signal the building cannot influence | No | Weather, pricing, carbon, temporal |
| 2 | **RL** | Residual Load — building's net passive demand after local generation | No | `non_shiftable_load − solar_generation` |

### RL (Residual Load) Token

PV generation is **folded into the RL token**: total non-flexible demand minus local generation. If demand = 100 kW and PV = 20 kW, RL = 80 kW. Can go negative when PV exceeds demand.

The agent sees a single value: "what the building needs that I can't control."

---

## Tokenizer Specification

This section is the implementation spec for the tokenizer module.

### Interface

```python
class ObservationTokenizer(nn.Module):
    """Converts a flat post-encoded observation vector into typed token embeddings."""

    def __init__(self, observation_names: List[str], action_names: List[str],
                 encoder_config: dict, tokenizer_config: dict, d_model: int):
        """
        Args:
            observation_names: Raw feature names for this building (from attach_environment)
            action_names: Action names for this building (from attach_environment)
            encoder_config: Loaded configs/encoders/default.json
            tokenizer_config: The algorithm.tokenizer section from YAML
            d_model: Embedding dimension for all tokens
        """
        # 1. Build index map: raw feature name → post-encoding slice indices
        # 2. Classify features into token groups using tokenizer_config
        # 3. Create per-type Linear projections

    def forward(self, encoded_obs: torch.Tensor) -> TokenizedObservation:
        """
        Args:
            encoded_obs: [batch, obs_dim] flat post-encoded observation vector

        Returns:
            TokenizedObservation with:
                ca_tokens:   [batch, N_ca, d_model]   — one per controllable asset
                sro_tokens:  [batch, N_sro, d_model]   — one per SRO group
                rl_token:    [batch, 1, d_model]        — residual load
                ca_types:    List[str]                   — type name per CA token (for output ordering)
                n_ca:        int                         — number of CA tokens
        """
```

### Configurable Token Type Registry

Token types are **defined in configuration** (not hardcoded). New asset types or SRO categories can be added without code changes.

```yaml
algorithm:
  tokenizer:
    ca_types:
      ev_charger:
        features:                                    # substring-matched against observation_names
          - electric_vehicle_charger_connected_state
          - electric_vehicle_soc
          - electric_vehicle_departure_time
          - electric_vehicle_required_soc_departure
          - electric_vehicle_battery_capacity
          - electric_vehicle_incoming_state
          - electric_vehicle_arrival_time
        action_name: electric_vehicle_storage        # substring-matched against action_names

      battery:
        features:
          - electrical_storage_soc
        action_name: electrical_storage

      washing_machine:
        features:
          - washing_machine_start_time_step
          - washing_machine_end_time_step
          - washing_machine_load_profile
        action_name: washing_machine

    sro_types:
      temporal:                                      # ← global time context
        features:
          - month
          - hour
          - day_type
      weather:
        features:
          - outdoor_dry_bulb_temperature
          - outdoor_relative_humidity
          - diffuse_solar_irradiance
          - direct_solar_irradiance
      pricing:
        features:
          - electricity_pricing
          - electricity_pricing_predicted_6h
          - electricity_pricing_predicted_12h
          - electricity_pricing_predicted_24h
      carbon:
        features:
          - carbon_intensity

    rl:
      demand_feature: non_shiftable_load
      generation_features:
        - solar_generation
```

### Classification & Grouping Algorithm

At init time (`attach_environment`):

1. Iterate over `observation_names` for this building
2. For each name, try substring match against `ca_types[*].features` — if matched, assign to that CA type
3. If a CA type has multiple instances (e.g., 2 EV chargers), CityLearn appends a device ID suffix (e.g., `_charger_15_1`). Features sharing the same suffix are grouped into one CA token. Features with no suffix form a single-instance CA.
4. If not a CA, try substring match against `sro_types[*].features` — if matched, assign to that SRO group
5. If not an SRO, check against `rl.demand_feature` and `rl.generation_features`
6. Any unmatched feature → log warning

Then build the **index map** using encoder config to determine how many post-encoding dimensions each raw feature occupies.

### Multi-Instance CA Detection

CityLearn appends device IDs as suffixes when a building has multiple instances of the same CA type:

| Scenario | Observation Name | Action Name |
|----------|-----------------|-------------|
| Single charger | `electric_vehicle_charger_connected_state` | `electric_vehicle_storage` |
| Multi-charger | `electric_vehicle_charger_connected_state_charger_15_1` | `electric_vehicle_storage_charger_15_1` |

The tokenizer uses **substring/contains matching** (not exact match), which handles suffixed names naturally. The suffix is used to group features of the same physical device into one token.

**Dataset evidence:** Building_15 has 2 chargers; `teste_181126_050126_15` has 21 chargers.

### Projection Layers (Per-Type Linear)

Each token type has its own `Linear(n_encoded_features, d_model)` projection:

```
CA projections (one Linear per CA type, shared across instances of that type):
  ev_charger:      Linear(59, d_model)   # 7 raw features → 59 post-encoding dims
  battery:         Linear(1, d_model)    # 1 raw feature  → 1 post-encoding dim
  washing_machine: Linear(3, d_model)    # 3 raw features → 3 post-encoding dims

SRO projections (one Linear per SRO group):
  temporal:  Linear(12, d_model)   # month(2) + hour(2) + day_type(8) = 12 post-encoding dims
  weather:   Linear(0, -)          # all weather features are RemoveFeature → 0 dims (skip)
  pricing:   Linear(4, d_model)    # 4 features × 1 dim each
  carbon:    Linear(1, d_model)    # 1 feature × 1 dim

RL projection:
  Linear(n, d_model)  # n = dims of (demand - generation) post-encoding
```

> **Note:** Weather features are dropped by `RemoveFeature` encoder → 0 dimensions → the weather SRO group won't produce a token. The tokenizer should handle this gracefully (skip groups with 0 features after encoding).

### Action Dimension

Every CA has exactly **1 scalar action** in `[-1, 1]` (normalized). `action_dimension[i]` = number of CAs in building `i`. Output action `j` corresponds to CA token `j` — structural by position.

---

## Architecture Direction

The agent will be implemented as **`AgentTransformerPPO`** — a PPO-based agent where the policy network is a Transformer that receives the tokenized observations described above.

### Actor-Critic in PPO

PPO uses both an actor and a critic:
- **Actor** (policy): Transformer encoder → actor head on CA positions → actions `[N_ca, 1]`
- **Critic** (value function): same Transformer encoder → critic head on pooled representation → scalar $V(s)$

Shared encoder, separate heads.

### Per-Building Agent Instances

Each building has its own `AgentTransformerPPO` instance (same class, separate weights). This mirrors the existing MADDPG pattern and fits the current runner/wrapper infrastructure, which already iterates per-building. The model architecture is identical across buildings — different buildings are different instances learning from their own observations and rewards.

> **Future: weight sharing.** The token-based architecture is explicitly designed so that a single shared model could process buildings with different asset compositions without any architectural changes. Enabling this would require modifications to the training loop (batching across buildings) and is listed under Deferred Features. The per-type projection layers and type embeddings will transfer directly when this is implemented.

### Core Contribution

The variable input/output tokenization architecture is algorithm-agnostic. PPO was chosen for stability and simplicity (on-policy, no replay buffer). The tokenizer and Transformer policy could be used with any RL algorithm.

---

## Key Codebase Integration Points

| File | Role | Tokenizer-Relevant? |
|------|------|:---:|
| `algorithms/agents/base_agent.py` | Contract: `predict()`, `update()`, `attach_environment()` | Yes — interface |
| `utils/wrapper_citylearn.py` | Training loop, encoding, observation delivery | Yes — understand data flow |
| `configs/encoders/default.json` | Encoder rules determining post-encoding dims | **Critical** |
| `algorithms/agents/maddpg_agent.py` | Reference implementation | Study pattern |
| `algorithms/registry.py` | Register: `"AgentTransformerPPO": AgentTransformerPPO` | Later |
| `utils/config_schema.py` | Pydantic validation for config | Later |
| `run_experiment.py` | Entrypoint | Reference only |

## Known Dataset Assets

| Dataset | Building | CAs | Notes |
|---------|----------|-----|-------|
| `citylearn_challenge_2022_phase_all_plus_evs` | Building_1 | 1 charger + 1 washing machine + 1 battery + PV | Full mix |
| | Building_15 | 2 chargers | Multi-instance |
| | Buildings 4,5,7,10,12 | 1 charger each | Single CA |
| `teste_181126_050126_15` | i-charging HQ | 21 chargers | Stress test |

## PoC Reference

The `experimentation/` folder contains a working proof-of-concept (pre-tokenizer, uses synthetic data):
- **poc_transformer.py**: `SimpleSetTransformerPolicy` — validates variable cardinality
- **policy.py**: `TransformerPolicy` — cleaner interface
- **test_policy.py**: Tests R1 (variable N_ca), R2 (output count = N_ca), R3 (cross-token conditioning)

## Evaluation Plan (from Professor)

Test on building archetypes: no CAs, EV only, BESS only, PV+BESS, EV+PV, EV+PV+BESS, multi-charger/office.

Evaluate: generalization across asset counts, robustness to connect/disconnect, token ordering sensitivity, representation quality, action stability.

## Deferred Features

- **Receding horizon** — 15-step planning output (future work)
- **GRU/LSTM temporal memory** — recurrence for temporal dependencies (future work)
- **Weight sharing across buildings** — transfer learning via shared weights (no architectural changes needed — per-type encoders transfer naturally)

---

## Decision Log

All architectural decisions are resolved. No blocking open questions.

| # | Question | Resolution |
|---|----------|------------|
| 1 | PV / Building Context token type | PV folded into RL token (demand − generation) |
| 2 | NFC naming | **RL** (Residual Load) — standard energy term |
| 3 | Encoded dimension expansion | Tokenizer uses raw names for classification + encoder config for post-encoding index mapping |
| 4 | Multi-building handling | One agent instance per building, same model class |
| 5 | Multi-instance CA detection | CityLearn appends device ID suffix; tokenizer uses substring matching |
| 6 | Action dimension per CA type | Always 1 scalar per CA, range [-1, 1] |
| 7 | PPO critic architecture | Shared Transformer encoder, separate actor/critic heads |
| 8 | Pre vs post encoding | Post-encoding confirmed; tokenizer must handle expanded indices |
| 9 | Per-type vs shared encoder | **Per-type** `Linear(n_features, d_model)` per CA/SRO type |
| 10 | Temporal features (month, hour, day_type) | Classified as SRO type `temporal` |
| 11 | Weather features after RemoveFeature | 0 dims post-encoding → weather SRO group skipped at runtime |
