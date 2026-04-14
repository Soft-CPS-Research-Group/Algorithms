# Universal Local Controller — Specification

## 1. Objective & Core Principles

### Objective

Build a **Universal Local Controller (ULC)** that manages energy assets within a building using a Transformer-based architecture where assets are represented as tokens. The controller must handle variable numbers and types of assets at runtime without retraining — buildings with/without EVs, batteries, different charger counts, assets that connect/disconnect dynamically.

The system processes three types of tokens:
- **CA (Controllable Assets)** — Devices the agent controls (EV chargers, batteries, washing machines). Each CA produces one action.
- **SRO (Shared Read-Only)** — External context signals the building cannot influence (weather, pricing, carbon, temporal).
- **NFC (Non-Flexible Context / Residual Load)** — Building's net passive demand after local generation (demand − PV generation).

### Core Principles

1. **Dynamic over heuristic** — Token boundaries are determined by explicit marker values in the observation tensor, not by substring matching or name-based inference.

2. **Runtime adaptability** — The architecture handles topology changes mid-episode. When observation count changes, the system detects it, triggers any pending PPO update, flushes the rollout buffer, and adapts.

3. **Modular components** — Enricher (portable), Tokenizer (generic), Agent (RL-specific) are separate, reusable classes.

4. **Portability** — Observation processing logic (enricher) must work identically in training and production inference repos.

5. **Per-type projections** — Each asset type has its own learned projection `Linear(input_dim, d_model)`. All types are known from config with explicit dimensions.

---

## 2. Data Pipeline & Flow

### End-to-End Flow

```
CityLearn env.step()
    │ raw observations: List[List[float]] (per building)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ WRAPPER (for Transformer agents only)                       │
│                                                             │
│ 1. Topology change detection                                │
│    - Compare current observation count vs previous          │
│    - If changed: rebuild enricher, rebuild encoders         │
│    - Notify agent (triggers PPO update + buffer flush)      │
│                                                             │
│ 2. Enrich values                                            │
│    - ObservationEnricher.enrich_values(raw_obs)            │
│    - Injects marker values: [1001, 0.65, 1002, 0.8, ...]   │
│                                                             │
│ 3. Encode                                                   │
│    - Apply encoders (markers use NoNormalization)           │
│    - Output: flat encoded vector with markers preserved     │
└─────────────────────────────────────────────────────────────┘
    │ List[np.ndarray] — flat encoded with markers
    ▼
┌─────────────────────────────────────────────────────────────┐
│ AGENT.predict()                                             │
│                                                             │
│ 4. Tokenize                                                 │
│    - ObservationTokenizer scans for marker values           │
│    - Splits tensor into CA/SRO/NFC groups                   │
│    - Projects each group to d_model                         │
│    - Returns TokenizedObservation                           │
│                                                             │
│ 5. Transformer backbone                                     │
│    - Concatenate: [CA tokens, SRO tokens, NFC token]        │
│    - Add type embeddings (CA=0, SRO=1, NFC=2)              │
│    - Self-attention over all tokens                         │
│    - Output: contextual embeddings                          │
│                                                             │
│ 6. Heads                                                    │
│    - Actor: CA embeddings → actions [N_ca, 1]              │
│    - Critic: mean-pooled embeddings → V(s) scalar          │
└─────────────────────────────────────────────────────────────┘
    │ actions: List[np.ndarray]
    ▼
CityLearn env.step(actions)
```

### Marker Value Scheme

| Token Family | Base Value | Instance Pattern | Example |
|--------------|------------|------------------|---------|
| CA | 1000 | 1001, 1002, 1003... | Battery=1001, EV1=1002, EV2=1003 |
| SRO | 2000 | 2001, 2002, 2003... | Temporal=2001, Pricing=2002, Carbon=2003 |
| NFC | 3000 | 3001 (single) | NFC=3001 |

Marker values are configurable in the tokenizer config.

### Marker Injection Order

The enricher injects markers in the order CityLearn expects actions. This ensures `tokenizer output[i]` maps directly to `action[i]` without reordering.

---

## 3. ObservationEnricher (Portable Module)

### Responsibility

Classifies raw observation features into token groups and injects marker values. Portable — works in both training wrapper and production inference.

### Interface

```python
class ObservationEnricher:
    def __init__(self, tokenizer_config: Dict[str, Any]) -> None:
        """
        Args:
            tokenizer_config: Loaded from configs/tokenizers/default.json
                Contains ca_types, sro_types, nfc, marker_values
        """

    def enrich_names(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> EnrichmentResult:
        """Classify features and produce enriched observation names.
        
        Called once per topology (cached until topology changes).
        
        Returns:
            EnrichmentResult with:
              - enriched_names: observation names with marker names inserted
              - marker_encoder_specs: encoder specs for markers (NoNormalization)
              - marker_positions: where markers are in the enriched list
        """

    def enrich_values(
        self,
        observation_values: List[float],
    ) -> List[float]:
        """Inject marker values at cached positions.
        
        Called every step. Uses positions cached from enrich_names().
        
        Returns:
            Enriched values with markers: [1001, 0.65, 1002, 0.8, ...]
        """

    def topology_changed(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> bool:
        """Check if topology differs from cached state."""
```

### Classification Logic

1. **CA classification:** Match features to CA types using configured `features` patterns. Group by device ID extracted from action names.
2. **SRO classification:** Match remaining features to SRO types using configured patterns.
3. **NFC classification:** Match remaining features to NFC patterns (demand, generation, extra).
4. **Unmatched:** Log warning, include in tensor but not tokenized.

### Portability Constraints

- **No imports** from `algorithms.*`, `utils.*`, or training-specific modules
- **No PyTorch/NumPy** — pure Python with stdlib + typing only
- Copy-pasteable to `energAIze_inference` repo

### File Location

`algorithms/utils/observation_enricher.py`

---

## 4. ObservationTokenizer (Generic, Reusable)

### Responsibility

Scans encoded observation tensor for marker values, splits into token groups, and projects each group to `d_model`. Generic — reusable by any Transformer-based agent.

### Interface

```python
class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        d_model: int,
    ) -> None:
        """
        Args:
            tokenizer_config: Loaded from configs/tokenizers/default.json
                Contains ca_types, sro_types, nfc with input_dim per type
            d_model: Embedding dimension for all tokens
        
        Creates:
            - Per-CA-type projection: Linear(input_dim, d_model)
            - Per-SRO-type projection: Linear(input_dim, d_model)
            - NFC projection: Linear(input_dim, d_model)
        """

    def forward(
        self,
        encoded_obs: torch.Tensor,  # [batch, obs_dim]
    ) -> TokenizedObservation:
        """
        Scans for marker values, splits into groups, projects to d_model.
        
        Returns:
            TokenizedObservation with:
              - ca_tokens: [batch, N_ca, d_model]
              - sro_tokens: [batch, N_sro, d_model]
              - nfc_token: [batch, 1, d_model]
              - ca_types: List[str] — type name per CA token
              - n_ca: int — number of CA tokens
        """
```

### Marker Scanning Logic

At forward time:
1. Scan tensor for values in CA range (1001-1999) → extract CA groups
2. Scan tensor for values in SRO range (2001-2999) → extract SRO groups
3. Scan tensor for NFC marker (3001) → extract NFC group
4. For each group: gather features, look up projection by type, project to `d_model`

### Marker Value → Type Mapping

The tokenizer needs to know which projection to use for each marker. This mapping is established by the enricher's injection order and communicated via a **marker registry**:

| Marker Value | Type | Projection Key |
|--------------|------|----------------|
| 1001 | First CA instance | Determined by `input_dim` match from config |
| 1002 | Second CA instance | Determined by `input_dim` match from config |
| 2001 | First SRO type | "temporal" (or by config order) |
| 2002 | Second SRO type | "pricing" (or by config order) |
| 3001 | NFC | "nfc" |

**Implementation approach:** The tokenizer counts features between markers. If a CA group has 1 feature, it's a battery (`input_dim: 1`). If it has 61 features, it's an EV charger (`input_dim: 61`). The `input_dim` in config uniquely identifies the type.

For SRO types, the enricher injects markers in a fixed config-defined order (temporal → pricing → carbon), so marker 2001 is always temporal, 2002 is always pricing, etc.

### Projection Layer Management

- All projections created at `__init__` based on types defined in config
- Each type has explicit `input_dim` in config — no inference needed
- Projections are `nn.ModuleDict` keyed by type name

### TokenizedObservation Dataclass

```python
@dataclass
class TokenizedObservation:
    ca_tokens: torch.Tensor     # [batch, N_ca, d_model]
    sro_tokens: torch.Tensor    # [batch, N_sro, d_model]
    nfc_token: torch.Tensor     # [batch, 1, d_model]
    ca_types: List[str]         # Type name per CA position
    n_ca: int                   # Number of CA tokens
```

### File Location

`algorithms/utils/observation_tokenizer.py`

---

## 5. Transformer Backbone

### Responsibility

Processes the concatenated token sequence through self-attention, producing contextual embeddings where each token has attended to all others.

### Interface

```python
class TransformerBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Creates:
            - Type embeddings: nn.Embedding(3, d_model) for CA=0, SRO=1, NFC=2
            - TransformerEncoder with specified layers
        """

    def forward(
        self,
        ca_tokens: torch.Tensor,    # [batch, N_ca, d_model]
        sro_tokens: torch.Tensor,   # [batch, N_sro, d_model]
        nfc_token: torch.Tensor,    # [batch, 1, d_model]
    ) -> TransformerOutput:
        """
        1. Concatenate: [CA tokens, SRO tokens, NFC token]
        2. Add type embeddings
        3. Self-attention through TransformerEncoder
        4. Return contextual embeddings
        """
```

### TransformerOutput Dataclass

```python
@dataclass
class TransformerOutput:
    all_embeddings: torch.Tensor   # [batch, N_total, d_model]
    ca_embeddings: torch.Tensor    # [batch, N_ca, d_model] — sliced from all
    pooled: torch.Tensor           # [batch, d_model] — mean over all tokens
    n_ca: int
```

### Design Decisions

- **No positional embeddings** — Pure set semantics; order doesn't matter within token types
- **Type embeddings only** — Distinguish CA/SRO/NFC via learned type vectors
- **Shared backbone** — Actor and critic both use the same contextual embeddings

### File Location

`algorithms/utils/transformer_backbone.py`

---

## 6. PPO Components

### Responsibility

PPO-specific components: Actor head, Critic head, Rollout buffer, and PPO loss computation. These are specific to the PPO algorithm and may be replaced for other RL implementations.

### Actor Head

```python
class ActorHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """
        MLP applied to each CA embedding independently.
        Outputs action mean + learned log_std per CA type.
        """

    def forward(
        self,
        ca_embeddings: torch.Tensor,  # [batch, N_ca, d_model]
        ca_types: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            actions: [batch, N_ca, 1] — sampled, tanh-squashed to [-1, 1]
            log_probs: [batch, N_ca] — log probability of sampled actions
            means: [batch, N_ca, 1] — action means (for deterministic mode)
        """
```

### Critic Head

```python
class CriticHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        """MLP that takes pooled embedding, outputs scalar V(s)."""

    def forward(
        self,
        pooled: torch.Tensor,  # [batch, d_model]
    ) -> torch.Tensor:
        """Returns: value [batch, 1]"""
```

### Rollout Buffer

```python
class RolloutBuffer:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        """On-policy buffer for PPO trajectories."""

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Store one transition."""

    def compute_returns_and_advantages(self) -> None:
        """Compute GAE advantages and discounted returns."""

    def get_batches(self, batch_size: int) -> Iterator[Batch]:
        """Yield minibatches for PPO epochs."""

    def clear(self) -> None:
        """Reset buffer after PPO update."""
    
    def flush_and_update(self) -> bool:
        """
        Called on topology change.
        Returns True if buffer had enough data to trigger update.
        """
```

### PPO Loss

```python
def compute_ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coeff: float,
    entropy_coeff: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO clipped surrogate loss + value loss + entropy bonus.
    
    Returns:
        total_loss: Combined loss for backprop
        metrics: Dict with policy_loss, value_loss, entropy for logging
    """
```

### File Location

`algorithms/utils/ppo_components.py`

---

## 7. AgentTransformerPPO

### Responsibility

Full agent implementation satisfying the `BaseAgent` contract. Combines the generic tokenizer with PPO-specific components.

### Interface

```python
class AgentTransformerPPO(BaseAgent):
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Reads from config:
            - algorithm.transformer: d_model, nhead, num_layers, etc.
            - algorithm.hyperparameters: lr, gamma, clip_eps, etc.
            - algorithm.tokenizer_config_path: path to tokenizer config
        
        Creates (per building):
            - ObservationTokenizer
            - TransformerBackbone
            - ActorHead, CriticHead
            - RolloutBuffer
            - Optimizer
        """

    def attach_environment(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Space],
        observation_space: List[Space],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Called by wrapper after environment setup.
        Stores metadata for validation (not used for tokenizer setup,
        since tokenizer works dynamically from marker values).
        """

    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[np.ndarray]:
        """
        For each building:
            1. Tokenize encoded observation
            2. Transformer backbone
            3. Actor head → actions
            4. If training: store in rollout buffer
        
        Returns: actions per building
        """

    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: List[bool],
        truncated: List[bool],
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> Dict[str, float]:
        """
        PPO on-policy update:
            1. Add transition to rollout buffer
            2. If update_step and buffer ready:
               - Compute GAE advantages
               - K epochs of minibatch updates
               - Clear buffer
        
        Returns: metrics dict for logging
        """

    def on_topology_change(self, building_idx: int) -> None:
        """
        Called by wrapper when topology changes mid-episode.
        Triggers PPO update (if buffer has data) then flushes buffer.
        """

    def export_artifacts(
        self,
        output_dir: Path,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Export ONNX models and return manifest metadata."""

    def save_checkpoint(self, output_dir: Path, step: int) -> None:
        """Save all network weights and optimizer state."""

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore from checkpoint."""
```

### Per-Building Architecture

Each building has its own:
- Tokenizer instance (same class, same projections, independent forward)
- Rollout buffer
- Potentially own optimizer (or shared, TBD based on training approach)

The Transformer backbone and heads are shared architecture but separate weight instances per building.

### File Location

`algorithms/agents/transformer_ppo_agent.py`

---

## 8. Wrapper Integration

### Responsibility

The wrapper (`utils/wrapper_citylearn.py`) handles enrichment and topology change detection for Transformer agents.

### New Wrapper Behavior (Transformer Agents Only)

```python
class CityLearnWrapper:
    def __init__(self, ...):
        # Existing init...
        
        # New: per-building enrichers (None for non-Transformer agents)
        self._enrichers: List[Optional[ObservationEnricher]] = []
        self._is_transformer_agent: bool = False

    def _setup_for_transformer_agent(self, tokenizer_config: Dict) -> None:
        """Called when agent is Transformer-based."""
        self._is_transformer_agent = True
        for i in range(self.num_buildings):
            self._enrichers.append(ObservationEnricher(tokenizer_config))

    def _detect_topology_change(self, building_idx: int) -> bool:
        """Compare current observation names with cached state."""
        if not self._is_transformer_agent:
            return False
        return self._enrichers[building_idx].topology_changed(
            self.observation_names[building_idx],
            self.action_names[building_idx],
        )

    def _handle_topology_change(self, building_idx: int) -> None:
        """Rebuild enricher/encoders and notify agent."""
        # 1. Re-enrich names
        enrichment = self._enrichers[building_idx].enrich_names(
            self.observation_names[building_idx],
            self.action_names[building_idx],
        )
        
        # 2. Rebuild encoders for this building
        self._rebuild_encoders(building_idx, enrichment)
        
        # 3. Notify agent
        self.model.on_topology_change(building_idx)

    def get_encoded_observations(self, building_idx: int, raw_obs: List[float]) -> np.ndarray:
        """Encode observations, with enrichment for Transformer agents."""
        if self._is_transformer_agent:
            enriched_obs = self._enrichers[building_idx].enrich_values(raw_obs)
            return self._encode(building_idx, enriched_obs)
        else:
            return self._encode(building_idx, raw_obs)
```

### Topology Change Detection Point

Detection happens at the start of each step, before encoding:
1. Wrapper receives new observation names from CityLearn
2. Compare with cached names
3. If changed → handle topology change → continue with new topology

### Non-Transformer Agents

When agent is MADDPG, RBC, or other non-Transformer:
- `_enrichers` list contains `None` for all buildings
- `_is_transformer_agent = False`
- Existing wrapper behavior unchanged

### File Location

`utils/wrapper_citylearn.py` (modifications to existing file)

---

## 9. Config Structure

### Tokenizer Config (New File)

**Location:** `configs/tokenizers/default.json`

> **Implementation Note:** The `input_dim` values below are calculated from the current encoder rules (`configs/encoders/default.json`) and dataset schema. These values should be verified once more during implementation to ensure they match the actual encoded dimensions.

```json
{
  "marker_values": {
    "ca_base": 1000,
    "sro_base": 2000,
    "nfc": 3001
  },
  "ca_types": {
    "battery": {
      "features": ["electrical_storage_soc"],
      "action_name": "electrical_storage",
      "input_dim": 1
    },
    "ev_charger": {
      "features": [
        "electric_vehicle_charger_connected_state",
        "connected_electric_vehicle_at_charger_battery_capacity",
        "connected_electric_vehicle_at_charger_departure_time",
        "connected_electric_vehicle_at_charger_required_soc_departure",
        "connected_electric_vehicle_at_charger_soc",
        "electric_vehicle_charger_incoming_state",
        "incoming_electric_vehicle_at_charger_estimated_arrival_time"
      ],
      "action_name": "electric_vehicle_storage",
      "input_dim": 61
    },
    "washing_machine": {
      "features": [
        "washing_machine_start_time_step",
        "washing_machine_end_time_step",
        "washing_machine_load_profile"
      ],
      "action_name": "washing_machine",
      "input_dim": 3
    }
  },
  "sro_types": {
    "temporal": {
      "features": ["month", "hour", "day_type"],
      "input_dim": 12
    },
    "pricing": {
      "features": [
        "electricity_pricing",
        "electricity_pricing_predicted_1",
        "electricity_pricing_predicted_2",
        "electricity_pricing_predicted_3"
      ],
      "input_dim": 4
    },
    "carbon": {
      "features": ["carbon_intensity"],
      "input_dim": 1
    }
  },
  "nfc": {
    "demand_features": ["non_shiftable_load"],
    "generation_features": ["solar_generation"],
    "extra_features": ["net_electricity_consumption"],
    "input_dim": 3
  }
}
```

**Dimension breakdown:**

| Type | Raw Features | Encoded Dims | Encoding Applied |
|------|--------------|--------------|------------------|
| battery | 1 | 1 | NoNormalization |
| ev_charger | 7 | 61 | 2×OneHot(2) + 2×OneHot(27) + 3×Normalize |
| washing_machine | 3 | 3 | NoNormalization |
| temporal | 3 | 12 | 2×Periodic(2) + 1×OneHot(8) |
| pricing | 4 | 4 | NoNormalization |
| carbon | 1 | 1 | NoNormalization |
| nfc | 3 | 3 | NoNormalization |

### Algorithm Config (Template)

**Location:** `configs/templates/transformer_ppo.yaml`

```yaml
algorithm:
  name: AgentTransformerPPO
  tokenizer_config_path: configs/tokenizers/default.json
  
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 128
    dropout: 0.1
  
  hyperparameters:
    learning_rate: 3.0e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    ppo_epochs: 4
    minibatch_size: 64
    entropy_coeff: 0.01
    value_coeff: 0.5
    max_grad_norm: 0.5
```

### Config Schema Update

**Location:** `utils/config_schema.py`

Add Pydantic models for:
- `TokenizerConfig` — validates tokenizer JSON structure
- `TransformerConfig` — d_model, nhead, num_layers, etc.
- `TransformerPPOHyperparameters` — PPO-specific params
- `TransformerPPOAlgorithmConfig` — combines above, adds to discriminated union

---

## 10. Test Plan

### Unit Tests (During Development)

Each component should have unit tests written alongside implementation:

**ObservationEnricher Tests** (`tests/test_observation_enricher.py`)

| Test | What it verifies |
|------|------------------|
| `test_enrich_names_single_ca` | Single battery: marker inserted before battery features |
| `test_enrich_names_multi_ca` | Battery + 2 EVs: separate markers with correct instance IDs (1001, 1002, 1003) |
| `test_enrich_names_sro_ordering` | SRO markers (2001, 2002...) inserted for temporal, pricing, carbon |
| `test_enrich_names_nfc` | NFC marker (3001) inserted before demand/generation features |
| `test_enrich_values_marker_injection` | Marker values appear at correct positions in enriched values |
| `test_enrich_values_preserves_originals` | Original observation values unchanged, just interleaved with markers |
| `test_topology_changed_detection` | Returns True when observation count changes |
| `test_topology_unchanged` | Returns False when same observations |
| `test_portability_no_external_deps` | Module imports only stdlib + typing |

**ObservationTokenizer Tests** (`tests/test_observation_tokenizer.py`)

| Test | What it verifies |
|------|------------------|
| `test_scan_markers_ca` | Correctly identifies CA markers (1001-1999 range) |
| `test_scan_markers_sro` | Correctly identifies SRO markers (2001-2999 range) |
| `test_scan_markers_nfc` | Correctly identifies NFC marker (3001) |
| `test_split_groups_correct_sizes` | Each group has expected number of features |
| `test_projection_shapes` | Output tokens have shape `[batch, N, d_model]` |
| `test_variable_ca_count` | Same tokenizer handles 1, 2, 3 CAs dynamically |
| `test_variable_sro_count` | Handles missing SRO types gracefully |
| `test_tokenized_observation_fields` | Returns correct `n_ca`, `ca_types` metadata |

**TransformerBackbone Tests** (`tests/test_transformer_backbone.py`)

| Test | What it verifies |
|------|------------------|
| `test_forward_shape` | Output shape matches input token count |
| `test_type_embeddings_applied` | CA/SRO/NFC get different type embeddings |
| `test_variable_token_count` | Same backbone handles different N_ca, N_sro |
| `test_ca_embeddings_sliced` | First N_ca outputs correctly extracted |
| `test_pooled_embedding` | Mean pooling over all tokens |
| `test_gradient_flow` | Gradients propagate through attention |

**PPO Components Tests** (`tests/test_ppo_components.py`)

| Test | What it verifies |
|------|------------------|
| `test_actor_output_shape` | Actions shape `[batch, N_ca, 1]` |
| `test_actor_output_range` | Actions in `[-1, 1]` after tanh |
| `test_actor_log_probs` | Log probabilities computed correctly |
| `test_critic_output_scalar` | Single value per batch |
| `test_rollout_buffer_add` | Transitions stored correctly |
| `test_rollout_buffer_gae` | GAE advantages computed |
| `test_rollout_buffer_batches` | Minibatch iteration works |
| `test_rollout_buffer_flush` | Triggers update and clears |
| `test_ppo_loss_clipping` | Clipped surrogate loss correct |

**AgentTransformerPPO Tests** (`tests/test_agent_transformer_ppo.py`)

| Test | What it verifies |
|------|------------------|
| `test_instantiation` | Agent creates with valid config |
| `test_predict_shape` | Output action count matches N_ca |
| `test_predict_deterministic` | Deterministic mode uses means |
| `test_predict_stochastic` | Stochastic mode samples and stores |
| `test_update_metrics` | Update returns loss metrics dict |
| `test_on_topology_change` | Buffer flushed, no crash |
| `test_checkpoint_save_load` | Round-trip preserves weights |
| `test_multi_building` | Independent per-building operation |

**Wrapper Integration Tests** (`tests/test_wrapper_transformer.py`)

| Test | What it verifies |
|------|------------------|
| `test_enricher_created_for_transformer` | Enrichers initialized when Transformer agent |
| `test_no_enricher_for_maddpg` | Enrichers None for non-Transformer agents |
| `test_topology_change_detected` | Wrapper detects observation count change |
| `test_topology_change_rebuilds_encoders` | Encoders rebuilt on change |
| `test_topology_change_notifies_agent` | `on_topology_change()` called |
| `test_encoded_obs_contains_markers` | Marker values present in encoded output |

### End-to-End Validation Tests

After all components are implemented:

**Full Pipeline Tests** (`tests/test_e2e_transformer_ppo.py`)

| Test | What it verifies |
|------|------------------|
| `test_single_building_single_episode` | Complete training loop runs without crash |
| `test_reward_not_constant` | Agent takes varying actions, rewards fluctuate |
| `test_actions_valid_range` | All actions in `[-1, 1]`, no NaN |
| `test_kpis_generated` | Output files created (result.json, summary.json) |
| `test_topology_change_mid_episode` | Simulated CA connect/disconnect handled |
| `test_multi_building_independence` | Buildings don't interfere with each other |

**Architecture Validation Tests**

| Test | What it verifies |
|------|------------------|
| `test_variable_ca_runtime` | Same model handles 1 CA, then 3 CAs in sequence |
| `test_output_count_matches_input` | N_actions == N_ca always |
| `test_sro_affects_context` | Perturbing SRO changes CA outputs (cross-attention works) |

---

## 11. File Structure & Work Packages

### New Files to Create

```
algorithms/
├── agents/
│   └── transformer_ppo_agent.py       # AgentTransformerPPO
├── utils/
│   ├── observation_enricher.py        # ObservationEnricher (portable)
│   ├── observation_tokenizer.py       # ObservationTokenizer (generic)
│   ├── transformer_backbone.py        # TransformerBackbone
│   └── ppo_components.py              # Actor, Critic, RolloutBuffer, PPO loss

configs/
├── tokenizers/
│   └── default.json                   # Tokenizer config (CA/SRO/NFC types)
├── templates/
│   └── transformer_ppo.yaml           # Algorithm config template

tests/
├── test_observation_enricher.py
├── test_observation_tokenizer.py
├── test_transformer_backbone.py
├── test_ppo_components.py
├── test_agent_transformer_ppo.py
├── test_wrapper_transformer.py
└── test_e2e_transformer_ppo.py
```

### Files to Modify

```
utils/
├── wrapper_citylearn.py               # Add enrichment for Transformer agents
├── config_schema.py                   # Add Pydantic models for new config

algorithms/
└── registry.py                        # Register AgentTransformerPPO
```

### Work Packages (Suggested Order)

| WP | Component | Dependencies | Deliverables |
|----|-----------|--------------|--------------|
| **WP1** | ObservationEnricher | None | `observation_enricher.py`, `test_observation_enricher.py` |
| **WP2** | Tokenizer Config | None | `configs/tokenizers/default.json` |
| **WP3** | ObservationTokenizer | WP1, WP2 | `observation_tokenizer.py`, `test_observation_tokenizer.py` |
| **WP4** | TransformerBackbone | None | `transformer_backbone.py`, `test_transformer_backbone.py` |
| **WP5** | PPO Components | None | `ppo_components.py`, `test_ppo_components.py` |
| **WP6** | Config Schema | WP2 | Updates to `config_schema.py` |
| **WP7** | Wrapper Integration | WP1, WP3 | Updates to `wrapper_citylearn.py`, `test_wrapper_transformer.py` |
| **WP8** | AgentTransformerPPO | WP3, WP4, WP5, WP7 | `transformer_ppo_agent.py`, `test_agent_transformer_ppo.py` |
| **WP9** | Registry & Template | WP6, WP8 | Updates to `registry.py`, `transformer_ppo.yaml` |
| **WP10** | E2E Validation | All above | `test_e2e_transformer_ppo.py`, validation runs |

### Dependency Graph

```
WP1 (Enricher) ──────┬──────────────────────────────────┐
                     │                                  │
WP2 (Config) ────────┼─────────┐                        │
                     │         │                        │
                     ▼         ▼                        │
               WP3 (Tokenizer) ──────┐                  │
                     │               │                  │
WP4 (Backbone) ──────┼───────────────┤                  │
                     │               │                  │
WP5 (PPO) ───────────┼───────────────┤                  │
                     │               │                  │
WP6 (Schema) ────────┼───────────────┤                  │
                     │               │                  │
                     ▼               ▼                  ▼
               WP7 (Wrapper) ────► WP8 (Agent) ◄───────┘
                                     │
                                     ▼
                              WP9 (Registry)
                                     │
                                     ▼
                              WP10 (E2E)
```

**Parallelizable:** WP1, WP2, WP4, WP5 can be developed in parallel.

---

## 12. Decisions Log

| # | Question | Decision |
|---|----------|----------|
| 1 | Marker approach | Value-based markers (1000s, 2000s, 3000s) scanned at forward time |
| 2 | CA instance differentiation | Different marker value per instance (1001, 1002, 1003...) |
| 3 | Marker collision risk | Accept low risk; use high magnitude values; make configurable |
| 4 | Projection strategy | Per-type projections with explicit `input_dim` from config |
| 5 | Topology change handling | Mid-episode support; trigger PPO update then flush buffer |
| 6 | Enrichment activation | Agent-type-driven (Transformer agents only) |
| 7 | Critic head pooling | Mean pooling over all tokens |
| 8 | Cross-topology checkpoints | Not required for production (validation test only) |
| 9 | Tokenizer config location | Separate file: `configs/tokenizers/default.json` |
| 10 | Action ordering | Marker order = action order (enricher injects in correct sequence) |
| 11 | Portability | Enricher must work in inference repo (pure Python, no ML deps) |
