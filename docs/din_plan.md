# DIN Plan — Dynamic Input Naming via Observation Enrichment

## 1. Context and Motivation

### The Goal

A single Transformer-PPO model must handle **variable numbers of Controllable Assets (CAs)** and **variable numbers of Shared REC Observations (SROs)** at runtime. When an EV charger connects or disconnects, the model adapts without rebuilding — the backbone, actor head, and critic head are already fully dynamic. The bottleneck is the **tokenizer**: how a flat encoded observation vector gets split into typed tokens.

### The Problem

The current `ObservationTokenizer` relies on `encoder_index_map.py` to figure out which indices in the post-encoded flat vector belong to which token group. This module **duplicates** the wrapper's encoder rule-matching logic (`_matches_rule`) to infer, from raw feature names and encoder specs, where each feature lands after encoding. This is:

1. **Fragile** — two independent rule-matching implementations must stay in sync.
2. **Heuristic-based** — feature classification uses substring matching on raw names, then computes encoded dimensions by re-applying encoder rules. Any mismatch between the wrapper's actual encoding and the index map's assumptions causes silent bugs.
3. **Fixed at init time** — if the observation list changes (asset connects/disconnects), the entire tokenizer must be rebuilt from scratch, including re-running the classification heuristics.

### The Root Cause

After encoding, we **lose track** of which features belong to which token. The flat encoded vector is just numbers — there's no structural information about token boundaries. The `encoder_index_map` exists solely to reconstruct this lost information, and it does so by guessing.

### The Solution: Observation Enrichment

Instead of guessing after the fact, we **inject explicit token-type markers into the observation list before encoding**. An `ObservationEnricher` class:

1. Classifies raw features by token type (using the existing tokenizer config patterns).
2. Injects marker feature names (e.g., `__tkn_ca_battery__`, `__tkn_sro_temporal__`, `__tkn_nfc__`) before each group.
3. The wrapper encodes these markers with `NoNormalization` (pass-through) — they become known positions in the encoded vector.
4. The tokenizer reads the enriched `observation_names` list, finds the `__tkn_*__` markers, and knows **exactly** where each token group starts and ends. No inference needed.

This eliminates the `encoder_index_map` module's classification role entirely.

---

## 2. Design Principles

1. **No heuristics or "magic" values.** The tokenizer must not scan the encoded tensor for sentinel numeric values. It uses feature **names** (which contain `__tkn_*__` markers) to identify token groups at init/rebuild time.

2. **Keep the flat vector format.** The `BaseAgent.predict()` interface stays `List[np.ndarray]`. No breaking changes to MADDPG, RBC, or any other agent.

3. **Subtle enrichment.** A simple pre-encoding step: classify features → inject markers → done. Not a complex state machine.

4. **Portable and self-contained.** The `ObservationEnricher` class must be copy-pasteable to the production inference repo (`energAIze_inference`) with zero dependencies on training-only code.

5. **Instance identity is NOT needed.** The model treats all batteries identically (same projection), all ev_chargers identically. Markers only encode family (CA/SRO/NFC) and type (battery/ev_charger/temporal/etc.) — not instance identity.

6. **Code hygiene.** Remove code that becomes unnecessary. Don't leave dead modules around.

---

## 3. Architecture: What's Already Dynamic vs What Changes

### Fully Dynamic (no changes needed)

| Component | Why it's fine |
|-----------|--------------|
| `TransformerBackbone` | Takes variable-length token sequences, processes via self-attention |
| `ActorHead` | Input `ca_embeddings: [batch, N_ca, d_model]`, applies same MLP per CA, `log_std` is per CA type (not instance) |
| `CriticHead` | Input `pooled: [batch, d_model]` (mean over all tokens), agnostic to token count |

### The Bottleneck — `ObservationTokenizer` (to be refactored)

| Current Mechanism | Problem |
|-------------------|---------|
| `__init__` builds `_index_map` via `build_encoder_index_map()` | Duplicates encoder rule matching |
| Classifies features with substring matching + device ID extraction | Same logic exists in the enricher's domain |
| Registers `_ca_idx_*`, `_sro_idx_*`, `_rl_*_idx` buffers as fixed tensors | Fixed at init; stale on topology change |
| `forward()` uses fixed buffers to slice `encoded_obs[:, idx_buf]` | Breaks if observation layout changes |

### After This Plan

The tokenizer will:
- Read enriched `observation_names` for `__tkn_*__` markers at init/rebuild time.
- Compute slicing metadata from marker positions + encoder config (which tells it how many encoded dims each feature produces).
- Use the same pre-registered buffer approach at forward time (but buffers are built from markers, not from heuristic classification).
- **Not import or use `encoder_index_map.py`** for classification.

---

## 4. Current vs Target Dataflow

### Current

```
CityLearn → List[List[float]] (raw obs per building)
    ↓
Wrapper.set_encoders() → builds encoder list from observation_names
Wrapper.predict() → get_all_encoded_observations() → List[np.ndarray] (flat encoded)
    ↓
Agent.predict(List[np.ndarray]) → per building:
    ObservationTokenizer.forward(encoded_obs)
        ↑ Uses encoder_index_map to compute slicing (duplicates encoder logic)
    → TokenizedObservation
    → Backbone → Actor/Critic
```

### Target

```
CityLearn → List[List[float]] (raw obs per building)
    ↓
ObservationEnricher.enrich_names(obs_names, action_names)
    → enriched_names (with __tkn_*__ markers)
    → encoder_specs for markers (NoNormalization)
Wrapper.set_encoders(enriched_names) → encoder list including pass-through for markers
    ↓
ObservationEnricher.enrich_values(obs_values)
    → enriched_values (marker values inserted at correct positions)
Wrapper.encode(enriched_values) → List[np.ndarray] (flat encoded, markers pass through)
    ↓
Agent.predict(List[np.ndarray]) → per building:
    ObservationTokenizer reads enriched observation_names for __tkn_*__ markers
    Builds slicing from marker positions (no heuristic classification)
    forward() slices encoded vector by marker-identified groups
    → TokenizedObservation
    → Backbone → Actor/Critic (unchanged)
```

---

## 5. Implementation Phases

### Phase A: `ObservationEnricher` Class

**File:** `algorithms/utils/observation_enricher.py` (new)

**Goal:** A standalone, portable class that classifies raw observation features into token groups and injects marker names/values.

#### A1. Marker naming convention

Markers follow the pattern: `__tkn_{family}_{type}__` or `__tkn_{family}_{type}__{device_id}__`

| Family | Type examples | Full marker name |
|--------|--------------|------------------|
| `ca` (single-instance) | `battery` | `__tkn_ca_battery__` |
| `ca` (multi-instance) | `ev_charger` with device `charger_1_1` | `__tkn_ca_ev_charger__charger_1_1__` |
| `sro` | `temporal`, `weather`, `pricing`, `carbon` | `__tkn_sro_temporal__` |
| `nfc` | (none — single token) | `__tkn_nfc__` |

Each marker is inserted **once** before the first feature of its group. For CA types with multiple instances, one marker is inserted **per instance**, and the marker **includes the device ID** (e.g., two EV chargers → `__tkn_ca_ev_charger__charger_1_1__` and `__tkn_ca_ev_charger__charger_1_2__`).

> **Why device ID in markers?** The tokenizer needs to know where each CA *instance* starts AND which device it corresponds to. The `_build_action_ca_map()` method matches actions to CA instances by device ID. Embedding the device ID in the marker preserves this information without requiring separate tracking.

#### A2. `ObservationEnricher` interface

```python
class ObservationEnricher:
    """Classifies observation features and injects token-type markers.

    Portable: no dependencies on training-only code. Can be used in
    both the training wrapper and the production inference preprocessor.
    """

    def __init__(self, tokenizer_config: Dict[str, Any]) -> None:
        """
        Args:
            tokenizer_config: The tokenizer section from config YAML.
                Must contain 'ca_types', 'sro_types', 'rl' keys.
        """

    def enrich_names(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> EnrichmentResult:
        """Classify features and inject marker names.

        Args:
            observation_names: Raw observation names for one building.
            action_names: Action names for one building (used to detect CA instances).

        Returns:
            EnrichmentResult with:
                - enriched_names: List[str] — observation names with __tkn_*__ markers inserted
                - marker_encoder_specs: List[Tuple[str, dict]] — (marker_name, encoder_spec)
                    pairs for each injected marker. encoder_spec is
                    {"type": "NoNormalization"} so the wrapper can build encoders.
                - marker_positions: Dict[str, List[int]] — marker_name → list of positions
                    in enriched_names (for debugging/validation)
        """

    def enrich_values(
        self,
        observation_values: List[float],
    ) -> List[float]:
        """Insert marker values at cached positions.

        Must be called AFTER enrich_names() for the same building.
        Uses cached positions from the last enrich_names() call.

        Marker values are 0.0 (NoNormalization passes them through as-is;
        the tokenizer ignores the actual value — it uses the name for classification).

        Args:
            observation_values: Raw observation values (same length as
                the original observation_names passed to enrich_names).

        Returns:
            Enriched values list (same length as enriched_names).
        """
```

#### A3. `EnrichmentResult` dataclass

```python
@dataclass
class EnrichmentResult:
    enriched_names: List[str]
    marker_encoder_specs: List[Tuple[str, dict]]
    marker_positions: Dict[str, List[int]]  
    # Marker name → list of positions where this marker appears.
    # E.g., {"__tkn_ca_ev_charger__charger_1_1__": [0], "__tkn_ca_ev_charger__charger_1_2__": [8]}
    # For CA markers with device IDs, each marker is unique, so each list has one element.
```

#### A4. Classification logic

The enricher reuses the **same classification logic** currently in `ObservationTokenizer.__init__` (lines 188-276 of `observation_tokenizer.py`):

1. **CA classification:** Extract device IDs from action names (via `_extract_device_ids`), match features to CA types using `_feature_matches_ca_type`, assign features to instances using `_contains_device_id`. These three helper functions should be **moved** from `observation_tokenizer.py` to `observation_enricher.py` (or a shared `_classification_helpers.py`) since the enricher now owns classification.

2. **SRO classification:** Match remaining unassigned features to SRO types using substring matching on configured `features` patterns.

3. **RL (NFC) classification:** Match remaining unassigned features to RL demand/generation/extra patterns.

4. **Unmatched features:** Kept as-is (no marker). They pass through encoding normally but are not assigned to any token. Log a warning, same as today.

After classification, the enricher builds the enriched names list by interleaving markers:

```
[__tkn_ca_battery__, electrical_storage_soc,
 __tkn_ca_ev_charger__charger_1_1__, connected_state_charger_1_1, departure_time_charger_1_1, ...,
 __tkn_ca_ev_charger__charger_1_2__, connected_state_charger_1_2, departure_time_charger_1_2, ...,
 __tkn_sro_temporal__, month, hour, day_type, daylight_savings_status,
 __tkn_sro_weather__, outdoor_dry_bulb_temperature, ...,
 __tkn_sro_pricing__, electricity_pricing,
 __tkn_sro_carbon__, carbon_intensity,
 __tkn_nfc__, non_shiftable_load, solar_generation, net_electricity_consumption,
 <any unmatched features>]
```

#### A5. Caching for topology change detection

```python
def enrich_names(self, observation_names, action_names) -> EnrichmentResult:
    key = (tuple(observation_names), tuple(action_names))
    if key == self._cache_key:
        return self._cached_result
    # ... full classification and enrichment ...
    self._cache_key = key
    self._cached_result = result
    self._insertion_positions = <positions for enrich_values>
    return result
```

`enrich_values()` uses `self._insertion_positions` to insert 0.0 values at the cached marker positions. This is O(n) and lightweight.

#### A6. Portability for `energAIze_inference`

The `ObservationEnricher` must have:
- **No imports** from `algorithms.*`, `utils.*`, or any training-specific module.
- **No dependency** on PyTorch, NumPy, or any ML framework.
- Only stdlib + typing imports.
- The classification helpers (`_extract_device_ids`, `_feature_matches_ca_type`, `_contains_device_id`) are pure Python string operations — they're already portable.

This means the three helper functions currently in `observation_tokenizer.py` should live in `observation_enricher.py` (and the tokenizer imports them from there).

---

### Phase B: Wrapper Integration

**File:** `utils/wrapper_citylearn.py`

**Goal:** The wrapper uses the enricher to inject markers before encoding. On topology change, it rebuilds the encoder list from enriched names.

#### B1. Instantiate enrichers (one per building)

In the wrapper's `__init__` or setup path (where `tokenizer_config` is accessible):

```python
from algorithms.utils.observation_enricher import ObservationEnricher

# One enricher PER BUILDING — each caches its own insertion positions
self._enrichers: List[Optional[ObservationEnricher]] = []
```

**Why per-building?** The enricher caches `_insertion_positions` for `enrich_values()`. If a single enricher is shared across buildings with different topologies, calling `enrich_names(building_1)` would overwrite building_0's cached positions, causing `enrich_values()` to insert markers at wrong positions.

**Important:** The wrapper only uses enrichers when the agent is a TransformerPPO agent. For MADDPG/RBC, enrichment is skipped entirely. Gate this on a config flag or agent type check.

#### B2. Modify `set_encoders()`

**Current** (`wrapper_citylearn.py:592-626`): Iterates `self.observation_names`, matches each name to an encoder rule, builds encoder list.

**New:** Before building encoders, enrich the observation names. Create one enricher per building:

```python
def set_encoders(self) -> List[List[Encoder]]:
    rules = _load_encoder_rules()
    encoders: List[List[Encoder]] = []
    self._enrichers = []  # Reset per-building enrichers
    self._enriched_observation_names: List[List[str]] = []  # Store for attach_environment

    for building_idx, (obs_group, act_group, space) in enumerate(zip(
        self.observation_names, self.action_names_by_agent, self.observation_space
    )):
        # --- Enrichment step (if enabled) ---
        if self._use_enrichment:  # Flag set based on agent type
            enricher = ObservationEnricher(self._tokenizer_config)
            enrichment = enricher.enrich_names(obs_group, act_group)
            enriched_names = enrichment.enriched_names
            self._enrichers.append(enricher)
            self._enriched_observation_names.append(enriched_names)
        else:
            self._enrichers.append(None)
            enriched_names = obs_group
            self._enriched_observation_names.append(obs_group)

        # Build encoders from enriched_names (markers get NoNormalization)
        group_encoders = []
        for index, name in enumerate(enriched_names):
            if name.startswith("__tkn_") and name.endswith("__"):
                # Marker feature: use NoNormalization encoder
                encoder = NoNormalization()  # pass-through
                group_encoders.append(encoder)
            else:
                # Normal feature: match against rules as before
                rule = next((r for r in rules if _matches_rule(name, r.get("match", {}))), None)
                if rule is None:
                    raise ValueError(f"No encoder rule for: {name}")
                encoder = _build_encoder(rule, space, index)
                group_encoders.append(encoder)
        encoders.append(group_encoders)

    return encoders
```

#### B3. Modify `get_encoded_observations()`

**Current** (`wrapper_citylearn.py:486-497`): Encodes raw observation values using `self.encoders[index]`.

**New:** Before encoding, inject marker values using the per-building enricher:

```python
def get_encoded_observations(self, index: int, observations: List[float]) -> np.ndarray:
    # Use the per-building enricher (if enabled)
    if self._enrichers[index] is not None:
        enriched_obs = self._enrichers[index].enrich_values(observations)
    else:
        enriched_obs = observations

    obs_array = np.array(enriched_obs, dtype=np.float64)
    encoded = np.hstack([
        encoder.transform(obs) if hasattr(encoder, "transform") else encoder * obs
        for encoder, obs in zip(self.encoders[index], obs_array)
    ]).astype(np.float64)
    return encoded[~np.isnan(encoded)]
```

#### B4. Pass enriched names to `attach_environment()`

The wrapper calls `agent.attach_environment(observation_names=..., ...)`. After enrichment, it must pass the **enriched** names so the tokenizer can find markers:

```python
self.model.attach_environment(
    observation_names=self._enriched_observation_names,  # NOT raw names
    action_names=...,
    action_space=...,
    observation_space=...,
    metadata=...,
)
```

#### B5. Topology change path (future)

When a topology change is detected (observation names change between steps):

1. The enricher's `enrich_names()` cache miss triggers re-classification.
2. The wrapper rebuilds the per-building enricher and encoder list from the new enriched names.
3. The wrapper calls `agent.reconfigure_building(i, new_enriched_names, new_action_names)` (existing flexy_plan mechanism).
4. The tokenizer rebuilds its slicing from the new enriched names.
5. The rollout buffer is flushed (existing mechanism).

This is future work — the enricher's cache makes it cheap, and the wrapper already has the hook points.

---

### Phase C: Tokenizer Refactor

**File:** `algorithms/utils/observation_tokenizer.py`

**Goal:** Replace heuristic-based classification + `encoder_index_map` with marker-based slicing.

#### C1. Remove `encoder_index_map` dependency

The tokenizer currently imports and uses `build_encoder_index_map()` (line 37-40, line 176-178). After enrichment, the tokenizer doesn't need this — it reads markers from observation names.

**Remove:**
```python
from algorithms.utils.encoder_index_map import (
    EncoderSlice,
    build_encoder_index_map,
)
```

**Replace with:** A local or imported helper that computes encoded dimensions per feature name using the encoder config. This is simpler than `build_encoder_index_map` because we don't need to re-match rules — we just need to know "how many encoded dims does this feature produce?"

We can keep the `_compute_encoded_dims()` function from `encoder_index_map.py` (lines 64-89) — it's a pure utility that computes dims from an encoder spec. Move it to a shared location or inline it.

#### C2. New `__init__` — marker-based slicing

The `__init__` method (currently lines 158-396) is refactored:

**Current flow:**
1. Build encoder index map (heuristic classification).
2. Classify features into CA/SRO/RL groups (substring matching).
3. Resolve post-encoding slices per group.
4. Create projections.
5. Register index buffers.

**New flow:**
1. Scan `observation_names` for `__tkn_*__` markers.
2. For each marker, collect subsequent feature names until the next marker (or end of list).
3. Compute post-encoding slice for each group using encoder config dims.
4. Create projections (same as today, using global dims).
5. Register index buffers (same as today, but built from markers).

```python
def __init__(
    self,
    observation_names: List[str],  # Now enriched names (with markers)
    action_names: List[str],
    encoder_config: Dict[str, Any],
    tokenizer_config: Dict[str, Any],
    d_model: int,
    global_ca_type_dims: Optional[Dict[str, int]] = None,
    global_sro_type_dims: Optional[Dict[str, int]] = None,
    max_rl_input_dim: Optional[int] = None,
) -> None:
    super().__init__()

    self.d_model = d_model
    self.observation_names = observation_names
    self.action_names = action_names

    # --- Step 1: Parse markers from enriched observation names ---
    token_groups = self._parse_markers(observation_names)
    # Returns: List[TokenGroup] where each has:
    #   - family: "ca" | "sro" | "nfc"
    #   - type_name: "battery" | "temporal" | etc.
    #   - feature_names: List[str] (the non-marker names in this group)

    # --- Step 2: Compute post-encoding positions ---
    # Build a simple name → encoded_dims mapping from encoder_config
    # (replaces build_encoder_index_map)
    encoded_dims_map = self._build_encoded_dims_map(observation_names, encoder_config)
    # Returns: OrderedDict[name, (start_idx, end_idx, n_dims)]

    # --- Step 3: Build slicing metadata from markers + dims ---
    # For CA groups: collect indices per instance
    # For SRO groups: collect indices per type
    # For NFC group: collect demand/gen/extra indices
    # (Same structure as today, but computed from markers, not heuristics)

    # --- Steps 4-5: Create projections, register buffers (same as today) ---
```

#### C3. `_parse_markers()` method

```python
@staticmethod
def _parse_markers(observation_names: List[str]) -> List[TokenGroup]:
    """Parse __tkn_*__ markers from enriched observation names.

    Returns ordered list of TokenGroup, one per marker encountered.
    Features before the first marker (if any) are unassigned.
    
    Marker format: __tkn_{family}_{type}__ or __tkn_{family}_{type}__{device_id}__
    Examples:
        - __tkn_ca_battery__ → family="ca", type_name="battery", device_id=None
        - __tkn_ca_ev_charger__charger_1_1__ → family="ca", type_name="ev_charger", device_id="charger_1_1"
        - __tkn_sro_temporal__ → family="sro", type_name="temporal", device_id=None
        - __tkn_nfc__ → family="nfc", type_name="", device_id=None
    """
    groups = []
    current_group = None

    for name in observation_names:
        if name.startswith("__tkn_") and name.endswith("__"):
            # Parse marker: __tkn_{family}_{type}__ or __tkn_{family}_{type}__{device_id}__
            inner = name[6:-2]  # strip __tkn_ and __
            
            # Check for device_id (indicated by double underscore in the middle)
            if "__" in inner:
                # Has device_id: e.g., "ca_ev_charger__charger_1_1"
                type_part, device_id = inner.rsplit("__", 1)
                parts = type_part.split("_", 1)
                family = parts[0]
                type_name = parts[1] if len(parts) > 1 else ""
            else:
                # No device_id: e.g., "ca_battery" or "sro_temporal" or "nfc"
                parts = inner.split("_", 1)
                family = parts[0]
                type_name = parts[1] if len(parts) > 1 else ""
                device_id = None
            
            current_group = TokenGroup(
                family=family, 
                type_name=type_name, 
                device_id=device_id,
                feature_names=[]
            )
            groups.append(current_group)
        elif current_group is not None:
            current_group.feature_names.append(name)
        else:
            # Feature before any marker — unassigned
            pass

    return groups
```

**`TokenGroup` dataclass:**

```python
@dataclass
class TokenGroup:
    family: str           # "ca", "sro", or "nfc"
    type_name: str        # "battery", "ev_charger", "temporal", etc.
    device_id: Optional[str]  # "charger_1_1" for multi-instance CAs, None otherwise
    feature_names: List[str]  # Features belonging to this group
```

#### C4. `_build_encoded_dims_map()` method

This replaces `build_encoder_index_map()`. It's simpler because it doesn't need to classify features — it just needs to compute encoded dimensions:

```python
@staticmethod
def _build_encoded_dims_map(
    observation_names: List[str],
    encoder_config: Dict[str, Any],
) -> OrderedDict[str, Tuple[int, int, int]]:
    """Map each feature name to (start_idx, end_idx, n_dims) in the encoded vector.

    For __tkn_*__ markers, uses NoNormalization → 1 dim.
    For regular features, matches against encoder rules.
    """
    rules = encoder_config.get("rules", [])
    result = OrderedDict()
    current_idx = 0

    for name in observation_names:
        if name.startswith("__tkn_") and name.endswith("__"):
            n_dims = 1  # NoNormalization → 1 dim
        else:
            rule = next((r for r in rules if _matches_rule(name, r.get("match", {}))), None)
            if rule is None:
                raise ValueError(f"No encoder rule matches: {name}")
            n_dims = _compute_encoded_dims(rule.get("encoder", {}))

        result[name] = (current_idx, current_idx + n_dims, n_dims)
        current_idx += n_dims

    return result
```

> **Note:** `_matches_rule` and `_compute_encoded_dims` are still needed here for computing **dimensions** (not classification). These ~35 lines should be **moved into `observation_tokenizer.py`** as private functions. The wrapper's `_matches_rule` in `wrapper_citylearn.py` is a separate copy and stays unchanged.

#### C5. Refactored slicing logic

After `_parse_markers()` and `_build_encoded_dims_map()`, the tokenizer builds its index buffers:

```python
# --- CA instances ---
self._ca_instances = []
self._ca_type_names = []

for group in token_groups:
    if group.family == "ca":
        indices = []
        for feat_name in group.feature_names:
            start, end, n_dims = encoded_dims_map[feat_name]
            if n_dims > 0:
                indices.extend(range(start, end))
        if indices:
            # Preserve device_id from marker for action mapping
            self._ca_instances.append((group.type_name, group.device_id, indices))
            self._ca_type_names.append(group.type_name)

# --- SRO groups ---
self._sro_groups = []
for group in token_groups:
    if group.family == "sro":
        indices = []
        for feat_name in group.feature_names:
            start, end, n_dims = encoded_dims_map[feat_name]
            if n_dims > 0:
                indices.extend(range(start, end))
        if indices:
            self._sro_groups.append((group.type_name, indices))

# --- RL (NFC) group ---
# Same residual logic as today, but features come from the nfc group
for group in token_groups:
    if group.family == "nfc":
        # Classify into demand/generation/extra using rl_config patterns
        # (same logic as current lines 254-271)
        ...
```

**Key change from current code:** The `device_id` is now preserved in `_ca_instances` (extracted from the marker), enabling `_build_action_ca_map()` to match actions correctly.

#### C6. `_build_action_ca_map()` — minimal changes

The action-to-CA mapping (`_build_action_ca_map`, lines 433-473) works the same way. It matches action names to CA instances using the `device_id` stored in `_ca_instances`. Since the marker now embeds the device ID (e.g., `__tkn_ca_ev_charger__charger_1_1__`), the tokenizer extracts it via `_parse_markers()` and stores it in `_ca_instances`. **No logic changes needed** — just ensure the existing code uses `device_id` from the tuple.

#### C7. `forward()` — no changes

The `forward()` method (lines 479-565) uses `_ca_idx_*`, `_sro_idx_*`, `_rl_*_idx` buffers and projections. These are still registered in the same way — they're just built from markers instead of heuristics. **No changes to forward().**

---

### Phase D: Agent Changes

**File:** `algorithms/agents/transformer_ppo_agent.py`

**Goal:** The agent's global vocabulary computation uses enriched names, and the agent passes enriched names to the tokenizer.

#### D1. `_compute_global_vocabulary()` simplification

**Current** (lines 145-329): Imports `build_encoder_index_map`, `_extract_device_ids`, `_feature_matches_ca_type`, `_contains_device_id` — duplicating classification logic a **third** time.

**New:** The agent creates an `ObservationEnricher` and uses it to classify features for all buildings. Then it reads the enrichment result to compute dims:

```python
def _compute_global_vocabulary(
    self,
    observation_names: List[List[str]],  # Already enriched names
    action_names: List[List[str]],
    encoder_config: Dict[str, Any],
    ...
) -> ...:
    """Compute global CA/SRO type dims from enriched observation names."""

    for i, (obs_names, act_names) in enumerate(zip(observation_names, action_names)):
        # obs_names are already enriched — parse markers
        token_groups = ObservationTokenizer._parse_markers(obs_names)
        dims_map = ObservationTokenizer._build_encoded_dims_map(obs_names, encoder_config)

        for group in token_groups:
            if group.family == "ca":
                dims = sum(dims_map[n][2] for n in group.feature_names if n in dims_map)
                # Track per-type dims...
            elif group.family == "sro":
                dims = sum(dims_map[n][2] for n in group.feature_names if n in dims_map)
                # Track per-type dims...
            elif group.family == "nfc":
                # Compute RL dims...
                ...
```

This eliminates the duplicated classification logic in the agent.

#### D2. `attach_environment()` receives enriched names

The wrapper passes enriched names (see Phase B4). The agent's `attach_environment()` passes them directly to the tokenizer — no changes needed beyond ensuring the names are enriched.

---

### Phase E: Cleanup

**Goal:** Remove code that's no longer necessary.

#### E1. `encoder_index_map.py` — delete entirely

The `build_encoder_index_map()` function and its `_matches_rule()` helper are no longer needed for classification.

**Action:** 
1. **Move** `_matches_rule()` and `_compute_encoded_dims()` (~35 lines total) into `observation_tokenizer.py` as private functions.
2. **Delete** `algorithms/utils/encoder_index_map.py` entirely.
3. **Delete** `tests/test_encoder_index_map.py`.

> **Note:** The wrapper's `_matches_rule` in `wrapper_citylearn.py` is a separate copy used for encoder building — it stays unchanged.

#### E2. Classification helpers — move to enricher

The three helper functions in `observation_tokenizer.py`:
- `_extract_device_ids()` (lines 72-115)
- `_contains_device_id()` (lines 118-139)
- `_feature_matches_ca_type()` (lines 142-147)

**Move** them to `observation_enricher.py`. The tokenizer no longer needs them (it reads markers). If the tokenizer's `_build_action_ca_map()` still needs device ID logic, import from the enricher.

#### E3. Remove duplicated classification from tokenizer `__init__`

Lines 180-276 of `observation_tokenizer.py` — the entire CA/SRO/RL classification block — is replaced by `_parse_markers()`. Remove it.

#### E4. Remove duplicated classification from agent

Lines 187-188 of `transformer_ppo_agent.py` — the imports of `_extract_device_ids`, `_contains_device_id`, `_feature_matches_ca_type` — are no longer needed. The entire classification loop in `_compute_global_vocabulary()` (lines 182-277) is replaced by marker parsing.

#### E5. Update `observation_tokenizer.py` docstring

The module docstring (lines 1-24) describes the current heuristic approach. Update it to reflect the marker-based approach.

---

## 6. Test Plan

### T1. `ObservationEnricher` unit tests

**File:** `tests/test_observation_enricher.py` (new)

| Test | What it verifies |
|------|-----------------|
| `test_enrich_names_single_ca` | Single battery building: enriched names contain `__tkn_ca_battery__` before battery features, `__tkn_sro_*__` before each SRO group, `__tkn_nfc__` before RL features |
| `test_enrich_names_multi_ca` | Building with battery + 2 EV chargers: markers include device IDs (`__tkn_ca_ev_charger__charger_1_1__`, `__tkn_ca_ev_charger__charger_1_2__`), one `__tkn_ca_battery__` marker |
| `test_enrich_names_no_ca` | Building with no controllable assets: no `__tkn_ca_*__` markers, SRO and NFC markers still present |
| `test_enrich_names_preserves_feature_order` | Features within each group appear in the same relative order as the original observation names |
| `test_enrich_names_unmatched_features` | Features matching no config pattern appear after all marker groups (no marker, not lost) |
| `test_enrich_values_correct_length` | `enrich_values()` output length equals `len(enriched_names)` |
| `test_enrich_values_marker_positions_zero` | Marker positions in enriched values contain 0.0 |
| `test_enrich_values_preserves_original_values` | Non-marker positions contain the original observation values in order |
| `test_cache_hit_same_topology` | Calling `enrich_names()` twice with same names returns cached result (same object) |
| `test_cache_miss_topology_change` | Calling `enrich_names()` with different names re-classifies and returns new result |
| `test_marker_naming_convention` | All injected markers match `__tkn_{family}_{type}__` or `__tkn_{family}_{type}__{device_id}__` pattern |
| `test_marker_device_id_extraction` | CA markers with device IDs correctly embed and allow extraction of device ID |
| `test_enricher_no_external_dependencies` | `ObservationEnricher` module imports only stdlib + typing (verify portability) |

### T2. Tokenizer marker parsing tests

**File:** `tests/test_observation_tokenizer.py` (extend existing)

| Test | What it verifies |
|------|-----------------|
| `test_parse_markers_basic` | `_parse_markers()` correctly identifies CA, SRO, NFC groups from enriched names |
| `test_parse_markers_extracts_device_id` | Marker `__tkn_ca_ev_charger__charger_1_1__` produces `TokenGroup` with `device_id="charger_1_1"` |
| `test_parse_markers_multi_instance` | Two `__tkn_ca_ev_charger__*__` markers with different device IDs produce two separate CA groups with correct device IDs |
| `test_parse_markers_empty` | Empty observation names → empty groups list |
| `test_parse_markers_no_markers` | Names without markers → empty groups list (graceful degradation) |
| `test_build_encoded_dims_map_with_markers` | Marker names get 1 dim (NoNormalization), regular features get correct dims |
| `test_tokenizer_init_from_enriched_names` | Tokenizer constructed with enriched names produces correct `n_ca`, `n_sro`, `ca_types` |
| `test_tokenizer_forward_enriched` | `forward()` with enriched encoded vector produces correct token shapes |
| `test_tokenizer_forward_matches_heuristic` | Given the same building config, tokenizer-from-enriched-names produces the **same** output tensors as the old heuristic tokenizer (regression test) |
| `test_tokenizer_no_encoder_index_map_import` | `observation_tokenizer.py` does not import from `encoder_index_map` (verify cleanup) |

### T3. Wrapper enrichment integration tests

**File:** `tests/test_wrapper_enrichment.py` (new)

| Test | What it verifies |
|------|-----------------|
| `test_set_encoders_with_enrichment` | `set_encoders()` produces correct number of encoders (original count + marker count) |
| `test_per_building_enrichers` | Each building gets its own enricher instance; `self._enrichers` is a list with length == num_buildings |
| `test_marker_encoders_are_no_normalization` | Encoders at marker positions are `NoNormalization` instances |
| `test_encoded_observation_shape` | Encoded observation dimension increases by number of markers (each marker → 1 dim) |
| `test_encoded_observation_values` | Encoded values at marker positions are 0.0; other positions match non-enriched encoding |
| `test_enrichment_disabled_for_non_transformer` | When agent is MADDPG/RBC, enricher is None and encoding is unchanged |
| `test_enriched_names_passed_to_agent` | `attach_environment()` receives enriched observation names (not raw) |

### T4. Agent global vocabulary with enrichment

**File:** `tests/test_runtime_adaptability.py` (extend existing)

| Test | What it verifies |
|------|-----------------|
| `test_global_vocab_from_enriched_names` | `_compute_global_vocabulary()` correctly computes CA/SRO dims from enriched names |
| `test_global_vocab_multi_building_enriched` | Global dims consistent across buildings with different topologies (enriched names) |
| `test_agent_predict_with_enrichment` | Full pipeline: enrich → encode → tokenize → predict → correct action count |

### T5. End-to-end regression tests

**File:** `tests/test_observation_tokenizer.py` or `tests/test_runtime_composition_changes.py` (extend)

| Test | What it verifies |
|------|-----------------|
| `test_e2e_enriched_vs_heuristic_same_output` | For a fixed building config, the enriched pipeline produces the **same** encoded vector and token embeddings as the old heuristic pipeline. This is the critical regression test. |
| `test_e2e_cross_topology_with_enrichment` | Train on building with 3 CAs (enriched), evaluate on building with 1 CA (enriched) → shapes correct, no crashes |

### T6. Cleanup verification tests

| Test | What it verifies |
|------|-----------------|
| `test_encoder_index_map_deleted` | `algorithms/utils/encoder_index_map.py` no longer exists |
| `test_no_heuristic_classification_in_tokenizer` | `observation_tokenizer.py` does not contain `_extract_device_ids` or `_feature_matches_ca_type` function definitions |

---

## 7. Implementation Order

Execute phases in this order:

1. **Phase A** — `ObservationEnricher` class + tests (T1). Self-contained, no existing code modified.
2. **Phase C1** (partial) — Add `_parse_markers()`, `TokenGroup`, and `_build_encoded_dims_map()` to tokenizer + tests (T2, parse-only). Also move `_matches_rule` and `_compute_encoded_dims` into tokenizer. No existing classification logic removed yet.
3. **Phase B** — Wrapper integration + tests (T3). Per-building enrichers active, wrapper passes enriched names.
4. **Phase D** — Agent changes + tests (T4). Agent uses enriched names for global vocabulary.
5. **Phase C2** (complete) — Replace tokenizer classification with marker-based logic. Run regression tests (T5). **Critical gate:** `test_e2e_enriched_vs_heuristic_same_output` must pass before proceeding.
6. **Phase E** — Cleanup: delete `encoder_index_map.py`, move helpers, remove duplicated code. Run cleanup tests (T6).
7. **Run full test suite** — all 191+ existing tests must pass.

> **Principle:** At every phase boundary, all tests pass. The enriched and heuristic paths coexist during phases 1-4. Phase 5 cuts over. Phase 6 cleans up.

---

## 8. Production Inference Impact

The `ObservationEnricher` is designed to be dropped into `energAIze_inference`:

**File:** `app/utils/observation_enricher.py` (copy from training repo)

**Integration in `preprocessor.py`:**

```python
class AgentPreprocessor:
    def __init__(self, observation_names, encoders, tokenizer_config, ...):
        self.enricher = ObservationEnricher(tokenizer_config)
        enrichment = self.enricher.enrich_names(observation_names, action_names)
        self.enriched_names = enrichment.enriched_names
        # Rebuild encoder list to include marker encoders
        ...

    def transform(self, raw_observations: Dict[str, float]) -> np.ndarray:
        values = [raw_observations[name] for name in self.original_names]
        enriched_values = self.enricher.enrich_values(values)
        # Encode using enriched encoder list
        ...
```

The manifest (`artifact_manifest.json`) should include the `tokenizer_config` so the inference preprocessor can construct the enricher. This is already partially done — the config is saved in `config.resolved.yaml`.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Enriched observation dim changes break existing checkpoints | Checkpoints save model weights, not observation layout. The tokenizer rebuilds slicing from names at init. No weight shape changes — projections use the same global dims. |
| Marker values (0.0) influence encoding of adjacent features | Markers use `NoNormalization` — they pass through independently. Each feature is encoded separately; markers don't affect neighbors. |
| Performance overhead of `enrich_values()` per step | O(n) list insertion. With ~30-50 features and ~10 markers, this is negligible (~microseconds). Cache ensures no re-classification. |
| Wrapper changes break MADDPG/RBC | Enrichment is gated on agent type. Non-Transformer agents see no change. |
| `encoder_index_map.py` is used elsewhere | Grep the codebase before deleting. Currently only imported by `observation_tokenizer.py` and `transformer_ppo_agent.py`. |
