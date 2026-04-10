# Runtime Topology Adaptability — Implementation Plan

## Motivation

The core goal of the Transformer architecture is to allow the **same base model** to work across buildings with different numbers and types of controllable assets (CAs). Currently, every topology-related data structure is frozen at `attach_environment()` time. If a building gains or loses a CA at runtime, the agent cannot adapt — it crashes or produces wrong outputs.

This plan fixes that by **pre-allocating** all learned parameters for the full type vocabularies from config, and adding a **`reconfigure_topology()`** path that rebuilds metadata without touching learned weights.

> **Scope note:** This plan covers **all three token families** — CA tokens, SRO tokens, and the RL token — plus critic/buffer interactions with variable token counts. The original plan only detailed CA changes; the sections below cover the full picture.

---

## Current Blockers (22 total)

### CA Blockers (13) — originally identified

| # | File | Lines | Component | Issue |
|---|------|-------|-----------|-------|
| 1 | `observation_tokenizer.py` | 278-291 | `_ca_instances` list | Fixed at init; new CAs never added |
| 2 | `observation_tokenizer.py` | 348-350 | `ca_projections` ModuleDict | Only contains types with active instances |
| 3 | `observation_tokenizer.py` | 367-368 | `_ca_idx_*` buffers | No buffers registered for new CAs |
| 4 | `observation_tokenizer.py` | 362-364 | `_action_ca_map` | Fixed at init; new actions unmapped |
| 5 | `observation_tokenizer.py` | 173-175 | `_index_map` / `total_encoded_dims` | Fixed observation layout; new features invisible |
| 6 | `ppo_components.py` | 51 | `ActorHead.log_std` | Shaped `(N_local_types,)` — IndexError for new type indices |
| 7 | `ppo_components.py` | 86 | `ActorHead.forward` log_std indexing | Out-of-bounds access with new type indices |
| 8 | `transformer_ppo_agent.py` | 172-173 | `_n_ca_per_building` | Cached count never updated |
| 9 | `transformer_ppo_agent.py` | 177-186 | `_ca_type_indices` tensor | Fixed shape/values; no new types |
| 10 | `transformer_ppo_agent.py` | 174 | `_action_ca_maps` | Fixed; wrong action count returned |
| 11 | `transformer_ppo_agent.py` | 516-519 | `load_state_dict` (strict) | Shape/key mismatch across topologies |
| 12 | `transformer_ppo_agent.py` | 270-277 | Rollout buffer push | Mixed tensor shapes if N_ca changes mid-rollout |
| 13 | `transformer_ppo_agent.py` | 202-203 | `_BuildingModel` construction | One-shot; no reconfiguration path |

### SRO Blockers (4) — NEW

| # | File | Lines | Component | Issue |
|---|------|-------|-----------|-------|
| 14 | `observation_tokenizer.py` | 352-354 | `sro_projections` ModuleDict | Only contains SRO types with matched features in this building |
| 15 | `observation_tokenizer.py` | 369-370 | `_sro_idx_*` buffers | Frozen; stale indices if features are reordered or new SRO features appear |
| 16 | `observation_tokenizer.py` | 352-354 | `sro_projections` `nn.Linear` input dim | `nn.Linear(N_matched_features, d_model)` — if feature count changes within an SRO type, weight matrix shape mismatches (hard crash: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`) |
| 17 | `observation_tokenizer.py` | 304-316 | `_sro_groups` list | Fixed at init; new SRO features fall to "unmatched" bucket or silently go to RL extras |

### RL Token Blockers (5) — NEW

| # | File | Lines | Component | Issue |
|---|------|-------|-----------|-------|
| 18 | `observation_tokenizer.py` | 356-359 | `rl_projection` `nn.Linear` input dim | `nn.Linear(rl_input_dim, d_model)` — if `extra_features` count changes, weight matrix shape mismatches (hard crash) |
| 19 | `observation_tokenizer.py` | 371-376 | `_rl_demand_idx` / `_rl_gen_idx` / `_rl_extra_idx` buffers | Frozen registered buffers; stale if features change |
| 20 | `observation_tokenizer.py` | 375-376 | `_rl_extra_idx` conditional registration | If extras go from 0 to >0, buffer doesn't exist → `AttributeError`; if extras disappear, stale buffer indexes wrong positions |
| 21 | `observation_tokenizer.py` | 356-359 | `rl_projection` conditional creation | If RL features go from 0 to >0, `rl_projection` is `None` — no projection exists to activate |
| 22 | `observation_tokenizer.py` | 335-345 | `rl_input_dim` computation | Demand/generation → 1 scalar (robust), but extras add their full encoded dims. Adding 1 extra feature changes `rl_input_dim`, breaking the `nn.Linear` |

### CA Removal — Soft Degradation (not a blocker, but important)

These don't cause crashes but affect learning quality when CAs are removed:

| Concern | File | Mechanism | Impact |
|---------|------|-----------|--------|
| Critic mean-pool shift | `transformer_backbone.py:150` | `embeddings.mean(dim=1)` pools over all tokens. Going from 6 tokens to 4 changes the weighting of SRO/RL tokens (from 1/6 to 1/4). | Learned value estimates become unreliable; critic needs fine-tuning |
| `evaluate_actions` shape | `transformer_ppo_agent.py:363-383` | `ca_embeddings` shape `[B, N_ca, d_model]` must match stored `old_actions` shape `[B, N_ca, 1]` | If N_ca changed mid-rollout, shapes mismatch even after buffer flush |

---

## Constraints

- Zero changes to wrapper, runner, or existing agents (MADDPG, RBC).
- Each building still gets its own agent instance (separate weights).
- All 178 existing tests must continue to pass.
- The Transformer backbone is already token-count-agnostic (attention works on any sequence length). No changes needed.
- The CriticHead MLP is dimension-agnostic (`[B, d_model] → [B, 1]`). No weight changes needed, but mean-pool semantics shift when token count changes (see soft degradation table above).

## Key Design Decision: Pre-allocation vs Reconstruction vs Max-Padding

Three strategies are available. We use a **hybrid** approach:

| Strategy | Used For | Why |
|----------|----------|-----|
| **Pre-allocation** | CA projections, SRO projections, `log_std` | Type vocabulary is known from config. We create `nn.Linear` for every type, even if inactive. Weights transfer across topologies. |
| **Max-padding** | RL projection, rollout buffer actions/log_probs | The RL extra_features dim and N_ca count are variable and not bounded by config vocabulary. Pad to a known max and mask. |
| **Reconstruction** | Index buffers, metadata (`_ca_instances`, `_sro_groups`, `_action_ca_map`, `_index_map`) | These are cheap, non-learned data structures. Rebuild from scratch in `reconfigure()`. |

**Why not just reconstruct everything?** Because `nn.Linear` weight matrices contain learned parameters. Reconstructing them means losing training progress or requiring weight surgery. Pre-allocation avoids this entirely for vocabulary-bounded dimensions. Max-padding handles the remaining cases where the dimension is data-dependent.

---

## Phase A: Global CA Type Vocabulary & Pre-allocation

**Goal:** All projection layers and `log_std` are always sized for the full config vocabulary, regardless of which CAs are active in a given building.

### A1. Compute global dims across all buildings

**File:** `transformer_ppo_agent.py` — `attach_environment()`

Currently, each building's tokenizer is constructed with only its own observation/action names. Change to a two-pass approach:

1. **Pass 1 (new):** Loop over all buildings' obs/action names. For each building, temporarily run instance detection + encoder index map to find which CA types exist and their encoded feature dims. Collect into:
   ```python
   global_ca_type_dims: Dict[str, int]  # e.g. {"battery": 2, "ev_charger": 14, "washing_machine": 4}
   ```
   This covers every CA type found across ANY building in the current environment.

2. **Fallback for unseen types:** For CA types defined in `tokenizer_config["ca_types"]` but not found in any building, add an auto-compute helper that resolves feature patterns against encoder rules to determine the expected encoded dim. This handles the edge case of a type that exists in config but has zero instances in the current dataset.

3. **Global `type_to_idx`:** Derive from `sorted(tokenizer_config["ca_types"].keys())`, constant across all buildings:
   ```python
   # Always the same, regardless of per-building active types:
   {"battery": 0, "ev_charger": 1, "washing_machine": 2}
   ```

Store as `self._global_type_to_idx` and `self._global_ca_type_dims` at the agent level.

### A2. Tokenizer pre-allocates all CA projections

**File:** `observation_tokenizer.py`

- Add `global_ca_type_dims: Dict[str, int]` parameter to `__init__`.
- `ca_projections` ModuleDict is populated for **every entry** in `global_ca_type_dims`, not just types with active instances in this building.
- Active instances still get index buffers and participate in `forward()`. Inactive projections are present in `state_dict()` but not called during forward.

**Resolves blockers:** #2 (ca_projections), partially #11 (checkpoint compatibility for projections).

### A2b. Tokenizer pre-allocates all SRO projections

**File:** `observation_tokenizer.py`

Currently (lines 352-354), `sro_projections` is populated only for SRO types that matched at least one feature in this building's observation names. A building with no pricing features has no `sro_projections["pricing"]` entry.

**Change:** Add `global_sro_type_dims: Dict[str, int]` parameter to `__init__`, computed the same way as CA dims (pass 1 across all buildings). Pre-allocate `sro_projections` for every SRO type in config:

```python
# OLD: only matched types
for sro_type_name, dims in sro_group_dims.items():
    self.sro_projections[sro_type_name] = nn.Linear(dims, d_model)

# NEW: all types from global vocabulary
for sro_type_name, dims in global_sro_type_dims.items():
    self.sro_projections[sro_type_name] = nn.Linear(dims, d_model)
```

**Challenge: SRO input dims can vary per building.** Unlike CA projections where all batteries share the same feature set (same `in_features`), SRO groups could theoretically have different feature counts across buildings if the dataset varies. In practice, all buildings in CityLearn share the same weather/time/pricing features, so dims are consistent.

**Safety check:** During pass 1, if an SRO type has different encoded dims across buildings, log a warning and use the maximum. At `reconfigure()` time, zero-pad shorter feature vectors to match the pre-allocated `nn.Linear` input dim.

**Resolves blockers:** #14 (sro_projections only for active), partially #16 (input dim — if dims are consistent across buildings, which they are in CityLearn).

### A3. ActorHead uses full vocabulary

**File:** `ppo_components.py`

- `n_ca_types` passed to `ActorHead.__init__` is now the **total from config** (e.g., 3), not the per-building count.
- `log_std = nn.Parameter(torch.zeros(n_ca_types))` always has shape `(3,)`.
- `ca_type_indices` tensor still maps active CA tokens to their global type index. For a battery-only building: `[0]`. For a full building: `[0, 1, 2]`. Indexing into `log_std[ca_type_indices]` works either way — it's a gather, not a slice.

**Resolves blockers:** #6, #7.

### A4. Agent stores global mapping

**File:** `transformer_ppo_agent.py`

- Pass `global_ca_type_dims` to each building's tokenizer.
- Pass `n_ca_types = len(global_ca_type_dims)` to each building's `ActorHead`.
- Build `_ca_type_indices` per building using the global `type_to_idx`.

**Resolves blockers:** #9.

---

## Phase A+: RL Token Max-Padding

**Goal:** Make `rl_projection` resilient to changing `extra_features` counts without requiring weight reconstruction.

### The Problem

The RL token is constructed from three sub-buckets:
1. **Demand features** → summed to 1 scalar (always contributes 1 dim, regardless of count) ✓ Naturally robust
2. **Generation features** → summed to 1 scalar (same) ✓ Naturally robust
3. **Extra features** → concatenated at full encoded dimensionality ✗ Variable

The `rl_projection` is `nn.Linear(rl_input_dim, d_model)` where:
```python
rl_input_dim = (1 if has_demand_or_gen else 0) + len(extra_encoded_indices)
```

If a building gains or loses an extra feature at runtime, `rl_input_dim` changes and the weight matrix shape is wrong. Pre-allocation doesn't help here because the dimension is data-dependent, not vocabulary-bounded.

### The Fix: Max-padding with mask

**File:** `observation_tokenizer.py`

1. **Compute `max_rl_extra_dim` in pass 1:** During the global scan in `attach_environment()`, compute `rl_input_dim` for each building. Take the maximum:
   ```python
   max_rl_input_dim = max(rl_input_dim_per_building.values())
   ```
   Pass this to each tokenizer.

2. **Allocate `rl_projection` to max dim:**
   ```python
   self.rl_projection = nn.Linear(max_rl_input_dim, d_model)
   ```

3. **In `forward()`, zero-pad RL input to max dim:**
   ```python
   rl_input = torch.cat(parts, dim=-1)  # [B, actual_dim]
   if rl_input.shape[-1] < self._max_rl_input_dim:
       padding = torch.zeros(batch, self._max_rl_input_dim - rl_input.shape[-1], device=device)
       rl_input = torch.cat([rl_input, padding], dim=-1)
   rl_token = self.rl_projection(rl_input).unsqueeze(1)
   ```

4. **In `reconfigure()`, rebuild RL index buffers** (demand/gen/extra), but do NOT rebuild `rl_projection`. If the new actual dim exceeds `max_rl_input_dim`, log a warning — this means the pre-allocated max was insufficient and requires a restart.

### Why this is acceptable

- The zero-padded dimensions contribute nothing to the output (zero input × learned weight = zero). The model learns to ignore padding positions naturally.
- The demand/generation residual (1 dim) is always in position 0, so its learned weight is stable across topologies.
- Extra features occupy positions 1..N. If a feature is removed, its position gets zero-padded; if added, it fills a previously-padded position. Feature ordering must be deterministic (sorted by name) to ensure consistency.

**Resolves blockers:** #18, #20, #21, #22. Partially #19 (buffers still rebuilt in `reconfigure()`).

**Limitation:** If `max_rl_input_dim` is much larger than the typical building's actual dim, unused weight columns waste parameters. In practice, RL extras are a small number of features (typically 0-5), so padding overhead is negligible.

---

## Phase B: `reconfigure_topology()` Method

**Goal:** Allow the tokenizer and agent to rebuild topology metadata at runtime without touching learned parameters. This is the method the deployment system calls when a building's assets change.

### B1. Tokenizer `reconfigure()`

**File:** `observation_tokenizer.py`

Add method:
```python
def reconfigure(self, observation_names, action_names, encoder_config):
```

This method rebuilds **all non-learned metadata** from new observation/action names:

**CA reconstruction:**
1. Rebuilds `_index_map` (encoder index map) from the new observation names.
2. Re-runs `_extract_device_ids()` and instance detection from new action names.
3. Rebuilds `_ca_instances`, `_ca_type_names`.
4. Deregisters old `_ca_idx_*` buffers, registers new ones for the new instance set.
5. Rebuilds `_action_ca_map`.

**SRO reconstruction:**
6. Re-runs SRO feature classification loop (lines 231-249) against new observation names.
7. Rebuilds `_sro_groups` list from newly matched features.
8. Deregisters old `_sro_idx_*` buffers, registers new ones.
9. Validates: for each SRO type, check that the new feature count matches the pre-allocated `nn.Linear` input dim. If shorter, store a flag to zero-pad in `forward()`. If longer, log an error — the pre-allocated max was insufficient.

**RL reconstruction:**
10. Re-runs RL feature classification (demand/generation/extra) against new observation names.
11. Rebuilds `_rl_demand_indices`, `_rl_generation_indices`, `_rl_extra_indices`.
12. Deregisters old `_rl_demand_idx` / `_rl_gen_idx` / `_rl_extra_idx` buffers, registers new ones.
13. Recomputes actual `rl_input_dim`. If it exceeds `_max_rl_input_dim`, log a warning.
14. Updates `total_encoded_dims`.

Does **NOT** touch (learned parameters preserved):
- `ca_projections` (pre-allocated for all types in Phase A2)
- `sro_projections` (pre-allocated for all types in Phase A2b)
- `rl_projection` (max-padded in Phase A+)

**Resolves blockers:** #1, #3, #4, #5 (CA); #15, #17 (SRO); #19 (RL buffers).

### B2. Agent `reconfigure_building()`

**File:** `transformer_ppo_agent.py`

Add method:
```python
def reconfigure_building(self, building_idx, observation_names, action_names, action_space, observation_space):
```

This method:
1. Calls `tokenizer.reconfigure(obs_names, action_names, encoder_config)` on the specified building's model.
2. Updates `_n_ca_per_building[building_idx]` with the new CA count.
3. Rebuilds `_ca_type_indices[building_idx]` using the global `type_to_idx`.
4. Updates `_action_ca_maps[building_idx]` from the tokenizer's new `action_ca_map`.
5. **Flushes the rollout buffer** for that building (mandatory — see CA Removal analysis below).
6. Logs the topology change with old/new CA counts.

**Resolves blockers:** #8, #10, #12, #13.

### B3. CA Removal — Failure Analysis & Mitigation

**Scenario:** Building goes from 3 CAs (battery + ev_charger + washing_machine) to 1 CA (battery only). Two EV chargers disconnect.

**What happens at each component:**

| Component | Behavior | Severity |
|-----------|----------|----------|
| **Tokenizer `forward()`** | Only 1 CA token produced instead of 3. SRO + RL tokens unchanged. Total tokens: 1 + 4 + 1 = 6 (was 3 + 4 + 1 = 8). | OK — backbone handles any sequence length |
| **Backbone** | Self-attention over 6 tokens instead of 8. Positional patterns change. | Minor — attention is permutation-equivariant, but learned patterns may be slightly off |
| **Mean pooling** | `embeddings.mean(dim=1)` — SRO/RL tokens now weighted 1/6 instead of 1/8. Pooled representation shifts. | **Soft degradation** — critic value estimates become unreliable |
| **Actor head** | Only 1 CA embedding → 1 action. `ca_type_indices = [0]` (battery). `log_std[0]` used. | OK — actor is per-token |
| **Critic head** | `nn.Linear(d_model, d_ff)` — same weights, different input distribution due to pool shift. | **Soft degradation** — needs fine-tuning |
| **Rollout buffer** | Old entries have `actions: [3, 1]`, new entries have `actions: [1, 1]`. `torch.stack` crashes. | **Hard crash** if not flushed |
| **PPO update** | `evaluate_actions(ca_embeddings, old_actions)` — shape mismatch between current model output and stored data. | **Hard crash** if buffer contains mixed shapes |

**Mitigation strategy:**

1. **Buffer flush is mandatory** in `reconfigure_building()` — discard any partial rollout data. This loses at most `rollout_steps` transitions. Acceptable because topology changes are rare events.

2. **Critic degradation** is expected and acceptable. The critic will adapt during continued training. For deployment (inference only), the critic is not used.

3. **Future improvement (not in this plan):** Replace mean pooling with a learned `[CLS]` token that is always present as the first token. The critic reads only the `[CLS]` token's output, making it independent of token count. This is a standard Transformer technique but requires architecture changes beyond the scope of runtime adaptability.

---

## Phase C: Checkpoint Compatibility

**Goal:** A checkpoint saved with one topology can be loaded into a model with a different topology.

**File:** `transformer_ppo_agent.py`

### C1. Separate learned params from transient buffers

Learned parameters are now the same shape across topologies (pre-allocated for full vocabulary):
- `tokenizer.ca_projections.{type}.weight/bias` — same for all buildings (pre-allocated)
- `tokenizer.sro_projections.{type}.weight/bias` — same for all buildings (pre-allocated)
- `tokenizer.rl_projection.weight/bias` — same for all buildings (max-padded to `max_rl_input_dim`)
- `backbone.*` — architecture-determined, same everywhere
- `actor.*` including `log_std` — same shape (full vocabulary)
- `critic.*` — same shape

Transient index buffers differ per topology:
- `tokenizer._ca_idx_0`, `_ca_idx_1`, etc. — different count and content per building
- `tokenizer._sro_idx_0`, `_sro_idx_1`, etc. — different if SRO features differ
- `tokenizer._rl_demand_idx`, `_rl_gen_idx`, `_rl_extra_idx` — different if RL features differ

### C2. Loading strategy

In `load_checkpoint()`:
1. Load the checkpoint state dict.
2. For each building model, filter out transient buffer keys (anything matching `_ca_idx_*`, `_sro_idx_*`, `_rl_demand_idx`, `_rl_gen_idx`, `_rl_extra_idx`).
3. Call `model.load_state_dict(filtered_dict, strict=False)` to load learned parameters.
4. Call `reconfigure_building()` (or just rely on the existing `attach_environment()` topology) to rebuild buffers.

Add a helper `_filter_transient_buffers(state_dict) -> state_dict` that removes index buffer keys using regex:
```python
TRANSIENT_PATTERN = re.compile(r"_ca_idx_\d+|_sro_idx_\d+|_rl_demand_idx|_rl_gen_idx|_rl_extra_idx")
```

**Resolves blocker:** #11.

---

## Phase D: Tests

### D1. New tests — CA adaptability

Add to `tests/test_runtime_adaptability.py`:

| Test | What it verifies |
|------|-----------------|
| `test_tokenizer_preallocates_all_ca_types` | Tokenizer with 1 active CA has projections for all 3 config types |
| `test_tokenizer_reconfigure_adds_cas` | `reconfigure()` with new action names → more CA instances, same projection weights |
| `test_tokenizer_reconfigure_removes_cas` | `reconfigure()` with fewer actions → fewer CA instances, projections preserved, forward produces fewer CA tokens |
| `test_actor_global_log_std_shape` | `log_std` shape always matches full vocabulary size (3), works with 1 or 3 CAs |
| `test_agent_reconfigure_building` | `reconfigure_building()` → predict returns correct number of actions |
| `test_checkpoint_cross_topology` | Save with 3 CAs, load into 1 CA building → weights load, reconfigure works |
| `test_global_type_to_idx_consistent` | Same mapping regardless of per-building active types |
| `test_auto_compute_dim_fallback` | CA type with no instances in dataset → dim computed from encoder rules + config |

### D2. New tests — SRO adaptability

| Test | What it verifies |
|------|-----------------|
| `test_tokenizer_preallocates_all_sro_types` | Tokenizer has `sro_projections` for every SRO type in config, even if no features matched in this building |
| `test_tokenizer_reconfigure_sro_features_change` | `reconfigure()` with different SRO feature set → new index buffers, same projection weights |
| `test_sro_projection_weight_stability` | After `reconfigure()`, `sro_projections["pricing"].weight` is the same tensor (not recreated) |
| `test_sro_zero_pad_shorter_features` | Building with fewer pricing features than global max → `forward()` zero-pads and projection still works |
| `test_checkpoint_sro_cross_topology` | Checkpoint from building with 4 SRO types loads into building with 2 active SRO types |

### D3. New tests — RL token adaptability

| Test | What it verifies |
|------|-----------------|
| `test_rl_projection_max_padded` | `rl_projection.in_features == max_rl_input_dim` (not building-specific dim) |
| `test_rl_forward_zero_padding` | Building with 2 extras and max_dim=5 → input is zero-padded to 5, output shape correct |
| `test_rl_reconfigure_extras_added` | `reconfigure()` with new extra features → RL token picks them up, output shape unchanged |
| `test_rl_reconfigure_extras_removed` | `reconfigure()` with fewer extras → zero-padding increases, no crash |
| `test_rl_demand_generation_robust` | Adding more demand features → residual still produces 1 scalar, projection unchanged |
| `test_rl_from_zero_to_nonzero` | Building starts with 0 RL features, reconfigure adds some → projection was pre-allocated to max, works |

### D4. New tests — CA removal failure analysis

| Test | What it verifies |
|------|-----------------|
| `test_ca_removal_buffer_flush` | `reconfigure_building()` flushes rollout buffer; no `torch.stack` crash |
| `test_ca_removal_predict_correct_actions` | After removing 2 CAs, `predict()` returns 1 action (not 3) |
| `test_ca_removal_critic_no_crash` | Critic produces a value after CA removal (may be different magnitude, but no crash) |
| `test_ca_removal_full_ppo_update` | Full PPO update cycle after reconfigure: collect transitions → update → no shape errors |
| `test_ca_removal_mid_episode_warning` | If `reconfigure_building()` called with non-empty buffer, it flushes and logs a warning |

### D5. Update existing tests

Some existing tests assert `len(tokenizer.ca_projections) == N_active_types`. These need updating to assert `== N_total_config_types`. Identify and update all affected assertions. Same for SRO projection count assertions.

---

## Phase E: Meaningful Training Run & Analysis

After Phases A-D are implemented and all tests pass:

1. **Run a full 1-episode training** (8760 steps) with `export_kpis_on_episode_end: true`.
2. **Analyze results:** reward trajectories, CityLearn KPIs (electricity cost, carbon, comfort).
3. **Test cross-topology inference:** take a trained model, call `reconfigure_building()` with different asset counts, verify it produces valid actions with correct shapes.
4. **Report:** whether the architecture goal (same base model across different topologies) is being achieved.

---

## Estimated Scope

| File | Changes |
|------|---------|
| `algorithms/utils/observation_tokenizer.py` | ~200 lines new/modified (pre-allocation for CA+SRO, max-padding for RL, `reconfigure()` method with SRO/RL reconstruction) |
| `algorithms/agents/transformer_ppo_agent.py` | ~100 lines new/modified (two-pass attach for global dims including SRO+RL max, `reconfigure_building()` with buffer flush, checkpoint filtering) |
| `algorithms/utils/ppo_components.py` | ~15 lines modified (global `n_ca_types` for `log_std`) |
| `tests/test_runtime_adaptability.py` | ~450 lines new tests (CA: 8, SRO: 5, RL: 6, CA removal: 5 = 24 tests) |
| Existing test files | ~30 lines updated assertions (CA + SRO projection counts) |

**Total:** ~795 lines. All 178 existing tests must continue to pass after changes.

---

## Blocker Resolution Matrix

| Blocker | Phase | Strategy |
|---------|-------|----------|
| #1 `_ca_instances` frozen | B1 | Reconstruction |
| #2 `ca_projections` incomplete | A2 | Pre-allocation |
| #3 `_ca_idx_*` stale | B1 | Reconstruction |
| #4 `_action_ca_map` frozen | B1 | Reconstruction |
| #5 `_index_map` frozen | B1 | Reconstruction |
| #6 `log_std` wrong shape | A3 | Pre-allocation |
| #7 `log_std` OOB indexing | A3 | Pre-allocation |
| #8 `_n_ca_per_building` stale | B2 | Reconstruction |
| #9 `_ca_type_indices` frozen | A4 | Pre-allocation |
| #10 `_action_ca_maps` stale | B2 | Reconstruction |
| #11 `load_state_dict` strict | C2 | Buffer filtering + `strict=False` |
| #12 Rollout buffer mixed shapes | B2 | Buffer flush (mandatory) |
| #13 `_BuildingModel` one-shot | B2 | `reconfigure_building()` path |
| #14 `sro_projections` incomplete | A2b | Pre-allocation |
| #15 `_sro_idx_*` stale | B1 | Reconstruction |
| #16 SRO `nn.Linear` dim mismatch | A2b | Pre-allocation (+ zero-pad if shorter) |
| #17 `_sro_groups` frozen | B1 | Reconstruction |
| #18 `rl_projection` dim mismatch | A+ | Max-padding |
| #19 RL index buffers stale | B1 | Reconstruction |
| #20 `_rl_extra_idx` conditional | A+ | Always pre-allocate to max |
| #21 `rl_projection` conditional None | A+ | Always create (max-padded) |
| #22 `rl_input_dim` data-dependent | A+ | Max-padding |
