# What Happens When a New EV Charger Appears? (Current System)

## Scenario

**Initial state:**
- Building_5 has 1 battery (1 CA)
- Observation features: `["month", "hour", ..., "electrical_storage_soc", ...]` (34 features)
- Action names: `["electrical_storage"]` (1 action)
- Agent is trained and running

**At timestep 500:**
- A new EV charger is installed (physical hardware added to building)
- Observation features now include: `["...", "electric_vehicle_charger_charger_1_1_connected_state", "connected_electric_vehicle_at_charger_charger_1_1_soc", ...]` (95 features)
- Action names now include: `["electrical_storage", "electric_vehicle_storage_charger_1_1"]` (2 actions)

---

## What Breaks (Step-by-Step Failure)

### 1. Wrapper receives new observation vector

```python
# Timestep 500 - CityLearn returns 95 features instead of 34
observations = env.step(actions)  # Returns dict with 95 values
```

**Status:** ✅ Wrapper receives it successfully

---

### 2. Wrapper tries to encode observations

```python
# wrapper_citylearn.py line 428
encoded_observations = self.get_all_encoded_observations(observations)
```

The wrapper tries to apply its pre-built encoder list (34 encoders) to 95 observations.

**Result:** ❌ **IndexError** or **ValueError**
- The encoder list has 34 entries (one per original feature)
- The observation list has 95 entries (including new EV features)
- `zip(self.encoders[index], obs_array)` at line 494 tries to pair them
- Either crashes or silently ignores the extra 61 features

**Why:** The wrapper's encoders were built at initialization (`set_encoders()` called once at startup) and never updated.

---

### 3. (If encoding somehow succeeded) Agent receives encoded vector

Assume the wrapper somehow produced a flat encoded vector of the new size (157 dims instead of original ~50).

```python
# transformer_ppo_agent.py line 237-240
obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
obs_tensor = obs_tensor.unsqueeze(0)  # [1, 157]
```

**Status:** ✅ Agent receives tensor successfully

---

### 4. Agent passes to tokenizer

```python
# transformer_ppo_agent.py line 253
ca_embeddings, sro_embeddings, rl_embedding, ca_token_counts = tokenizer(obs_tensor)
```

The tokenizer tries to slice the vector using pre-computed index buffers:

```python
# observation_tokenizer.py line 480
idx_buf: torch.Tensor = getattr(self, f"_ca_idx_{i}")  # e.g., [10, 11] for battery
features = encoded_obs[:, idx_buf]  # Slice positions 10-11
```

**Result:** ❌ **Wrong features extracted**
- The index buffers were computed for the old observation layout (34 features → ~50 encoded dims)
- With 95 features, the encoding layout is completely different
- `idx_buf=[10,11]` might now be pointing at `hour` instead of `electrical_storage_soc`
- The battery token gets garbage data

**Why:** The `_index_map` was built once at `attach_environment()` and stores slice positions for the old layout.

---

### 5. (If tokenization somehow succeeded) New EV features are invisible

Even if the battery token extracted correct data, the new EV charger features are **not in `_ca_instances`**:

```python
# observation_tokenizer.py line 278-291
# Built once at __init__:
self._ca_instances = [
    ("battery", None, [10, 11])  # Only 1 CA instance
]
```

The `forward()` loop (line 477-484) only iterates this list. The EV charger features exist in the encoded vector but are never looked up — **they're invisible to the tokenizer**.

**Result:** ❌ **Agent produces 1 action instead of 2**
- The tokenizer produces 1 CA token (battery only)
- The actor head produces 1 action
- The environment expects 2 actions (battery + EV charger)

---

### 6. Environment rejects actions

```python
# Environment expects: [battery_action, ev_charger_action]
# Agent returns: [battery_action]
```

**Result:** ❌ **ValueError** from CityLearn
- Action shape mismatch: expected (2,), got (1,)
- Training crashes

---

## Summary: Complete System Failure

| Step | Component | What Breaks |
|------|-----------|-------------|
| 2 | Wrapper encoding | Encoder list has wrong length → IndexError |
| 4 | Tokenizer slicing | Index buffers point to wrong positions → Garbage data |
| 5 | Tokenizer instances | New CA type not in `_ca_instances` → Invisible features |
| 6 | Environment step | Action count mismatch → ValueError |

---

## What `flexy_plan.md` Fixes

The plan makes the system **adaptive** by:

1. **Pre-allocating projections** for all CA types (battery, ev_charger, washing_machine) even if not currently present
2. **Adding `reconfigure()` method** that rebuilds:
   - Index map (Step 4 fix)
   - CA instances list (Step 5 fix)
   - Action mapping (Step 6 fix)
3. **Wrapper awareness** (optional): Either wrapper gets a `reconfigure_encoders()` method, or we require a new `attach_environment()` call when topology changes

After the fixes, when the EV charger appears:
1. System calls `agent.reconfigure_building(building_idx, new_obs_names, new_action_names, ...)`
2. Tokenizer rebuilds its instance list and index buffers
3. EV charger projection layer (already pre-allocated) is now used
4. Agent produces 2 actions
5. ✅ Training continues without crashing
