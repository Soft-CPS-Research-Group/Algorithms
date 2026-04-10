# AgentTransformerPPO — Dataflow Walkthrough

## Overview Diagram

```
                     ┌─────────────────────────┐
                     │  0. Setup               │  ← Once at startup
                     │  (attach_environment)   │
                     └────────────┬────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│                         CITYLEARN ENVIRONMENT                        │
│                    (Building with batteries, EVs, etc.)              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  1. Raw Observations    │  ← Every timestep
                    │  (feature name → value) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  2. Encoding            │  (Wrapper applies encoder rules)
                    │  (normalize, one-hot)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  3. Encoded Obs Vector  │  (Flat numeric vector)
                    │  [0.23, 0.87, ...]      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  4. Encoder Index Map   │  (Which slice = which feature?)
                    │  (name → slice indices) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  5. Observation         │  (Group features by asset/context)
                    │     Tokenizer           │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  CA Tokens       │      │  SRO + RL Tokens │
        │  (per battery,   │      │  (weather, time, │
        │   EV, etc.)      │      │   pricing, etc.) │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  6. Token Embeddings    │  (All projected to d_model dims)
                 │     [batch, N, 64]      │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │  7. Transformer         │  (Self-attention encoder)
                 │     Backbone            │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  8a. CA          │      │  8b. Pooled      │
    │      Embeddings  │      │      Embedding   │
    │  (contextual)    │      │  (global state)  │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  9. Actor Head   │      │ 10. Critic Head  │
    │  (per-CA action) │      │  (state value)   │
    └────────┬─────────┘      └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ 11. Actions      │  → Back to Environment
    │ [-0.5, 0.8, ...] │
    └──────────────────┘
```

---

## Step-by-Step Breakdown

Each step includes: **what it does**, **why it's needed**, **real example**, and **basketball analogy**.

---

### Step 0: Setup — `attach_environment()`

**When:** Once, at startup, before any timestep runs.

**What it does:**
The runner calls `agent.attach_environment()` passing each building's observation names, action names, action space, and observation space. The agent uses this metadata to build all internal structures: the encoder index map, the tokenizer (with its projections), the Transformer backbone, actor, critic, and optimizer. After this step, the agent knows the exact topology of each building — how many CAs, what types, what features each has.

**Code:** `transformer_ppo_agent.py` — `attach_environment()` method. For each building it:
1. Loads the encoder config from `configs/encoders/default.json`
2. Creates an `ObservationTokenizer` (which internally builds the encoder index map, detects CA instances from action names, classifies features into CA/SRO/RL groups, and creates `nn.Linear` projections)
3. Creates a `TransformerBackbone`
4. Creates `ActorHead` and `CriticHead`
5. Bundles them into a `_BuildingModel`

**Why it's needed:**
Your POC (`policy.py`) skips this entirely — it receives pre-made `[N_ca, d_model]` tensors. But in reality, the agent starts with nothing: just a list of feature names like `["month", "hour", "electrical_storage_soc", ...]` and action names like `["electrical_storage", "electric_vehicle_storage_charger_1_1"]`. From these strings alone, it must figure out: "this building has 1 battery and 1 EV charger, the battery's features are at these positions in the encoded vector, the EV charger's features are at those positions." That's what setup does — it builds the translation layer between CityLearn's flat world and the Transformer's token world.

**Real example — Building_1 (index 0):**
```
observation_names: ["month", "hour", "day_type", "daylight_savings_status",
                    "outdoor_dry_bulb_temperature", ...,
                    "electrical_storage_soc",
                    "electric_vehicle_charger_charger_1_1_connected_state",
                    "connected_electric_vehicle_at_charger_charger_1_1_soc",
                    ...,
                    "washing_machine_1_start_time_step", ...]

action_names: ["electrical_storage",
               "electric_vehicle_storage_charger_1_1",
               "washing_machine_1"]
```

From this, the tokenizer figures out:
- **3 CA instances:** battery (from action `electrical_storage`), ev_charger (from `electric_vehicle_storage_charger_1_1`), washing_machine (from `washing_machine_1`)
- **4 SRO groups:** temporal (month, hour, day_type, daylight_savings_status), weather (temperatures, humidity, irradiance), pricing (electricity_pricing), carbon (carbon_intensity)
- **1 RL token:** demand (non_shiftable_load) - generation (solar_generation) + extras (net_electricity_consumption)

**Basketball analogy:**
This is **team registration day** before the season starts. The league office tells each franchise: "Here's your roster list and the positions they can play." From those names alone, the coaching staff has to figure out: "We have 2 point guards, 3 shooting guards, and 2 centers — here's how we'll organize our playbook." They don't run any plays yet — they're building the game plan based on who's on the roster. After registration, the team knows exactly how to run its offense for any given lineup.

---

### Step 1: Raw Observations

**When:** Every timestep (8760 per year-episode).

**What it does:**
CityLearn simulates the building for one hour and produces a list of raw floating-point values — one per observation feature. These are physical measurements: temperatures in Celsius, SoC as 0.0-1.0, pricing in $/kWh, etc. Each value is a named feature that the environment exposes.

**Code:** The wrapper calls `env.step(actions)` which returns `observations` — a `List[List[float]]` where `observations[i]` is the raw observation for building `i`.

**Why it's needed:**
This is the ground truth from the simulation. Everything downstream transforms and processes these raw numbers. Without raw observations, the agent has nothing to work with.

**Real example — Building_1, timestep 100 (roughly April 11th, 4am):**
```
Feature name                           │ Raw value │ Meaning
───────────────────────────────────────┼───────────┼─────────────────────────
month                                  │ 4.0       │ April
hour                                   │ 4.0       │ 4:00 AM
day_type                               │ 1.0       │ Monday
daylight_savings_status                │ 1.0       │ DST active
outdoor_dry_bulb_temperature           │ 12.3      │ 12.3°C outside
electrical_storage_soc                 │ 0.65      │ Battery at 65%
ev_charger_charger_1_1_connected_state │ 1.0       │ EV is plugged in
connected_ev_at_charger_1_1_soc        │ 0.42      │ EV battery at 42%
non_shiftable_load                     │ 1.85      │ 1.85 kWh demand
solar_generation                       │ 0.0       │ No sun at 4am
electricity_pricing                    │ 0.08      │ $0.08/kWh
...                                    │ ...       │ ...
```

The raw observation is just the flat list of values: `[4.0, 4.0, 1.0, 1.0, 12.3, ..., 0.65, 1.0, 0.42, ..., 1.85, 0.0, 0.08, ...]`

**Basketball analogy:**
This is the **live game stats feed**. Every minute the scoreboard shows raw numbers: "Player A has 12 points, 4 rebounds, 2 assists. Player B has 8 points, 7 rebounds, 5 assists. Shot clock: 14s. Score: 65-58." These are the raw measurements — real, physical values from the game. Nobody has analyzed them yet; they're just the facts on the ground.

---

### Step 2: Encoding

**When:** Every timestep, immediately after raw observations arrive.

**What it does:**
The wrapper applies **encoder rules** from `configs/encoders/default.json` to transform each raw feature into a neural-network-friendly format. Different features get different treatments:

| Encoder Type | What it does | Input → Output dims | Example |
|---|---|---|---|
| `PeriodicNormalization` | Converts cyclical values to (sin, cos) | 1 → **2** | month=4 → [sin(4/12·2π), cos(4/12·2π)] = [0.87, -0.5] |
| `OnehotEncoding` | Converts categorical to one-hot vector | 1 → **N classes** | day_type=1 → [1,0,0,0,0,0,0,0] (8 dims) |
| `Normalize` | Scales to [0,1] range | 1 → **1** | electrical_storage_soc=0.65 → 0.65 |
| `NormalizeWithMissing` | Like Normalize but handles -0.1 (missing) | 1 → **1** | ev_soc=0.42 → 0.42; no EV → 0.0 |
| `RemoveFeature` | Drops the feature entirely | 1 → **0** | outdoor_dry_bulb_temperature → gone |
| `NoNormalization` | Pass-through | 1 → **1** | net_electricity_consumption → as-is |

**Code:** `wrapper_citylearn.py:486-497` — `get_encoded_observations()`:
```python
encoded = np.hstack([
    encoder.transform(obs) if hasattr(encoder, "transform") else encoder * obs
    for encoder, obs in zip(self.encoders[index], obs_array)
])
```
It walks through each raw feature, applies the matching encoder's `.transform()`, and `hstack`s all the results into one flat vector.

**Why it's needed:**
Raw values are messy for neural networks. A month value of `12` has no special relationship with `1` (January), even though December and January are adjacent in time. `PeriodicNormalization` fixes this by mapping to a circle: December and January are close in (sin, cos) space. Similarly, a `day_type` of `3` doesn't mean "3x as much Wednesday as Monday" — one-hot encoding makes each category independent. And features like `outdoor_dry_bulb_temperature` are deliberately removed because they don't help the agent's decisions (or were found to be noisy).

**Crucially:** Encoding is done by the **wrapper**, not our agent. The agent receives the already-encoded flat vector. This is the same encoding pipeline that MADDPG and RuleBasedPolicy use — we didn't change it. Our innovation starts at Step 4.

**Real example — Building_1, same timestep:**
```
Raw feature              │ Encoder             │ Encoded output
─────────────────────────┼─────────────────────┼──────────────────────
month = 4                │ PeriodicNormalization│ [0.87, -0.50]         (2 dims)
hour = 4                 │ PeriodicNormalization│ [0.87, 0.50]          (2 dims)
day_type = 1             │ OnehotEncoding(8)    │ [1,0,0,0,0,0,0,0]    (8 dims)
daylight_savings = 1     │ OnehotEncoding(2)    │ [0, 1]                (2 dims)
outdoor_dry_bulb = 12.3  │ RemoveFeature        │ (nothing)             (0 dims)
...                      │ ...                  │ ...
electrical_storage_soc   │ Normalize            │ [0.65]                (1 dim)
ev_connected_state = 1   │ OnehotEncoding(2)    │ [0, 1]                (2 dims)
ev_departure_time = 7    │ OnehotEncoding(27)   │ [0,..,1,..,0]         (27 dims)
ev_soc = 0.42            │ NormalizeWithMissing │ [0.42]                (1 dim)
...                      │ ...                  │ ...
electricity_pricing=0.08 │ NoNormalization      │ [0.08]                (1 dim)
```

After encoding, all outputs are `hstack`ed into a single flat vector of length 97 (for Building_1).

**Connection to the flexy plan:**
This encoding step is one of the things that makes runtime adaptability hard. If a building gains a new EV charger at runtime, its raw observation list gets longer (new features appear). The wrapper would produce a different-length encoded vector. Our `_index_map` (built at Step 0) would be stale because it was computed for the old feature list. That's blockers #5 and #19 in the flexy plan — the metadata computed at setup time doesn't match the new reality. The `reconfigure()` method in Phase B rebuilds the index map from the new feature names.

**Basketball analogy:**
This is the **stats processing department** normalizing the raw feed before coaches see it. Raw stats say "Player A: 12 points." But coaches need context: is 12 points good or bad? Depends on the game. So the stats department normalizes: "Player A is scoring at the 75th percentile tonight" (like `Normalize`). For time-related stats, they convert to cyclic form: "We're in the 4th quarter" becomes a position on a clock face (like `PeriodicNormalization`). For categorical stuff like "zone defense vs man-to-man", they create binary flags (like `OnehotEncoding`). And some raw stats — like the arena temperature — are dropped entirely because coaches don't need them (like `RemoveFeature`). The processed stat sheet that reaches the coach's tablet is **the encoded vector**: a standardized, fixed-format summary that's easy to work with.

---

### Step 3: Encoded Observation Vector

**When:** Every timestep, immediately after encoding. This is what our agent actually receives.

**What it does:**
This is the **output of encoding / input to our agent**. It's a flat numpy array of floating-point numbers, with a fixed length per building. The agent's `predict()` method receives this vector and must turn it into actions. Everything from here onwards (Steps 4-11) is our architecture's responsibility.

**Code:** The wrapper stores the encoded vector and passes it to the agent:
```python
# wrapper_citylearn.py:428
encoded_observations = self.get_all_encoded_observations(observations)
# ...
# transformer_ppo_agent.py:248 (in predict)
obs_tensor = torch.tensor(observations[i], dtype=torch.float32)  # [obs_dim]
```

**Why it's needed:**
This is the boundary between "what the platform gives us" and "what we do with it." The flat vector is what **every** agent in the system receives — MADDPG gets the same vector and feeds it directly into an MLP. The difference is what happens next: MADDPG treats the vector as an opaque blob. Our agent knows the **structure** inside the vector.

**Real example — Building_1:**
```
encoded_obs = [0.87, -0.50, 0.87, 0.50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, ..., 0.65, 0, 1, ..., 0.08, ...]
               ┃ month  ┃   ┃ hour   ┃  ┃     day_type (8)      ┃  ┃DST┃      ┃bat┃ ┃EV connect┃     ┃price┃
               positions    positions    positions                   pos.       pos   positions        pos
               [0:2]        [2:4]        [4:12]                      [12:14]    [?]   [?:?]            [?]
```

Shape: `[97]` — a flat vector with 97 numbers. This is the same every timestep (for this building). The numbers change; the positions don't.

**The problem this creates (and why we built Steps 4-5):**
MADDPG feeds all 97 numbers into a single fully-connected layer. The network has to learn from scratch that positions [0:2] are "month", positions [67] is "battery SoC", etc. Worse, if a different building has 2 EV chargers instead of 1, it has a different obs_dim (e.g., 156 instead of 97). MADDPG needs a **separate network** for each building because `nn.Linear(97, hidden)` can't accept a 156-dimensional input.

Our approach: instead of treating this as an opaque blob, we use the **encoder index map** (Step 4) to know exactly where each feature lives, then the **tokenizer** (Step 5) groups features by asset type and projects them to a common dimension. That's how the same Transformer handles buildings with different obs_dims.

This is exactly what your POC assumes is already done — in `policy.py`, you receive `ca_embeddings: [N_ca, d_model]` already projected. Steps 3→4→5→6 are the pipeline that gets us from this flat `[97]` blob to those pre-made tokens.

**Connection to the flexy plan:**
The flat vector length (97) is determined by the feature list established at setup time. If features change at runtime, the vector length changes. The tokenizer's registered index buffers (`_ca_idx_0`, `_sro_idx_0`, etc.) contain hard-coded positions within this vector. A new 120-length vector with different feature positions would cause the buffers to index wrong positions or go out-of-bounds. That's why `reconfigure()` (Phase B1) rebuilds ALL index buffers from the new feature names — it's essentially re-answering the question "where is each feature in the new encoded vector?"

**Basketball analogy:**
The encoded vector is the **standardized stat sheet** that arrives on the coach's tablet every minute. It's a long row of processed numbers: `[0.87, -0.50, 0.87, 0.50, 1, 0, 0, 0, ..., 0.65, ...]`. The old-school coach (MADDPG) glances at the entire sheet and makes a gut decision — the sheet is just "a bunch of numbers." Our coach (TransformerPPO) knows the layout: "columns 1-2 are the game clock, columns 3-10 are the defensive scheme, column 67 is our center's energy level, columns 68-95 are our EV charger stats..." and organizes them into player cards (tokens). If a team adds a new player mid-season, the stat sheet gets longer with new columns — the old-school coach needs a whole new playbook (new network), but our coach just reads the updated column labels and reorganizes the player cards (reconfigure).

---

## What We've Covered So Far

| Step | Happens | Who does it | Your POC equivalent |
|------|---------|-------------|---------------------|
| 0. Setup | Once at startup | Agent's `attach_environment()` | Not covered — POC receives pre-made tokens |
| 1. Raw Obs | Every timestep | CityLearn environment | Not covered |
| 2. Encoding | Every timestep | Wrapper (shared with MADDPG) | Not covered |
| 3. Encoded Vector | Every timestep | Wrapper output → Agent input | Not covered — this is where POC begins conceptually, but POC starts with already-tokenized `[N_ca, d_model]` tensors, skipping Steps 4-6 |

**The gap between your POC and reality is Steps 0-6.** Your POC proves the Transformer core works. Steps 0-6 are the plumbing that converts CityLearn's flat world into the token format your POC expects. Steps 7-11 are essentially your POC (backbone + head + action extraction) plus the PPO training machinery.

---

### Step 4: Encoder Index Map — Translating Feature Names to Positions

**When:** Once at startup, during `attach_environment()`, before the tokenizer is built.

**What it does:**
The `build_encoder_index_map()` function processes the list of raw observation names together with the encoder configuration and produces an ordered dictionary that maps each feature name to its post-encoding slice `(start_idx, end_idx, n_dims)` in the flat encoded vector. This is the **translation key** that tells the tokenizer where to find each feature after the wrapper encodes the raw observations.

**Code:** `algorithms/utils/encoder_index_map.py` — `build_encoder_index_map()`:
1. Iterate through observation names in order (this order matters — CityLearn always produces them in the same sequence).
2. For each name, find the matching encoder rule from `configs/encoders/default.json` using the same `_matches_rule()` logic the wrapper uses (equals, contains, prefixes, suffixes).
3. Compute how many post-encoding dimensions the encoder produces: `PeriodicNormalization` → 2 (sin, cos), `OnehotEncoding` → len(classes), `RemoveFeature` → 0, everything else → 1.
4. Assign the current running index as `start_idx`, add `n_dims` to get `end_idx`, record the slice, increment the running index.
5. Return the ordered mapping `{name: EncoderSlice(start, end, n_dims)}`.

**Why it's needed:**
The tokenizer needs to know "where is the battery SoC in this 97-number flat vector?" Without the index map, the tokenizer has no way to extract features from the encoded vector — it would be like trying to read a book where all the words are written as a single line of numbers with no spaces or punctuation. The index map provides the punctuation: it says "positions 0-2 are month, positions 2-4 are hour, position 67 is battery SoC."

Crucially, the index map is computed from the same encoder config that the wrapper uses, so they're always in sync. If the wrapper applies `OnehotEncoding(8)` to `day_type`, the index map knows to expect 8 dimensions starting at that position.

**Real example — Building_1:**

```
Observation names (first 10):
["month", "hour", "day_type", "daylight_savings_status", "outdoor_dry_bulb_temperature", ..., "electrical_storage_soc", ...]

Encoder config rules (excerpt):
- {"match": {"equals": ["month"]}, "encoder": {"type": "PeriodicNormalization"}}
- {"match": {"equals": ["hour"]}, "encoder": {"type": "PeriodicNormalization"}}
- {"match": {"equals": ["day_type"]}, "encoder": {"type": "OnehotEncoding", "params": {"classes": [0,1,2,3,4,5,6,7]}}}
- {"match": {"suffixes": ["_soc"]}, "encoder": {"type": "Normalize"}}
- {"match": {"equals": ["outdoor_dry_bulb_temperature"]}, "encoder": {"type": "RemoveFeature"}}

Resulting index map (excerpt):
{
    "month":                           EncoderSlice(start_idx=0,  end_idx=2,  n_dims=2),   # PeriodicNorm → 2
    "hour":                            EncoderSlice(start_idx=2,  end_idx=4,  n_dims=2),   # PeriodicNorm → 2
    "day_type":                        EncoderSlice(start_idx=4,  end_idx=12, n_dims=8),   # OnehotEncoding(8) → 8
    "daylight_savings_status":         EncoderSlice(start_idx=12, end_idx=14, n_dims=2),   # OnehotEncoding(2) → 2
    "outdoor_dry_bulb_temperature":    EncoderSlice(start_idx=14, end_idx=14, n_dims=0),   # RemoveFeature → 0 (no space)
    ...
    "electrical_storage_soc":          EncoderSlice(start_idx=67, end_idx=68, n_dims=1),   # Normalize → 1
    ...
}
```

Notice how `outdoor_dry_bulb_temperature` has `start_idx == end_idx` (both 14) and `n_dims=0` — it takes up no space in the encoded vector because it was removed. The next feature starts at index 14, not 15.

Given the encoded vector `[0.87, -0.50, 0.87, 0.50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, ..., 0.65, ...]` from Step 3, we can now extract:
- `month` → `[0.87, -0.50]` (indices 0:2)
- `hour` → `[0.87, 0.50]` (indices 2:4)
- `day_type` → `[1, 0, 0, 0, 0, 0, 0, 0]` (indices 4:12)
- `electrical_storage_soc` → `[0.65]` (index 67:68)

**Connection to the flexy plan:**
The index map is **topology-specific** — it's computed from a particular building's observation names at startup. If the building gains a new EV charger at runtime, its observation names list gets longer (new EV features appear), the wrapper's encoded vector gets longer, and the old index map is stale. The flexy plan's Phase B (`reconfigure()`) rebuilds the index map from the new observation names, updating all the indices to match the new encoded vector structure. This is blocker #5 in the flexy plan — the index map is a hard-coded snapshot of the startup topology and doesn't automatically adapt to changes.

**Basketball analogy:**
The index map is the **stat sheet column legend**. When the stats department hands the coach a processed stat sheet (the encoded vector), it's just a long row of numbers. Without a legend, the coach doesn't know which columns are which — is column 5 "points scored" or "fouls committed"? The index map is the legend that says: "Column 0-1: game clock (sin, cos), Column 2-9: defensive formation (one-hot), Column 67: center's energy level." With the legend, the coach can extract the right stats for each decision. If the team adds a new player mid-season, the stat sheet gets new columns and the legend must be updated — that's why `reconfigure()` rebuilds the map when the roster changes.

---

### Step 5: Observation Tokenizer — Feature Classification

**When:** Once at startup, during the tokenizer's `__init__()` method, immediately after building the encoder index map.

**What it does:**
The tokenizer sorts the raw feature names into **token groups**: CA tokens (one per device instance), SRO tokens (shared read-only context), and the RL token (demand-generation residual). This classification determines the token structure that the Transformer will process. The tokenizer uses pattern matching on feature names (configured in `configs/tokenizer_config.json`) to decide which group each feature belongs to.

**Code:** `algorithms/utils/observation_tokenizer.py` — `ObservationTokenizer.__init__()` (lines 177-273):

**Classification happens in three phases:**

**Phase 1 — CA classification (lines 187-229):**
- Extract device IDs from **action names** (e.g., `"electric_vehicle_storage_charger_1_1"` → device ID `"charger_1_1"`).
- For each CA type (battery, ev_charger, washing_machine), check each observation name:
  - Does the name match any of this CA type's feature patterns? (e.g., `"_soc"`, `"connected_state"` for EV chargers)
  - If yes, does the name contain the device ID for one of this type's instances? (e.g., does `"electric_vehicle_charger_charger_1_1_connected_state"` contain `"charger_1_1"`?)
- Group features by `(ca_type, device_id)` pairs — each pair becomes one CA token.
- Mark all matched features as "assigned."

**Phase 2 — SRO classification (lines 232-249):**
- For each SRO type (temporal, weather, pricing, carbon), check each **unassigned** observation name against the SRO type's feature patterns.
- If a name matches (e.g., `"month"` matches temporal, `"electricity_pricing"` matches pricing), add it to that SRO group.
- Mark matched features as "assigned."

**Phase 3 — RL classification (lines 252-268):**
- Check all **remaining unassigned** names for demand features (`"non_shiftable_load"`), generation features (`"solar_generation"`), or extra features (`"net_electricity_consumption"`).
- All RL-related features go into a single RL group.
- Mark matched features as "assigned."
- Warn about any features that remain unmatched.

**Why it's needed:**
This step implements the core architectural design: **structural separation** of controllable assets, context information, and demand signals. By grouping features into typed tokens, we enforce the one-to-one CA input/output mapping (design goal #2 from the plan) and allow the Transformer to learn shared representations for asset types across different buildings. A battery in Building_2 and a battery in Building_4 both produce "battery tokens" processed by the same projection layer — the Transformer learns what a "battery token" means in general, not just for one specific building.

Without classification, all features would be in one opaque blob and the Transformer couldn't distinguish "this is a battery feature" from "this is a weather feature." Classification gives the model **structure** to learn from.

**Real example — Building_1:**

Building_1 has 3 CAs (battery, 1 EV charger, 1 washing machine), 4 SRO groups (temporal, weather, pricing, carbon), and 1 RL token.

```
Raw observation names (97 total):
["month", "hour", "day_type", "daylight_savings_status",
 "outdoor_dry_bulb_temperature", "outdoor_dry_bulb_temperature_predicted_6h", ...,
 "electrical_storage_soc",
 "electric_vehicle_charger_charger_1_1_connected_state",
 "electric_vehicle_charger_charger_1_1_departure_time",
 "connected_electric_vehicle_at_charger_charger_1_1_soc",
 ...,
 "washing_machine_1_start_time_step",
 "washing_machine_1_end_time_step",
 ...,
 "electricity_pricing",
 "carbon_intensity",
 "non_shiftable_load",
 "solar_generation",
 "net_electricity_consumption"]

Action names (3 total):
["electrical_storage",
 "electric_vehicle_storage_charger_1_1",
 "washing_machine_1"]

Classification result:

CA groups:
  battery (device_id=None):
    - "electrical_storage_soc"                                   → 1 dim after encoding
  
  ev_charger (device_id="charger_1_1"):
    - "electric_vehicle_charger_charger_1_1_connected_state"     → 2 dims (one-hot)
    - "electric_vehicle_charger_charger_1_1_departure_time"      → 27 dims (one-hot)
    - "connected_electric_vehicle_at_charger_charger_1_1_soc"    → 1 dim
    ... (10 more EV features)                                     → 14 dims total
  
  washing_machine (device_id="1"):
    - "washing_machine_1_start_time_step"                        → 1 dim
    - "washing_machine_1_end_time_step"                          → 1 dim
    ... (2 more WM features)                                      → 4 dims total

SRO groups:
  temporal:
    - "month", "hour", "day_type", "daylight_savings_status"     → 14 dims total
  
  weather:
    - (none active — outdoor_dry_bulb_temperature removed)       → 0 dims
  
  pricing:
    - "electricity_pricing"                                      → 1 dim
  
  carbon:
    - "carbon_intensity"                                         → 1 dim

RL group:
  demand: "non_shiftable_load"                                   → 1 dim
  generation: "solar_generation"                                 → 1 dim
  extra: "net_electricity_consumption"                           → 1 dim
  → RL input = (demand - generation) + extra = 1 + 1 = 2 dims
```

Token structure: **3 CA tokens + 3 SRO tokens (weather skipped, 0 dims) + 1 RL token = 7 tokens** total.

**Connection to the flexy plan:**
Classification is **action-driven** for CAs — device IDs come from action names, not observation names. If a building gains an EV charger at runtime, a new action appears (`"electric_vehicle_storage_charger_2_1"`), and `reconfigure()` (Phase B2) re-runs classification to detect the new CA instance and assign its features to a new token. SRO and RL groups are stable (they don't depend on actions), but CA groups are dynamic. This is why blockers #8 and #9 in the flexy plan require updating the `_ca_instances` list and the `_action_ca_map` — the number and identity of CA tokens can change.

**Basketball analogy:**
Classification is **roster organization**. The coach receives a long list of players and stats, and needs to organize them into functional groups: "These 5 players are the starting lineup (CA tokens), these stats are about the game clock and opponent (SRO tokens), and this stat is our team's energy deficit (RL token)." Each starting player gets their own player card (one CA token per player), and shared context (weather, clock) goes on a separate board (SRO tokens). If a new player joins mid-season, the coach adds a new player card — that's reconfiguration. The classification step is building the initial roster organization from the player names (observation names) and position assignments (action names).

---

### Step 6: Token Projection — Embedding Features into d_model Space

**When:** Every timestep, during the tokenizer's `forward()` method, after receiving the flat encoded observation vector.

**What it does:**
For each token group (CAs, SROs, RL), the tokenizer:
1. **Gathers** the feature values from the encoded vector using the index buffers registered at startup (e.g., `_ca_idx_0` contains `[67]` for battery SoC).
2. **Projects** the gathered features to `d_model` dimensions (typically 64) using per-type `nn.Linear` layers.
3. **Stacks** all tokens into a single sequence `[batch, N_tokens, d_model]` ready for the Transformer.

This is where we convert from **feature space** (different dimensions per asset type — battery=1, EV=14, washing machine=4) to **token space** (all tokens have `d_model=64` dims).

**Code:** `algorithms/utils/observation_tokenizer.py` — `ObservationTokenizer.forward()` (lines 460-539):

**For CA tokens (lines 477-489):**
```python
for i, (ca_type, device_id, _indices) in enumerate(self._ca_instances):
    idx_buf = getattr(self, f"_ca_idx_{i}")          # Registered buffer: [67] for battery
    features = encoded_obs[:, idx_buf]               # [batch, 1] — gather battery SoC
    projection = self.ca_projections[ca_type]        # nn.Linear(1, 64) for battery type
    token = projection(features)                     # [batch, 64] — projected embedding
    ca_token_list.append(token)
ca_tokens = torch.stack(ca_token_list, dim=1)        # [batch, N_ca, 64]
```

**For SRO tokens (lines 492-503):**
```python
for i, (sro_type, _indices) in enumerate(self._sro_groups):
    idx_buf = getattr(self, f"_sro_idx_{i}")         # e.g., [0,1,2,3,...,13] for temporal
    features = encoded_obs[:, idx_buf]               # [batch, 14] — gather temporal features
    projection = self.sro_projections[sro_type]      # nn.Linear(14, 64) for temporal type
    token = projection(features)                     # [batch, 64]
    sro_token_list.append(token)
sro_tokens = torch.stack(sro_token_list, dim=1)      # [batch, N_sro, 64]
```

**For RL token (lines 506-531):**
```python
# Compute residual = demand - generation
demand = encoded_obs[:, demand_idx].sum(dim=-1, keepdim=True)       # [batch, 1]
generation = encoded_obs[:, gen_idx].sum(dim=-1, keepdim=True)      # [batch, 1]
residual = demand - generation                                      # [batch, 1]

# Gather extra features
extra = encoded_obs[:, extra_idx]                                   # [batch, n_extra]

# Concatenate and project
rl_input = torch.cat([residual, extra], dim=-1)                     # [batch, rl_input_dim]
rl_token = self.rl_projection(rl_input).unsqueeze(1)                # [batch, 1, 64]
```

**Why it's needed:**
The Transformer's self-attention mechanism requires all tokens to have the same dimension. A battery (1 encoded feature) and an EV charger (14 encoded features) can't be processed together in their raw form — they're different sizes. By projecting both to `d_model=64`, we create a common representation space where the Transformer can compare and relate them via attention weights.

The projection layers are **per-type, not per-instance** — all batteries use the same `ca_projections["battery"]` layer, all EV chargers use `ca_projections["ev_charger"]`, etc. This is the key to cross-topology transfer (design goal #1): when a checkpoint trained on Building_4 (2 CAs) loads into Building_2 (1 CA), the battery projection weights transfer because they're shared across all battery instances. If projections were per-instance, weights wouldn't transfer.

**Real example — Building_1 at timestep 100:**

Encoded observation vector (97 dims):
```
[0.87, -0.50, 0.87, 0.50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, ..., 0.65, ..., 0, 1, ..., 0.42, ..., 0.08, ..., 1.85, 0.0, 0.35]
```

**CA token 0 (battery):**
- Gather: `idx_buf = [67]` → `features = [0.65]` (shape: `[batch=1, 1]`)
- Project: `ca_projections["battery"]` is `nn.Linear(1, 64)` with learned weights `W_battery: [1, 64]`
- Result: `token = W_battery @ [0.65]` → `[0.13, -0.42, 0.78, ..., 0.22]` (shape: `[1, 64]`)

**CA token 1 (EV charger charger_1_1):**
- Gather: `idx_buf = [68, 69, ..., 81]` → `features = [0, 1, ..., 0.42]` (shape: `[1, 14]`)
- Project: `ca_projections["ev_charger"]` is `nn.Linear(14, 64)` with learned weights `W_ev: [14, 64]`
- Result: `token = W_ev @ [0, 1, ..., 0.42]` → `[0.55, -0.12, 0.03, ..., -0.31]` (shape: `[1, 64]`)

**CA token 2 (washing machine 1):**
- Gather: `idx_buf = [82, 83, 84, 85]` → `features = [0, 24, 0, 0]` (shape: `[1, 4]`)
- Project: `ca_projections["washing_machine"]` is `nn.Linear(4, 64)` with learned weights `W_wm: [4, 64]`
- Result: `token = W_wm @ [0, 24, 0, 0]` → `[-0.88, 0.67, -0.15, ..., 0.44]` (shape: `[1, 64]`)

**SRO token 0 (temporal):**
- Gather: `idx_buf = [0,1,2,3,...,13]` → `features = [0.87, -0.50, ..., 0, 1]` (shape: `[1, 14]`)
- Project: `sro_projections["temporal"]` is `nn.Linear(14, 64)`
- Result: `token = [...]` (shape: `[1, 64]`)

**SRO token 1 (pricing):**
- Gather: `idx_buf = [86]` → `features = [0.08]` (shape: `[1, 1]`)
- Project: `sro_projections["pricing"]` is `nn.Linear(1, 64)`
- Result: `token = [...]` (shape: `[1, 64]`)

**SRO token 2 (carbon):**
- Gather: `idx_buf = [87]` → `features = [0.12]` (shape: `[1, 1]`)
- Project: `sro_projections["carbon"]` is `nn.Linear(1, 64)`
- Result: `token = [...]` (shape: `[1, 64]`)

**RL token:**
- Compute residual: `demand=[1.85]`, `generation=[0.0]` → `residual = [1.85]`
- Gather extra: `extra = [0.35]`
- Concatenate: `rl_input = [1.85, 0.35]` (shape: `[1, 2]`)
- Project: `rl_projection` is `nn.Linear(2, 64)`
- Result: `token = [...]` (shape: `[1, 64]`)

**Final token sequence:**
```
ca_tokens:  [batch=1, N_ca=3, d_model=64]    (battery, EV, WM)
sro_tokens: [batch=1, N_sro=3, d_model=64]   (temporal, pricing, carbon)
rl_token:   [batch=1, 1, d_model=64]

Combined sequence for Transformer input: [batch=1, 7, 64]
  token 0: battery          [0.13, -0.42, 0.78, ..., 0.22]
  token 1: EV charger       [0.55, -0.12, 0.03, ..., -0.31]
  token 2: washing machine  [-0.88, 0.67, -0.15, ..., 0.44]
  token 3: temporal         [...]
  token 4: pricing          [...]
  token 5: carbon           [...]
  token 6: RL               [...]
```

This is exactly the format your POC expects: a sequence of `[N_tokens, d_model]` embeddings where the first `N_ca` tokens are CA tokens producing actions.

**Connection to the flexy plan:**
The index buffers (`_ca_idx_0`, `_sro_idx_0`, etc.) are **registered as PyTorch buffers**, which means they're part of the model's state dict but NOT learned parameters. When a checkpoint is loaded, these buffers are loaded too — but if the building topology is different, the buffers point to wrong positions in the encoded vector (blocker #14). Phase B3 of the flexy plan re-registers all buffers after reconfiguration to match the new topology. Phase A pre-allocates all projection layers for the full vocabulary (even inactive types), so the learned weights always transfer — only the buffers need updating.

The projection layers (`nn.Linear`) are **shared across instances of the same type** — all batteries use the same projection, all EV chargers use the same projection. This is why a checkpoint trained on Building_4 (1 battery) can load into Building_15 (1 battery + 2 EVs): the battery projection weights transfer directly, and the EV projection weights are already pre-allocated (Phase A) so they're present in both models.

**Basketball analogy:**
Token projection is **creating player cards** for the playbook. Each player's raw stats (height, weight, shooting%, assists) are different dimensions — the center has 1 stat (rebounds=12), the point guard has 5 stats (assists=7, steals=3, turnovers=2, points=14, fouls=1). You can't compare them directly. So the analytics team creates a standardized "player card" for each — a 64-number summary that captures their contribution in a common format. The center's card is computed by projecting their [12 rebounds] through a "center projection formula" → 64-number card. The point guard's card is computed by projecting their [7,3,2,14,1] through a "point guard projection formula" → 64-number card. Now the coach can lay out all player cards on the table (the token sequence) and compare them side-by-side — they're all the same size, even though the raw stats were different sizes. The "projection formula" for centers is shared across all centers on all teams — it's learned once and reused everywhere (that's why weights transfer across buildings).

---

### Step 7: Transformer Backbone — Self-Attention Over Tokens

**When:** Every timestep, during `predict()`, after tokenization.

**What it does:**
The Transformer backbone processes the token sequence `[batch, N_tokens, d_model]` through multiple layers of self-attention and feedforward networks. Each layer allows tokens to "communicate" with each other — the battery token can attend to the pricing token, the EV charger token can attend to the RL token, etc. This contextual processing enriches each token's representation with information from all other tokens. The output is a contextual embedding sequence with the same shape as the input: `[batch, N_tokens, d_model]`.

**Code:** `algorithms/utils/transformer_backbone.py` — `TransformerBackbone.forward()`:
```python
def forward(self, token_seq: torch.Tensor) -> TransformerOutput:
    """
    Parameters
    ----------
    token_seq : Tensor[batch, N, d_model]
    
    Returns
    -------
    TransformerOutput with:
      - ca_embeddings: [batch, N_ca, d_model] (first N_ca tokens)
      - pooled: [batch, d_model] (mean over all tokens)
    """
    # Multi-head self-attention + feedforward layers
    x = token_seq  # [batch, N, d_model]
    for layer in self.layers:
        x = layer(x)  # Each layer: self-attention → add&norm → FFN → add&norm
    
    # Split output: first N_ca tokens are CA embeddings
    ca_embeddings = x[:, :N_ca, :]  # Computed dynamically from token_seq shape
    
    # Pool all tokens for critic
    pooled = x.mean(dim=1)  # [batch, d_model]
    
    return TransformerOutput(ca_embeddings=ca_embeddings, pooled=pooled)
```

**Why it's needed:**
Without the Transformer, each token would be processed independently — the battery wouldn't know the current electricity price, the EV charger wouldn't know if the RL token indicates high demand, etc. Self-attention lets the model learn relationships: "when pricing is high (pricing token) AND demand is high (RL token), the battery token should encode 'prepare to discharge'." These contextual embeddings are much richer than the initial projected tokens from Step 6.

The Transformer is **topology-agnostic** — it doesn't care how many CA tokens there are or which types they represent. It just processes whatever sequence it receives. This is why the same Transformer weights can handle Building_2 (1 CA) and Building_15 (3 CAs) — the self-attention mechanism naturally adapts to different sequence lengths.

**Real example — Building_1 at timestep 100:**

Input token sequence (from Step 6):
```
[batch=1, 7 tokens, 64 dims]
  token 0: battery          [0.13, -0.42, 0.78, ..., 0.22]
  token 1: EV charger       [0.55, -0.12, 0.03, ..., -0.31]
  token 2: washing machine  [-0.88, 0.67, -0.15, ..., 0.44]
  token 3: temporal         [...]
  token 4: pricing          [...]
  token 5: carbon           [...]
  token 6: RL               [...]
```

After Transformer backbone (2 layers, 4 attention heads):
```
[batch=1, 7 tokens, 64 dims]  (same shape, but values are contextualized)
  token 0: battery (contextualized)       [0.22, -0.35, 0.91, ..., 0.18]
  token 1: EV charger (contextualized)    [0.61, -0.08, 0.15, ..., -0.22]
  token 2: washing machine (contextualized) [-0.75, 0.72, -0.10, ..., 0.39]
  token 3: temporal (contextualized)      [...]
  token 4: pricing (contextualized)       [...]
  token 5: carbon (contextualized)        [...]
  token 6: RL (contextualized)            [...]
```

The contextualized battery token now "knows" about high demand (from RL token), high pricing (from pricing token), and EV charger status (from EV token) through attention weights. For example, attention weights might show:
- Battery token attends strongly to: pricing (0.35), RL (0.28), temporal (0.20), self (0.17)
- EV token attends strongly to: temporal (0.42), self (0.25), RL (0.18), battery (0.15)

**Connection to the flexy plan:**
The Transformer backbone is **completely topology-agnostic** — it's a pure sequence-to-sequence model with no hard-coded assumptions about token count or types. This is the key to cross-topology transfer (design goal #1). When a checkpoint trained on Building_4 loads into Building_2, the Transformer weights transfer directly without modification. The backbone doesn't care that the sequence length changed from 7 tokens to 5 tokens — self-attention naturally handles variable-length sequences. This is why Phase C's checkpoint compatibility only needs to filter transient buffers; the Transformer itself needs no special handling.

**Basketball analogy:**
The Transformer is the **team huddle** before each play. Initially, each player knows only their own stats (the projected tokens from Step 6). During the huddle, they share information: "I'm the point guard, I see the defense is in zone formation (from temporal/context tokens). Center, you're at 85% energy (battery token). Shooting guard, you're hot tonight with 18 points (EV token). Coach says we're down by 5 (RL token) and we need to be aggressive." After the huddle, each player's understanding is **contextualized** — the center now knows "I should conserve energy because we're behind and the point guard needs me for late-game rebounds." The contextualized player cards (output embeddings) incorporate information from the entire team, not just individual stats. The huddle process (self-attention) works the same whether you have 5 players or 7 — it's a universal communication mechanism.

---

### Step 8a: CA Embeddings — Slicing Contextual Tokens for Actions

**When:** Every timestep, immediately after the Transformer backbone processes the token sequence.

**What it does:**
The Transformer outputs `[batch, N_tokens, d_model]` for all tokens, but only the **first `N_ca` tokens** correspond to controllable assets that need actions. This step extracts those CA embeddings by slicing: `ca_embeddings = transformer_out[:, :N_ca, :]`. These embeddings are passed to the actor head to generate actions.

**Code:** `algorithms/utils/transformer_backbone.py` — inside `TransformerBackbone.forward()`:
```python
# After all Transformer layers
x = self.layers[-1](x)  # [batch, N_tokens, d_model]

# Extract CA embeddings (first N_ca positions)
# N_ca is determined from the input token sequence shape
# (the tokenizer ensures CA tokens come first)
ca_embeddings = x[:, :N_ca, :]  # [batch, N_ca, d_model]
```

And in `transformer_ppo_agent.py` — inside `predict()`:
```python
transformer_out = model.backbone(tokenized)  # TransformerOutput object
ca_embeddings = transformer_out.ca_embeddings  # [batch, N_ca, d_model]

# Pass to actor head
actions, log_probs, entropy = model.actor(
    ca_embeddings,
    ca_type_indices=ca_type_idx,
    deterministic=is_deterministic,
)
```

**Why it's needed:**
The Transformer processes all tokens (CAs + SROs + RL) together, but only CA tokens need actions — SRO tokens are read-only context, and the RL token is just an input signal. By slicing out only the CA embeddings, we ensure the actor head produces exactly `N_ca` actions, one per controllable asset. This enforces the **strict one-to-one CA input/output mapping** (design goal #2).

Crucially, the tokenizer ensures CA tokens always come first in the sequence (see `observation_tokenizer.py:460-489`), so slicing `[:, :N_ca, :]` always extracts the right tokens. If Building_1 has 3 CAs, we slice `[:, :3, :]`. If Building_2 has 1 CA, we slice `[:, :1, :]`. The slice size adapts to the building topology.

**Real example — Building_1:**

Transformer output (from Step 7):
```
[batch=1, 7 tokens, 64 dims]
  token 0: battery (contextualized)       [0.22, -0.35, 0.91, ..., 0.18]
  token 1: EV charger (contextualized)    [0.61, -0.08, 0.15, ..., -0.22]
  token 2: washing machine (contextualized) [-0.75, 0.72, -0.10, ..., 0.39]
  token 3: temporal (contextualized)      [...]
  token 4: pricing (contextualized)       [...]
  token 5: carbon (contextualized)        [...]
  token 6: RL (contextualized)            [...]
```

CA embeddings (slice `[:, :3, :]`):
```
[batch=1, 3 CAs, 64 dims]
  CA 0: battery          [0.22, -0.35, 0.91, ..., 0.18]
  CA 1: EV charger       [0.61, -0.08, 0.15, ..., -0.22]
  CA 2: washing machine  [-0.75, 0.72, -0.10, ..., 0.39]
```

For Building_2 (1 CA), the slice would be `[:, :1, :]` → `[batch=1, 1 CA, 64 dims]`.

**Connection to the flexy plan:**
The CA embedding slice size (`N_ca`) is **determined at runtime** from the tokenizer's output shape. When the topology changes (e.g., a new EV is added), the tokenizer produces a different `N_ca` (e.g., 3 → 4), and the slice automatically adapts. No code needs to change — the slicing logic `[:, :N_ca, :]` is generic. This is how the architecture achieves runtime adaptability (design goal #1) without recompilation.

However, the `N_ca` value is not arbitrary — it must match the number of CA tokens the tokenizer produced. If the tokenizer wasn't reconfigured after a topology change, `N_ca` would be stale. That's why Phase B (`reconfigure()`) rebuilds the tokenizer when the building gains/loses CAs — it ensures `N_ca` stays in sync with the action space.

**Basketball analogy:**
CA embeddings are the **starting lineup's player cards** pulled from the team stack. After the huddle (Transformer), the coach has contextualized cards for all 12 players on the roster (7 tokens). But only 5 players are on the court right now (CAs) — those are the ones who need play instructions (actions). The coach picks out the 5 starting player cards from the top of the stack (slice `[:5]`) and sets aside the bench players and context stats (SROs, RL). If a player gets injured and the lineup shrinks to 4, the coach just picks the top 4 cards (slice `[:4]`). The slicing adapts to the active roster size.

---

### Step 8b: Pooled Embedding — Global State for the Critic

**When:** Every timestep, immediately after the Transformer backbone processes the token sequence (parallel to Step 8a).

**What it does:**
The Transformer outputs `[batch, N_tokens, d_model]` for all tokens. To estimate the state value for the critic, we need a **single global representation** that summarizes the entire building state. This is computed by **mean-pooling** over all tokens: `pooled = transformer_out.mean(dim=1)`. The result is a `[batch, d_model]` vector that captures information from all CAs, SROs, and the RL token in an aggregate form.

**Code:** `algorithms/utils/transformer_backbone.py` — inside `TransformerBackbone.forward()`:
```python
# After all Transformer layers
x = self.layers[-1](x)  # [batch, N_tokens, d_model]

# Mean-pool over all tokens for critic
pooled = x.mean(dim=1)  # [batch, d_model]

return TransformerOutput(
    ca_embeddings=x[:, :N_ca, :],
    pooled=pooled,
)
```

And in `transformer_ppo_agent.py` — inside `predict()`:
```python
transformer_out = model.backbone(tokenized)
pooled = transformer_out.pooled  # [batch, d_model]

# Pass to critic head
value = model.critic(pooled)  # [batch, 1]
```

**Why it's needed:**
The critic estimates **V(s)** — the expected cumulative reward from the current state. This value depends on the **entire building state**: all CA states (battery SoC, EV connection status, etc.), all context (time, pricing, weather), and the RL signal (demand-generation residual). No single token captures all this information — each token is specialized (battery token focuses on battery features, pricing token focuses on price, etc.).

By mean-pooling over all tokens, we create a summary embedding that blends information from all sources. The pooled vector doesn't privilege any particular asset or context — it's a **democratic average** that gives the critic a holistic view of the building state.

**Real example — Building_1:**

Transformer output (from Step 7):
```
[batch=1, 7 tokens, 64 dims]
  token 0: battery          [0.22, -0.35, 0.91, ..., 0.18]
  token 1: EV charger       [0.61, -0.08, 0.15, ..., -0.22]
  token 2: washing machine  [-0.75, 0.72, -0.10, ..., 0.39]
  token 3: temporal         [0.12, 0.45, -0.33, ..., 0.67]
  token 4: pricing          [0.88, -0.21, 0.56, ..., -0.14]
  token 5: carbon           [-0.33, 0.19, 0.44, ..., 0.29]
  token 6: RL               [0.77, -0.52, 0.08, ..., 0.11]
```

Pooled embedding (mean over all 7 tokens):
```
pooled = (token0 + token1 + ... + token6) / 7
       = [0.22, -0.11, 0.27, ..., 0.18]  (shape: [batch=1, 64])
```

Each dimension of the pooled vector is the average of that dimension across all tokens. For example:
- `pooled[0] = (0.22 + 0.61 - 0.75 + 0.12 + 0.88 - 0.33 + 0.77) / 7 = 0.22`
- `pooled[1] = (-0.35 - 0.08 + 0.72 + 0.45 - 0.21 + 0.19 - 0.52) / 7 = -0.11`

The pooled vector is a **lossy summary** — individual token details are blurred — but it captures the overall "state of the building" in a fixed-size representation. This is exactly what the critic needs: a global view to estimate "how good is this state overall?"

**Connection to the flexy plan:**
Mean-pooling is **topology-agnostic** — it works for any `N_tokens`. If Building_2 has 5 tokens and Building_15 has 8 tokens, the pooling operation adapts automatically: `mean(dim=1)` computes the average regardless of the sequence length. This is another reason why cross-topology transfer works seamlessly (design goal #1) — the pooled representation has the same shape `[batch, d_model]` for all buildings, even though they have different numbers of CAs and different token counts. The critic head receives the same input shape regardless of topology, so critic weights transfer across buildings without modification.

**Basketball analogy:**
Pooled embedding is the **team summary stats card** handed to the GM in the booth. The GM doesn't need to know "the center has 12 rebounds, the point guard has 7 assists, the shooting guard has 18 points..." — that's too much detail for a high-level decision like "should we rest our starters or keep them in?" Instead, the analytics team computes an aggregate "team performance score" by averaging all player contributions (mean-pooling): "Overall team effectiveness: 0.22, defensive intensity: -0.11, offensive rhythm: 0.27, ..." The GM uses this summary to make strategic decisions (estimate state value). The summary is the same size (64 numbers) whether you have 5 players or 7 — it's just the average changes based on who's contributing.

---

### Step 9: Actor Head — Per-CA Action Distribution

**When:** Every timestep, during `predict()`, immediately after extracting CA embeddings from the Transformer output.

**What it does:**
The actor head is a small 2-layer MLP that processes each CA embedding independently and outputs a **Gaussian distribution** over actions. For each CA, it computes:
1. **Mean (μ)**: The central tendency of the action distribution (before squashing).
2. **Log-std (log σ)**: The exploration noise level, learned per CA type (e.g., batteries have one log-std parameter, EV chargers have another).
3. **Sample action**: Sample `u ~ N(μ, σ)`, then squash through `tanh` to get actions in `[-1, 1]`.
4. **Log-probability**: Compute `log π(a|s)` with tanh correction for PPO loss.
5. **Entropy**: Measure of exploration (used in PPO loss to encourage exploration).

**Code:** `algorithms/utils/ppo_components.py` — `ActorHead.forward()`:
```python
def forward(self, ca_embeddings, ca_type_indices=None, deterministic=False):
    """
    Parameters
    ----------
    ca_embeddings : Tensor[batch, N_ca, d_model]
    ca_type_indices : Tensor[N_ca] — index into self.log_std for each CA
    deterministic : bool — if True, return mean action (no sampling)
    
    Returns
    -------
    actions : Tensor[batch, N_ca, 1] in [-1, 1]
    log_probs : Tensor[batch, N_ca, 1]
    entropy : Tensor[batch, N_ca, 1]
    """
    x = self.norm(ca_embeddings)              # LayerNorm
    x = F.gelu(self.fc1(x))                   # [batch, N_ca, d_ff]
    mu = self.fc2(x)                          # [batch, N_ca, 1]
    
    # Per-CA-type std (learned parameter)
    log_std = self.log_std[ca_type_indices]   # [N_ca]
    std = log_std.exp()                       # [N_ca]
    
    dist = Normal(mu, std)
    u = mu if deterministic else dist.rsample()  # [batch, N_ca, 1]
    
    actions = torch.tanh(u)                   # Squash to [-1, 1]
    log_probs = dist.log_prob(u) - torch.log(1 - actions**2 + 1e-6)
    entropy = dist.entropy()
    
    return actions, log_probs, entropy
```

**Why it's needed:**
Each CA needs its own action — we can't produce a single scalar action for all assets. The actor head processes each CA embedding independently (via element-wise MLP application) to produce one action per CA. The per-CA-type log-std allows different asset types to have different exploration strategies: batteries might need less noise (more confident actions), while EV chargers might need more exploration (uncertain optimal charging patterns).

The squashing through `tanh` ensures actions are always in `[-1, 1]`, matching CityLearn's action space constraints. The log-prob calculation includes a tanh correction term (`-log(1 - a²)`) to account for the squashing transformation — this is critical for correct PPO gradient estimates.

**Real example — Building_1:**

CA embeddings (from Step 8a):
```
[batch=1, 3 CAs, 64 dims]
  CA 0: battery          [0.22, -0.35, 0.91, ..., 0.18]
  CA 1: EV charger       [0.61, -0.08, 0.15, ..., -0.22]
  CA 2: washing machine  [-0.75, 0.72, -0.10, ..., 0.39]
```

CA type indices (from global vocabulary):
```
ca_type_indices = [0, 1, 2]  # battery=0, ev_charger=1, washing_machine=2
```

Learned log-std parameters (per CA type):
```
self.log_std = Parameter([log(0.5), log(0.8), log(0.3)])  # [n_ca_types=3]
                        = [-0.69,     -0.22,     -1.20]
```

**Actor head forward pass:**

1. **MLP processes each CA embedding:**
   - Battery: `mu_battery = fc2(gelu(fc1(norm([0.22, -0.35, ...]))))` → `mu_battery = -0.35` (scalar)
   - EV: `mu_ev = fc2(gelu(fc1(norm([0.61, -0.08, ...]))))` → `mu_ev = 0.62`
   - WM: `mu_wm = fc2(gelu(fc1(norm([-0.75, 0.72, ...]))))` → `mu_wm = 0.12`

2. **Assign std based on CA type:**
   - Battery: `std_battery = exp(-0.69) = 0.50`
   - EV: `std_ev = exp(-0.22) = 0.80`
   - WM: `std_wm = exp(-1.20) = 0.30`

3. **Sample from Gaussian distributions:**
   - Battery: `u_battery ~ N(-0.35, 0.50)` → sample: `u = -0.52`
   - EV: `u_ev ~ N(0.62, 0.80)` → sample: `u = 1.10`
   - WM: `u_wm ~ N(0.12, 0.30)` → sample: `u = 0.23`

4. **Squash through tanh:**
   - Battery: `action_battery = tanh(-0.52) = -0.48`
   - EV: `action_ev = tanh(1.10) = 0.80`
   - WM: `action_wm = tanh(0.23) = 0.23`

5. **Compute log-probs (for PPO):**
   - Battery: `log_prob = log N(-0.52 | -0.35, 0.50) - log(1 - (-0.48)²) = -1.12 - 0.11 = -1.23`
   - EV: `log_prob = log N(1.10 | 0.62, 0.80) - log(1 - 0.80²) = -1.05 - 0.48 = -1.53`
   - WM: `log_prob = log N(0.23 | 0.12, 0.30) - log(1 - 0.23²) = -1.08 - 0.03 = -1.11`

6. **Compute entropy (for exploration bonus):**
   - Entropy = `0.5 * log(2πe * σ²)`
   - Battery: `0.5 * log(2πe * 0.5²) = 0.45`
   - EV: `0.5 * log(2πe * 0.8²) = 0.83`
   - WM: `0.5 * log(2πe * 0.3²) = 0.18`

**Final output:**
```
actions = [[-0.48], [0.80], [0.23]]    # [batch=1, N_ca=3, 1]
log_probs = [[-1.23], [-1.53], [-1.11]]
entropy = [[0.45], [0.83], [0.18]]
```

**Connection to the flexy plan:**
The actor head processes `N_ca` embeddings, where `N_ca` is determined at runtime. When the topology changes, `N_ca` changes, and the actor head automatically adapts — it doesn't have a fixed input size because it processes each CA independently via element-wise MLP application. However, the `log_std` parameter is sized to the **full global vocabulary** (3 CA types in the example), not per-building count. This is Phase A's pre-allocation — even if Building_2 only has 1 CA (battery), the `log_std` tensor still has 3 elements (battery, ev_charger, washing_machine), so when a checkpoint loads into a building with 2 CAs (battery + EV), the EV's log-std parameter is already present and transfers seamlessly.

Without pre-allocation, `log_std` would be sized to the training building's CA count, and loading into a building with more CAs would fail (missing parameters). Phase A ensures all CA type parameters are always present.

**Basketball analogy:**
The actor head is **the coach's play-calling system** for each player. After the huddle (Transformer) and pulling out the starting lineup cards (CA embeddings), the coach assigns a play to each player: "Center (battery), I want you to go for a rebound with 50% confidence — don't force it, but be ready (μ=-0.35, σ=0.50 → action=-0.48)." "Shooting guard (EV), be aggressive and drive to the basket (μ=0.62, σ=0.80 → action=0.80)." "Point guard (WM), play it safe and pass (μ=0.12, σ=0.30 → action=0.23)." Each player gets a personalized instruction (action) based on their contextualized card (embedding). The coach also tracks "how confident am I in this call?" (log-prob) and "how much freedom does this player have to improvise?" (entropy). Different player types (centers vs guards) get different confidence levels (per-type log-std) — centers are more predictable (low std), guards have more creative freedom (high std).

---

### Step 10: Critic Head — State Value Estimation

**When:** Every timestep, during `predict()`, immediately after computing the pooled embedding from the Transformer output (parallel to Step 9).

**What it does:**
The critic head is a small 2-layer MLP that processes the pooled embedding and outputs a **scalar state value** `V(s)` — the expected cumulative discounted reward from the current state. This value is used for:
1. **GAE (Generalized Advantage Estimation)**: Computing advantages during PPO updates.
2. **Value loss**: Training the critic to predict accurate returns.
3. **Baseline subtraction**: Reducing variance in policy gradient estimates.

**Code:** `algorithms/utils/ppo_components.py` — `CriticHead.forward()`:
```python
def forward(self, pooled):
    """
    Parameters
    ----------
    pooled : Tensor[batch, d_model]
    
    Returns
    -------
    Tensor[batch, 1] — state value V(s)
    """
    x = self.norm(pooled)      # LayerNorm
    x = F.gelu(self.fc1(x))    # [batch, d_ff]
    return self.fc2(x)         # [batch, 1]
```

And in `transformer_ppo_agent.py` — inside `predict()`:
```python
transformer_out = model.backbone(tokenized)
pooled = transformer_out.pooled  # [batch, d_model]

value = model.critic(pooled)  # [batch, 1]
```

**Why it's needed:**
PPO is an actor-critic algorithm — it needs both a policy (actor) and a value function (critic). The critic estimates "how good is this state?" without knowing which actions will be taken. This value is used to compute advantages: `A(s,a) = Q(s,a) - V(s)`, which tells the actor "this action was better/worse than expected." The critic also provides a baseline that reduces variance in policy gradient estimates, making training more stable.

The critic receives the **pooled embedding** (not individual CA embeddings) because it needs a global view of the building state. The value of a state depends on all factors: battery SoC, EV status, time of day, pricing, demand, etc. The pooled embedding captures all this information in a single vector.

**Real example — Building_1:**

Pooled embedding (from Step 8b):
```
pooled = [0.22, -0.11, 0.27, ..., 0.18]  (shape: [batch=1, 64])
```

**Critic head forward pass:**

1. **Layer normalization:**
   - `normalized = LayerNorm(pooled)` → `[0.25, -0.09, 0.30, ..., 0.20]`

2. **First linear + GELU:**
   - `hidden = gelu(fc1(normalized))` → `[batch=1, 128]` (assuming `d_ff=128`)
   - Example hidden activations: `[0.78, -0.12, 0.45, ..., 0.91]`

3. **Second linear (output):**
   - `value = fc2(hidden)` → `[batch=1, 1]`
   - Example value: `value = [3.52]` (scalar, unbounded)

**Interpretation:**
The critic estimates that from the current state (Building_1 at timestep 100 with battery at 65% SoC, EV connected at 42%, pricing at $0.08/kWh, demand at 1.85 kWh, etc.), the expected cumulative discounted reward is **3.52**. This is an abstract number — it's only meaningful relative to other states. A higher value means "this is a better state to be in," a lower value means "this state will likely lead to worse rewards."

During training, if the agent takes actions and receives actual rewards `[r_100, r_101, ..., r_8760]`, the returns `G_100 = Σ γ^k * r_{100+k}` are computed. The critic is trained to predict `V(s_100) ≈ G_100`. Over time, the critic learns to accurately estimate "what cumulative reward will I get from this state?"

**Connection to the flexy plan:**
The critic receives the **pooled embedding**, which has a fixed shape `[batch, d_model]` regardless of building topology. This means the critic head has the same input/output dimensions for all buildings, so critic weights transfer seamlessly across topologies (design goal #1). When a checkpoint trained on Building_4 loads into Building_2, the critic weights transfer directly — both buildings produce `[batch, 64]` pooled embeddings, even though they have different numbers of CAs and different token sequences.

If the critic had to process individual CA embeddings (like the actor), it would need topology-specific logic ("sum over N_ca embeddings"). By using the pooled embedding, the critic is completely topology-agnostic.

**Basketball analogy:**
The critic is the **team's win probability estimator** sitting in the analytics booth. Based on the team summary stats card (pooled embedding — "team effectiveness: 0.22, defensive intensity: -0.11, offensive rhythm: 0.27, ..."), the estimator predicts: "Given the current game state (score, time remaining, player energy levels, opponent strength), our probability of winning is 3.52" (actually a proxy for expected final score differential, but conceptually similar). The estimator doesn't know what plays the coach will call (actions) — it just evaluates "how good is our position right now?" If the team is ahead with fresh players and good momentum (high SoC, low demand, low pricing), the estimate is high. If behind with tired players and bad matchups (low SoC, high demand, high pricing), the estimate is low. During the game (training), the estimator learns to predict final outcomes by comparing its pre-play estimates to actual results.

---

### Step 11: Action Mapping — Reordering to Match CityLearn's Expected Order

**When:** Every timestep, at the end of `predict()`, after the actor head produces actions.

**What it does:**
The actor head produces actions in **CA token order** (the order tokens appear in the sequence: battery, EV, washing machine), but CityLearn expects actions in **action name order** (the order defined in the schema's `actions` list). This step reorders the actions using the `action_ca_map` computed during tokenizer setup. The map is a list where `action_ca_map[action_idx] = ca_token_idx` — it tells us "the action at position `action_idx` corresponds to the CA at token position `ca_token_idx`."

**Code:** `transformer_ppo_agent.py` — inside `predict()`:
```python
# Actor produces actions in CA-token order
actions, log_probs, entropy = model.actor(
    transformer_out.ca_embeddings,
    ca_type_indices=ca_type_idx,
    deterministic=is_deterministic,
)  # actions: [batch=1, N_ca, 1]

actions_flat = actions.squeeze(0).squeeze(-1)  # [N_ca]

# Reorder from CA-token order to action-name order
action_ca_map = self._action_ca_maps[i]  # [action_idx] -> ca_token_idx
action_list = [0.0] * len(action_ca_map)
for act_idx, ca_idx in enumerate(action_ca_map):
    action_list[act_idx] = actions_flat[ca_idx].item()
```

**Why it's needed:**
The tokenizer organizes CA features by **device type** (all batteries first, then all EV chargers, then all washing machines) to enable weight sharing and cross-topology transfer. But the schema defines actions in **arbitrary order** — it might list `["washing_machine_1", "electrical_storage", "electric_vehicle_storage_charger_1_1"]` (WM, battery, EV). If we returned actions in CA-token order, the environment would interpret `action[0]` as the washing machine's action when it actually expects the battery's action — **the actions would be applied to the wrong devices**.

The `action_ca_map` is built during tokenizer setup by matching action names to CA instances and recording the correspondence. This ensures actions are always delivered in the order CityLearn expects.

**Real example — Building_1:**

**Action names (from schema):**
```
action_names = [
    "electrical_storage",                      # action index 0
    "electric_vehicle_storage_charger_1_1",    # action index 1
    "washing_machine_1"                        # action index 2
]
```

**CA token order (from tokenizer):**
```
CA token 0: battery (device_id=None, type="battery")
CA token 1: ev_charger (device_id="charger_1_1", type="ev_charger")
CA token 2: washing_machine (device_id="1", type="washing_machine")
```

**Action-CA map (computed during tokenizer setup):**
```
action_ca_map = [0, 1, 2]
# action_ca_map[0] = 0  (action "electrical_storage" → CA token 0)
# action_ca_map[1] = 1  (action "electric_vehicle_storage_charger_1_1" → CA token 1)
# action_ca_map[2] = 2  (action "washing_machine_1" → CA token 2)
```

In this case, the order happens to match (battery→battery, EV→EV, WM→WM), so no reordering is needed. But consider Building_15 with 2 EV chargers:

**Action names (Building_15):**
```
action_names = [
    "electric_vehicle_storage_charger_2_1",    # action index 0
    "electrical_storage",                      # action index 1
    "electric_vehicle_storage_charger_1_1",    # action index 2
]
```

**CA token order (Building_15):**
```
CA token 0: battery (device_id=None, type="battery")
CA token 1: ev_charger (device_id="charger_1_1", type="ev_charger")
CA token 2: ev_charger (device_id="charger_2_1", type="ev_charger")
```

**Action-CA map (Building_15):**
```
action_ca_map = [2, 0, 1]
# action_ca_map[0] = 2  (action "...charger_2_1" → CA token 2)
# action_ca_map[1] = 0  (action "electrical_storage" → CA token 0)
# action_ca_map[2] = 1  (action "...charger_1_1" → CA token 1)
```

**Actor produces (CA-token order):**
```
actions_flat = [-0.48, 0.80, -0.25]  # [battery, EV_charger_1_1, EV_charger_2_1]
```

**Reordering (action-name order):**
```
action_list = [0.0, 0.0, 0.0]
action_list[0] = actions_flat[2] = -0.25  # "...charger_2_1"
action_list[1] = actions_flat[0] = -0.48  # "electrical_storage"
action_list[2] = actions_flat[1] = 0.80   # "...charger_1_1"

final action_list = [-0.25, -0.48, 0.80]
```

Now the environment receives actions in the order it expects: charger_2_1 first, battery second, charger_1_1 third.

**Connection to the flexy plan:**
The `action_ca_map` is **topology-specific** and stored as a regular Python list (not a PyTorch buffer), so it's not part of the model checkpoint. When `reconfigure()` (Phase B) rebuilds the tokenizer for a new topology, it also recomputes the `action_ca_map` to match the new action names. This ensures the mapping stays correct even if the building gains/loses CAs or if action names are reordered.

If the map weren't updated during reconfiguration, actions would be delivered to the wrong devices after a topology change — a critical bug. That's why Phase B explicitly updates `_action_ca_maps` (blocker #9 in the flexy plan).

**Basketball analogy:**
Action mapping is the **scorekeeper's playbook translation**. The coach calls plays using jersey numbers: "Tell #23 to shoot, #15 to rebound, #7 to defend." But the official scorecard lists players by name in alphabetical order: "Johnson, Miller, Smith." The scorekeeper has a translation sheet (action_ca_map): "#23 = Smith (position 2 on scorecard), #15 = Johnson (position 0), #7 = Miller (position 1)." When the coach calls "#23 shoot," the scorekeeper writes "Smith: shot attempt" at position 2 on the scorecard. Without the translation sheet, the scorekeeper would write the coach's instructions in jersey number order (#7, #15, #23), but the league office expects alphabetical order (Johnson, Miller, Smith) — the stats would be attributed to the wrong players. The translation ensures everyone interprets actions correctly.

---

## Summary — Complete Dataflow

| Step | Operation | Input | Output | Topology-Dependent? |
|------|-----------|-------|--------|---------------------|
| 0 | Setup | Observation/action names | Tokenizer, models | Yes — built per building |
| 1 | Raw Observations | Simulation | `[N_obs]` floats | Yes — N_obs varies |
| 2 | Encoding | Raw obs | `[encoded_dim]` floats | Yes — dim varies by features |
| 3 | Encoded Vector | Encoded obs | Flat array | Yes — length varies |
| 4 | Index Map | Obs names + encoder config | Feature→slice mapping | Yes — computed from obs names |
| 5 | Tokenizer Classification | Index map + action names | CA/SRO/RL groups | Yes — depends on action count |
| 6 | Token Projection | Encoded vector | `[N_tokens, d_model]` | Yes — N_tokens varies |
| 7 | Transformer Backbone | Token sequence | `[N_tokens, d_model]` (contextualized) | **No** — topology-agnostic |
| 8a | CA Embeddings Slice | Transformer output | `[N_ca, d_model]` | Yes — slice size = N_ca |
| 8b | Pooled Embedding | Transformer output | `[d_model]` | **No** — always same shape |
| 9 | Actor Head | CA embeddings | `[N_ca, 1]` actions | Yes — N_ca actions produced |
| 10 | Critic Head | Pooled embedding | `[1]` value | **No** — always scalar |
| 11 | Action Mapping | Actor actions | Reordered list | Yes — depends on action_ca_map |

**Key insight:** Steps 7, 8b, and 10 (Transformer backbone, pooling, critic) are **completely topology-agnostic** — they work the same regardless of building topology. This is why checkpoint transfer works: these components have no hard-coded assumptions about CA count or types. Steps 0-6, 8a, 9, and 11 are topology-dependent but **reconfigurable** (Phase B rebuilds them) or **pre-allocated** (Phase A ensures all type-specific parameters exist upfront). Together, these design choices enable the three architectural goals:

1. **Variable CA count** (goal #1) — Transformer processes variable-length sequences; actor produces variable-length action vectors.
2. **Strict CA input/output mapping** (goal #2) — Each CA token → exactly one action via actor head's per-embedding MLP.
3. **Additional context without spurious outputs** (goal #3) — SRO tokens and RL token are inputs only; actor only processes CA embeddings.
