# Runtime Asset Adaptation: Solution Proposals

## Problem Statement

The system must handle changes in the set of controllable assets (CAs) at runtime -- an EV charger connects or disconnects, a battery is added, a washing machine comes online. Today, the observation layout and encoder list are fixed at `attach_environment()` time. If the number or type of assets changes mid-run, the pipeline crashes:

```
Raw sensors  -->  Wrapper (encoding)  -->  Agent (tokenization)  -->  Actions
     ^                 ^                        ^
     |                 |                        |
  dynamic         fixed at init           fixed at init
```

The root cause: the wrapper builds encoders for a fixed list of observation names, and the tokenizer builds index buffers for a fixed encoded vector layout. A new asset introduces new observation features and new actions, breaking both.

This document proposes four solutions grouped into two families:

- **Family 1 (pre-encoder):** Act before or during encoding to make the observation vector flexible.
- **Family 2 (post-encoder):** Keep the encoder fixed and handle variability downstream.

---

## Solution A: Dynamic Encoder Rebuild (Pre-Encoder)

**Idea:** When assets change, rebuild the wrapper's encoder list and the tokenizer's index map to match the new observation space. The agent receives a correctly-sized encoded vector at every step.

**How it works:**

1. CityLearn (or a middleware layer) signals that the observation space changed.
2. The wrapper calls `set_encoders()` again with the new `observation_names` and `observation_space`.
3. The wrapper calls `agent.reconfigure_building(i, new_obs_names, new_action_names, ...)` which rebuilds the tokenizer's index buffers, CA/SRO/RL classification, and action maps.
4. Learned projection weights are preserved (pre-allocated for all types in config). Only non-learned metadata is rebuilt.
5. Rollout buffer is flushed (mixed-shape transitions would crash PPO).

**This is the approach already designed in `flexy_plan.md` (Phases A-C).**

| Pros | Cons |
|------|------|
| Encoder output is always correct for the current topology | Requires a signal/hook when the observation space changes |
| Agent receives a clean, properly-encoded vector | Encoder rebuild is synchronous; blocks the training step |
| Existing encoder rules (`default.json`) work unchanged | Rollout buffer must be flushed, losing partial trajectory data |
| Projection weight pre-allocation is already implemented (Phase A done) | Wrapper and agent must coordinate the reconfigure call |
| Tokenizer `reconfigure()` is cheap (non-learned data only) | If CityLearn changes observation order silently, index buffers become stale without detection |

**Effort:** Medium. The pre-allocation is done. What remains is the `reconfigure()` path in the tokenizer and wrapper-side encoder rebuild hook.

---

## Solution B: Fixed Superset Encoding (Pre-Encoder)

**Idea:** Pre-define a superset observation vector that includes slots for every possible asset type and instance, based on the maximum topology from the config. Features for assets that are not currently present are filled with sentinel values (e.g., zero or a configured missing indicator). The encoder list and vector size never change.

**How it works:**

1. At `attach_environment()`, scan the config to determine the maximum possible topology: max number of EV chargers, batteries, washing machines, etc.
2. Build an encoder list and observation name list for this superset.
3. At every step, the wrapper (or a pre-processing layer) maps the actual CityLearn observations to the superset positions. Missing assets get their sentinel values.
4. The tokenizer always operates on the same fixed-size vector. CA instances that are "empty" produce tokens from sentinel-encoded features.
5. The actor head produces actions for all slots; downstream masking zeroes out actions for absent assets.

| Pros | Cons |
|------|------|
| Observation vector size never changes; no rebuild needed | Wastes encoded dimensions on absent assets (padding overhead) |
| No rollout buffer flush; transitions always have the same shape | Requires a mapping layer that knows which CityLearn features map to which superset slot |
| No coordination signal needed; the superset handles all topologies | Sentinel values propagate through the network; the model must learn to ignore them |
| Simple to reason about; no runtime reconfiguration path | Maximum topology must be known at init time; cannot exceed it without restart |
| Works identically at training and inference time | Action masking adds a post-processing step that must be correct for safety |

**Effort:** Medium-high. Requires building the superset mapping layer and modifying how observations are fed to the encoder. The tokenizer and agent core are unchanged.

---

## Solution C: Per-Token Independent Encoding (Post-Encoder)

**Idea:** Move encoding inside the tokenizer itself. Instead of receiving a flat pre-encoded vector, the tokenizer receives raw observation values grouped by token type, and each token's projection layer includes its own normalization.

**How it works:**

1. The wrapper passes raw observations (or a dict keyed by feature name) directly to the agent. The agent sets `use_raw_observations = True`.
2. The tokenizer receives a dict `{feature_name: raw_value}`.
3. For each CA instance, the tokenizer looks up its features by name, applies per-feature encoding (periodic, onehot, normalize, etc.) inline, and passes the result to the CA projection layer.
4. SRO and RL tokens do the same.
5. When assets change, the tokenizer's `reconfigure()` only updates which feature names to look up. The encoding logic is embedded in the tokenizer, not in a separate wrapper encoder list.
6. The flat encoded vector is never constructed; each token is independently encoded and projected.

| Pros | Cons |
|------|------|
| No dependency on the wrapper's encoder list | Duplicates encoding logic that already exists in the wrapper/preprocessor |
| Adding or removing assets only requires updating the feature name lookup | Breaks the current architecture where encoding is a separate, shared responsibility |
| No padding or sentinel values; only real data is processed | Raw observations must include feature names (dict, not flat array); changes the predict/update signatures |
| Each token is self-contained; natural fit for the Transformer token paradigm | Onehot and periodic encoders are stateful (need class lists, x_max); tokenizer must carry encoder params per feature |
| Inference service already has this pattern (preprocessor applies encoders per feature name) | Cannot reuse the wrapper's `get_encoded_observations()` for other agents (MADDPG, RBC) |

**Effort:** High. Requires restructuring how observations flow from the environment to the agent. The wrapper's encoding role changes or is bypassed for this agent type.

---

## Solution D: Attention Mask with Fixed Architecture (Post-Encoder)

**Idea:** Keep the encoder and tokenizer exactly as they are. Handle asset presence/absence purely through attention masking and action gating. Absent assets are not removed from the sequence; they are masked so the Transformer ignores them.

**How it works:**

1. The observation vector always has the same size (same as init). When an asset goes offline, its features become zero (CityLearn's default behavior for disconnected EVs).
2. The tokenizer produces CA tokens for all initially-configured assets, including absent ones.
3. An attention mask is added to the Transformer backbone: tokens for absent assets are masked out of self-attention (like padding tokens in NLP). Other tokens do not attend to masked positions.
4. The actor head still produces actions for all CA positions, but an output mask zeroes out actions for absent assets.
5. For the critic, the mean pool excludes masked positions.
6. Presence/absence is determined by checking a flag feature (e.g., `connected_state`) at each step.

| Pros | Cons |
|------|------|
| Zero changes to the encoder or tokenizer | Cannot handle genuinely new assets that were not in the initial topology |
| No rebuild, no reconfigure, no flush | Masked tokens still consume memory and compute in the sequence |
| Standard Transformer pattern; well-understood in NLP (padding masks) | The model must learn that zero-feature tokens are "absent" -- masking helps but is not a training-time signal |
| Clean separation: attention mask for the model, action mask for the environment | Requires a reliable presence indicator feature (e.g., `connected_state`) for every CA type |
| Works at inference time without any changes to the serving pipeline | Does not handle changes in the number of SRO or RL features |

**Effort:** Low-medium. Requires adding attention mask support to the backbone (a few lines) and action gating logic. No structural changes to the pipeline.

---

## Comparison Matrix

| Criterion | A: Dynamic Rebuild | B: Fixed Superset | C: Per-Token Encoding | D: Attention Mask |
|-----------|--------------------|-------------------|----------------------|-------------------|
| Handles new asset types at runtime | Yes (if in config vocab) | Yes (if in superset) | Yes (if in config vocab) | No |
| Handles asset removal at runtime | Yes | Yes (via sentinels) | Yes | Yes (via masking) |
| Encoder changes required | Rebuild | None (pre-built) | Replace with inline | None |
| Tokenizer changes required | `reconfigure()` method | None | Major restructure | Mask support |
| Rollout buffer impact | Flush on change | No flush needed | Flush on change | No flush needed |
| Padding/waste overhead | None | High (all absent slots encoded) | None | Moderate (masked tokens in sequence) |
| Implementation effort | Medium | Medium-high | High | Low-medium |
| Inference service compatibility | Compatible (manifest reflects current topology) | Compatible (fixed manifest) | Requires serving-side changes | Compatible (actions masked post-model) |
| Risk of stale metadata | Medium (must detect space changes) | Low (superset is static) | Low (dict-based lookup) | Low (mask derived per step) |

---

## Recommendation

For the thesis scope, **Solution A (Dynamic Rebuild)** is the most natural fit because Phase A (pre-allocation) is already implemented and the `reconfigure()` path is a well-scoped extension. It is also the cleanest architecture: the encoder produces correct data, the tokenizer processes correct data, and no masking or sentinel logic is needed.

**Solution D (Attention Mask)** is worth implementing as a complementary mechanism for the common case where assets temporarily go offline but the observation space does not actually change (e.g., EV disconnects but CityLearn still reports its features as zero). It is cheap to add and provides graceful degradation without any rebuild.

Solutions B and C are alternatives for production systems with different constraints. Solution B suits scenarios where the maximum topology is well-known and stable. Solution C is architecturally cleaner in the long term but requires significant restructuring that goes beyond the thesis scope.
