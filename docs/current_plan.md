# Current Plan — Validation & Flexy Implementation

## Context for a New Agent

### Why Transformers?

The overall goal of introducing Transformers into this RL system is to address three
fundamental design considerations that traditional MLP-based agents (like MADDPG) cannot
handle:

1. **Adapt to changes in the number of CAs at runtime without retraining**, supporting
   variable numbers of CA inputs and outputs. An MLP has a fixed input/output dimension —
   if a building gains or loses a controllable asset, the entire network must be rebuilt
   and retrained. A Transformer processes a variable-length token sequence via
   self-attention, naturally handling different numbers of assets.

2. **Ensure a strict one-to-one mapping between each CA input and its corresponding
   control output.** Each controllable asset's observation features are tokenized into a
   dedicated token. The actor head produces one action per CA token. There is no mixing of
   battery actions into EV charger outputs — the architecture enforces this structurally.

3. **Incorporate additional inputs — a single global NFC input (the RL token) and a
   variable-sized set of SROs (shared read-only context) — while keeping all outputs
   aligned with their respective CAs.** The Transformer backbone processes CA, SRO, and RL
   tokens together via self-attention, so every CA token can attend to weather, pricing,
   and demand signals. But only CA token outputs produce actions — SRO and RL tokens
   inform decisions without generating spurious outputs.

The thesis goal: **a single model class handles different prosumer configurations (varying
numbers and types of controllable assets per building) without retraining from scratch.**
Weights for shared asset types (e.g., battery projections) transfer across topologies, and
the self-attention mechanism naturally adapts to different token counts.

### What exists today

All 6 implementation phases are complete (178 tests pass):

| Layer | File | What it does |
|-------|------|-------------|
| Encoder Index Map | `algorithms/utils/encoder_index_map.py` | Maps raw feature names -> post-encoding slice indices |
| Observation Tokenizer | `algorithms/utils/observation_tokenizer.py` | Groups flat encoded vector into typed tokens (CA/SRO/RL) |
| Transformer Backbone | `algorithms/utils/transformer_backbone.py` | Self-attention over tokens; produces contextual embeddings |
| PPO Components | `algorithms/utils/ppo_components.py` | Actor, Critic, Rollout Buffer, PPO loss |
| Agent | `algorithms/agents/transformer_ppo_agent.py` | Full BaseAgent implementation (predict/update/export) |
| Config | `configs/templates/transformer_ppo_local.yaml` | Template with all hyperparameters |

### What hasn't been done

1. **No meaningful training run** — only a 10-step smoke test has been executed.
2. **No cross-topology loading** — checkpoint from one building topology can't load into a
   different topology (flexy plan Phase A + C needed).
3. **Dataflow walkthrough incomplete** — Steps 0-3 written, Steps 4-11 pending.
4. **No validation results** — the architecture works mechanically but hasn't been proven
   to learn.

### Key references

| Doc | Purpose |
|-----|---------|
| `docs/plan.md` | Original 6-phase development plan (all phases complete) |
| `docs/base.md` | Architecture design spec |
| `docs/flexy_plan.md` | Runtime topology adaptability plan (22 blockers identified) |
| `docs/dataflow_walkthrough.md` | Step-by-step data pipeline explanation (Steps 0-3 done) |
| `AGENTS.md` | Agent contract + validation phase table |

### Building topology in the dataset

| Topology | CAs | Buildings |
|----------|-----|-----------|
| Battery only | 1 | 2, 3, 6, 8, 9, 11, 13, 14, 16, 17 |
| Battery + 1 EV | 2 | 4, 5, 7, 10, 12 |
| Battery + 1 EV + 1 Washing Machine | 3 | 1 |
| Battery + 2 EVs | 3 | 15 |

---

## Execution Order

```
Step 1: Write dataflow Steps 4-6 (tokenizer internals)
Step 2: Implement flexy plan Phase A (pre-allocation)
Step 3: Implement flexy plan Phase C (checkpoint compatibility)
Step 4: Create test data files (schema variants for single-building runs)
Step 5: Run Validation Phase 1 (Building_4, 1 episode, 8760 steps)
Step 6: Write dataflow Steps 7-11 (backbone -> actions)
Step 7: Run Validation Phase 2 (cross-topology checkpoint loading)
Step 8: Validation Phase 3 (KPI analysis, tuning if needed)
Step 9: Validation Phase 4 (multi-building)
```

---

## Step 1: Dataflow Walkthrough — Steps 4-6

**Goal:** Document the tokenizer pipeline so you can explain it to others.

Write Steps 4-6 in `docs/dataflow_walkthrough.md`:

- **Step 4: Encoder Index Map** — How `build_encoder_index_map()` maps each raw feature
  name to its post-encoding slice. Real example: `"month" -> (0, 2, 2)` because
  PeriodicNormalization produces 2 dims. Why: the tokenizer needs to know where each
  feature lives in the flat encoded vector.

- **Step 5: Observation Tokenizer — Feature Classification** — How features are sorted
  into CA tokens (per device instance), SRO tokens (shared read-only context), and the RL
  token (demand-generation residual). Real example: `electrical_storage_soc` matches
  battery CA type -> goes to battery token; `month` matches temporal SRO type -> goes to
  temporal token; `non_shiftable_load` matches demand -> goes to RL token.

- **Step 6: Token Projection** — How each group's features are gathered from the encoded
  vector using registered index buffers, then projected to `d_model` dims via per-type
  `nn.Linear` layers. Real example: battery token gathers `[0.65]` (1 dim) ->
  `nn.Linear(1, 64)` -> `[64]` embedding. EV charger token gathers 14 encoded dims ->
  `nn.Linear(14, 64)` -> `[64]`.

Each step needs: what, why, real data example, basketball analogy, connection to flexy plan.

---

## Step 2: Flexy Plan Phase A — Pre-allocation

**Goal:** All projection layers and `log_std` are sized for the full config vocabulary,
regardless of which CAs/SROs are active in a given building.

### A1. Global vocabulary computation

**File:** `transformer_ppo_agent.py` — `attach_environment()`

Add a two-pass approach:

1. **Pass 1:** Before building per-building models, loop over all buildings' observation
   and action names. For each building, temporarily run encoder index map + instance
   detection to determine which CA types exist and their encoded feature dims. Collect:
   ```python
   global_ca_type_dims: Dict[str, int]   # {"battery": 1, "ev_charger": 14, "washing_machine": 4}
   global_sro_type_dims: Dict[str, int]  # {"temporal": 14, "weather": 0, "pricing": 1, "carbon": 1}
   global_type_to_idx: Dict[str, int]    # {"battery": 0, "ev_charger": 1, "washing_machine": 2}
   max_rl_input_dim: int                 # max across all buildings
   ```

2. **Pass 2 (existing):** Build per-building models, now passing global dims to tokenizer
   and actor head.

**Fallback for unseen types:** If a CA type is in config but has zero instances across all
buildings, compute its expected encoded dim from config feature patterns + encoder rules.

### A2. Tokenizer pre-allocates all projections

**File:** `observation_tokenizer.py`

- Add `global_ca_type_dims` parameter to `__init__`.
- `ca_projections` ModuleDict populated for **every** type in `global_ca_type_dims`, not
  just active types.
- Same for `sro_projections` with `global_sro_type_dims`.
- `rl_projection` sized to `max_rl_input_dim`, zero-padded in `forward()` if actual dim
  is smaller.

### A3. ActorHead uses full vocabulary

**File:** `ppo_components.py`

- `n_ca_types` = total config types (3), not per-building active count.
- `log_std = nn.Parameter(torch.zeros(3))` always.
- `ca_type_indices` still maps active CAs -> global type index (gather, not slice).

### A4. Update existing tests

Some tests assert `len(tokenizer.ca_projections) == N_active_types`. Update to assert
`== N_total_config_types`. Identify all affected assertions. Run full test suite.

### Verification

```bash
pytest -v  # All 178 existing tests must pass (with updated assertions)
```

Plus new tests in `tests/test_runtime_adaptability.py`:
- `test_tokenizer_preallocates_all_ca_types`
- `test_tokenizer_preallocates_all_sro_types`
- `test_actor_global_log_std_shape`
- `test_global_type_to_idx_consistent`
- `test_rl_projection_max_padded`

---

## Step 3: Flexy Plan Phase C — Checkpoint Compatibility

**Goal:** A checkpoint saved with one topology can load into a model with a different
topology.

### C1. Buffer filtering

**File:** `transformer_ppo_agent.py` — `load_checkpoint()`

Add helper to filter transient index buffers from checkpoint state dict:
```python
TRANSIENT_PATTERN = re.compile(
    r"_ca_idx_\d+|_sro_idx_\d+|_rl_demand_idx|_rl_gen_idx|_rl_extra_idx"
)
```

### C2. Loading with strict=False

After filtering, call `model.load_state_dict(filtered_dict, strict=False)` so that:
- Extra keys in checkpoint (e.g., `_ca_idx_2` for a 3-CA building loading into 1-CA) are
  silently ignored.
- Missing keys in checkpoint (e.g., building has new CAs) get default initialization.
- All learned parameters (projections, backbone, actor, critic) transfer because they're
  pre-allocated to full vocabulary (Phase A).

### Verification

New tests in `tests/test_runtime_adaptability.py`:
- `test_checkpoint_cross_topology` — Save 3-CA model, load into 1-CA model, verify
  predict works and weights match.
- `test_checkpoint_sro_cross_topology` — Same for SRO variation.
- `test_checkpoint_filter_transient_buffers` — Verify only index buffers are filtered.

```bash
pytest tests/test_runtime_adaptability.py -v
pytest -v  # Full suite still passes
```

---

## Step 4: Test Data Files

**Goal:** Create schema variants for single-building and cross-topology validation.
**Constraint:** Do NOT modify existing data files. Create new files alongside them.

### Schema files to create

All in `datasets/citylearn_challenge_2022_phase_all_plus_evs/`:

| File | Buildings included | Purpose |
|------|-------------------|---------|
| `schema_building4_only.json` | Building_4 (battery + 1 EV) | Phase 1: single-building training |
| `schema_building2_only.json` | Building_2 (battery only) | Phase 2: 1-CA evaluation target |
| `schema_building15_only.json` | Building_15 (battery + 2 EVs) | Phase 2: 3-CA evaluation target |
| `schema_building1_only.json` | Building_1 (battery + EV + WM) | Phase 2: 3-CA with washing machine |

Each file is a copy of `schema.json` with all buildings set to `"include": false` except
the target building.

### Config files to create

In `configs/`:

| File | Schema | Episodes | Purpose |
|------|--------|----------|---------|
| `validation_phase1.yaml` | `schema_building4_only.json` | 1 | Phase 1 run config |
| `validation_phase2_train.yaml` | `schema_building4_only.json` | 1 | Phase 2 training |
| `validation_phase2_eval.yaml` | `schema_building2_only.json` | 1 | Phase 2 cross-topology eval |

Each extends the transformer_ppo_local template with the appropriate schema path.

---

## Step 5: Validation Phase 1 — Does It Learn?

**Goal:** Run TransformerPPO on Building_4 for 1 episode (8760 steps). Determine if the
agent shows any learning signal.

### Setup

- Building: Building_4 (battery + 1 EV charger, 2 CAs)
- Episodes: 1 (8760 hourly timesteps)
- Config: `configs/validation_phase1.yaml`
- KPI export: enabled
- No baselines — just TransformerPPO

### Execution

```bash
python run_experiment.py --config configs/validation_phase1.yaml
```

### Success criteria

1. **No crashes** — completes all 8760 steps.
2. **Reward signal** — rewards are not constant (agent is taking different actions).
3. **Actions vary** — not stuck at 0 or +/-1 the entire episode.
4. **KPIs generated** — `result.json` and `summary.json` exist in output.

### Analysis

After the run, examine:
- Reward trajectory over the episode (does it trend anywhere?)
- Action distribution (histogram of battery and EV charger actions)
- KPI values (electricity cost, carbon emissions, comfort metrics)
- PPO loss components (policy loss, value loss, entropy) over update steps

**Note:** With only 1 episode, we don't expect convergence. We're looking for mechanical
correctness and basic responsiveness. If the agent outputs constant actions or NaN values,
something is wrong. If actions vary and rewards fluctuate reasonably, Phase 1 passes.

---

## Step 6: Dataflow Walkthrough — Steps 7-11

**Goal:** Document the backbone -> action pipeline.

Write Steps 7-11 in `docs/dataflow_walkthrough.md`:

- **Step 7: Transformer Backbone** — Self-attention over the token sequence. How CA tokens
  attend to SRO tokens (and vice versa) to produce context-enriched embeddings.

- **Step 8a: CA Embeddings** — Slicing the first N_ca outputs from the backbone as
  per-asset contextual representations.

- **Step 8b: Pooled Embedding** — Mean-pooling all token outputs for the critic's global
  state representation.

- **Step 9: Actor Head** — Per-CA-token MLP producing mean + learned log_std -> Gaussian
  sampling -> tanh squashing -> actions.

- **Step 10: Critic Head** — MLP on pooled embedding -> scalar state value V(s).

- **Step 11: Action Mapping** — Reordering per-CA-token actions back to CityLearn's
  expected action order via `action_ca_map`.

Each step: what, why, real data example, basketball analogy.

---

## Step 7: Validation Phase 2 — Cross-Topology Transfer

**Goal:** Prove the core thesis — same model weights work across different prosumer
configurations without retraining.

### Setup

1. **Train** on Building_4 (2 CAs: battery + 1 EV) for 1 episode.
2. **Save checkpoint.**
3. **Load checkpoint** into Building_2 (1 CA: battery only).
4. **Run inference** — does the loaded model produce valid battery actions?
5. **Load checkpoint** into Building_15 (3 CAs: battery + 2 EVs).
6. **Run inference** — does the loaded model produce valid actions for all 3 CAs?

### Success criteria

1. **Checkpoint loads without crash** — `strict=False` + buffer filtering works.
2. **Battery projection weights are identical** — the battery's `nn.Linear` weights in
   the loaded model match the trained checkpoint exactly.
3. **Actions are valid** — within [-1, 1] range, not NaN.
4. **Shared projections transfer** — SRO projections (temporal, pricing, carbon) carry
   over since they're the same across buildings.

### What this does NOT prove (yet)

- That the transferred model performs *well* on the new topology (that's Phase 3).
- That fine-tuning on the new topology converges faster than training from scratch.

---

## Step 8: Validation Phase 3 — KPI Analysis & Tuning

**Goal:** Assess whether transferred weights produce reasonable energy management behavior.

### Setup

1. Take the Phase 2 checkpoint (trained on Building_4).
2. Fine-tune on Building_2 for 1 episode.
3. Compare KPIs: transferred+fine-tuned vs. trained-from-scratch on Building_2.
4. If results are poor, investigate and tune hyperparameters.

### Metrics to compare

- Electricity cost
- Carbon emissions
- Battery utilization patterns
- Convergence speed (reward curve slope in early timesteps)

---

## Step 9: Validation Phase 4 — Multi-Building

**Goal:** Scale up. Run TransformerPPO on multiple buildings simultaneously.

### Setup

1. Select 3-4 buildings with different topologies (e.g., Building_2, Building_4,
   Building_15).
2. Create a schema with these buildings included.
3. Train for 1 episode.
4. Verify all buildings learn independently, no cross-building interference.
