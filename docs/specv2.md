## Universal Local Controller — Specification v2 (Entity Interface, Dynamic Topology)

> Reference v1: `docs/spec.md` and `docs/implementation_summary.md`.
> Sample observation payload: `datasets/tmp_entity_obs_full_step2200_named.json`.
> Target dataset: `citylearn_three_phase_dynamic_assets_only_demo`.
> Reusable v1 source code is **not** present on this branch. It lives on
> branch `gj/plan-c` (pinned to commit
> `3e9a6737d7306b04f3516738f86a98ea52106ef5`). Implementing agents must check
> that branch out (or `git show gj/plan-c:<path>`) to read the v1 modules
> referenced below as “reuse from `gj/plan-c`”.
> **No legacy (`flat`) interface logic should be added.** Anything new must
> use the `entity` interface and `topology_version`-based topology tracking
> introduced in `softcpsrecsimulator 0.3.0`.

---

## 0. Conventions

- File paths are repo-relative.
- Line references are pinned to current `HEAD` of this branch unless prefixed
  with `gj/plan-c:` (e.g. `gj/plan-c:algorithms/utils/transformer_backbone.py`).
- The phrase “the wrapper” means `utils/wrapper_citylearn.py:Wrapper_CityLearn`.
- The phrase “the adapter” means
  `utils/entity_adapter.py:EntityContractAdapter`.

### 0.1 Token-family vocabulary

v2 has **three** token families (down from four in earlier drafts — CTX
has been collapsed into SRO):

| Family | Cardinality per building | Has action output? | Examples |
|---|---|---|---|
| **CA** (Controllable Asset) | variable (one per asset) | Yes — exactly one scalar in `[-1, 1]` | `storage`, `charger` |
| **NFC** (Non-Flexible Context) | exactly **one** per building | No | the scalar `non_shiftable_load - solar_generation` |
| **SRO** (Shared Read-Only) | variable, multiple **types** per building | No | `district_pricing_current`, `district_weather_forecast`, `building_storage_state`, `pv`, `ev_connected`, … |

The type embedding table has **3** entries: `SRO=0`, `NFC=1`, `CA=2`.

“CTX” no longer appears anywhere in the design. Per-asset read-only
streams that earlier drafts called CTX (`pv`, `ev_connected`,
`ev_incoming`) are now SRO **types** with `cardinality: per_asset` (see
§7.3). Per-feature-group district splits (`district_pricing_current`,
`district_weather_forecast`, …) are SRO types with
`cardinality: singleton`. The Transformer treats them all the same way
at attention time; the per-type `Linear` projection is what gives them
distinct embedding subspaces.

### 0.2 Authoritative feature names

The regex catalogs in §13.1 are derived from
`datasets/tmp_entity_obs_full_step2200_named.json` (district 46
features, building 38, charger 16, ev 8, storage 9, pv 3). Any new
dataset version that introduces feature names not matched by an
existing pattern must hard-fail at config validation time (see §13.4
rule 1). The recommended fix is documented in the error message.

---

## 1. Objective & Core Principles

### Objective

Build a **Universal Local Controller (ULC)** that controls per-building energy
assets using a Transformer-based architecture in which every asset (and every
shared context source) is a token. The same trained model must handle, at
runtime and without retraining:

- Buildings with or without batteries / PV.
- Variable numbers of EV chargers per building.
- Mid-episode topology changes (assets entering / leaving) on the
  `softcpsrecsimulator 0.3.0` entity interface.
- There are **CA tokens** (controllable assets, one action output per
  token), **SRO tokens** (shared / read-only context — variable count,
  multiple types per building), and exactly **one NFC token per
  building** (the scalar uncontrollable net load `non_shiftable_load -
  solar_generation`).
The agent produces one action per CA token, so the CA order must be consistent between observation and action sides. SRO and NFC tokens have **no action outputs** but feed into shared self-attention so they shape every CA action.

The end goal is identical to v1. What changes in v2 is **how data flows in
and out of the agent**: the simulator now exposes a structured `entity`
contract with explicit `meta.topology_version`, so we no longer need
sentinel marker values to recover token boundaries, and we no longer use
asset/feature counting to detect topology changes.

### Core Principles

1. **Structure over sentinels.** Token boundaries are resolved from the
   per-feature *origin metadata* the wrapper already produces (e.g.
   `charger::Building_1/charger_1_1::connected_state`). No sentinel values
   are injected into the observation tensor.
2. **Topology-version-driven adaptation.** Topology changes are detected
   exclusively via `meta.topology_version` increments. On change, the
   wrapper rebuilds the per-building observation layout, the agent rebuilds
   its tokenizer’s segment table, and the rollout buffer is flushed.
3. **Wrapper owns the entity contract.** The agent receives a per-building
   `np.ndarray` and returns a per-building `List[float]`. The wrapper
   translates between the simulator’s entity payload and these flat vectors
   in both directions via `EntityContractAdapter`.
4. **Per-building agent.** Same architecture as v1 — independent
   tokenizer / backbone / actor / critic / rollout buffer per building.
5. **Reusable v1 components.** TransformerBackbone, ActorHead, CriticHead,
   RolloutBuffer and the PPO loss are imported verbatim from `gj/plan-c`.
   The marker-based ObservationEnricher and marker-scanning Tokenizer are
   replaced by entity-aware modules.
6. **Portability.** The new layout-builder module is pure-Python (stdlib +
   typing only) so the production inference repo can reuse it without
   pulling Torch.

---

## 2. Repo Delta from `gj/plan-c`

### 2.1 What this checkout already has (entity side)

| Path | Purpose | Lines |
|---|---|---|
| `utils/wrapper_citylearn.py` | Entity-mode detection, topology tracking, metadata attach, action conversion delegation | `:147-175`, `:309-365`, `:733-740` |
| `utils/entity_adapter.py` | Per-building observation flattening (`to_agent_observations`) and flat→entity action conversion (`to_entity_actions`) | `:145-347`, `:486-552` |
| `algorithms/agents/base_agent.py` | The exact agent contract this work must satisfy | `:16-84` |
| `utils/config_schema.py` | Pydantic models, including the existing `topology_mode='dynamic'` ⇒ `interface='entity'` validator and the MADDPG-vs-`entity+dynamic` guardrail | `:117-118`, `:157-158`, `:344-356` |
| `configs/templates/rule_based_entity_dynamic_assets_only_local.yaml` | Reference template shape we must mirror | full file |

### 2.2 What is on `gj/plan-c` and **must be brought across or recreated**

These files do **not** exist on this branch. The implementing agent must
either copy them (preferred when reuse is verbatim) or rewrite them.

| Path on `gj/plan-c` | Reuse policy in v2 |
|---|---|
| `algorithms/utils/transformer_backbone.py` | **Copy verbatim**, then update for v2 token families: 3-entry type embedding (`SRO=0, NFC=1, CA=2`) and `forward(sros, nfc, cas)` signature. |
| `algorithms/utils/ppo_components.py` (Actor, Critic, RolloutBuffer, `compute_ppo_loss`) | **Copy verbatim.** No API change. |
| `algorithms/agents/transformer_ppo/update_helper.py` | **Copy verbatim.** |
| `algorithms/agents/transformer_ppo/export_helper.py` | Copy and adapt: input shape now derived from `BuildingTokenLayout`. |
| `algorithms/agents/transformer_ppo/state_helper.py` | Copy, then **drop the marker registry**; add a `BuildingTokenLayout` field per building. |
| `algorithms/agents/transformer_ppo_agent.py` | Copy, then surgically replace the enricher/marker-tokenizer wiring with the new entity layout/tokenizer. |
| `algorithms/registry.py` change | Re-apply the `"AgentTransformerPPO"` registration. |

### 2.3 Files created **net-new** in v2

| Path | Purpose |
|---|---|
| `algorithms/utils/entity_token_layout.py` | Pure-Python `EntityTokenLayoutBuilder`, `BuildingTokenLayout`, `TokenSegment`. Replaces the v1 `ObservationEnricher`. |
| `algorithms/utils/entity_observation_tokenizer.py` | Torch `EntityObservationTokenizer`. Replaces the v1 marker-scanning tokenizer. |
| `utils/wrapper_transformer/__init__.py` + `transformer_observation_coordinator.py` | New coordinator hooks the layout builder into the wrapper’s entity lifecycle. |
| `configs/tokenizers/entity_default.json` | New tokenizer config (entity-type based, no marker fields). |
| `configs/templates/transformer_ppo_entity_dynamic.yaml` | New algorithm template; full repo-valid shape (see §13.2). |
| Tests under `tests/` (see §16). | |

### 2.4 Files modified

| Path | Modification |
|---|---|
| `utils/wrapper_citylearn.py` | Add a `TransformerObservationCoordinator` hook (idempotent for non-Transformer agents). Generalise the `MADDPG`-specific dynamic guardrail (see §12.4). |
| `utils/config_schema.py` | Add `EntityTokenizerConfig`, `TransformerConfig`, `TransformerPPOHyperparameters`, `TransformerPPOAlgorithmConfig`. Generalise the dynamic-topology guardrail to use a `supports_dynamic_topology` allow-list (keeping MADDPG’s rejection unchanged). Add JSON-validation for the file pointed to by `algorithm.tokenizer_config_path` (see §13.4). |
| `algorithms/registry.py` | Register `"AgentTransformerPPO"`. |

---

## 3. Entity Interface Data Format

Sample: `datasets/tmp_entity_obs_full_step2200_named.json` (step 2200,
`topology_version = 7`). The payload has three top-level fields: `tables`,
`edges`, `meta`.

### 3.1 `tables`

Each table has `features` (column names), `units`, and `rows` (list of dicts
with `id` + values). Tables observed in the demo dataset:

| Table | Cardinality | Feature count (sample) | Role | Tokenization |
|---|---|---|---|---|
| `district` | 1 | 46 | Community-wide context: weather, prices, carbon, time, community-level KPIs, `topology_version`, `active_*_count` | **Multiple SRO tokens** (one per semantic group: time, weather current, weather forecast, carbon, pricing current, pricing forecast, community energy, community headroom, community history, meta) |
| `building` | N | 38 | Per-building load / generation / storage SOC, headroom, energy book-keeping | **NFC token** (`non_shiftable_load - solar_generation` scalar) + **multiple SRO tokens** (storage state, charging phase one-hots, charging headroom, charging violation, energy current, energy history, building meta) |
| `charger` | M | 16 | Per-charger state and EV-related context | **CA token** per charger |
| `storage` | up to N | 9 | Per-building battery state | **CA token** per storage |
| `pv` | up to N | 3 | Per-building PV state | **SRO token** per PV (per-asset cardinality) |
| `ev` | K | 8 | EV state (SOC, capacity, ratios) | **SRO token** per EV (split into `ev_connected` / `ev_incoming` per parent charger) |

Counts above match `datasets/tmp_entity_obs_full_step2200_named.json` and
must be reverified during implementation against the active dataset’s
`entity_specs` (see §8.4).

### 3.2 `edges`

Topology graph; the wrapper consumes these to slice the global tables down
to per-building views.

| Edge name | Meaning |
|---|---|
| `district_to_building` | Always 1→all. |
| `building_to_charger` | Which chargers belong to a building. |
| `building_to_storage` | Which storage units belong to a building. |
| `building_to_pv` | Which PVs belong to a building. |
| `charger_to_ev_connected` (+ `_mask`) | Currently-connected EV per charger; `-1` = none. |
| `charger_to_ev_incoming` (+ `_mask`) | Reserved/incoming EV per charger; `-1` = none. |

### 3.3 `meta`

```json
{
  "time_step": 2200,
  "endogenous_time_step": 2199,
  "spec_version": "entity_v1",
  "temporal_semantics": {"exogenous": "t", "endogenous": "t_minus_1_settled"},
  "topology_version": 7
}
```

`topology_version` is the **single source of truth** for layout invalidation.

### 3.4 Action contract (current ground truth)

The adapter (`utils/entity_adapter.py:486-552`) returns:

```python
{
  "tables": {
    "building": np.zeros((n_buildings, len(building_action_features)), dtype=np.float32),
    "charger": np.zeros((n_chargers, len(charger_action_features)), dtype=np.float32),
  }
}
```

It does **not** emit a `"map"` field today. The schema in `softcpsrecsimulator
0.3.0`’s example uses `tables + map`, but the env accepts the tables-only
form already in production via `to_entity_actions`. **v2 keeps the
tables-only payload and does not add a `map` field**; if a future version
of the simulator requires it, the change is local to
`EntityContractAdapter.to_entity_actions` and invisible to the agent.

Source of truth for shape and ordering:

- `env.entity_specs["actions"]["building"]["features"]` — building action feature
  names (e.g. `electrical_storage`).
- `env.entity_specs["actions"]["charger"]["features"]` — charger action feature
  names (e.g. `electric_vehicle_storage`).
- `env.entity_specs["actions"]["building"]["ids"]` and `["charger"]["ids"]` —
  row order of the action tables.
- `env.action_space["tables"]["building"|"charger"].shape` — current shapes
  (change on topology mutation).

---

## 4. End-to-End Data Flow

```
                                                      step()
                                                         │
   ┌──────────────────────────────────────────────────────┴──────────────────────┐
   │                       softcpsrecsimulator 0.3.0 (CityLearnEnv)              │
   │  interface=entity                                                           │
   │  topology_mode=dynamic                                                      │
   └────────────────────────────────┬────────────────────────────────────────────┘
                                    │ entity payload {tables, edges, meta}
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ Wrapper_CityLearn (utils/wrapper_citylearn.py)                                 │
 │                                                                                │
 │ 1. EntityContractAdapter.to_agent_observations(payload)                        │
 │    → for each building: a flat np.ndarray + observation_names + space          │
 │    → updates self._entity_topology_version                                     │
 │                                                                                │
 │ 2. Topology change?  (self._entity_topology_version != previous_version)       │
 │    YES (and previous_version is not None)                                      │
 │      → self.encoders = self.set_encoders()      (placeholder NoNormalization)  │
 │      → wrapper._attach_model_environment_metadata()   (re-emits entity_specs   │
 │            + new observation_names + new action_names BEFORE layout rebuild)   │
 │      → coordinator.handle_topology_change(self) → for each building:           │
 │            agent.on_topology_change(b)                                         │
 │              ├─ flush rollout_buffer[b] (PPO update on previous layout)        │
 │              ├─ layouts[b] = layout_builder.build(                             │
 │              │      building_id[b], observation_names[b], action_names[b])    │
 │              ├─ re-pre-register tokenizer index buffers                        │
 │              └─ assert layouts[b].ca_action_names == action_names[b]           │
 │                                                                                │
 │ 3. agent.predict(encoded_obs, deterministic) → List[List[float]]               │
 │    where encoded_obs[b] = adapter.normalize_observation(b, raw_obs[b], ...)    │
 │    (see utils/wrapper_citylearn.py:964-988 — current behaviour)                │
 │                                                                                │
 │ 4. EntityContractAdapter.to_entity_actions(actions, action_names)              │
 │    → {"tables": {"building": np.ndarray, "charger": np.ndarray}}               │
 └────────────────────────────────┬───────────────────────────────────────────────┘
                                  │ flat vectors per building
                                  ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │ AgentTransformerPPO.predict() — per building b                                 │
 │                                                                                │
 │  EntityObservationTokenizer.forward(encoded_b, layout_b)                       │
 │     ├─ Slice encoded vector by `layout_b.segments` via torch.index_select      │
 │     ├─ NFC segment: gather 2 source features, apply NfcExpression              │
 │     │     (subtract) → scalar → project via projections["building_nfc"]        │
 │     ├─ All other segments: per-type Linear projection → d_model                │
 │     └─ Returns TokenizedObservation:                                           │
 │           {sro_tokens: N_sro, nfc_token: 1, ca_tokens: N_ca}                   │
 │                                                                                │
 │  TransformerBackbone.forward(sros, nfc, cas)                                   │
 │     ├─ Concat tokens [sros…, nfc, cas…], add type embedding                    │
 │     │     (SRO=0, NFC=1, CA=2 — 3 entries)                                     │
 │     ├─ TransformerEncoder (Pre-LN, GELU)                                       │
 │     └─ Returns all_embeddings + ca_embeddings + pooled                         │
 │                                                                                │
 │  ActorHead(ca_embeddings, ca_types) → actions [N_ca_b, 1] in [-1, 1]           │
 │  CriticHead(pooled) → V(s) scalar                                              │
 │                                                                                │
 │  if training: RolloutBuffer.add(...)                                           │
 │  on update_step: PPO update (GAE, clipped surrogate, K epochs)                 │
 └────────────────────────────────────────────────────────────────────────────────┘
```

> Reminder: although the entity payload is community-wide, the
> `EntityContractAdapter` already produces **one flat vector per building**
> (`utils/entity_adapter.py:145`). The agent therefore stays per-building.

---

## 5. Definitive Observation Contract

This section is the authoritative reference for the layout builder and
tokenizer tests. It mirrors the emission order of
`EntityContractAdapter.to_agent_observations` (`utils/entity_adapter.py:213-329`).

### 5.1 Per-building emission order

For each building `b` (in district-to-building edge order), names are
appended to `observation_names[b]` in this exact order:

1. **District block** — one feature per `district` table column, prefixed
   `district__<feature>` (`utils/entity_adapter.py:215`).
2. **Building block** — one feature per `building` table column,
   **unprefixed** (`utils/entity_adapter.py:223`). Example names from the
   sample payload: `non_shiftable_load`, `solar_generation`,
   `electrical_storage_soc`, `net_electricity_consumption`,
   `charging_phase_one_hot_charger_15_1_L1`, …
3. **Per-storage blocks** — for each storage row attached to this building,
   prefixed `storage::<storage_id>::<feature>`
   (`utils/entity_adapter.py:239`).
4. **Per-PV blocks** — for each PV row attached to this building, prefixed
   `pv::<pv_id>::<feature>` (`utils/entity_adapter.py:250`).
5. **Per-charger blocks**, in this order per charger
   (`utils/entity_adapter.py:258-296`):
   1. Charger features prefixed `charger::<charger_id>::<feature>`.
   2. Connected-EV context features prefixed
      `charger::<charger_id>::connected_ev::<feature>`. **The label is
      exactly `connected_ev`** (`utils/entity_adapter.py:291`).
   3. Incoming-EV context features prefixed
      `charger::<charger_id>::incoming_ev::<feature>`. **The label is
      exactly `incoming_ev`** (`utils/entity_adapter.py:294`).
6. **Active-counters block** — three unprefixed features:
   `active_chargers_count`, `active_storages_count`, `active_pvs_count`
   (`utils/entity_adapter.py:298-300`).
7. **Legacy-charger-aliases block** — zero or more unprefixed features whose
   names come from `EntityContractAdapter._LEGACY_CHARGER_ALIASES`
   (`utils/entity_adapter.py:302-329`). These are RBC-compatibility aliases
   sourced from the building’s first connected charger. **In v2 these are
   excluded from the token catalog** (see §5.4 / §13.1
   `excluded_features`); they remain in `observation_names` for adapter
   compatibility but are not bound to any token.

### 5.2 Token-family classification

Classification is **regex-driven and config-driven** (see §13.1 for the
full tokenizer JSON). The high-level rule per feature is:

| Feature pattern | Family | Type name | Instance id | Notes |
|---|---|---|---|---|
| Matches an SRO type’s `feature_patterns` (district / building scope) | `sro` | the SRO type name | building id (since SRO patterns are evaluated per building) | Singleton SRO types yield one token per building. |
| Per-asset SRO `entity_table` prefix in name (e.g. `pv::<id>::…`, `charger::<id>::connected_ev::…`, `charger::<id>::incoming_ev::…`) | `sro` | `pv` / `ev_connected` / `ev_incoming` | `<id>` | Variable count per building. |
| `storage::<id>::<feature>` | `ca` | `storage` | `<id>` | Action `electrical_storage`. |
| `charger::<id>::<feature>` (no `connected_ev`/`incoming_ev` segment) | `ca` | `charger` | `<id>` | Action `electric_vehicle_storage`. |
| Matches `nfc.expression` source features (`non_shiftable_load`, `solar_generation`) | `nfc` | `building` | `<building_id>` | Two source features collapse into **one** scalar token via the configured expression. |
| Matches `excluded_features` patterns | (excluded) | — | — | Removed before classification (see §5.4). |
| Anything else | (error) | — | — | Hard-fail at validation (§13.4 rule 1). |

Notes:

- **One feature → one match**: tokenizer JSON validation rule 2
  (uniqueness, §13.4) guarantees no feature is matched by patterns from
  more than one SRO type.
- **NFC is a scalar**: the NFC token has dim 1 — the value of
  `non_shiftable_load - solar_generation`. The two source features are
  consumed and **do not** appear in any SRO group.
- **SRO type input dim** is the count of features that match its
  `feature_patterns` (for table-scoped SRO types) or
  `len(entity_specs.tables[<entity_table>].features)` (for per-asset
  SRO types). This count is fixed per topology and used to size the
  per-type `Linear` projection (§8).
- **Non-contiguous indices**: an SRO type’s feature positions in
  `observation_names` may be non-contiguous (e.g.
  `district_weather_forecast` mixes `outdoor_dry_bulb_temperature_predicted_*`
  and `direct_solar_irradiance_predicted_*`). The layout records the
  full index list per group; the tokenizer slices via
  `torch.index_select` (§8.3).

### 5.3 Why no marker injection

v1 used numeric markers because the legacy `flat` interface emitted a single
unstructured vector. With `interface=entity`, the wrapper already emits
deterministic per-feature names (see §5.1). We classify them once per
topology via the regex catalog (§13.1) and slice the encoded tensor at
fixed integer indices every step.

### 5.4 Excluded features

The tokenizer config carries an `excluded_features` list of patterns
(see §13.1). Features matching any pattern are removed *before*
classification. Use cases:

- **`topology_version`** (in the district table) — already conveyed via
  `meta.topology_version`; redundant as an observation feature.
- **Legacy charger aliases** (`electric_vehicle_charger_state`,
  `electric_vehicle_soc`, `electric_vehicle_required_soc_departure`,
  `electric_vehicle_departure_time`, `electric_vehicle_is_flexible`)
  emitted by `EntityContractAdapter._LEGACY_CHARGER_ALIASES`
  (`utils/entity_adapter.py:302-321`) — kept for RBC compatibility but
  redundant with the per-charger CA tokens for the Transformer.

This is the primary "feature engineering" knob: to remove a feature
from the agent’s view, add a pattern to `excluded_features`. Removed
features are still emitted by the adapter (we do not modify the
adapter); they are simply not bound to any token.

The exclusion list is part of the tokenizer config so that exclusions
travel with the model (training and inference must agree).

---

## 6. Encoding Story (entity mode)

This is intentionally minimal in the current code base.

### 6.1 Current behaviour

- `Wrapper_CityLearn.set_encoders()` returns a list of `NoNormalization()`
  placeholders when `interface=entity`
  (`utils/wrapper_citylearn.py:1100-1108`).
- `Wrapper_CityLearn.get_encoded_observations(index, observations)` ignores
  those placeholders and instead delegates to
  `EntityContractAdapter.normalize_observation(...)`
  (`utils/wrapper_citylearn.py:964-988`).
- `normalize_observation` returns a numpy array of the **same length and
  ordering** as `observation_names[index]`, scaled per
  `simulator.entity_encoding`.

### 6.2 Implication for v2

There is **no encoded-dimension expansion**. Encoded index = raw feature
position in `observation_names[building]`. The layout builder therefore
records `feature_indices` directly as positions in
`observation_names[building]`, and those same indices are used at slice
time on the encoded tensor. We do **not** introduce an `encoded_index_map`
parameter in v2 (it was an unnecessary abstraction in the first draft of
this spec).

### 6.3 When encoders are rebuilt

Encoders (the placeholder list) are rebuilt from
`Wrapper_CityLearn._apply_entity_layout` whenever `_apply_entity_layout`
runs — i.e. on every reset and on every step that produces a new payload.
The rebuild is cheap (it allocates one `NoNormalization()` per feature
name). The agent’s layout rebuild, however, is gated on actual
`topology_version` changes (see §12).

---

## 7. EntityTokenLayoutBuilder (replaces v1 ObservationEnricher)

### 7.1 Responsibility

Given the per-building `observation_names` (post-adapter, pre-encoding —
which equals the post-encoding ordering, see §6.2), build a token segment
table. Pure Python so it can be reused by the inference repo.

### 7.2 Interface

```python
@dataclass(frozen=True)
class TokenSegment:
    family: str                          # "sro" | "nfc" | "ca"
    type_name: str                       # e.g. "district_pricing_current",
                                          #      "building_storage_state",
                                          #      "pv", "ev_connected",
                                          #      "storage", "charger"
    instance_id: Optional[str]           # asset id (per-asset SRO / CA),
                                          # building id (singleton SRO / NFC)
    feature_indices: Tuple[int, ...]     # positions in observation_names
                                          # (may be non-contiguous)
    feature_names: Tuple[str, ...]
    derived: Optional[NfcExpression] = None
                                          # Only set on the NFC segment.
                                          # Tells the tokenizer to compute a
                                          # scalar from `feature_indices`
                                          # (currently: subtract).


@dataclass(frozen=True)
class NfcExpression:
    op: str                              # "subtract" (others reserved)
    left_index_in_segment: int           # offset within feature_indices
    right_index_in_segment: int


@dataclass(frozen=True)
class BuildingTokenLayout:
    building_id: str
    segments: Tuple[TokenSegment, ...]   # ordered: sros…, nfc, cas…
    n_sro: int
    n_ca: int                            # n_nfc is always 1
    ca_action_names: Tuple[str, ...]     # action_field per CA segment, in
                                          # segment order — equal to
                                          # action_names[building] by
                                          # construction.
    excluded_feature_names: Tuple[str, ...]
                                          # Features dropped before
                                          # classification (see §5.4).


class EntityTokenLayoutBuilder:
    def __init__(self, tokenizer_config: Mapping[str, Any]) -> None: ...

    def build(
        self,
        building_id: str,
        observation_names: Sequence[str],
        action_names: Sequence[str],
    ) -> BuildingTokenLayout:
        """Return the cached layout if (observation_names, action_names) matches,
        else recompute. Owns CA ordering — see §7.3."""

    def topology_changed(
        self,
        building_id: str,
        observation_names: Sequence[str],
        action_names: Sequence[str],
    ) -> bool: ...
```

### 7.3 Classification rules

For each building, in order:

1. **Drop excluded features**: any name matching `excluded_features`
   patterns (§5.4) is recorded in `excluded_feature_names` and skipped
   from the rest of classification. The remaining names form the
   classifier input.
2. **Detect NFC**: locate the indices of all features named in
   `nfc.expression` source fields (default:
   `non_shiftable_load`, `solar_generation`). If any source feature is
   missing → `ValueError`. If both are present, emit one
   `TokenSegment(family="nfc", type_name="building",
   instance_id=building_id, feature_indices=(idx_left, idx_right),
   derived=NfcExpression(op="subtract", left_index_in_segment=0,
   right_index_in_segment=1))`. These two source features are consumed
   by NFC and not eligible for any SRO group.
3. **Match SRO patterns**: for each remaining name, walk the SRO type
   table and pick the unique match. The tokenizer JSON validation
   guarantees uniqueness (§13.4 rule 2). Per-asset SRO types
   (`pv`, `ev_connected`, `ev_incoming`) are matched by their adapter
   prefix (e.g. `pv::<id>::…`); the `<id>` becomes the segment’s
   `instance_id`. Singleton SRO types are scoped to the table prefix
   (`district__…` for district-table SROs; unprefixed for
   building-table SROs) and use `building_id` as `instance_id`.
4. **Match CA prefixes**: `storage::<id>::…` → CA `storage`;
   `charger::<id>::…` (with no `connected_ev`/`incoming_ev` segment) →
   CA `charger`.
5. **Reject leftovers**: any name that survives steps 1–4 unmatched →
   `ValueError` listing the name and table-of-origin (recoverable from
   the adapter prefix). This is the runtime mirror of the validation
   rule in §13.4 rule 1; both must hard-fail to keep training and
   inference layouts identical.

Within each matched group, `feature_indices` are appended in the order
the names appear in `observation_names`. Output ordering of `segments`:

1. All SRO segments, sorted by `(sro_type_declaration_order_in_config,
   instance_id)`. The declaration order in the tokenizer JSON is
   stable, so this gives a deterministic SRO order (e.g.
   `district_time` before `district_weather_current` before
   `district_weather_forecast`, then per-asset SRO types sorted by
   `instance_id`).
2. The single NFC segment.
3. All CA segments, ordered to **exactly match `action_names[building]`**
   (see below).

**CA ordering — single rule, single owner.** `EntityTokenLayoutBuilder`
is the *only* component that decides CA segment order. For each CA
candidate it derives the `action_field` from its `type_name` via the
tokenizer config (`storage` ⇒ `electrical_storage`, `charger` ⇒
`electric_vehicle_storage`) combined with the `instance_id`, then sorts
the CA segments so that the resulting `ca_action_names` tuple is
**element-wise equal to `action_names[building]`**. If no permutation
satisfies that constraint (e.g. an action name has no matching CA
candidate, or vice-versa), `build()` raises `ValueError` with both
sequences in the message.

This makes CA order *defined by `action_names[building]`* — there is no
secondary sort key, no lexicographic fallback, no instance-id ordering.
The startup check in §10.1 then becomes a pure post-condition assertion.

### 7.4 Portability constraints

- No imports from `algorithms.*`, `utils.*`, no torch/numpy.
- Pure stdlib + `typing` + `re`.
- File location: `algorithms/utils/entity_token_layout.py`.

---

## 8. EntityObservationTokenizer (replaces v1 marker tokenizer)

### 8.1 Responsibility

Slice the encoded per-building tensor into token groups using a
`BuildingTokenLayout`, then project each group to `d_model` via per-type
`Linear` layers. NFC is the special case: its segment carries an
`NfcExpression` and is reduced to a scalar before projection.

### 8.2 Interface

```python
@dataclass
class TokenizedObservation:
    sro_tokens: torch.Tensor    # [batch, N_sro, d_model]
    nfc_token: torch.Tensor     # [batch, 1,     d_model]
    ca_tokens: torch.Tensor     # [batch, N_ca,  d_model]
    sro_types: List[str]        # one per SRO token (e.g. "district_pricing_current",
                                # "pv", "ev_connected", …) — order = layout
    ca_types: List[str]         # one per CA token (e.g. "storage", "charger") — order = layout
    n_sro: int
    n_ca: int


class EntityObservationTokenizer(nn.Module):
    def __init__(
        self,
        tokenizer_config: Mapping[str, Any],
        d_model: int,
        type_input_dims: Mapping[str, int],
    ) -> None:
        """Create one nn.Linear per type. type_input_dims maps every
        declared type name (every SRO type, every CA type, plus the
        single NFC type) to its raw feature count.

        For SRO and CA types the input dim is taken from entity_specs at
        attach time (see §8.5). For the NFC type the input dim is always
        1 because the NfcExpression collapses its source features to a
        scalar before projection."""

    def forward(
        self,
        encoded_obs: torch.Tensor,        # [batch, obs_dim]
        layout: BuildingTokenLayout,
    ) -> TokenizedObservation: ...
```

### 8.3 Slicing implementation

The tokenizer pre-registers, per segment, an `index_buffer = torch.tensor(
segment.feature_indices, dtype=torch.long)` as a non-trainable buffer (so
it follows the module to GPU). At forward time it does
`group = encoded_obs.index_select(dim=1, index=index_buffer)`, which
handles the non-contiguous case (e.g. forecasts interleaved with
current-step features inside `district_weather_forecast`).

For the **NFC segment specifically**, the tokenizer reads
`segment.derived` (an `NfcExpression`), gathers the two source features
via `index_select`, computes the configured op (currently only
`subtract`: `nfc_scalar = group[..., left] - group[..., right]`),
unsqueezes to shape `[batch, 1]`, then projects via
`projections["building_nfc"]` → `[batch, d_model]`.

### 8.4 Variable token counts (zero new parameters)

A new charger reuses `projections["charger"]`. A new PV reuses
`projections["pv"]`. A new building with the same `district` schema
reuses every `projections["district_*"]`. Only the layout’s segment
tuple grows — not the parameter set.

### 8.5 Where input dimensions come from

- **NFC**: always `1` (the scalar produced by `NfcExpression`). The
  tokenizer config entry for NFC has no `input_dim_fallback` field —
  the dim is hard-coded.
- **CA types** (`storage`, `charger`): read from
  `metadata["entity_specs"]["tables"]["<entity_table>"]["features"]`
  at `attach_environment` time and use `len(features)`.
- **Per-asset SRO types** (`pv`, `ev_connected`, `ev_incoming`): same
  as CA — read from the corresponding `entity_table` (`pv`, `ev`).
- **Singleton SRO types** (declared with `feature_patterns` against a
  table such as `district` or `building`): the per-type input dim is
  the count of features in that table that match the type’s
  `feature_patterns` (computed once at `attach_environment` time using
  `entity_specs.tables[<entity_table>].features`). For the
  `building`-scoped SRO types the count is taken **after subtracting
  excluded features and the NFC source features**.

The tokenizer config provides `input_dim_fallback` per type for
inference contexts where `entity_specs` is not available. Fallback
values must equal the dim derived from `entity_specs`; mismatches raise
`ValueError` listing both sides at instantiation. (NFC excepted as
above.)

---

## 9. TransformerBackbone (reuse from `gj/plan-c`, extended)

`gj/plan-c:algorithms/utils/transformer_backbone.py` is copied with two
changes:

- Type embedding table size = **3**: `SRO=0`, `NFC=1`, `CA=2`. (v1 had 3
  for different families; the count happens to be unchanged but the
  semantics differ. The CTX entry from earlier v2 drafts is **not**
  introduced.)
- `forward(sros, nfc, cas)` takes three tensors:
  - `sros`: `[batch, N_sro, d_model]`
  - `nfc`:  `[batch, 1,     d_model]`
  - `cas`:  `[batch, N_ca,  d_model]`

  Concatenates in fixed order `[sros…, nfc, cas…]` and slices
  `ca_embeddings` at offsets `[N_sro + 1 : N_sro + 1 + N_ca]`. Mean
  pooling for the critic spans **all** tokens (SRO + NFC + CA).

Pre-LN, GELU, no positional embeddings — unchanged.

---

## 10. PPO Components & Action Contract

### 10.1 CA token order ↔ action order

CA order is owned exclusively by `EntityTokenLayoutBuilder.build()`,
which receives `action_names[building]` and constructs `segments` so
that `BuildingTokenLayout.ca_action_names == tuple(action_names[building])`
by construction (see §7.3).

`AgentTransformerPPO.predict()` returns one scalar per CA token, in the
order produced by `BuildingTokenLayout.segments`. The wrapper then
forwards that flat list to
`EntityContractAdapter.to_entity_actions(actions, action_names)`. The
adapter resolves each action *by name* via
`action_names[building][position]`, so position-equality between
`ca_action_names` and `action_names[building]` is exactly what the
adapter requires.

`attach_environment` performs a startup post-condition assertion
(belt-and-braces; the builder already enforces it):

```
for b in range(len(action_names)):
    expected = tuple(action_names[b])
    actual = layout_builder.build(
        building_id[b], observation_names[b], action_names[b]
    ).ca_action_names
    if expected != actual:
        raise ValueError(
            f"BuildingTokenLayout.ca_action_names {actual!r} does not match "
            f"action_names[{b}] {expected!r}"
        )
```

The same assertion runs inside `on_topology_change(b)` after the layout
is rebuilt (see §12.2).

### 10.2 Reused PPO modules (verbatim from `gj/plan-c`)

- `ActorHead` — MLP per CA embedding; tanh-squashed Gaussian; learnable
  shared `log_std` per CA *type*.
- `CriticHead` — MLP from pooled embedding → scalar V(s).
- `RolloutBuffer` — On-policy buffer with GAE.
- `compute_ppo_loss` — Clipped surrogate + value loss + entropy bonus.

### 10.3 Action range and clipping

`predict()` returns values in `[-1, 1]`. The wrapper clips to per-agent
action-space bounds via the existing `_clip_actions`
(`utils/wrapper_citylearn.py:743-779`).

---

## 11. AgentTransformerPPO (v2)

### 11.1 `BaseAgent` contract — exact signatures

Copied from `algorithms/agents/base_agent.py:16-84`. Note `terminated` /
`truncated` are **scalar booleans** (not lists), `output_dir` is `str`,
`predict` returns `List[List[float]]`, and `attach_environment` uses
keyword-only arguments.

```python
class AgentTransformerPPO(BaseAgent):
    supports_dynamic_topology: ClassVar[bool] = True

    def __init__(self, config: Dict[str, Any]) -> None: ...

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build per-building EntityObservationTokenizer / TransformerBackbone /
        ActorHead / CriticHead / RolloutBuffer; store entity_specs; build the
        initial BuildingTokenLayout per building; validate CA-order vs
        action_names (see §10.1)."""

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: Optional[bool] = None,
    ) -> List[List[float]]: ...

    def update(
        self,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float64]],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None: ...

    def on_topology_change(self, building_idx: int) -> None:
        """Triggered by the wrapper after _entity_topology_version increments
        (excluding the first attach — see §12.1).
        1. If rollout buffer has data → run a PPO update on collected trajectory.
        2. Clear the rollout buffer.
        3. Rebuild the BuildingTokenLayout from new observation_names + action_names.
        4. Re-run §13.4 rules 1–5 against new entity_specs (coverage,
           uniqueness, NFC sources, pattern compilation, action-field coverage).
        5. The §10.1 CA-order post-condition is implicit in rule 5."""

    def export_artifacts(self, output_dir: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]: ...
    def load_checkpoint(self, checkpoint_path: str) -> None: ...
```

### 11.2 Per-building structure

Each building owns: tokenizer, backbone, actor, critic, rollout buffer,
optimizer, cached `BuildingTokenLayout`. Same as v1.

---

## 12. Lifecycle: Reset, Topology Change, Buffer Flush, PPO

This section makes the boundaries the reviewer asked for explicit.

### 12.1 First attach (wrapper construction → `set_model`)

The wrapper attaches the agent in **two stages** in the current code:

1. `Wrapper_CityLearn.__init__` calls `_apply_entity_layout(payload,
   force_attach=False)` (`utils/wrapper_citylearn.py:306`). At this
   point `self.model is None`, so the inner
   `_attach_model_environment_metadata()` short-circuits
   (`utils/wrapper_citylearn.py:344-346`). The wrapper has populated
   `observation_names`, `observation_space`, `action_space`,
   `action_names`, `encoders`.
2. `run_experiment.py` later instantiates the agent and calls
   `wrapper.set_model(agent)`, which assigns `self.model = agent` and
   immediately calls `_attach_model_environment_metadata()`
   (`utils/wrapper_citylearn.py:418-423`). **This is the real first
   `attach_environment(...)` invocation for the agent.**

The Transformer agent therefore implements `attach_environment` so that
it works in this order:

1. Receive `observation_names`, `action_names`, `action_space`,
   `observation_space`, and `metadata` (which contains
   `entity_specs`, `interface`, `topology_mode`, `building_names`).
2. Build per-building tokenizer / backbone / actor / critic / rollout
   buffer / optimizer.
3. Build layouts: `layouts[b] = layout_builder.build(building_id[b],
   observation_names[b], action_names[b])`.
4. Run the post-condition assertion in §10.1.
5. Do **not** treat this as a topology change — buffers are empty by
   definition; no PPO update is attempted.

The agent MUST be tolerant of a second `attach_environment` call with
identical `observation_names`/`action_names` (no-op, returns without
mutation) because the wrapper may re-emit metadata on episode reset
without a topology change.

### 12.2 Genuine topology change mid-episode

Call site: `Wrapper_CityLearn._apply_entity_layout`
(`utils/wrapper_citylearn.py:309-342`) detects
`previous_version is not None and self._entity_topology_version != previous_version`.

The wrapper is the **single owner** of the rebuild trigger. The agent
owns layout reconstruction. Sequence (must run in this order):

1. Wrapper rebuilds `observation_names`, `observation_space`,
   `action_space`, `action_names`, `encoders` (already happens at
   `utils/wrapper_citylearn.py:317-326`).
2. Wrapper calls `_attach_model_environment_metadata()` so the agent
   sees the **new** `observation_names`/`action_names`/`entity_specs`
   *before* it rebuilds layouts. (This replaces the existing call at
   `utils/wrapper_citylearn.py:339-340`, which already runs in this
   slot.)
3. Wrapper calls
   `TransformerObservationCoordinator.handle_topology_change(self)`,
   which loops `for b in range(len(self.observation_names))` and calls
   `agent.on_topology_change(b)`.
4. Inside `on_topology_change(b)`:
   a. If `rollout_buffer[b]` has data → run a single PPO update on the
      collected trajectory using the **previous** layout (cached on the
      agent until step 4c). PPO is on-policy and the data was collected
      under the old policy/topology, so this is the only sound moment
      to flush.
   b. Clear `rollout_buffer[b]`.
   c. Rebuild `layouts[b] = layout_builder.build(building_id[b],
      observation_names[b], action_names[b])`. Re-pre-register
      tokenizer index buffers.
   d. Re-run §13.4 rules 1–5 against the new `entity_specs` (coverage,
      uniqueness, NFC sources, pattern compilation already cached, action-
      field coverage). Raise on any mismatch.
   e. Re-run the §10.1 post-condition assertion.

`attach_environment` does **not** rebuild layouts on its own when called
during a topology change — the coordinator path above is the single
entry point for layout rebuilds. This avoids the double-build that the
reviewer flagged.

> Edge case: a topology change in the middle of an unfinished
> `update_step` window is handled here (the buffer is force-flushed). The
> next regularly-scheduled `update_step` will then operate on a fresh
> buffer.

### 12.3 No-op steps

When `_apply_entity_layout` runs but `topology_version` did not change, the
coordinator does **nothing**. The agent does **not** rebuild layouts,
buffers, or anything else.

### 12.4 Dynamic-topology guardrail (generalised)

The current MADDPG-specific raise (`utils/wrapper_citylearn.py:333-338` and
`utils/config_schema.py:352-356`) is replaced by a single, generic rule:

- Each agent class declares `supports_dynamic_topology: ClassVar[bool]`
  (default `False`). `MADDPG` keeps `False`; `RuleBasedPolicy` is `True`
  (it already works dynamically); `AgentTransformerPPO` is `True`.
- `utils/config_schema.py` reads the registry at validation time and
  raises if `simulator.topology_mode == "dynamic"` and the chosen
  algorithm class has `supports_dynamic_topology is False`. The error
  message keeps the existing wording for MADDPG to preserve test output.
- `utils/wrapper_citylearn.py` keeps a runtime double-check (defence in
  depth) but uses the same flag rather than a hard-coded `MADDPG` literal.

---

## 13. Configuration

### 13.1 Tokenizer config

`configs/tokenizers/entity_default.json`

The config is the **single source of truth** for token taxonomy,
feature exclusion, and NFC computation. Adding a new SRO grouping or
removing a feature from the agent’s view is a config-only change — no
code edit required.

```json
{
  "type_embeddings": { "SRO": 0, "NFC": 1, "CA": 2 },

  "excluded_features": {
    "patterns": [
      "^district__topology_version$",
      "^electric_vehicle_charger_state$",
      "^electric_vehicle_soc$",
      "^electric_vehicle_required_soc_departure$",
      "^electric_vehicle_departure_time$",
      "^electric_vehicle_is_flexible$"
    ]
  },

  "nfc": {
    "type_name": "building_nfc",
    "entity_table": "building",
    "expression": {
      "op": "subtract",
      "left":  { "feature": "non_shiftable_load" },
      "right": { "feature": "solar_generation" }
    }
  },

  "ca_types": {
    "storage": {
      "entity_table": "storage",
      "action_field": "electrical_storage",
      "input_dim_fallback": 9
    },
    "charger": {
      "entity_table": "charger",
      "action_field": "electric_vehicle_storage",
      "input_dim_fallback": 16
    }
  },

  "sro_types": {
    "district_time": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__month$",
        "^district__day_type$",
        "^district__hour$"
      ],
      "input_dim_fallback": 3
    },
    "district_weather_current": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__outdoor_dry_bulb_temperature$",
        "^district__outdoor_relative_humidity$",
        "^district__diffuse_solar_irradiance$",
        "^district__direct_solar_irradiance$"
      ],
      "input_dim_fallback": 4
    },
    "district_weather_forecast": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__outdoor_dry_bulb_temperature_predicted_\\d+$",
        "^district__outdoor_relative_humidity_predicted_\\d+$",
        "^district__diffuse_solar_irradiance_predicted_\\d+$",
        "^district__direct_solar_irradiance_predicted_\\d+$"
      ],
      "input_dim_fallback": 12
    },
    "district_carbon": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__carbon_intensity$" ],
      "input_dim_fallback": 1
    },
    "district_pricing_current": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__electricity_pricing$" ],
      "input_dim_fallback": 1
    },
    "district_pricing_forecast": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__electricity_pricing_predicted_\\d+$" ],
      "input_dim_fallback": 3
    },
    "district_community_energy": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_(net|import|export|pv|bess|ev)_(power_kw|energy_kwh_step)$"
      ],
      "input_dim_fallback": 12
    },
    "district_community_headroom": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_(building|phase)(_export)?_headroom_kw$"
      ],
      "input_dim_fallback": 4
    },
    "district_community_history": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [
        "^district__community_net_prev_\\d+_(kwh_step|mean_kwh_step)$"
      ],
      "input_dim_fallback": 2
    },
    "district_meta": {
      "entity_table": "district",
      "cardinality": "singleton",
      "feature_patterns": [ "^district__active_(buildings|chargers|evs)_count$" ],
      "input_dim_fallback": 3
    },

    "building_storage_state": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^electrical_storage_soc$",
        "^electrical_storage_soc_ratio$"
      ],
      "input_dim_fallback": 2
    },
    "building_charging_phase_onehot": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [ "^charging_phase_one_hot_.+$" ],
      "input_dim_fallback": 6
    },
    "building_charging_headroom": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^charging_(building|phase_L\\d+)(_export)?_headroom_kw$"
      ],
      "input_dim_fallback": 8
    },
    "building_charging_violation": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [ "^charging_constraint_violation_kwh$" ],
      "input_dim_fallback": 1
    },
    "building_energy_current": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^net_electricity_consumption$",
        "^(net|import|export|load|pv|bess|ev_charging)_(power_kw|energy_kwh_step)$"
      ],
      "input_dim_fallback": 15
    },
    "building_energy_history": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^(net|import|export)_energy_prev_\\d+_(kwh_step|mean_kwh_step)$"
      ],
      "input_dim_fallback": 4
    },
    "building_meta": {
      "entity_table": "building",
      "cardinality": "singleton",
      "feature_patterns": [
        "^active_chargers_count$",
        "^active_storages_count$",
        "^active_pvs_count$"
      ],
      "input_dim_fallback": 3
    },

    "pv": {
      "entity_table": "pv",
      "cardinality": "per_asset",
      "adapter_prefix": "pv::",
      "input_dim_fallback": 3
    },
    "ev_connected": {
      "entity_table": "ev",
      "cardinality": "per_asset",
      "adapter_prefix": "charger::",
      "adapter_label": "connected_ev",
      "input_dim_fallback": 8
    },
    "ev_incoming": {
      "entity_table": "ev",
      "cardinality": "per_asset",
      "adapter_prefix": "charger::",
      "adapter_label": "incoming_ev",
      "input_dim_fallback": 8
    }
  },

  "validation": {
    "unmatched_features": "fail",
    "ambiguous_pattern_match": "fail",
    "input_dim_mismatch": "fail"
  }
}
```

#### Coverage accounting (matches the sample payload exactly)

Numbers below are derived from
`datasets/tmp_entity_obs_full_step2200_named.json` (district 46,
building 38).

| Table | Bucket | Count |
|---|---|---|
| district | `district_time` | 3 |
| district | `district_weather_current` | 4 |
| district | `district_weather_forecast` | 12 |
| district | `district_carbon` | 1 |
| district | `district_pricing_current` | 1 |
| district | `district_pricing_forecast` | 3 |
| district | `district_community_energy` | 12 |
| district | `district_community_headroom` | 4 |
| district | `district_community_history` | 2 |
| district | `district_meta` | 3 |
| district | excluded (`topology_version`) | 1 |
| **district total** | | **46** ✓ |
| building | NFC sources (`non_shiftable_load`, `solar_generation`) | 2 |
| building | `building_storage_state` | 2 |
| building | `building_charging_phase_onehot` | 6 |
| building | `building_charging_headroom` | 8 |
| building | `building_charging_violation` | 1 |
| building | `building_energy_current` | 15 |
| building | `building_energy_history` | 4 |
| **building table total** | | **38** ✓ |
| Adapter-emitted trailing (per-building) | `building_meta` (`active_*_count`) | 3 |
| Adapter-emitted trailing (per-building) | excluded (legacy charger aliases) | 4–5 |

Per-asset SRO and CA token counts vary at runtime with topology.

#### "Cardinality" semantics

- `singleton` SRO types yield exactly **one token per building** (the
  features matched by `feature_patterns` form one segment). Used for
  district and building scoped groupings.
- `per_asset` SRO types yield **N tokens per building** where N is
  determined by the adapter prefix (`pv::<id>::…`,
  `charger::<id>::connected_ev::…`, `charger::<id>::incoming_ev::…`).
  One segment per matched `<id>`.

#### NFC config

The `nfc` block declares the **single** NFC token. `expression.op`
must be one of: `subtract`. `left.feature` and `right.feature` must
exist in the `entity_table`. The tokenizer collapses these two source
features into a scalar at forward time (§8.3) — the source features
do not appear in any SRO group (§7.3 step 2).

To change the NFC definition (e.g. add a third term, or use a
different expression), modify this block — no code change required.

### 13.2 Algorithm template (full repo-valid shape)

`configs/templates/transformer_ppo_entity_dynamic.yaml`

```yaml
metadata:
  experiment_name: "transformer_ppo_entity_dynamic_template"
  run_name: "Transformer PPO Entity Dynamic Local"
  community_name: "default_community"
  description: "Per-building Transformer PPO over entity interface with dynamic topology"

runtime:
  log_dir: null
  job_dir: null
  mlflow_uri: null
  job_id: null
  run_id: null
  run_name: null
  tracking_uri: null
  experiment_id: null
  mlflow_run_url: null

tracking:
  mlflow_enabled: true
  log_level: "INFO"
  log_frequency: 1
  mlflow_step_sample_interval: 10
  mlflow_artifacts_profile: minimal
  progress_updates_enabled: true
  progress_update_interval: 5
  system_metrics_enabled: false
  system_metrics_interval: 10

checkpointing:
  resume_training: false
  checkpoint_run_id: null
  checkpoint_artifact: "transformer_ppo_checkpoint.pt"
  use_best_checkpoint_artifact: false
  reset_replay_buffer: false
  freeze_pretrained_layers: false
  fine_tune: false
  checkpoint_interval: null

bundle:
  bundle_version: null
  description: null
  alias_mapping_path: null
  require_observations_envelope: false
  artifact_config: {}
  per_agent_artifact_config: {}

simulator:
  dataset_name: citylearn_three_phase_dynamic_assets_only_demo
  dataset_path: ./datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json
  central_agent: false
  interface: entity
  topology_mode: dynamic
  entity_encoding:
    enabled: true
    normalization: minmax_space
    clip: true
  reward_function: RewardFunction
  reward_function_kwargs: {}
  episodes: 1
  simulation_start_time_step: 0
  simulation_end_time_step: 3400
  episode_time_steps: 3401
  export:
    mode: end
    export_kpis_on_episode_end: true
    session_name: null
  wrapper_reward:
    enabled: false
    profile: cost_limits_v1
    clip_enabled: true
    clip_min: -10.0
    clip_max: 10.0
    squash: none

training:
  seed: 22
  steps_between_training_updates: 1
  target_update_interval: 0

topology:
  num_agents: null
  observation_dimensions: null
  action_dimensions: null
  action_space: null

algorithm:
  name: "AgentTransformerPPO"
  tokenizer_config_path: configs/tokenizers/entity_default.json
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
  networks: null
  replay_buffer: null
  exploration: null

execution: null
```

This template mirrors the field set of
`configs/templates/rule_based_entity_dynamic_assets_only_local.yaml` so it
will pass `validate_config(...)` without further plumbing.

### 13.3 Hyperparameter naming (canonical)

The single canonical name for the optimizer learning rate is
**`learning_rate`** (matching v1 spec §9 and the YAML above). Internally the
agent may alias `self.lr = self.config["algorithm"]["hyperparameters"]["learning_rate"]`,
but no schema, config or docstring uses `lr` as the canonical key.

### 13.4 Tokenizer JSON validation

`utils/config_schema.py` is extended with `EntityTokenizerConfig` (Pydantic).
After successfully loading a YAML config that names
`AgentTransformerPPO`, `validate_config(...)` opens the file at
`algorithm.tokenizer_config_path`, parses it as JSON, constructs
`EntityTokenizerConfig.model_validate(...)`, and then enforces the
following **five hard-fail rules** against a real entity-payload sample
(loaded from the configured dataset, or from
`datasets/tmp_entity_obs_full_step2200_named.json` if the simulator
hasn't been instantiated yet):

1. **Coverage.** Every feature in every `entity_table` referenced by
   the tokenizer (district, building, plus per-asset tables) must be
   either (a) matched by exactly one SRO type's `feature_patterns`,
   (b) consumed by `nfc.expression`, or (c) matched by an
   `excluded_features.patterns` entry. Unmatched features → `ValueError`
   listing each unmatched feature, its source table, and the
   recommended fix ("add to an existing SRO group, define a new SRO
   type, or add to `excluded_features.patterns`").
2. **Uniqueness.** No feature may match patterns from more than one
   SRO type, or simultaneously match an SRO type and
   `excluded_features.patterns`. Conflicts → `ValueError` listing the
   feature, the matching SRO types / exclusion, and which patterns
   matched.
3. **NFC sources exist.** `nfc.expression.left.feature` and
   `nfc.expression.right.feature` must appear in the tokenizer's
   `nfc.entity_table`. Missing → `ValueError`.
4. **Pattern compilation.** Each `feature_patterns` entry and every
   `excluded_features.patterns` entry must compile as a Python regex
   (`re.compile`). Bad regex → `ValueError` reporting the JSON path
   and the regex error message.
5. **Action-field coverage.** Every `ca_types[*].action_field` must
   appear in `action_names[building]` for every building reported by
   the dataset. Missing → `ValueError` (this is the existing §10.1
   post-condition, generalised).

These same rules also run at runtime inside `attach_environment` and
`on_topology_change` against the live `entity_specs`, so feature
additions / renames in a new dataset version trigger an explicit
failure, not silent drops.

Validation is performed in a single authoritative place
(`validate_config`) — no caller may instantiate
`AgentTransformerPPO` with a malformed tokenizer config.

### 13.5 Schema additions summary

- `EntityTokenizerConfig` — fields per §13.1:
  - `type_embeddings: Mapping[Literal["SRO","NFC","CA"], int]` (must equal
    `{"SRO": 0, "NFC": 1, "CA": 2}`).
  - `excluded_features: ExcludedFeaturesConfig` (with `patterns: List[str]`).
  - `nfc: NfcConfig` (`type_name`, `entity_table`, `expression`).
  - `nfc.expression: NfcExpressionConfig` (`op: Literal["subtract"]`,
    `left: NfcOperandConfig`, `right: NfcOperandConfig`).
  - `ca_types: Mapping[str, CaTypeConfig]` with at minimum `storage` and
    `charger`.
  - `sro_types: Mapping[str, SroTypeConfig]` — values discriminated on
    `cardinality: Literal["singleton","per_asset"]`. Singleton variants
    require `feature_patterns: List[str]` and `entity_table: str`.
    Per-asset variants require `entity_table: str`, `adapter_prefix: str`,
    optional `adapter_label: str`.
  - `validation: Mapping[str, Literal["fail"]]` — currently three keys:
    `unmatched_features`, `ambiguous_pattern_match`,
    `input_dim_mismatch`. All must be `"fail"` (no soft modes today —
    reserved for the future).
- `TransformerConfig` — `d_model`, `nhead`, `num_layers`, `dim_feedforward`,
  `dropout` (positive ints / unit-interval floats).
- `TransformerPPOHyperparameters` — `learning_rate`, `gamma`, `gae_lambda`,
  `clip_eps`, `ppo_epochs`, `minibatch_size`, `entropy_coeff`,
  `value_coeff`, `max_grad_norm`.
- `TransformerPPOAlgorithmConfig` — `name: Literal["AgentTransformerPPO"]`,
  `tokenizer_config_path: str`, `transformer: TransformerConfig`,
  `hyperparameters: TransformerPPOHyperparameters`. Added to
  `ProjectConfig.algorithm` discriminated union.
- Generic dynamic-topology guardrail (replaces MADDPG-specific check, see
  §12.4).

---

## 14. Export & Checkpoint Contract (dynamic topology)

### 14.1 ONNX export

Per-building, per call to `export_artifacts(output_dir, context=...)`. For
each building `b`:

- File: `<output_dir>/onnx_models/agent_<b>__topology_v<v>.onnx`
  where `v` is `self._entity_topology_version_at_export`.
- Inputs:
  - `encoded_obs`: `Tensor[float32, (1, obs_dim_b)]` — obs_dim_b is the
    current observation dimension for building `b`.
  - `segment_offsets`: `Tensor[int64, (n_segments_b + 1,)]` — prefix-sum
    index into a flattened concatenation of `feature_indices`. (We embed
    the layout into the graph because ONNX has no native ragged-tuple
    type.)
  - `segment_indices`: `Tensor[int64, (sum_of_segment_sizes,)]`.
- Output:
  - `actions`: `Tensor[float32, (1, n_ca_b)]` — deterministic policy means.
- Opset: 17, dynamic axes only on `obs_dim_b` (so the same file remains
  valid if encoded length changes inside the same topology, which it
  doesn’t today but is cheap to guard against).

### 14.2 Manifest entries

`export_artifacts` MUST return the canonical artifact payload required
by `utils/artifact_manifest.py:69-84` and validated by
`utils/bundle_validator.py:108-172`. The required keys are `format`
(top-level) and `artifacts` (a non-empty list, one entry per building,
each with `agent_index: int >= 0`, `path: str` pointing to a file that
exists under `output_dir`, and a JSON-serialisable `config` object).
For ONNX artifacts the file MUST end in `.onnx`. The number of entries
in `artifacts` MUST equal `topology.num_agents` (set by
`run_experiment.py:691-697` from the wrapper just before
`export_artifacts` runs).

Returned shape:

```json
{
  "format": "onnx",
  "artifacts": [
    {
      "agent_index": 0,
      "path": "onnx_models/agent_0__topology_v7.onnx",
      "format": "onnx",
      "config": {
        "building_id": "Building_1",
        "topology_version": 7,
        "obs_dim": 187,
        "n_sro": 18,
        "n_ca": 2,
        "sro_types": [
          "district_time", "district_weather_current",
          "district_weather_forecast", "district_carbon",
          "district_pricing_current", "district_pricing_forecast",
          "district_community_energy", "district_community_headroom",
          "district_community_history", "district_meta",
          "building_storage_state", "building_charging_phase_onehot",
          "building_charging_headroom", "building_charging_violation",
          "building_energy_current", "building_energy_history",
          "building_meta", "pv", "ev_connected"
        ],
        "ca_types": ["storage", "charger"]
      }
    }
  ],
  "tokenizer_config_path": "configs/tokenizers/entity_default.json",
  "supports_dynamic_topology": true,
  "agent_models": [
    {
      "building_index": 0,
      "building_id": "Building_1",
      "topology_version": 7,
      "onnx_path": "onnx_models/agent_0__topology_v7.onnx",
      "obs_dim": 187,
      "n_sro": 18,
      "n_ca": 2,
      "sro_types": [
        "district_time", "district_weather_current",
        "district_weather_forecast", "district_carbon",
        "district_pricing_current", "district_pricing_forecast",
        "district_community_energy", "district_community_headroom",
        "district_community_history", "district_meta",
        "building_storage_state", "building_charging_phase_onehot",
        "building_charging_headroom", "building_charging_violation",
        "building_energy_current", "building_energy_history",
        "building_meta", "pv", "ev_connected"
      ],
      "ca_types": ["storage", "charger"]
    }
  ]
}
```

The `format` and `artifacts` fields are mandatory (validated by
`bundle_validator`). The remaining fields (`tokenizer_config_path`,
`supports_dynamic_topology`, `agent_models`) are supplemental metadata
used by debugging and tooling and are passed through verbatim by
`build_manifest` (`utils/artifact_manifest.py:41`).

We export **only the topology version current at export time**; no
cross-topology weight portability is required for production.

### 14.3 Checkpoints

- File: `<output_dir>/checkpoints/transformer_ppo_step<step>.pt`.
- Payload (`torch.save`):
  ```python
  {
    "step": int,
    "topology_version": int,
    "config": dict,        # the algorithm sub-config
    "agents": [
      {
        "building_id": str,
        "tokenizer_state": state_dict,
        "backbone_state": state_dict,
        "actor_state": state_dict,
        "critic_state": state_dict,
        "optimizer_state": state_dict,
        "layout_signature": tuple[str, ...],   # sorted observation_names tuple
      }
    ],
  }
  ```
- `load_checkpoint` rejects loading into a topology whose
  `layout_signature` differs from the saved one, with an error pointing to
  the field that disagrees. Cross-topology resumption is **not** supported.

---

## 15. File Structure & Work Packages

### 15.1 Files

(See §2 for the full delta. Summary view here.)

```
algorithms/
├── agents/
│   ├── transformer_ppo_agent.py                 # PORT from gj/plan-c, surgical edits
│   └── transformer_ppo/
│       ├── __init__.py                          # PORT
│       ├── state_helper.py                      # PORT, drop marker registry
│       ├── update_helper.py                     # PORT verbatim
│       └── export_helper.py                     # PORT, drive shapes from layout
├── utils/
│   ├── entity_token_layout.py                   # NEW
│   ├── entity_observation_tokenizer.py          # NEW
│   ├── transformer_backbone.py                  # PORT, extend type embedding to 4
│   └── ppo_components.py                        # PORT verbatim

configs/
├── tokenizers/
│   └── entity_default.json                      # NEW
└── templates/
    └── transformer_ppo_entity_dynamic.yaml      # NEW

utils/
├── wrapper_citylearn.py                         # MODIFY (coordinator hook, generic guardrail)
├── wrapper_transformer/
│   ├── __init__.py                              # NEW
│   └── transformer_observation_coordinator.py   # NEW
└── config_schema.py                             # MODIFY (new models, JSON validation, generic guardrail)

algorithms/registry.py                           # MODIFY (register AgentTransformerPPO)

tests/
├── test_entity_token_layout.py                  # NEW
├── test_entity_observation_tokenizer.py         # NEW
├── test_transformer_backbone.py                 # PORT + extend
├── test_ppo_components.py                       # PORT
├── test_agent_transformer_ppo_entity.py         # NEW
├── test_wrapper_transformer_entity.py           # NEW
└── test_e2e_transformer_ppo_entity_dynamic.py   # NEW

docs/
└── specv2.md                                    # this document
```

### 15.2 Work packages

| WP | Component | Depends on | Deliverables |
|----|---|---|---|
| WP0 | Bring sources from `gj/plan-c` | — | Backbone, PPO components, helpers (see §2.2) |
| WP1 | `EntityTokenLayoutBuilder` | — | `entity_token_layout.py` + tests |
| WP2 | Tokenizer config + Pydantic schema + JSON validation | — | `entity_default.json`, `EntityTokenizerConfig`, validator hook |
| WP3 | `EntityObservationTokenizer` | WP1, WP2 | `entity_observation_tokenizer.py` + tests |
| WP4 | TransformerBackbone update (3-entry type embedding, new forward signature `(sros, nfc, cas)`) | WP0 | extended backbone + tests |
| WP5 | PPO components verification | WP0 | re-run ported tests |
| WP6 | `TransformerObservationCoordinator` | WP1 | coordinator + tests |
| WP7 | Wrapper hooks + generic guardrail | WP6 | wrapper diff + integration tests |
| WP8 | `AgentTransformerPPO` (CA-order check, attach lifecycle) | WP3, WP4, WP5, WP7 | agent + helpers + tests |
| WP9 | Algorithm template, registry, schema additions | WP2, WP8 | yaml, registry, schema |
| WP10 | E2E validation on `citylearn_three_phase_dynamic_assets_only_demo` | All | smoke test producing KPIs and ONNX |

Parallelizable: WP0, WP1, WP2.

---

## 16. Test Plan

### 16.1 `EntityTokenLayoutBuilder` (`tests/test_entity_token_layout.py`)

| Test | Verifies |
|---|---|
| `test_classifies_district_time_to_sro_singleton` | `district__hour` → `(sro, "district_time", building_id)` |
| `test_classifies_district_pricing_current_separately_from_forecast` | `district__electricity_pricing` → `district_pricing_current`; `district__electricity_pricing_predicted_2` → `district_pricing_forecast` |
| `test_classifies_district_carbon_separately_from_pricing` | `district__carbon_intensity` → `district_carbon`, not `district_pricing_*` |
| `test_classifies_building_storage_state_to_sro` | `electrical_storage_soc`, `electrical_storage_soc_ratio` → `(sro, "building_storage_state")` |
| `test_classifies_per_asset_pv_to_sro` | `pv::Building_1/pv::generation_power_kw` → `(sro, "pv", "Building_1/pv")` |
| `test_classifies_per_asset_ev_connected_to_sro` | `charger::B/c::connected_ev::soc` → `(sro, "ev_connected", "B/c")` |
| `test_classifies_per_asset_ev_incoming_to_sro` | `charger::B/c::incoming_ev::soc` → `(sro, "ev_incoming", "B/c")` |
| `test_classifies_storage_prefix_to_ca` | `storage::Building_1/electrical_storage::soc` → `(ca, "storage", "Building_1/electrical_storage")` |
| `test_classifies_charger_prefix_to_ca` | `charger::B/c::connected_state` → `(ca, "charger", "B/c")` |
| `test_nfc_segment_has_two_source_indices_and_subtract_op` | NFC segment has `feature_indices=(idx_nsl, idx_solar)`; `derived.op == "subtract"` |
| `test_nfc_source_features_not_in_any_sro_group` | `non_shiftable_load` and `solar_generation` do not appear in any SRO segment’s `feature_names` |
| `test_excluded_features_dropped_before_classification` | `district__topology_version` and legacy charger aliases appear in `excluded_feature_names` and not in any segment |
| `test_unmatched_feature_raises` | Inject a never-seen feature `district__some_new_feature` → `ValueError` listing the feature and table |
| `test_ambiguous_pattern_raises` | Two SRO types claim the same feature → `ValueError` |
| `test_sro_segment_order_follows_config_declaration` | `district_time` segment appears before `district_weather_current`, etc. |
| `test_per_asset_sro_segments_sorted_by_instance_id` | Two PVs → segments in lexicographic order of `instance_id` |
| `test_segment_overall_order` | `[sros…, nfc, cas…sorted_by_action_position]` |
| `test_topology_changed_when_names_differ` | Adding a new charger → True |
| `test_topology_unchanged_for_identical_names` | False |
| `test_layout_is_cached` | Repeated `build()` returns the same instance |
| `test_no_external_imports` | Module imports only stdlib + typing + re |
| `test_uses_real_sample_payload` | Build layout from `datasets/tmp_entity_obs_full_step2200_named.json` (passed through the adapter) succeeds for every building, with all 46 district + 38 building features accounted for |
| `test_coverage_accounting_matches_spec` | Per-table feature counts in §13.1 coverage table are reproduced exactly |

### 16.2 `EntityObservationTokenizer` (`tests/test_entity_observation_tokenizer.py`)

| Test | Verifies |
|---|---|
| `test_forward_shapes_baseline` | sro_tokens=[1,N_sro,d], nfc_token=[1,1,d], ca_tokens=[1,N_ca,d] for the sample payload |
| `test_forward_shapes_extra_charger` | ca_tokens=[1,2,d] when a second charger is present |
| `test_projection_is_per_type_no_new_params_on_topology_grow` | Adding chargers / PVs does NOT add new parameters |
| `test_index_select_handles_non_contiguous_sro_segment` | Mock encoded vector with sentinel values; verify `district_weather_forecast` slice gathers correct interleaved positions |
| `test_nfc_token_value_equals_subtract_op` | Mock `non_shiftable_load=5.0`, `solar_generation=2.0` → NFC scalar = 3.0 (pre-projection) |
| `test_nfc_input_dim_is_one` | `projections["building_nfc"].in_features == 1` |
| `test_input_dim_mismatch_raises` | type_input_dims != entity_specs dim → clear ValueError naming both sides and the affected type |
| `test_dtype_and_device_propagation` | Input cuda → output cuda; input float32 → output float32 |

### 16.3 TransformerBackbone update (`tests/test_transformer_backbone.py`)

| Test | Verifies |
|---|---|
| `test_type_embedding_table_size_3` | `nn.Embedding(3, d_model)` (SRO=0, NFC=1, CA=2) |
| `test_concat_order_sros_nfc_cas` | First N_sro slots are sro tokens, slot N_sro is nfc, last block is ca |
| `test_ca_embeddings_sliced_at_correct_offset` | Slice = `[N_sro + 1 : N_sro + 1 + N_ca]` |
| `test_pooled_includes_sro_and_nfc_tokens` | Pooling spans all tokens (sros + nfc + cas) |
| `test_gradient_flow_through_sro_tokens` | Backward through any sro token reaches the corresponding per-type projection |
| `test_gradient_flow_through_nfc_token` | Backward through nfc reaches `projections["building_nfc"]` and the NfcExpression source indices |

### 16.4 `AgentTransformerPPO` (`tests/test_agent_transformer_ppo_entity.py`)

| Test | Verifies |
|---|---|
| `test_attach_environment_builds_layouts` | One `BuildingTokenLayout` per building, cached |
| `test_attach_environment_validates_ca_action_order` | Mismatched `action_names` order raises |
| `test_attach_environment_does_not_treat_first_call_as_topology_change` | No PPO update, no buffer flush |
| `test_predict_action_count_matches_n_ca` | `len(actions[b]) == n_ca[b]` |
| `test_predict_returns_list_of_lists_of_float` | Matches `BaseAgent.predict` signature |
| `test_deterministic_uses_means` | Two deterministic calls → identical actions |
| `test_stochastic_samples` | Two stochastic calls → different actions (>1e-6) |
| `test_update_signature_uses_scalar_terminated_truncated` | Calling with `bool` works; calling with `List[bool]` raises a clear TypeError from upstream |
| `test_update_returns_none` | Matches `BaseAgent.update` |
| `test_on_topology_change_runs_update_then_flushes_buffer` | Buffer length == 0 after; PPO step counter incremented |
| `test_on_topology_change_rebuilds_layout` | New layout reflects new observation_names |
| `test_on_topology_change_validates_input_dim` | Raises if entity_specs dim != tokenizer dim |
| `test_supports_dynamic_topology_flag` | `AgentTransformerPPO.supports_dynamic_topology is True` |
| `test_checkpoint_round_trip_same_topology` | Save → load → identical weights |
| `test_checkpoint_load_rejects_layout_signature_mismatch` | Loading into a different topology raises |

### 16.5 Wrapper integration (`tests/test_wrapper_transformer_entity.py`)

| Test | Verifies |
|---|---|
| `test_coordinator_initialised_only_for_transformer_agent` | `RuleBasedPolicy` doesn't trigger coordinator hooks |
| `test_first_attach_does_not_trigger_topology_change` | `agent.on_topology_change` not called on initial reset |
| `test_topology_version_increment_triggers_rebuild` | Mock env where `meta.topology_version` flips → encoders + layouts + agent rebuilds happen exactly once |
| `test_topology_unchanged_does_not_rebuild` | Repeated steps with same `topology_version` → no rebuilds |
| `test_action_conversion_uses_entity_adapter_tables_only` | Wrapper outputs `{"tables": {"building": ndarray, "charger": ndarray}}`; no `"map"` key |
| `test_dynamic_guardrail_uses_supports_dynamic_topology_flag` | Setting flag to False on a dummy agent triggers the existing schema/runtime error; True does not |
| `test_maddpg_dynamic_error_message_unchanged` | The exact MADDPG error string remains so existing tests continue to pass |

### 16.6 Tokenizer JSON validation (`tests/test_entity_tokenizer_config_schema.py`)

| Test | Verifies |
|---|---|
| `test_valid_json_loads` | The shipped `entity_default.json` validates against `EntityTokenizerConfig` and passes all 5 hard-fail rules on the sample payload |
| `test_missing_ca_type_raises` | Removing `ca_types.charger` fails Pydantic validation |
| `test_unknown_field_raises` | Extra unknown field at any level fails (strict mode) |
| `test_validate_config_loads_tokenizer_json` | Top-level `validate_config(...)` invokes the JSON loader and reports the JSON path on failure |
| `test_rule1_unmatched_feature_fails` | Add a fake feature `district__some_new_thing` to the sample payload → `ValueError` listing it |
| `test_rule2_ambiguous_pattern_fails` | Two SRO types with overlapping patterns → `ValueError` listing the colliding feature and types |
| `test_rule3_missing_nfc_source_fails` | Tokenizer references `nfc.expression.left.feature = "does_not_exist"` → `ValueError` |
| `test_rule4_bad_regex_fails` | A `feature_patterns` entry like `"^[unclosed"` → `ValueError` reporting the JSON path and regex error |
| `test_rule5_missing_action_field_fails` | Set `ca_types.charger.action_field = "no_such_action"` → `ValueError` |
| `test_excluded_feature_pattern_removes_topology_version` | After validation, `district__topology_version` is in `excluded_feature_names` for every building |
| `test_excluded_feature_cannot_match_an_sro_type` | Same feature in both `excluded_features.patterns` and an SRO `feature_patterns` → rule 2 conflict |

### 16.7 End-to-end (`tests/test_e2e_transformer_ppo_entity_dynamic.py`)

Run on `citylearn_three_phase_dynamic_assets_only_demo` for a small horizon.

| Test | Verifies |
|---|---|
| `test_smoke_run_completes` | 200 steps, no crash |
| `test_actions_in_valid_range` | All actions in `[-1, 1]`, no NaN |
| `test_topology_changes_observed_during_run` | At least one `topology_version` increment in 200 steps (the assets-only demo guarantees this) |
| `test_kpi_files_generated` | `runs/jobs/<id>/results/{result,summary}.json` exist |
| `test_artifact_manifest_includes_onnx_per_building` | `artifact_manifest.json` lists per-building ONNX with the naming from §14.2 |
| `test_buffer_flush_on_topology_change_does_not_crash` | The PPO update triggered by topology change runs and the agent continues training |

---

## 17. Decisions Log

| # | Question | Decision |
|---|---|---|
| 1 | Token boundary signal | Per-feature origin prefixes emitted by `EntityContractAdapter`, classified once per topology by `EntityTokenLayoutBuilder`. No marker values in tensor. |
| 2 | Topology-change detection | `meta.topology_version` increment, exclusively. No asset/feature counting. |
| 3 | Per-building vs centralized | Per-building, like v1. |
| 4 | Action payload | `{"tables": {"building": ndarray, "charger": ndarray}}` only. **No `"map"` field**, matching the current adapter (`utils/entity_adapter.py:548-552`). |
| 5 | CA token order | Sorted by position in `action_names[building]`, validated at `attach_environment` and `on_topology_change`. |
| 6 | Graph structure usage | Edges consumed by the wrapper for slicing only. The Transformer treats the result as a set of typed tokens. |
| 7 | EV / PV handling | Per-asset SRO tokens. EV split into `ev_connected` and `ev_incoming` (matching `connected_ev` / `incoming_ev` adapter labels). PV is its own SRO type. |
| 8 | RL algorithm | PPO, as in v1. Rollout buffer, GAE, clipped surrogate, K epochs. |
| 9 | Encoder rebuilds | `set_encoders()` is called every `_apply_entity_layout`; cheap (placeholder `NoNormalization`). Agent layout rebuilds are gated on `topology_version` change. |
| 10 | Type embeddings | 3 families: SRO, NFC, CA. CTX from earlier drafts collapsed into SRO with per-type projections distinguishing semantic groups. |
| 21 | NFC scope | Strict: NFC = single scalar per building, value of `non_shiftable_load - solar_generation` computed by `NfcExpression`. Other building-table features become singleton SRO types (`building_storage_state`, `building_charging_*`, `building_energy_*`, `building_meta`). |
| 22 | SRO grouping | Driven by `feature_patterns` (regex) in tokenizer JSON. District split into 10 semantic SROs (time, weather current/forecast, carbon, pricing current/forecast, community energy/headroom/history, meta). Pricing and carbon are separate. Current and forecast are separate. |
| 23 | Feature exclusion | `excluded_features.patterns` in tokenizer JSON. Excludes `topology_version` (redundant with `meta`) and legacy charger aliases (redundant with per-charger CA tokens). Adding/removing features is config-only. |
| 24 | Validation rules | Five hard-fail rules in §13.4 enforced both at config-validation time and at runtime on every topology change. No silent dropping of features. |
| 11 | Reuse policy | Backbone, PPO components, helpers ported from `gj/plan-c@3e9a673`. Marker-based modules replaced. |
| 12 | Cross-topology checkpoints/exports | Not supported. Checkpoint loader rejects layout-signature mismatch. ONNX file name encodes `topology_version`. |
| 13 | Dynamic-topology guardrail | Replaced by generic `agent.supports_dynamic_topology` flag, enforced both in `validate_config` and at runtime. MADDPG error message preserved. |
| 14 | Tokenizer input dims | Read from `entity_specs` at attach time; tokenizer config’s `input_dim_fallback` exists only for inference. |
| 15 | Tokenizer JSON validation | Performed inside `validate_config(...)` after YAML validation succeeds. |
| 16 | Hyperparameter naming | `learning_rate` is canonical. `lr` is never used as a config key. |
| 17 | `terminated`/`truncated` types | Scalar `bool`, matching `BaseAgent.update` (`algorithms/agents/base_agent.py:33`). |
| 18 | `attach_environment` arguments | Keyword-only, matching `BaseAgent.attach_environment` (`algorithms/agents/base_agent.py:58-65`). |
| 19 | Encoder dimension expansion | Not supported in v2; encoded length equals raw `observation_names` length (current entity-mode behaviour). No `encoded_index_map`. |
| 20 | Portability | `EntityTokenLayoutBuilder` stays pure-Python; `EntityObservationTokenizer` (torch) lives only in training repo. |
