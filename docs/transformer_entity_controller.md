# Transformer Entity Controller Contract

> Status: **Current implementation**
> Scope: Shared entity and Transformer contracts for controllers in this repository.
> Last reviewed: 2026-08-16
> Reviewed `main` commit: `f2809313c7550405eccd4d9b276adbf8a9103a5c`
> Maintainer: Algorithms maintainers

This document is the algorithm-independent reference for an entity-interface
Transformer controller. The [Transformer PPO specification](transformer_ppo_spec.md)
adds PPO, behavior-cloning, safety, checkpoint, and export rules. Future
Transformer algorithms should reuse this document and state their differences.

The authoritative sources are the runtime [entity adapter](../utils/entity_adapter.py),
[wrapper](../utils/wrapper_citylearn.py), [layout builder](../algorithms/transformer_shared/entity_token_layout.py),
[tokenizer](../algorithms/transformer_shared/entity_observation_tokenizer.py), and
[backbone](../algorithms/transformer_shared/transformer_backbone.py), together with
the shipped [tokenizer configuration](../configs/tokenizers/entity_default.json)
and its [schema fixture](../configs/tokenizers/fixtures/entity_obs_sample.json).
The tested simulator package is `softcpsrecsimulator==1.5.6` from
[`requirements.txt`](../requirements.txt). Historical PRs, including closed
PR 23, are not normative.

## 1. Entity payload and adapter boundary

The simulator runs with `simulator.interface: entity`. A reset or step returns
an entity payload with these top-level fields:

| Field | Contract |
|---|---|
| `tables` | Entity tables. Each table has feature names, rows, and values. |
| `edges` | Relationships used to select assets for each building. |
| `meta` | Runtime metadata. `meta.topology_version` identifies layout invalidation. |

The implementation requires entity tables, edges, metadata, dynamic topology
signals, and active action-table shapes. These concepts are separate from the
simulator package version.

`EntityContractAdapter` owns conversion in both directions:

```text
entity payload -> one observation vector per building -> controller
controller action vectors -> entity action tables -> simulator
```

The controller does not parse tables or edges. It receives one vector and one
action vector per building. The wrapper supplies observation names, action names,
spaces, building names, entity specifications, and encoded observation names
through `attach_environment`.

The active action table shape and row order come from the environment entity
specification and action space. The adapter currently emits a tables-only action
payload. It does not emit a `map` field.

## 2. Per-building observation and action names

The adapter emits names in this order for each building:

1. District features, prefixed `district__`.
2. Building features, unprefixed.
3. Storage blocks, `storage::<asset_id>::<feature>`.
4. PV blocks, `pv::<asset_id>::<feature>`.
5. Deferrable-appliance blocks,
   `deferrable_appliance::<asset_id>::<feature>`.
6. Charger blocks, followed by connected-EV and incoming-EV context blocks for
   each charger.
7. The four active-asset counters for chargers, storage, PV, and deferrable
   appliances.
8. Legacy charger aliases, or their zero-valued defaults when no charger is
   active, followed by `electric_vehicle_is_flexible`.
9. Conditional `minute` and `solar_generation` compatibility aliases when the
   corresponding canonical names are absent.

The exact list is topology-dependent. The tokenizer layout uses feature-origin
names, not numeric values, to map these names to token segments. Excluded names
remain in the adapter vector but are not consumed by the controller.

The controller returns `List[List[float]]`. For building `b`, action position
`i` refers to the CA token whose `action_field` matches `action_names[b][i]`.
This position invariant applies to every controller using this contract.

## 3. Token-family terminology

| Family | Meaning | Cardinality | Action output |
|---|---|---:|---:|
| SRO | Shared read-only context | Variable | None |
| NFC | Non-flexible context | Exactly one per building | None |
| CA | Controllable asset | Variable | One value per token |

Examples of SRO types include district time, weather, pricing, building energy,
PV, deferrable appliances, connected EV, and incoming EV. The current tokenizer
classifies deferrable appliances as SRO context. Current CA types are `storage`
and `charger`. A future controller that acts on deferrable appliances must add
an explicit CA type and action-order contract. The NFC type is `building_nfc`,
computed as `non_shiftable_load - solar_generation`.

The backbone uses three fixed type IDs: `SRO=0`, `NFC=1`, and `CA=2`.

## 4. Layout construction and ordering

`EntityTokenLayoutBuilder` constructs a deterministic `BuildingTokenLayout` from
`building_id`, observation names, and action names.

The construction order is:

1. Remove names matching `excluded_features.patterns`.
2. Find the two configured NFC source names and build one subtract expression.
3. Classify every remaining name as a configured singleton SRO, per-asset SRO,
   or CA. Unmatched names fail.
4. Order SRO segments by tokenizer declaration order and instance ID.
5. Insert the NFC segment.
6. Order CA segments by action names. Exact action-field matches and simulator
   action-field prefixes with asset suffixes are supported.
7. Verify that CA segment order matches the action vector position by position.

The layout records segment family, type, instance ID, feature indices, feature
names, the NFC expression, excluded names, and CA action fields. Non-contiguous
feature indices are valid and are selected with `torch.index_select`.

The tokenizer configuration uses strict Pydantic models. The validator defines
five rules:

1. Every feature in the NFC table and singleton-SRO tables is covered by an
   exclusion, NFC source, or singleton-SRO rule.
2. Singleton SRO patterns are not ambiguous.
3. Both NFC source features exist.
4. All configured regular expressions compile.
5. Every declared CA action field is covered by the supplied action-field set.

`validate_config` runs the feature rules against the pinned fixture. It supplies
the configured CA action fields as a synthetic action-field set for rule 5; it
does not inspect live simulator action names. On environment attachment, the
layout builder enforces actual CA count, order, and exact-or-prefixed field
matching against each building's active `action_names`.

Rule 1 does not validate each per-asset SRO or CA column. The layout builder
classifies those blocks by their adapter prefixes. Tokenizer construction and
runtime topology handling then reject a per-type feature width that does not
match the existing projection.

At runtime, a changed layout re-runs the layout checks against the active names.
The controller intentionally skips startup-only rule 5 on runtime mutation. An
asset can disappear while its configured CA type remains valid.
Feature-schema drift still fails.

## 5. Encoding and tokenizer contract

The current TPPO templates use the `minmax_space` encoding profile. This profile,
and disabled normalization, preserve feature order and width between the entity
observation vector and the controller input. Min-max normalization changes
values but not feature positions. A layout built from raw observation names can
therefore index the TPPO encoded vector directly.

MADDPG-style encoding profiles do not have this guarantee. They can derive,
remove, or rename features. A future Transformer controller that uses one of
these profiles must define tokenizer coverage over the encoded feature set and
build its layout from `metadata.encoded_observation_names` or the matching entry
in `metadata.profiled_encoded_observation_names`.

`EntityObservationTokenizer` has one linear projection per declared type, not
per asset instance. A new instance of an existing type reuses the type weights.
NFC is reduced to one scalar before its projection. Its projection input width
is always one. Per-asset input widths use the configured fallback when a type
has no active instance, so a later first instance can reuse the projection.

Dynamic cardinality is supported when all of these remain true:

- The type is declared in the tokenizer configuration.
- Its feature width matches the existing projection.
- The active layout is covered by the runtime rules.
- CA order still matches the action names.

Adding or removing instances is portable. Adding a new type, changing a type's
feature width, renaming a feature, or changing the NFC expression is a schema
change and requires a new compatible model or a restart from scratch.

## 6. Transformer backbone contract

The shared sequence is:

```text
[SRO tokens, NFC token, CA tokens]
        -> type embeddings -> Transformer encoder
```

The backbone accepts `[batch, token_count, d_model]` banks with variable SRO and
CA counts. It returns CA embeddings in CA order and a mean-pooled representation
over all tokens. The CA embeddings drive a controller's action head. The pooled
representation is a per-building state representation. It is not a joint
community representation and is not sufficient by itself for a centralized
multi-agent critic. An algorithm with a centralized critic must define how it
combines buildings, actions, and any community context.

The backbone has no fixed asset-count parameter. `d_model`, attention heads,
layer count, feed-forward width, and other neural settings remain fixed for a
trained model.

## 7. Topology signals and wrapper responsibilities

The wrapper reads `meta.topology_version` through the entity adapter. A version
change causes the wrapper to rebuild observation names, spaces, encoders, action
bounds, and model metadata. The wrapper then calls `attach_environment` with the
new per-building contract.

The wrapper also enforces algorithm capability. A controller in dynamic entity
mode must declare dynamic-topology support. Controllers without that capability
fail when the topology mutates.

For a controller that supports transactional adaptation, the wrapper can:

- snapshot wrapper and controller state;
- record the old-layout transition before replacing the model layout;
- attach the new environment metadata;
- restore both snapshots if adaptation fails.

The exact rollout and optimizer behavior belongs to the algorithm document.

The controller may receive a building-count change. This is a full rebuild
boundary. Per-building model state cannot be matched safely when the number or
order of buildings changes unless the algorithm defines an explicit remapping.

## 8. Shared action and deployment invariants

- One action value exists for each active CA token.
- Action output order equals the environment action-name order.
- Action values are passed back through the adapter without changing building
  or asset row ownership.
- Compatibility signatures are algorithm-specific. Current BC signatures
  include ordered segments and CA action names. Current TPPO checkpoint layout
  signatures use sorted observation names and validate action names and bounds
  separately.
- Feature-schema changes are not silently adapted.
- An exported graph must declare its input feature width and layout metadata.
  Layout indices are topology-specific; a deployment target must select a model
  whose metadata matches its current layout.
- Any controller-side safety or post-processing layer must be declared outside
  the neural graph unless the export implementation proves that it is embedded.

These invariants are the reuse boundary for future Transformer MATD3 work. They
do not prescribe replay, target-network, exploration, or critic semantics.

## 9. Evidence and change control

The shared contract is covered by:

- `tests/test_entity_token_layout.py`
- `tests/test_entity_observation_tokenizer.py`
- `tests/test_transformer_backbone.py`
- `tests/test_entity_tokenizer_config_schema.py`
- `tests/test_wrapper_entity_mode.py`
- `tests/test_entity_adapter.py`

Regenerate `configs/tokenizers/fixtures/entity_obs_sample.json` when the
simulator schema, entity table columns, asset types, or adapter emission order
changes:

```bash
python scripts/dump_entity_obs_sample.py \
  --config configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml \
  --output configs/tokenizers/fixtures/entity_obs_sample.json
```
