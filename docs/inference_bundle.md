# Policy bundle contract v2

Algorithms 0.4 produces `algorithms.bundle.v2` manifests through the common job
finalizer. Every registered family uses the same transaction. Training completion,
successful export and deployment eligibility are distinct fields. An ONNX file
alone does not establish that the whole evaluated policy was exported.

The reference consumer and replay command ship with Algorithms:

```sh
pip install '.[training,export]'  # from an Algorithms 0.4 checkout
algorithms-bundle /path/to/job/bundle
```

This validates paths, inventory hashes, composition and compatibility, loads the
actual models with ONNX Runtime CPU (or the registered JSON policy adapter), and
replays chronological samples captured during deterministic evaluation. Missing
files, changed models, unknown versions, incompatible shapes and differing actions
fail. There is no unknown-policy fallback to zero actions.

## Contents and identity

`artifact_manifest.json` describes:

* `manifest_version: 2` and `contract: algorithms.bundle.v2`;
* `pipeline` and an explicit `algorithm` summary; `execution.nodes` identifies each
  stage/member independently of its local `agent_index`;
* `agent.artifacts`, with unique `artifact_id`, `node_id`, portable `path`, format
  and deployment status;
* ordered raw/encoded observation names, encoders, entity layout, action names and
  bounds in `environment`; units absent from the simulator remain undeclared;
* `evaluation`, named seed streams, input identity, final topology and reward;
* `build`: code commit, package/runtime versions and image digest when supplied;
* `provenance`: immutable workspace/community/dataset/site mapping, retaining
  `mapping_status: unmapped` for laboratory data without platform IDs;
* `files`: SHA-256 for every delivered file, including sidecars and reference data;
* `evidence`: initial configuration, resolved pre-run configuration, input-file
  inventory and final KPI scorecard. Final topology is separate from resolved
  configuration so runtime membership changes do not rewrite its meaning;
* `deployment`: eligibility, consumer contract and reasons for restrictions.

Wheels embed the commit; Docker/Union CI injects `ALGORITHMS_COMMIT`. The launcher
may inject `ALGORITHMS_IMAGE_DIGEST` at execution. An image cannot contain its own
final registry digest: absent injection is recorded as `not_provided_by_launcher`,
never inferred from a mutable image tag.

Loadable paths are relative POSIX files inside the bundle. Absolute paths, parent
traversal, symlinks, unknown files and unsealed dependencies are rejected. Training
input paths in provenance are source references, not files the consumer should
open. Hashes provide integrity relative to a trusted manifest, not authenticity.

JSON is strict. Numeric limits use `{"$number":"+inf"}` or `"-inf"` for unbounded
limits, and `"nan"` for unavailable values where needed. They are not converted to
zero. KPI missing values use `value: null` with a reason. Executable JSON state has
a versioned array/mapping/record codec with a fixed allowlist, not arbitrary
imports, expressions or pickle execution.

## Execution and preprocessing

`algorithms.export.consumer.BundleRuntime` consumes **ordered vectors**, in the
raw/encoded form declared for each node. Application adapters use the manifest's
names and preprocessing metadata to construct those vectors. Mixed pipelines may
supply both raw and encoded observations. Ensemble context vectors are routed to
the corresponding member. `episode_step` is explicit for temporal signal policies.
Final clipping/rejection follows the exported action bounds and policy.

Pipelines run every stage in order and pass its signal onward. Ensembles execute
each member on its observation slice. Joint Transformer graphs consume all
members together. Choosing a file using only `agent_index` is not a v2 composition
executor.

References contain policy emissions and, at the root, submitted actions after
common clipping. `verification.json` records count, tolerances, maximum error,
actual opsets and runtime/provider. Defaults: absolute tolerance `1e-5`, relative
`1e-4`, at most 32 evaluation steps per layout. This is bounded regression evidence,
not proof for every possible observation or hardware. Topology changes restart
references for the exported layout; fixed-layout ONNX files do not promise to
adapt to future membership changes.

## Delivery formats

| Family | Export and consumer status |
|---|---|
| MADDPG, MATD3, MASAC, PPO, TD3, IPPO, MAPPO, HAPPO | ONNX; base policies compared with actual evaluation. PPO/TD3 ensembles use explicit composition. |
| TransformerMADDPG | Joint ONNX including learned CA/SRO/NFC tokenizer and actor, static layout; research checkpoint retained for restoration. |
| AgentTransformerPPO / AgentTransformerMATD3 | ONNX per exported building/layout, tokenizer embedded by existing exporter; base policies verified. |
| IQLEntityAgent / CQLEntityAgent | ONNX including learned mean/std and observation/action permutation; restoration sidecars retained. |
| BuildingAgent | ONNX with learned observation normalization and appended scalar context; tested with a signal manager and two members. |
| RBC, BAU/Normal, random and signal-aware variants | `algorithms_policy` executable JSON state, exact registered implementation and Algorithms version. Old generic rule descriptions are not equivalent executable policies. |
| FixedPriceSignal / CausalPriceSignal | Executable JSON state, including schedules/interval state; verified in whole pipelines. |
| CCLevel1, CCLevel2, CommunityCoordinator | ONNX components retained and loadable; generic consumer still lacks the full context/history/interval adapter. Explicit research status. Downstream workers are not advertised as independent replacements for the hierarchy. |
| CCLevel2Capacity | Research Torch payload and restoration metadata. |
| BCAgent / flat IQLAgent | Historical Building_5 policy plus OfflineRBC companions; research export, not an interchangeable actor for arbitrary communities. |
| TIMARL | Typed actor/deployment payload, interface registry and feasibility metadata retained; requires typed runtime/compiler/projector adapter, research status in this consumer. |
| Oracle replay policies | Reference schedules/perfect-foresight research artifacts; no automatic production promotion. |

Variants may add external residual teachers, price conditioning, local safety or
other runtime state. Existing exporter guards remain active. Such variants must
export their complete semantics or retain explicit restrictions; the core never
drops dependencies to obtain a passing ONNX. `bundle.require_deployable: true`
fails an unverified/research delivery while recording that training completed.

Rules/schedules use executable JSON, neural policies use ONNX where equivalent,
and specialized research formats retain their requirements. All share manifest,
inventory, provenance and status semantics.

## Version compatibility and inference handoff

The [v1 description](inference_bundle_v1.md) remains for historical bundles.
Legacy structural validation remains available with portable path checks. The
new executable consumer rejects v1: re-export from its checkpoint/model directory
using an evaluation job to obtain references and a v2 manifest. Renaming a version
field cannot manufacture missing preprocessing or evaluation evidence.

Unknown major versions fail. Optional descriptive fields may be added within v2;
execution semantics changes require a new version. JSON policies pin the
Algorithms implementation version because behavior lives in that package.

The shared [fixtures](../contracts/bundle/v2/README.md) are the handoff for
[SOF-100](https://linear.app/softcps/issue/SOF-100) and
[SOF-101](https://linear.app/softcps/issue/SOF-101). Integration into the separate
inference service and backend remains tracked there. This delivery verifies the
Algorithms reference consumer, not an unmodified production inference service.
