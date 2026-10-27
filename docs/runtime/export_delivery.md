# Stage 4 — final outputs and portable policy contract

This delivery implements the 21 actions assigned to stage 4 of the original
100-action plan. It keeps training algorithms behind the common runtime and
closes the Algorithms producer/reference-consumer boundary. It does not certify
scientific quality or the separate production inference application.

## Acceptance matrix

| Action | Implementation and evidence |
|---|---|
| A04.01 | Nullable export flags resolved in schema and controller; real final/diagnostic job files checked. |
| A04.02 | Normal profile writes requested KPIs only at final deterministic evaluation; time series opt-in. No final evaluation means no eligible scorecard. |
| A04.03 | Progress metrics, KPIs, time series, checkpoints and bundle remain independent requests; common job profile matrix covers none/checkpoint/bundle/both. |
| A04.04 | BAU calculation retained independently of BAU time-series writing. Final KPI jobs work without BAU files. |
| A04.05 | Explicit diagnostic profile: three training exports and 128 MiB default limits; diagnostic files excluded from canonical scorecard. The byte guard runs after an episode's export and may overshoot by that export. |
| A04.06 | `algorithms.outputs.v2`, `algorithms.summary.v2`, `algorithms.kpis.v1`: scope, units or explicit undeclared unit, missing-value reasons, final evaluation identity and CSV hash. |
| A04.07 | 27 measured jobs, 24/168/720 steps, three profiles and repetitions; raw timings/bytes in `export_benchmark.json`. |
| A07.01 | `algorithms.bundle.v2`: explicit execution graph, node/artifact identities and pipeline/algorithm summary; legacy contract retained with re-export migration rule. |
| A07.02 | Strict JSON and typed nonfinite numeric codec; infinity keeps unbounded meaning. Duplicate keys/nonstandard JSON rejected. |
| A07.03 | Initial and resolved configuration, final topology, inputs and frozen site mapping separated; laboratory mapping remains explicitly unmapped. Node identity prevents repeated local index collisions. |
| A07.04 | Wheel/build commit, dirty state where known, image digest injection or explicit absence, package/runtime versions, inputs, seed streams, reward and evaluated policy/checkpoint evidence. |
| A07.05 | Sealed dependency inventory, embedded tokenizer/normalizers, portable offline restoration sidecars and oracle schedules; input preflight preserves missing-model rejection. Known specialised runtime families are rejected before training when deployability is required. |
| A07.06 | Confined POSIX paths, exact inventory, hashes, no traversal/absolute paths/symlinks; relocated bundles execute. |
| A07.07 | Shared ONNX and JSON fixtures, nine transport corruption cases plus structural inconsistency regressions; v1 migration and v2 compatibility documented for SOF-100/101. |
| A08.01 | Training, export and deployment statuses separate; export failure preserves completed training. KPI-only jobs remain valid. |
| A08.02 | Registered JSON policy/state executor preserves RBC/BAU semantics and package version. Unknown types fail instead of producing zero actions. |
| A08.03 | Ordered raw/encoded observation metadata, bounds/encoders, learned normalization/permutations, scalar context, temporal state and root action clipping retained; undeclared units explicit. Unsupported variants remain restricted. |
| A08.04 | Bounded chronological observations/outputs captured from the actual final policy; real ONNX/JSON replay, shape/finiteness/tolerance checks and root submitted actions. |
| A08.05 | ONNX checker and ONNX Runtime CPU load every ONNX component; actual opset/runtime versions recorded. CPU parity is not GPU/ARM certification. |
| A08.06 | Pipeline/Ensemble graph, stage/member routing and signal context tested as a whole; a worker file is not a standalone replacement for its hierarchy. |
| A08.07 | Ten specialised/reference families keep explicit research status; no automatic promotion from successful training or an ONNX component alone. |

## Verification and support boundaries

The common job suites exercise all eight standard actor–critic names, both entity
Transformers, recovered Transformer MADDPG and entity IQL/CQL, BuildingAgent,
all rule/signal variants, manager pipelines, capacity coordination and TI-MARL.
Flat offline BC/IQL and the three oracle replays have family-level common export
coverage with their historical/synthetic fixtures. This accounts for all 38
registered names without treating every family as interchangeable.

The tested base configurations cover **14 ONNX names, 14 executable JSON names
and 10 specialised research names**. See the [format matrix](../inference_bundle.md)
for exact families and composition restrictions. Neural exports include learned
preprocessing where required. External residual teachers, safety projections,
price conditioning and specialised layouts retain their existing guards; each
new variant must establish its own equivalence rather than inherit a blanket
certificate from the family name.

Final checks include the full unit/regression suite, the four common end-to-end
job suites, lint/format/type checks, a wheel installed outside the checkout, and
execution of relocated ONNX/JSON fixtures. Check totals and immutable integration
references are recorded in the PR and stage delivery record. Existing skips are
four unavailable CUDA checks and fourteen unavailable historical assets.

The reference consumer accepts ordered raw/encoded vectors at the declared
boundary. Mapping application requests to those vectors and incorporating the
contract into the separate inference/backend repositories remain SOF-100 and
SOF-101. Fixture success here does not claim those applications were changed.

Stage 5 retains the original 18 actions: exhaustive family/scientific/hardware
certification, performance, dependency review and final acceptance. No new stage
was inserted and no algorithm tuning was started here.
