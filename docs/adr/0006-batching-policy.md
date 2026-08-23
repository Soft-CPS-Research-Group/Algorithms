# ADR-0006 — Batching policy

Status: accepted
Date: 2026-08-18
Depends on: ADR-0005
Related: ADR-0004, ADR-0007

## Context

The Transformer backbone processes tokens in one homogeneous shape per
forward pass because it has no padding-mask contract. TPPO's
`RolloutBuffer.get_batches` also stacks tensors directly, which requires
uniform batch shapes.

We must decide:

- 6a — batching strategy.
- 6b — behavior when the current-signature bucket is under-full.
- 6c — sampler API surface.

## Plain-language

Analogy: filing cabinet with folders labeled by schema fingerprint.
For any training step, we pull records only from the folder matching
today's fingerprint. Old folders remain for analytics and future extensions.

## Decisions

### 6a — batching strategy: B1 signature-bucketed

Every sampled batch is drawn from exactly one signature bucket. The
default bucket is the current LayoutSignature. Batches are always
shape-homogeneous.

### 6b — under-full bucket: U1 wait

If the current-signature bucket has fewer than `batch_size`
transitions, the learning step is skipped. This matches the established MATD3
replay warm-up pattern. Bounded wait — normal
environment stepping refills the bucket within `~batch_size` steps
after a topology change.

### 6c — sampler API: expose iterate-all-buckets

The replay wrapper exposes:

- `sample(signature: LayoutSignature, k: int) -> Batch`
- `signatures() -> Iterable[LayoutSignature]`
- `bucket_size(signature: LayoutSignature) -> int`
- `total_size() -> int`

BC-A (ADR-0007) uses current-signature sampling. Analytics and
future extensions may iterate all buckets.

## Consequences

- `SignatureBucketedReplayBuffer` is a dedicated replay implementation.
- Training step gate:
  `if buffer.bucket_size(current_sig) < batch_size: return`.
- No backbone upgrades required (ADR-0004 confirmed no-op).

## Evidence

- Legacy MATD3 gates learning on replay size.
- TPPO `RolloutBuffer.get_batches` stacks batch tensors directly.

## Future improvements

- B2 padded + masked batches, contingent on ADR-0004 upgrades.
- U2 reduced-batch mode if U1's dead time proves operationally
  costly. Requires validation that TD3 twin-critic training remains
  stable under smaller batches.
- U3 nearest-bucket fallback — research-flavored; requires a defined
  "similarity" metric between LayoutSignatures.
