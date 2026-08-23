# Transformer MATD3 architecture decisions

These ADRs record why the Transformer MATD3 architecture has its current shape.
They are historical decision records, not configuration references. The
implementation and Pydantic schema remain authoritative for current field
names and defaults.

Read the [operational guide](../transformer_matd3.md) first. Read the
[technical specification](../transformer_matd3_spec.md) when changing behavior
or invariants. Consult an ADR when reconsidering its decision or trade-offs.

| ADR | Decision |
|---|---|
| [0001](0001-shared-package-extraction.md) | Shared Transformer/entity package ownership |
| [0002](0002-actor-ownership.md) | Per-building actor ownership |
| [0003](0003-centralized-twin-critic.md) | Centralized twin-critic architecture |
| [0004](0004-backbone-upgrades.md) | No backbone upgrades in v1 |
| [0005](0005-replay-representation.md) | Encoded replay and full layout signatures |
| [0006](0006-batching-policy.md) | Signature-homogeneous replay batches |
| [0007](0007-behavior-cloning.md) | BC-A and BC-B boundaries |
| [0008](0008-residual-policy.md) | Residual-policy composition |
| [0009](0009-local-price-adapter.md) | Pre-tokenization local price conditioning |
| [0010](0010-checkpoint.md) | Strict format-5 checkpoints |
| [0011](0011-onnx-export.md) | Per-building, per-topology ONNX export |
| [0012](0012-schema-registry-wrapper.md) | Schema, registry, and wrapper integration |

## Change policy

- Keep accepted ADRs. Do not rewrite their original rationale to match a new
  decision.
- Add a superseding ADR when an accepted decision changes.
- Update the guide, specification, templates, tests, and code in the same PR.
- Replace volatile line-number references with stable file and symbol names
  when touching an ADR.
