# Ownership, contributions and provenance

Changes follow the common contract and require a pull request plus the repository's
required tests. Roles below define review responsibility; they do not assign
people or teams that have not accepted ownership.

| Area | Responsibility |
| --- | --- |
| `algorithms/configuration`, catalogue, registry, contracts | Core maintainers: job inputs, capabilities, migration and extension compatibility |
| `algorithms/runtime` | Core maintainers: simulator lifecycle, execution, resources, tracking and final outputs |
| Agent packages and their parameter schemas | Algorithm authors: numerical correctness, buffer/optimizer semantics and truthful capabilities |
| Simulator package and datasets | Simulator/data owners: observations, actions, termination, physical timing and dataset provenance |
| Artifact/inference consumers | Export/inference maintainers: format compatibility and deployment certification |
| Research commands in `scripts/` | Experiment authors: campaign selection, collection, multi-seed analyses and benchmark interpretation |

New agents should use the [extension guide](agents.md). Common changes should
include a focused regression test and run the affected representative job.
Keep generated results in `runs/`, local campaign recipes outside tracked
reusable templates, and accepted architecture decisions in documentation.
IDE settings and generated build/metadata directories are not repository inputs.

## Provenance and licensing

The [branch consolidation record](../consolidation/README.md) and its inventories
retain the source commits, original branches and recovery decisions. The archived
heads preserve student work and authorship. Stage 2 moves the existing runtime
and validation implementation into explicit owners; it introduces no external
algorithm implementation. Replay/rollout mathematics are not being replaced.

The audited repository has no root license granting general reuse. This document
does **not** supply one, infer contributor permission, or relicense dependencies.
A maintainer/institution decision is required before publishing a new reuse
license. Simulator/CityLearn, PyTorch and other installed dependencies retain
separate licenses; their package metadata and source notices remain authoritative.
Recovered code provenance is retained in Git and the consolidation inventory;
publication of that code does not itself establish a new license grant.

No speculative CODEOWNERS identities have been added. The repository's existing
access and review controls continue to determine who can merge.
