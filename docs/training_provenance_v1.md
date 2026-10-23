# Training provenance v1 — 2026-10-23 (SOF-122)

Artifact manifest v1 gains an additive `provenance` section and file SHA-256 index.
The same provenance and manifest digest appear in result.json and summary.json.
The exact saved runtime config is hashed separately from the original catalogue
config. A result is completed only after artifact export and contract validation.

The reserved `tracking.tags.energaize_training` object is supplied by Training,
with workspace/community IDs, catalogue configuration identity, frozen dataset
versions and building-to-site mappings. The Orchestrator adds the logical job ID
and, for isolated executions, the attempt number. Existing images accept this
tracking tag; only provenance-aware images produce verified candidates.

Mapping uses actual simulator entity building IDs (or exact flat schema keys),
never display names or filename parsing. Decentralized agent indices are joined
only after cardinality and index validation. Central/shared models preserve all
controlled sites. Pipeline stages retain stage identity, including coordinator
scope. Dynamic topology records its final building order, version and entity-spec
hash; the full entity schema remains in the environment section.

Missing, ambiguous, duplicate or newly introduced buildings yield an explicit
unmapped export. Laboratory/legacy experiments continue to run, but the Training
backend will not register such exports as verified community candidates.
Non-deployable formats remain training artifacts. This contract does not claim
runtime support for every model or define a BAU benchmark.

Validation includes stable IDs under renaming/reordering, central and pipeline
models, invalid mappings, config/file hashes, runner output consistency and a real
two-house dataset → simulator → export → candidate integration test in the Training
repository. Filesystem artifacts remain on the shared volume; SQL stores metadata
and immutable origins, not model binaries.
