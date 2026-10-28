# Shared bundle v2 fixtures

Producer: Algorithms 0.3. Consumer integration tasks:
[SOF-100](https://linear.app/softcps/issue/SOF-100) (inference validation) and
[SOF-101](https://linear.app/softcps/issue/SOF-101) (backend provenance).

`fixtures/valid_onnx` is a synthetic linear graph with three known input/output
pairs. `fixtures/valid_policy` is a deterministic RandomPolicy JSON snapshot.
Both are portable, sealed and replayable. They are contract fixtures, not model
quality evidence. `fixtures/invalid_cases.json` specifies language-neutral
mutations consumers must reject: missing file, bad hash, absolute/traversal path,
unsupported version/format, duplicate node, unsealed extra file and symlink.

The authoritative validator is `algorithms.export.validation.validate_v2`, called
by `utils.bundle_validator.validate_bundle_contract`. File/hash/execution-graph
checks cannot be replaced by JSON structure checks alone. Replay with
`algorithms-bundle <fixture-directory>`.

Tests copy valid fixtures elsewhere, replay them, apply every invalid case and
verify that unknown policy names cannot fall back to zero actions. Regenerate
with `python -m scripts.generate_bundle_contract_fixtures`. Consumers should pin
a released producer commit and run these fixtures in CI. This handoff does not
claim that the separate service/backend integrations are complete.
