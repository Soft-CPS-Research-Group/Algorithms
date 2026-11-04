# Working on Algorithms

Start with [the agent contract](docs/runtime/agents.md). Agents own numerical decisions, learning, and serialization. The shared runtime owns job setup, the simulator loop, results and bundle finalization.

- Register the implementation, lazy schema and actual capabilities in `algorithms/catalogue.py`. Extend `BaseAgent`. Keep numerical imports out of configuration validation.
- Use the shared stage context and update schedule. Preserve existing pipeline, ensemble and raw observation contracts. Offline preparation belongs to the agent.
- Put reusable recipes in `configs/templates/` using `schema_version: algorithms.config.v1` and an ordered `pipeline`. Check them through `algorithms-job --validate` and, when a dataset is available, `--dry-run`.
- Export preprocessing, temporal state and dependencies needed to reproduce a decision. Declare specialized research adapters explicitly. Follow [the output contract](docs/runtime/exports.md) and [the bundle specification](docs/inference_bundle.md).
- Test through the public job entry point. For entity or Transformer changes, check layout changes and action bindings against the [entity contract](docs/transformer_entity_controller.md).
- When the simulator entity schema changes, regenerate `configs/tokenizers/fixtures/entity_obs_sample.json` with `scripts/dump_entity_obs_sample.py` using a supported entity config. Run the tokenizer schema tests.
- Keep the documentation focused on running, extending and comparing controllers. Keep per-job results and drafts outside the versioned guides. Preserve the source history of recovered research implementations.
