# Algorithms consolidation: first delivery

This change consolidates the implementation families from the 26 October 2026
branch snapshot onto the current main runtime. It preserves the original heads
through archive tags and records every decision in the [branch map](branch-map.md).
The scope is integration and a working minimum, not model-quality improvement.

## What was already in main

Most student contributions had already reached main through merged, squashed or
restacked PRs. The existing 30 registered agents, hierarchical pipeline, entity
adapter, job runner, runtime fixes and current simulator pin remain the base.
In particular, the current CCLevel2 and AgentTransformerMATD3 are not replaced by
older files bearing similar names.

## What was still missing

| Source head | Recovered implementation | Integration decision |
|---|---|---|
| `69316e57414516ae688a0515fd8a0d5b04dac730` | BC, flat IQL, entity IQL/CQL, transition datasets, trainers, collectors and tests | Registered through the shared job lifecycle; optional offline preparation precedes simulator evaluation. PR #13 had merged into this branch, not main. |
| `e4dd4e73b3bf29a1a04472268ac12bad1768c751` | Historical TransformerMADDPG, tokenizer, replay and networks | A distinct experimental agent, separate from Transformer MATD3. Current pipeline schema and complete checkpoint export. |
| `a7a395d743c9031499eff3fa590ee3b3942acb93` | Capacity-aware coordination, directional and community signal workers, reward variants, frozen-worker coupling | Named CCLevel2Capacity variant preserves its teacher/network without overwriting current CCLevel2; optional coupling defaults off. |
| Local main and uncommitted work | Portugal dataset catalogue, TI-MARL protocol, diagnostic evidence and selection rules | Original three commits replayed and local patch preserved separately before integration. |

The [agent catalogue](../agent_catalogue.md) lists all 38 registered names and
explains the recovered families' capabilities. Data collectors are utilities,
not additional trainable policies. `SingleAgentRL` remains an explicit rejected
placeholder, not an implemented model.

## Common runtime boundary

```text
job config + simulator dataset + optional offline transitions
  -> validate config / create job directories / initialize simulator
  -> build agent or pipeline
  -> optional prepare_training (offline gradient updates; no extra simulator)
  -> attach environment / optional resume
  -> shared wrapper: episodes, predict, update, final deterministic evaluation
  -> shared results, KPIs, artifact manifest and bundle validation
```

Online agents inherit a no-op preparation hook. Pipeline and Ensemble forward it
to their children with separate subdirectories. Offline agents choose exactly one
policy source: `model_dir` for evaluation, or `offline_training` for optimization
inside this job. Their environment updates are no-ops because learning already
consumed the offline dataset. No new agent-specific runner is required.

The recovered historical command-line collectors, trainers and benchmark tools
remain available for reproducing research. The supported job entry point is still
`run_experiment.py`; those tools do not define a competing production job contract.

## Integration corrections

- Existing inference-only offline constructors now also accept the pipeline config.
- Offline exports include architecture and normalization files, can be reloaded,
  validate required sidecars, and preserve named feature/action order. Ambiguous
  policy groups and incompatible runtime layouts fail explicitly.
- TransformerMADDPG requires equal nonzero action widths and static topology.
  Its checkpoint includes actor, critic, targets, optimizers and tokenizer. The
  historical token-level ONNX exporter now preserves variable sequence lengths;
  it is still not an end-to-end serving bundle.
- CCLevel2Capacity receives battery capacities through additive wrapper metadata;
  it does not change the existing CCLevel2 encoding layout or learned model shape.
- `signal_aware_actor_obs` is restricted to frozen workers. Training a manager and
  a conditioned worker jointly needs a replay-contract review in the next phase.
- `research_torch` explicitly means `deployable: false`. Hierarchical signals and
  optional worker observation/price coupling are also marked non-deployable until
  the inference runtime implements their complete semantics. Composite manifests
  retain that flag and resolve nested sidecar paths correctly.
- The old Simulator 1.0.2 monkeypatch is not restored: its patched method no longer
  exists in the pinned 1.8.3 implementation. Dataset helpers reuse the current
  runner's portable schema resolution.
- The broad catch-and-ignore RNG restoration proposed on the CC branch is not
  adopted, because it would silently change training-resume guarantees.

The restored calibration and flat RBC collector scripts had lost indentation in
`2e5e7fb`. Their valid pre-rename `_v2.py` sources were recovered and migrated to
the current module names; script compilation is now included in CI.

## Verification gates

Use Python 3.10 and `requirements-test.txt` for verification. The runtime image
continues to use `requirements.txt`; ONNX Runtime is a test-only addition.
CI retains the existing tests and additionally runs five real short job contracts:
TransformerMADDPG, capacity coordination, directional coordination, entity IQL and
entity CQL. They check completed jobs, validated model bundles, deterministic final
evaluation, final-episode-only export and nonempty KPI files. Synthetic offline
training tests also cover flat BC/IQL and reload parity for all four policies.
These are execution checks, not evidence that any model is competitive.

Optional historical tests require campaign datasets/checkpoints absent from the
repository and are skipped explicitly. Large campaigns, GPU performance and the
other platforms' full runtime matrix remain separate work.

## Branch policy

- Main is the shared implementation source. Add a family under its agent package
  and register it; do not use a permanent branch as the algorithm selector.
- Use short feature branches and a PR into main. Run the full suite and relevant
  real-job contract before merging. The existing repository protection remains.
- Experiments belong in configs and evidence directories; checkpoints/results stay
  outside the source tree. Preserve author/source commit attribution.
- Before retiring a branch, record its full SHA, verify its archive tag remotely,
  verify the main destination, and refuse deletion if the remote head advanced.
- Archive tags are historical evidence, not supported alternative release lines.
  They preserve prototypes, superseded recipes and reports omitted from main.
- An open PR is closed as obsolete only after its content is accounted for.
  #3 is already ancestral; #16 is superseded by the current CCLevel2; #24 is
  superseded by this selective integration and explicitly named variants.

## Next stages, retained in order

1. Finish the shared configuration/agent lifecycle contract and remove obsolete
   paths without losing useful research tooling. Specify offline feature identity,
   group dispatch, hierarchy signals and capability declarations uniformly.
2. Standardize the intended export policy: deterministic final episode by default,
   explicit exceptions, consistent KPI/model manifests and disabled MLflow usage
   unless requested. This delivery's new recipes already use final-only export;
   it does not change every existing experiment's meaning.
3. Audit determinism, seeds, checkpoint/resume guarantees, correctness and failure
   recovery across all model families. Review cached-token learning in historical
   TransformerMADDPG, offline group semantics and coordinator math before tuning.
4. Review dependency/platform profiles, startup cost and long-job observability,
   including offline preparation progress. Reconsider MLflow's dependency footprint.
5. Evaluate model quality under the agreed datasets and experimental protocol.
6. Extend energAIze inference only against an explicit deployable bundle contract.

The wider audit and all planned actions remain under Linear SOF-241/SOF-251. This
stage does not reinstate MLflow services, schedule campaigns, expire artifacts,
resume Deucalion or modify inference deployments.
