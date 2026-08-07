# Transformer PPO Behavior-Cloning PR Cleanup Plan

## Goal

Turn PR #22 into a reviewable, correct behavior-cloning change by removing
stale artifacts, retaining only correctness work required by BC, and moving
independent runtime and experiment work out of the feature diff.

## Target Scope

### PR #22: Transformer PPO behavior cloning

Keep:

- behavior-cloning configuration and validation;
- deterministic teacher construction and demonstration collection;
- actor-only pretraining and auxiliary BC loss;
- per-building and per-layout demonstration isolation;
- raw-observation context required by the teacher;
- topology-transition handling required to retain valid on-policy boundaries;
- pending-action/log-probability identity required for valid PPO ratios;
- transactional topology and checkpoint restoration required to make BC
  checkpoints safe;
- one local dynamic BC template;
- focused unit, integration, and configuration tests for those behaviors.

Remove from PR #22:

- deleted Wave-A campaign configurations and their tests;
- deleted local canary/smoke runbooks, runner, and their tests;
- the deleted 15-minute smoke recipe and its topology-event offset support;
- duplicate 15-minute week/month/year experiment recipes;
- experiment-result/status documents;
- brittle tests that assert exact prose in documentation;
- duplicate or trivial tests that do not add a distinct behavioral path;
- unrelated repository housekeeping.

Restore unchanged from the base branch:

- `configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml`, because it
  remains a documented non-BC template and is used by existing E2E coverage.

### Follow-up PR: TPPO GPU and runtime safeguards

Move independently reviewable CUDA selection, CUDA checkpoint/export handling,
update-duration guards, CUDA timing synchronization, and watchdog attachment
instrumentation into a dedicated runtime PR. Preserve its focused tests there.

### Follow-up PR: experiment recipes

Add 15-minute week/month/year recipes and any campaign runbooks only after the
core BC contract is merged. Keep recipe validation table-driven and avoid
checking exact documentation wording.

## Required Invariants

The cleanup must retain tests proving:

1. PPO stores the exact collection-time action, pre-tanh action, log
   probability, and value used by the environment transition.
2. Termination, truncation, and topology boundaries do not lose or relabel an
   on-policy transition.
3. A failed topology mutation or PPO update restores every affected model,
   optimizer, buffer, counter, and RNG state.
4. Teacher actions are normalized into the actor's tanh domain and are never
   executed during PPO or deterministic evaluation.
5. Demonstrations remain isolated by building and compatible layout signature.
6. Pretraining fails before PPO when an active building has no usable samples.
7. Checkpoint validation rejects incompatible BC state before mutating the
   receiving agent.

## Execution Steps

1. Remove dangling campaign/local-runner tests and stale smoke support.
2. Restore the base non-BC dynamic template unchanged.
3. Remove experiment-only recipes and status material from PR #22.
4. Consolidate duplicate validation, checkpoint, and numerical tests while
   retaining one case per distinct failure stage.
5. Correct documentation so historical compatible layouts are described
   consistently.
6. Run focused BC/TPPO tests, then the complete test suite and `git diff --check`.
7. Compare final file and line counts with the original 46-file, 11,507-line
   diff and prepare logically separated commits/PRs.

## Acceptance Criteria

- PR #22 CI-equivalent test suite passes.
- No test references a deleted file.
- No production option exists solely for a removed recipe.
- The BC specification, agent guidance, schema, and runtime behavior agree.
- The final diff is materially smaller and every remaining changed file maps
  directly to BC or a documented correctness prerequisite.
