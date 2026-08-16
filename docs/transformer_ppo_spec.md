# AgentTransformerPPO Specification

> Status: **Current implementation**
> Scope: `AgentTransformerPPO` on the entity interface with dynamic topology.
> Last reviewed: 2026-08-16
> Reviewed `main` commit: `f2809313c7550405eccd4d9b276adbf8a9103a5c`
> Maintainer: Algorithms maintainers

This is a current implementation specification. It documents code on the
reviewed commit. It does not restore closed PR 23, define Transformer MATD3,
or turn historical proposals into requirements. See the shared
[Transformer Entity Controller Contract](transformer_entity_controller.md) for
entity, token, layout, and backbone rules.

## 1. Status and applicability

TPPO is registered as `AgentTransformerPPO`. It supports
`simulator.interface: entity`, including `simulator.topology_mode: dynamic`.
The tested package is `softcpsrecsimulator==1.5.6`; the conceptual contract is
the entity payload (`tables`, `edges`, `meta`), `meta.topology_version`, and
active action-table shapes.

The implementation sources are [agent.py](../algorithms/transformer_ppo/agent.py),
[behavior_cloning.py](../algorithms/transformer_ppo/behavior_cloning.py),
[PPO components](../algorithms/transformer_ppo/ppo_components.py), the
[wrapper](../utils/wrapper_citylearn.py), the [schema](../utils/config_schema.py),
and the [manifest and bundle validators](../utils/artifact_manifest.py).

## 2. Objective and non-goals

TPPO learns one local policy per building. It maps the shared entity-derived
token sequence to one bounded action per controllable asset. It supports
variable asset cardinality when the shared contract and tokenizer type widths
remain valid.

This document does not define Transformer MATD3, replay, target actors, twin
critics, delayed policy updates, target smoothing, or exploration noise. Those
are future design choices. The shared contract identifies the reuse boundary.

## 3. Dependency on the shared contract

TPPO receives one encoded vector per building from `Wrapper_CityLearn`. The
wrapper owns entity conversion, topology detection, action-table conversion,
and feature encoding. TPPO uses the shared token order `[SRO, NFC, CA]` and the
CA/action-name position invariant. TPPO is the final pipeline stage because it
validates the action returned by `predict` against the action passed to `update`.

## 4. TPPO model structure

TPPO owns one independent stack per building:

- one `EntityObservationTokenizer`;
- one `TransformerBackbone`;
- one `ActorHead`;
- one `CriticHead`;
- one PPO optimizer;
- one optional behavior-cloning optimizer;
- one `RunningValueNormalizer`; and
- one on-policy `RolloutBuffer`.

The tokenizer and backbone feed both heads. The critic consumes the mean-pooled
representation and estimates scalar `V(s)`. The actor consumes CA embeddings
and emits one scalar per CA token. The actor MLP is applied independently to CA
embeddings. `ActorHead.log_std` is one learned scalar shared by all CA tokens;
it is not one value per CA type or per asset.

The Transformer dropout value is required to be `0.0`. PPO old/new probability
ratios require the same deterministic representation for a stored rollout.

## 5. Action and value semantics

During stochastic prediction, the actor samples a Gaussian in pre-tanh space,
applies `tanh`, then maps the result to each action space with:

```text
action = low + (tanh_sample + 1) * (high - low) / 2
```

The stored transition keeps the exact pre-tanh sample and its log probability.
The log probability includes the tanh correction and the affine action-scale
correction. Deterministic prediction uses the squashed mean.

If local action safety is enabled, projection runs after the affine mapping.
The projected value becomes the pending action and must be the action passed to
`update`. The wrapper performs its normal finite-space clipping before sending
the action to the simulator. The pending-action equality check prevents an
unrelated pipeline stage from changing an on-policy action.

Rewards retain their raw value for diagnostics. TPPO applies a lower floor of
`-10.0` through the implementation default `reward_clip=10.0` before adding the
sample to PPO. `reward_clip` is not a schema-supported YAML field and is not a
supported configuration option. A separate code decision is required before
exposing it.

The value normalizer tracks return mean, variance, and count per building. It
normalizes PPO target values for the critic loss and denormalizes values used by
prediction and bootstrap calculations. It is not updated by behavior cloning.

## 6. Training lifecycle without behavior cloning

The normal training path is:

1. The wrapper attaches observation names, action names, spaces, and metadata.
   TPPO builds one stack per building and action bounds.
2. The wrapper calls `on_episode_start` with the episode number and training
   flag. Evaluation episodes do not collect rollouts.
3. `predict` tokenizes each encoded observation, runs the actor and critic, and
   creates one pending decision per building.
4. The wrapper sends the returned action through its action adapter and the
   simulator. The same action is retained for the transition.
5. `update` validates row counts and compares each supplied action with its
   pending decision. A mismatch raises before collection.
6. TPPO stores the observation, executed action, exact pre-tanh action, old log
   probability, value, clipped reward, termination flags, and raw reward.
7. On a scheduled `update_step`, TPPO updates only when a buffer reaches the
   configured minibatch size. It computes GAE, updates the value normalizer,
   runs the configured PPO epochs, then clears the buffer after success.
8. At an episode boundary, TPPO flushes each non-empty buffer, including a
   one-sample rollout. Terminated transitions bootstrap with zero. A truncated
   or non-terminal transition bootstraps from its next observation when present.
9. The wrapper consumes the latest TPPO metrics and clears the consumed cache.
10. The checkpoint manager calls TPPO preflight at its configured safe boundary.

`terminated` and `truncated` may be scalar booleans or per-building arrays.
Both forms are validated. Each building stores its own flags. `terminated`
causes zero bootstrap; `truncated` is retained in the rollout contract but does
not by itself force a zero bootstrap when a next observation exists.

## 7. Training lifecycle with behavior cloning

Behavior cloning is active only when the pipeline stage contains a
`behavior_cloning` mapping with `enabled: true`. An absent mapping and a mapping
with `enabled: false` disable BC.

The current teacher is deterministic `RBCSmartPolicy`. The BC lifecycle is:

1. Before the first PPO rollout, the configured demonstration episodes run with
   teacher actions. TPPO uses raw observation context for teacher computation.
2. TPPO stores encoded observations, normalized teacher actions in tanh space,
   action masks implied by the layout, and a complete layout signature. Samples
   are immutable and kept in a bounded per-building reservoir.
3. At the end of the last demonstration episode, actor-only pretraining runs
   before PPO. It updates tokenizer, backbone, and actor parameters. It does
   not update the critic, PPO optimizer, or value-normalizer statistics.
4. If any building has no usable compatible demonstration, pretraining fails
   before the first PPO rollout. Historical demonstrations are grouped by their
   stored layout signature. Every compatible group can be pre-trained with its
   stored layout.
5. After pretraining, PPO rollouts and evaluation are actor-controlled. Teacher
   actions are not blended into environment actions and the teacher does not
   replace actor actions probabilistically.
6. After all PPO epochs for a building, TPPO may run one separate auxiliary BC
   optimizer step. This step is actor-only and does not change the PPO optimizer,
   critic, or value normalizer.
7. The BC weight follows `weight`, `min_weight`, `decay_start_step`, and
   `decay_steps`. The persisted `bc_actor_training_step` is the decay clock.

Current TPPO does not implement teacher blending, probabilistic teacher
replacement, or `RBCCommunityPolicy` as a BC teacher. Those names are historical
or belong to other algorithms.

## 8. Dynamic-topology transaction

There is no public `AgentTransformerPPO.on_topology_change` API. The current
transaction is split between the wrapper and real controller hooks:

1. The wrapper detects a changed `meta.topology_version` through the entity
   adapter.
2. It records the old-layout transition before the new model layout is attached
   when training requires that transition.
3. It snapshots wrapper layout state and, when supported, the controller state.
4. It rebuilds observation names, spaces, encoders, action names, and bounds.
5. `attach_environment` compares cached observation and action names.
6. Changed buildings enter the private `_handle_topology_change` path.
7. TPPO flushes and clears the old rollout, builds the new layout, and retains
   compatible per-type tokenizer weights, neural weights, optimizer state,
   value normalizer, and counters.
8. Runtime tokenizer validation runs without startup-only rule 5. New types or
   changed feature widths fail instead of silently reusing incompatible weights.
9. The BC teacher and local safety adapters are reattached to the new metadata.
10. If any step fails, wrapper and TPPO snapshots are restored atomically.

A building-count change is a full rebuild. Old per-building neural stacks,
rollout buffers, optimizers, and value normalizers are not remapped. The BC
regularizer object and its stored demonstrations may remain available, but only
demonstrations compatible with the new building index and layout are usable.

## 9. Local action safety

Local safety is a TPPO-side projection over raw entity observations. It is
disabled by default in the TPPO schema. When enabled, the wrapper must provide
raw observation context before `predict`; normalized actor inputs are unsuitable
for physical power and SOC constraints.

The projection order is:

```text
actor -> tanh -> affine action bounds -> local safety projection
      -> pending/executed action -> wrapper clip/entity conversion
```

The projection can protect EV minimum service, EV service targets, required
deferrable starts, and electrical headroom. It respects action bounds and the
configured EV minimum mode. An infeasible projection raises `RuntimeError` when
`local_action_safety_fail_on_infeasible=true`; otherwise the result and its
infeasible reasons are returned for execution and diagnostics.

The supported safety fields are listed in §10. The adapter is rebuilt after a
topology reattachment. Deterministic evaluation uses the deterministic actor,
then the same safety projection. Demonstration episodes bypass the actor and
return deterministic teacher actions; TPPO does not run those teacher actions
through the TPPO local safety adapter.

Safety diagnostics are cumulative until the process ends:

- `TPPO/local_action_safety_projections`
- `TPPO/local_action_safety_interventions`
- `TPPO/local_action_safety_infeasible`
- `TPPO/local_action_safety_reason_<reason>`

The ONNX graph contains the neural deterministic actor and affine bounds only.
It does not contain this raw-observation safety projection. Deployment must
reproduce the safety layer separately or disable the option. No inference parity
claim is made when deployment omits that layer.

## 10. Configuration reference

The schema source is `TransformerPPOStageConfig` in
[`utils/config_schema.py`](../utils/config_schema.py). Unknown fields are not
part of the supported contract.

### 10.1 Stage and Transformer fields

| Field | Type and default | Constraints | Runtime effect |
|---|---|---|---|
| `algorithm` | literal `AgentTransformerPPO` | Required | Selects TPPO. |
| `count` | integer, `1` | Exactly `1` | TPPO owns one controller stage. |
| `frozen` | boolean, `false` | None | Stage freeze flag. |
| `tokenizer_config_path` | non-empty string | Required | Loads type, feature, NFC, and validation rules. |
| `transformer.d_model` | integer | `>=1` | Token embedding width. |
| `transformer.nhead` | integer | `>=1` | Attention head count. |
| `transformer.num_layers` | integer | `>=1` | Encoder depth. |
| `transformer.dim_feedforward` | integer | `>=1` | Encoder feed-forward width. |
| `transformer.dropout` | float, `0.0` | `0.0` only | Nonzero values fail schema validation because PPO ratios require deterministic representations. |

### 10.2 PPO hyperparameters

| Field | Type and default | Constraints | Runtime effect |
|---|---|---|---|
| `require_cuda` | boolean, `false` | None | Fail initialization when CUDA is unavailable if true. |
| `learning_rate` | float | `>0` | PPO and BC optimizer learning rate. |
| `gamma` | float | `(0,1]` | Return discount. |
| `gae_lambda` | float | `(0,1]` | GAE trace parameter. |
| `clip_eps` | float | `>0` | PPO ratio clipping interval. |
| `ppo_epochs` | integer | `>=1` | PPO passes over each rollout. |
| `minibatch_size` | integer | `>=1` | Minimum rollout size for scheduled PPO and batch size cap for boundary flush. |
| `entropy_coeff` | float | `>=0` | Entropy term coefficient. |
| `value_coeff` | float | `>=0` | Critic loss coefficient. |
| `max_grad_norm` | float | `>0` | Gradient clipping bound. |
| `actor_log_std_init` | float, `-0.5` | No schema range | Initial shared actor log standard deviation. Runtime clamps it to `[-2.0, 0.5]` when constructing the distribution. |

`reward_clip` is read as an implementation fallback of `10.0` but is not a
schema field. Do not add it to a validated TPPO YAML file.

### 10.3 Local action-safety fields

| Field | Type and default | Constraints | Applies when |
|---|---|---|---|
| `local_action_safety_enabled` | boolean, `false` | None | Safety is enabled. |
| `local_action_safety_fail_on_infeasible` | boolean, `false` | None | An infeasible projection should raise. |
| `local_action_safety_protect_ev_minimum` | boolean, `true` | None | EV minimum action is protected. |
| `local_action_safety_ev_minimum_mode` | enum, `average` | `average` or `deadline_feasible` | Selects EV minimum calculation. |
| `local_action_safety_protect_ev_service_target` | boolean, `false` | None | EV required-SOC service target limits are applied. |
| `local_action_safety_protect_deferrable_must_start` | boolean, `true` | None | Required deferrable starts are reserved. |
| `local_action_safety_allow_discretionary_deferrable_start` | boolean, `false` | None | Allows optional deferrable starts. |
| `local_action_safety_headroom_reserve_kw` | float, `0.0` | `>=0` | Reserves this electrical headroom before projection. |

Safety applies in actor-controlled training and evaluation. It requires raw
observation context. It is external to ONNX.

### 10.4 Behavior-cloning fields

The `behavior_cloning` block is optional. It is active only with
`enabled: true`. When enabled, `demonstration_episodes` must be at least one
and `min_weight` must not exceed `weight`.

| Field | Type and default | Constraints | Runtime effect |
|---|---|---|---|
| `behavior_cloning.enabled` | boolean, `true` in a present block | Enabled block requires one or more demonstration episodes. | Enables teacher collection, pretraining, and auxiliary BC. |
| `demonstration_episodes` | integer, `1` | `>=0`; `>=1` when enabled | Number of deterministic teacher episodes. |
| `max_samples_per_building` | integer, `4096` | `>=1` | Reservoir capacity per building. |
| `pretraining_epochs` | integer, `4` | `>=1` | Actor-only pretraining epochs per compatible layout group. |
| `batch_size` | integer, `64` | `>=1` | BC batch size. |
| `weight` | float, `0.0` | `>=0` | Auxiliary BC loss weight. |
| `min_weight` | float, `0.0` | `>=0`, `<= weight` | Final auxiliary BC weight. |
| `decay_start_step` | integer, `0` | `>=0` | BC decay start on the persisted BC actor clock. |
| `decay_steps` | integer, `0` | `>=0` | Linear decay duration; zero keeps the configured weight. |
| `ev_multiplier` | float, `1.0` | `>=0` | Relative BC weight for charger CA actions. |
| `storage_multiplier` | float, `1.0` | `>=0` | Relative BC weight for storage CA actions. |
| `teacher.policy` | literal `RBCSmartPolicy`, default same | Only supported teacher. | Builds the deterministic demonstration teacher. |
| `teacher.hyperparameters` | mapping, `{}` | Extra teacher settings are passed to the teacher. | Configures the teacher implementation. |

### 10.5 Shipped templates

The complete validated examples are:

- [TPPO without BC](../configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml)
- [TPPO with BC](../configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml)

Both use entity mode, dynamic topology, `dropout: 0.0`, and the bundled entity
tokenizer. The BC template uses deterministic `RBCSmartPolicy` demonstrations
and contains no teacher-blending fields.

## 11. Metrics and diagnostics

TPPO emits its consumed training metrics with the final `TPPO/` prefix. Building
metrics use `TPPO/building_<index>/...`. The wrapper consumes and clears the
latest training cache; safety status metrics are cumulative.

### 11.1 Rollout and PPO update

For each building update, TPPO can emit:

`update_count`, `rollout_size`, `policy_loss`, `value_loss`, `entropy`,
`clip_fraction`, `approx_kl`, `ratio_error_max`, `explained_variance`,
`actor_grad_norm`, `critic_grad_norm`, `raw_reward_mean`, `raw_reward_min`,
`raw_reward_max`, `clipped_reward_mean`, `clipped_reward_min`,
`clipped_reward_max`, `value_residual_p50`, `value_residual_p90`,
`value_residual_p99`, `episode_training`, and `teacher_action_execution`.

For example, `TPPO/building_0/policy_loss` is a per-building latest value. A
metric is absent when no update emitted it. There are no promised skipped-batch
or actor-only evaluation metrics.

### 11.2 Behavior cloning

BC metrics use the `TPPO/` prefix after consumption:

`behavior_cloning_teacher_enabled`, `behavior_cloning_demonstration_samples`,
`behavior_cloning_effective_weight`, `behavior_cloning_loss`,
`behavior_cloning_weighted_loss`, `behavior_cloning_valid_samples`,
`behavior_cloning_pretraining_epochs`,
`behavior_cloning_incompatible_demonstration_samples`,
`behavior_cloning_rejected_at_record`, and
`behavior_cloning_pretraining_batches`.

Pretraining also emits per-building
`behavior_cloning_building_<building_id>_usable_samples`,
`..._trained_batches`, and `..._zero_action_samples`. These are latest values
from the completed lifecycle event, not lifetime counters unless the name says
the underlying reservoir or rejection count.

### 11.3 Local safety

Safety status names are already emitted as `TPPO/local_action_safety_*` and are
cumulative: projections, interventions, infeasible results, and reason-code
counters. The wrapper may additionally emit generic `Action/*`,
`Deferrable/*`, reward-component, runtime, and system metrics. Those are wrapper
metrics, not TPPO algorithm metrics.

## 12. Checkpoint and resume contract

The current checkpoint format version is `4`; versions `1` through `4` are
accepted by the loader. The payload categories are:

- format version, save step, global learning step, PPO update count, and episode;
- BC pretraining flag, BC actor-training clock, latest metrics, and BC state;
- Python, NumPy, Torch, and CUDA RNG state when applicable;
- per-building tokenizer, backbone, actor, critic, PPO optimizer, and optional
  BC optimizer state;
- per-building layout signature, action names, action bounds, and value
  normalizer state.

BC state includes immutable encoded demonstrations, targets, layout signatures,
reservoir counts, sampler state, lifecycle metrics, and rejection counts. The
live teacher is not serialized; `attach_environment` rebuilds it on restore.

`preflight_checkpoint` defers a save when any rollout buffer is non-empty. This
preserves on-policy correctness. The checkpoint manager can retry at an update
or episode boundary and handles `DeferredCheckpointError` without treating it
as a failed run.

Restore validates format, building count, BC compatibility, layout signatures,
action names, and action bounds before changing model state. Cross-cardinality
and cross-layout resume is rejected. A failure while applying state restores the
complete previous runtime snapshot. Legacy formats are accepted only when their
stored fields meet the current compatibility checks.

## 13. ONNX and manifest contract

`export_artifacts` exports one deterministic neural actor per current building.
The graph has exactly:

- input `encoded_obs`, with fixed width for the exported topology;
- output `actions`, with dynamic batch axis;
- dynamic batch axis on `encoded_obs`;
- opset `17`.

Layout indices, tokenizer projections, Transformer, actor MLP, `tanh`, and
affine action bounds are baked into the graph. The graph uses only the current
topology. It does not include local raw-observation safety projection.

Files use `onnx_models/agent_<building_index>__topology_v<version>.onnx`.
`context.topology_version` overrides the filename and metadata version when
provided; otherwise TPPO uses the per-building topology counter.

The returned metadata includes `format: onnx`, `artifacts`,
`tokenizer_config_path`, `supports_dynamic_topology`, and `agent_models`.
Each `agent_models` entry uses `model_path`, not `onnx_path`, and includes
`building_index`, `building_id`, `topology_version`, `obs_dim`, `n_sro`, `n_ca`,
`sro_types`, `ca_types`, and `ca_action_names`. Artifact entries also carry
`config.ca_action_names`.

Only the current topology is exported. A dynamic deployment must select a new
export after a topology mutation or implement an external model-routing policy.
The manifest alone does not make one fixed graph portable to every future
cardinality.

## 14. Supported templates and run examples

Validate both shipped templates through the current schema and template tests:

```bash
pytest -q tests/test_template_transformer_ppo_entity_dynamic.py \
  tests/test_template_transformer_ppo_bc_entity_dynamic.py
```

Run the focused TPPO suites:

```bash
pytest -q tests/test_agent_transformer_ppo.py \
  tests/test_agent_transformer_ppo_behavior_cloning.py \
  tests/test_agent_transformer_ppo_wrapper_integration.py
```

Start local training with the shipped examples:

```bash
python run_experiment.py \
  --config configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml \
  --job_id tppo-local

python run_experiment.py \
  --config configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml \
  --job_id tppo-bc-local
```

Outputs are under `runs/jobs/<job_id>/`: metrics are in `logs/` or the local
metrics stream, checkpoints in `checkpoints/`, ONNX files in `onnx_models/`,
and the final artifact manifest in `artifact_manifest.json`.

On the reviewed commit, the optional dynamic end-to-end smoke reaches the
runtime topology transaction but fails for the bundled dataset because
`Building_2` changes the active `charger` feature width from `16` to `63`.
This is a separate runtime defect, not a documentation change. The focused
unit, schema, template, integration, export, and bundle tests pass; no
end-to-end success claim is made until that implementation issue is resolved.

## 15. Known limits

- TPPO supports the entity interface only.
- Feature-schema changes are not dynamically portable.
- Cross-layout and cross-cardinality checkpoint restore is rejected.
- Export covers only the current topology.
- Local action safety is external to ONNX.
- TPPO must be the final pipeline stage.
- Functional tests do not provide a performance, stability, or convergence
  guarantee.
- The bundled dynamic end-to-end smoke is currently blocked by the documented
  charger feature-width mutation; this needs a separate implementation task.
- `reward_clip` remains an implementation default, not a schema-supported
  configuration field.

## 16. Requirement-to-test traceability

The following names were checked against the current test files on the reviewed
commit. A row marked `Gap` would identify an uncovered requirement; no current
row is presented as a planned test.

| ID | Behavior | Test | Level |
|---|---|---|---|
| TPPO-01 | Registry name and construction | `tests/test_agent_transformer_ppo.py::test_registered_under_canonical_name`, `::test_create_agent_via_registry` | unit |
| TPPO-02 | CUDA requirement and device selection | `tests/test_agent_transformer_ppo.py::test_device_defaults_to_cpu_when_cuda_is_unavailable`, `::test_require_cuda_raises_when_cuda_is_unavailable` | unit |
| TPPO-03 | Actor output shape and finite action range | `tests/test_agent_transformer_ppo.py::test_predict_shape_and_range` | unit |
| TPPO-04 | Deterministic prediction repeatability | `tests/test_agent_transformer_ppo.py::test_predict_deterministic_is_repeatable` | unit |
| TPPO-05 | Pending action validation | `tests/test_agent_transformer_ppo.py::test_update_rejects_action_that_differs_from_pending_decision` | unit |
| TPPO-06 | Scheduled update and buffer clear | `tests/test_agent_transformer_ppo.py::test_update_appends_to_buffer_then_ppo_step_clears` | unit |
| TPPO-07 | One-sample episode flush | `tests/test_agent_transformer_ppo.py::test_episode_boundary_trains_one_sample_rollout` | unit |
| TPPO-08 | Scalar/per-building metrics remain distinct | `tests/test_agent_transformer_ppo.py::test_training_metrics_keep_each_building_result` | unit |
| TPPO-09 | Dynamic topology support declaration | `tests/test_agent_transformer_ppo.py::test_supports_dynamic_topology_classvar_true` | unit |
| TPPO-10 | Layout rebuild preserves compatible weights | `tests/test_agent_transformer_ppo.py::test_topology_change_rebuilds_layout_and_preserves_weights` | integration |
| TPPO-11 | Feature-count drift fails | `tests/test_agent_transformer_ppo.py::test_topology_change_feature_count_drift_hard_fails` | integration |
| TPPO-12 | Building-count change flushes boundary rollout | `tests/test_agent_transformer_ppo.py::test_building_count_change_flushes_one_sample_rollout` | integration |
| TPPO-13 | Wrapper detects and attaches dynamic topology | `tests/test_agent_transformer_ppo_wrapper_integration.py::test_wrapper_topology_change_triggers_agent_rebuild` | integration |
| TPPO-14 | Wrapper/controller rollback is atomic | `tests/test_agent_transformer_ppo_wrapper_integration.py::test_learn_rolls_back_wrapper_and_agent_when_deferred_attach_fails`, `::test_learn_rolls_back_wrapper_when_agent_snapshot_fails` | integration |
| TPPO-15 | Safety projection becomes executed action | `tests/test_agent_transformer_ppo.py::test_local_action_safety_projection_is_used_for_executed_action` | unit |
| TPPO-16 | Safety constraints and infeasible behavior | `tests/test_local_action_safety.py::test_urgent_ev_reports_infeasible_when_headroom_cannot_supply_minimum`, `::test_deferrable_must_start_is_reserved_and_unavailable_is_infeasible` | unit |
| TPPO-17 | BC teacher-only demonstrations | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_demo_episode_executes_teacher_only_records_immutable_demo_and_no_ppo` | integration |
| TPPO-18 | Evaluation remains actor-controlled | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_evaluation_at_episode_zero_uses_actor_not_teacher` | integration |
| TPPO-19 | BC pretraining precedes PPO | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_final_demo_end_pretrains_actor_then_ppo_uses_only_actor_actions` | integration |
| TPPO-20 | Missing usable BC demonstrations fail early | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_final_demo_lifecycle_rejects_zero_usable_demonstrations`, `::test_pretraining_rejects_each_building_without_usable_demonstrations` | integration |
| TPPO-21 | Compatible historical layout groups train | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_final_demo_boundary_pretrains_every_stored_topology_group` | integration |
| TPPO-22 | Auxiliary BC is actor-only and post-PPO | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_auxiliary_bc_update_changes_actor_and_tokenizer_but_not_critic`, `::test_auxiliary_bc_runs_after_all_ppo_epochs` | unit |
| TPPO-23 | BC checkpoint lifecycle and rollback | `tests/test_agent_transformer_ppo_behavior_cloning.py::test_checkpoint_restores_bc_demonstrations_phase_and_decay_progress`, `::test_checkpoint_apply_failure_restores_complete_runtime_state` | integration |
| TPPO-24 | Checkpoint safe-boundary deferral | `tests/test_checkpoint_manager.py::test_checkpoint_manager_defers_only_nonempty_tppo_rollout` | integration |
| TPPO-25 | Checkpoint round trip and layout rejection | `tests/test_agent_transformer_ppo.py::test_checkpoint_round_trip`, `::test_checkpoint_layout_signature_mismatch_rejected`, `::test_checkpoint_signature_mismatch_same_cardinality` | integration |
| TPPO-26 | ONNX artifact names and metadata | `tests/test_agent_transformer_ppo.py::test_export_artifacts_writes_files_and_returns_manifest` | integration |
| TPPO-27 | ONNX bundle structure validation | `tests/test_bundle_validator.py::test_validate_bundle_contract_accepts_onnx_bundle`, `tests/test_artifact_manifest.py::test_manifest_contains_core_sections_and_normalized_artifacts` | integration |
| TPPO-28 | Token layout, variable cardinality, and action ordering | `tests/test_entity_token_layout.py::test_segment_overall_order`, `::test_per_asset_sro_segments_sorted_by_instance_id`, `::test_ca_count_mismatch_raises` | unit |
| TPPO-29 | Tokenizer non-contiguous slicing and topology reuse | `tests/test_entity_observation_tokenizer.py::test_index_select_handles_non_contiguous_sro_segment`, `::test_projection_is_per_type_no_new_params_on_topology_grow` | unit |
| TPPO-30 | Backbone variable token count and pooling | `tests/test_transformer_backbone.py::test_variable_token_count_supported`, `::test_pooled_is_mean_over_all_tokens` | unit |
| TPPO-31 | Five tokenizer validation rules | `tests/test_entity_tokenizer_config_schema.py::test_rule1_unmatched_feature_fails`, `::test_rule2_ambiguous_pattern_fails`, `::test_rule3_missing_nfc_source_fails`, `::test_rule4_bad_regex_fails`, `::test_rule5_missing_action_field_fails` | schema |
| TPPO-32 | No-BC shipped template | `tests/test_template_transformer_ppo_entity_dynamic.py::test_template_passes_schema_validation`, `::test_template_tokenizer_path_validates_against_bundled_sample` | template |
| TPPO-33 | BC template uses demonstrations without blending | `tests/test_template_transformer_ppo_bc_entity_dynamic.py::test_local_bc_template_uses_demonstrations_without_action_blending` | template |
| TPPO-34 | Dynamic end-to-end topology and export smoke; currently blocked by the `Building_2` charger-width mutation described in §14 | `tests/e2e/test_e2e_transformer_ppo_entity_dynamic.py::test_smoke_run_completes`, `::test_topology_changes_observed_during_run`, `::test_artifact_manifest_includes_onnx_per_building` | end-to-end / Gap |

## 17. Current decisions

| ID | Status | Decision and reason | Evidence | Future Transformer consequence |
|---|---|---|---|---|
| D-01 | accepted | Entity payload conversion stays in the wrapper. | `EntityContractAdapter`, wrapper integration tests | MATD3 reuses the same adapter boundary. |
| D-02 | accepted | Feature-origin names build layouts; numeric sentinels are not used. | layout and tokenizer tests | New controllers consume layout metadata, not marker values. |
| D-03 | accepted | Token projections are shared per type. | tokenizer topology-growth test | Variable cardinality can reuse type weights. |
| D-04 | accepted | CA order follows action names. | layout action-order tests and agent action checks | Every actor must preserve this mapping. |
| D-05 | accepted | TPPO uses a pooled critic and one scalar shared actor log standard deviation. | `ppo_components.py`, actor tests | MATD3 may replace only the algorithm-specific heads and critic rules. |
| D-06 | accepted | `dropout=0.0` is required by the PPO ratio contract. | `TransformerPPOTransformerConfig` validator | Other algorithms need their own stochastic-representation decision. |
| D-07 | accepted | Dynamic changes flush old rollouts and preserve only compatible type weights. | topology integration tests | Replay across layouts needs a separate representation design. |
| D-08 | accepted | BC uses deterministic `RBCSmartPolicy` demonstrations and actor-only updates. | BC lifecycle and template tests | Future algorithms may reuse storage but must define their own loss semantics. |
| D-09 | accepted | Safety runs outside the neural ONNX graph. | safety adapter and export implementation | Deployment needs an explicit post-processing contract. |
| D-10 | accepted | Checkpoints cannot save non-empty on-policy rollouts. | checkpoint manager test and `preflight_checkpoint` | Replay algorithms need a different checkpoint boundary rule. |
| D-11 | deferred | Exposing reward clipping as a schema field requires a separate implementation decision. | Agent reads fallback; `TransformerPPOHyperparameters` does not declare it. | Do not copy this hidden PPO option into another algorithm. |

### Historical notes

Closed PR 23 is historical context only. Its `RBCCommunityPolicy`, teacher
blending, probabilistic teacher replacement, and Transformer MATD3 proposals are
not current TPPO behavior.

### Transformer MATD3 readiness statement

The repository is ready for a separate Transformer MATD3 specification when it
reuses §1–§8 of the shared contract and explicitly defines deterministic actor
semantics, twin-critic centralization, replay transition representation across
layouts, target networks, delayed actor updates, target smoothing, exploration,
BC policy, checkpoint differences, export differences, metrics, and acceptance
tests. This document intentionally does not make those decisions.
