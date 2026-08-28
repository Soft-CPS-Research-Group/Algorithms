# AgentTransformerMATD3

`AgentTransformerMATD3` is an off-policy controller for CityLearn's entity
interface. It uses one Transformer actor stack per building and one independent
twin-critic pair per controlled building. Each critic observes the joint state
and actions of all buildings.

Use this guide for configuration and operation. Use the
[technical specification](transformer_matd3_spec.md) for invariants and data
flows. Use the [ADRs](adr/README.md) when changing an architectural decision.
Terms specific to this controller are defined in the
[glossary](transformer_matd3_glossary.md).

## Supported contract

- The simulator must use `interface: entity`.
- The controller supports `topology_mode: dynamic`.
- The pipeline must contain one `AgentTransformerMATD3` stage, in final
  position, with `count: 1`.
- The tokenizer must satisfy the shared
  [Transformer entity contract](transformer_entity_controller.md).
- Action position always follows the attached `action_names` order.
- CUDA is optional. Set `require_cuda: true` to reject CPU execution.

The registered implementation is
`algorithms/transformer_matd3/agent.py::AgentTransformerMATD3`. Configuration
validation is defined by `TransformerMATD3StageConfig` and its nested models in
`utils/config_schema.py`. These code symbols are authoritative when this guide
and the code disagree.

## Start a run

Choose the smallest template that provides the required behavior:

| Template | Purpose |
|---|---|
| [`transformer_matd3_entity_dynamic.yaml`](../configs/templates/dynamic/transformer_matd3_entity_dynamic.yaml) | Core Transformer MATD3 |
| [`transformer_matd3_entity_dynamic_residual.yaml`](../configs/templates/dynamic/transformer_matd3_entity_dynamic_residual.yaml) | Actor corrections around an RBC base policy |
| [`transformer_matd3_entity_dynamic_bc.yaml`](../configs/templates/dynamic/transformer_matd3_entity_dynamic_bc.yaml) | Replay BC, demonstration BC, and local action safety |
| [`transformer_matd3_entity_dynamic_cost4_faithful.yaml`](../configs/templates/dynamic/transformer_matd3_entity_dynamic_cost4_faithful.yaml) | Fifteen-minute cost4 translation over `RBCCommunityPolicy` |

The `cost4_realistic_pilot`, `cost4_realistic_fast_pilot`, and
`cost4_realistic_speed_pilot` templates are bounded diagnostics. They are not
promotion configurations. Matching `RBCCommunityPolicy` and `RBCSmartPolicy`
baseline templates use the same dynamic 15-minute dataset and reward.

Copy a template and change the dataset, duration, seed, and hyperparameters.
Do not enable optional runtime paths without their required inputs.

### Training diagnostics

With MLflow disabled, training diagnostics are written to `logs/metrics.jsonl`.
`TransformerMATD3/replay_action_q_mean` and
`replay_action_q_abs_mean` describe critic values at replay actions and persist
across critic-only updates. `building_<index>_target_*`, `td_abs_*`,
`gap_abs_*`, and `critic_*_grad_norm` describe the target, absolute TD error,
twin-critic gap, and critic gradient norms for that building. The corresponding
`*_max` values are tail diagnostics, not clipped training values.

`policy_replay_q_abs_gap` is the mean absolute difference between policy-action
and replay-action Q tensors from one actor-update event. It is not the
difference between aggregate means. `actor_update_event_count` identifies the
actor event that produced the value; critic-only flushes retain that event's
value instead of replacing it with a synthetic zero. A zero gap is valid only
when the paired tensors are equal.

Every 16th critic update, the agent records
`building_<index>_storage_critic_dq_da_abs_{mean,p95,max}`. These are absolute
partial derivatives of critic 1 with respect to that building's stored
electrical-storage action components, evaluated at replay actions. They are
local critic sensitivity diagnostics, not environment perturbation results, and
are emitted only when the derivative is available and finite. The companion
`storage_action_count` and `storage_critic_dq_da_available` fields distinguish
no storage action or a structurally disconnected critic input from a measured
zero sensitivity; disconnected derivatives are not reported as numeric zero.

Run locally:

```bash
python run_experiment.py \
  --config configs/templates/dynamic/transformer_matd3_entity_dynamic.yaml \
  --job_id transformer-matd3-local
```

For remote execution, use the normal OPEVA preparation and submission flow in
`scripts/manage_remote_experiment.py`. A merge-validation run should pin the
exact commit image, dataset, worker, seed, episode count, and time-step window.
The dataset interval alone does not define the run length:

```text
total environment steps = episodes * episode_time_steps
```

For example, two episodes with `episode_time_steps: 3401` execute 6,802 steps.
Two full 15-minute years with 35,040 steps each execute 70,080 steps.

## Core configuration

The pipeline stage has four sections:

```yaml
pipeline:
  - algorithm: AgentTransformerMATD3
    count: 1
    frozen: false
    tokenizer_config_path: configs/tokenizers/entity_default.json
    transformer:
      d_model: 64
      nhead: 4
      num_layers: 2
      dim_feedforward: 128
      dropout: 0.0
    hyperparameters:
      require_cuda: false
      learning_rate: 3.0e-4
      gamma: 0.99
      tau: 0.005
      batch_size: 128
      buffer_capacity: 100000
      max_grad_norm: 1.0
      target_policy_smoothing: true
      target_policy_noise: 0.2
      target_policy_noise_clip: 0.5
      actor_update_interval: 2
      actor_policy_loss_weight: 1.0
      sigma: 0.1
      sigma_decay: 0.9995
      min_sigma: 0.01
      bias: 0.0
      random_exploration_steps: 256
      end_initial_exploration_time_step: 256
    behavior_cloning:
      replay_based:
        enabled: false
      demonstration_based:
        enabled: false
```

Important rules:

- `d_model` must be divisible by `nhead`.
- `buffer_capacity` must be at least `batch_size`.
- `min_sigma` cannot exceed `sigma`.
- `random_exploration_steps` controls random actions.
- `end_initial_exploration_time_step` controls when learning may start.
- `actor_policy_loss_weight` must be non-negative. It scales the MATD3 policy
  term before optional BC terms. The cost4 translation uses `0.085`.
- `n_step_gamma` defaults to `gamma`.
- Unknown fields fail validation.

The templates and Pydantic models contain the complete field set and defaults.
Do not copy field lists from an ADR into a new config.

## Learning model

Each building owns independent online and target actor stacks:

```text
encoded observation
  -> entity tokenizer
  -> Transformer backbone
  -> per-CA actor head
  -> tanh and affine action bounds
```

Each controlled building also owns two independent centralized critics. A
critic encodes each building, injects its action after tokenization, aggregates
building embeddings with a permutation-invariant Deep Sets block, and returns
one Q value. Learning uses the minimum target Q, delayed actor updates, target
policy smoothing, and soft target updates.

Replay stores encoded transitions in buckets keyed by the full layout
signature. A gradient batch contains one signature only. If the current bucket
has fewer than `batch_size` transitions, the update is skipped and reported in
metrics.

## Dynamic topology

An asset-count change within existing buildings preserves compatible neural
weights, optimizer state, replay history, and training clocks. The controller
increments the topology version for changed buildings. New instances reuse
their asset-type projections and per-CA head.

The topology transaction rejects these schema changes before committing state:

- an existing segment changes order, family, type, instance identity, feature
  names, feature width, or NFC expression;
- a new entity type has no existing tokenizer projection;
- an existing building changes identity.

On every committed topology boundary, pending n-step transitions are flushed
as truncated. A building-count change performs a full controller rebuild and
clears replay, optimizer state, exploration state, normalizer state, and BC
state. Any failed attachment restores the previous controller and random state.

## Optional behavior

### Residual control

Set `residual_policy_enabled: true` and provide
`warm_start_policy_name`. The warm-start policy supplies a base action. The
Transformer actor supplies a bounded correction scaled by the configured
authority schedule. Replay, online critics, target critics, and actor policy-Q
all use the same proposed action. Runtime safety and service-teacher adapters
may produce a different executed action.

Set `local_action_safety_service_teacher_enabled: true` to preserve EV and
deferrable actions from the warm-start teacher. Set
`local_action_safety_service_teacher_eval_enabled: true` to apply the same
preservation during deterministic evaluation.

### Replay behavior cloning (BC-A)

Enable `behavior_cloning.replay_based`. BC-A trains only the actor stack from
cloning targets stored with replay transitions. Supported teachers are
`warm_start` and `replay_action`; `external` is rejected during config
validation. BC-A samples the current layout signature only.

### Demonstration behavior cloning (BC-B)

Enable `behavior_cloning.demonstration_based`. BC-B collects deterministic
`RBCSmartPolicy` demonstrations into a separate per-signature reservoir, then
pretrains the actor before RL. It can continue as an auxiliary actor loss.
Pretraining fails before RL when a building has no compatible demonstrations.

BC-A and BC-B use the actor optimizer for actor-only updates. Neither path
updates critics, targets, normalizer statistics, or replay state.

### Local action safety and price conditioning

The local safety adapter projects executed actions using raw CityLearn context.
The price adapter rewrites a copy of the encoded current or successor
observation before tokenization. Price conditioning requires the
`minmax_space` entity-encoding profile.

Both adapters are outside the exported ONNX graph.

## Checkpoints and resume

Checkpoint settings belong in the top-level `checkpointing` block, not inside
the algorithm stage:

```yaml
checkpointing:
  resume_training: false
  checkpoint_artifact: transformer_matd3_checkpoint.pt
  checkpoint_mode: full
  checkpoint_interval: 2048
```

Format 5 supports:

- `full`: actor and target stacks, critics, optimizers, replay, n-step queue,
  exploration, reward normalization, random generators, and enabled BC state;
- `inference`: actor stacks, action bounds, topology metadata, and the
  operational step needed by the residual schedule.

Restore is strict. Building count, building identity, layout signature, action
names, action bounds, checkpoint mode, enabled BC paths, and compatible BC-B
reservoir capacity must match. All validation completes before live state
changes. An inference checkpoint can only load into a frozen stage.

## Export and deployment

Export writes one opset-17 ONNX actor per building for the current topology:

```text
onnx_models/agent_<building-index>__topology_v<version>.onnx
```

The graph includes tokenization, the Transformer actor, `tanh`, and affine
action bounds. It excludes exploration, critics, residual composition, local
safety, and local price conditioning.

If residual, safety, or price behavior is enabled, export fails unless its
matching `*_runtime_only_export` flag is true. Opting in creates experiment
evidence marked `deployable: false`; it does not make the graph behaviorally
complete. Production serving must reproduce every required external adapter.

Run outputs follow the repository contract under `runs/jobs/<job_id>/`,
including `results/`, `checkpoints/`, `onnx_models/`, the resolved config, and
`artifact_manifest.json`.

## Verification

Run focused validation after changing this controller or its documentation:

```bash
pytest -q \
  tests/test_transformer_matd3_components.py \
  tests/test_transformer_matd3_replay.py \
  tests/test_agent_transformer_matd3.py \
  tests/test_agent_transformer_matd3_behavior_cloning.py \
  tests/test_agent_transformer_matd3_checkpoint.py \
  tests/test_agent_transformer_matd3_export.py \
  tests/test_agent_transformer_matd3_integration.py \
  tests/test_agent_transformer_matd3_residual.py \
  tests/test_agent_transformer_matd3_wrapper_integration.py \
  tests/test_template_transformer_matd3_entity_dynamic.py
```

Run the slow end-to-end contract explicitly:

```bash
pytest -q -o addopts='' -m slow \
  tests/e2e/test_e2e_transformer_matd3_entity_dynamic.py
```

## Known limits

- Only the entity interface is supported.
- Checkpoints cannot restore across layouts or building counts.
- Export contains only the current topology.
- Residual, safety, and price behavior require external serving logic.
- BC-A cannot train across historical signatures.
- Batches cannot mix layout signatures.
- Building-count changes reset all learned state.
# Critic loss and diagnostic index mapping

`hyperparameters.critic_loss_type` selects `mse` (the backward-compatible
default) or PyTorch Smooth L1/Huber loss. `critic_huber_delta` is the positive
Smooth L1 transition delta and defaults to `1.0`. The selected loss is applied
identically to both centralized critics. Per-building telemetry keys use
zero-based indices: `building_14_*` therefore refers to simulator `Building_15`.
