# Transformer PPO Recovery Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a mathematically valid TPPO implementation and six committed Wave A configurations for remote baseline, plain TPPO, and Smart-teacher BC screening.

**Architecture:** Keep the per-building Transformer architecture. Repair the collection/update boundary so PPO stores exact collection-time actions, log probabilities, and values. Move teacher execution into a separate demonstration episode and use teacher data only for supervised pretraining or an auxiliary loss while PPO executes actor-sampled actions. Use the wrapper's deterministic final episode as frozen evaluation.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pydantic, PyYAML, pytest, CityLearn entity interface.

---

## File Map

### PPO Core

- Modify `algorithms/utils/ppo_components.py`: rollout terminal masks, scalar value normalization, Huber critic loss, KL and explained-variance metrics.
- Modify `algorithms/agents/agent_transformer_ppo.py`: device placement, exact pending-decision cache, rollout lifecycle, BC phase lifecycle, topology boundaries, checkpoint state, diagnostics.
- Modify `algorithms/utils/behavior_cloning.py`: replace action phaseout with bounded demonstration storage and supervised training accessors.

### Runtime Contract

- Modify `algorithms/execution_unit.py`: optional episode lifecycle hook.
- Modify `algorithms/agents/base_agent.py`: inherit the lifecycle default.
- Modify `algorithms/pipeline.py`: delegate episode lifecycle hooks.
- Modify `utils/wrapper_citylearn.py`: announce training/evaluation phases and flush final rollouts through lifecycle hooks.
- Modify `utils/config_schema.py`: validate TPPO correctness and BC pretraining fields.

### Tests

- Modify `tests/test_ppo_components.py`: Huber gradient, masks, normalization, diagnostics.
- Modify `tests/test_agent_transformer_ppo.py`: exact collection cache, dropout/device behavior, partial rollouts, topology, checkpointing.
- Rewrite relevant tests in `tests/test_agent_transformer_ppo_behavior_cloning.py`: teacher-only demonstration collection, no teacher actions in PPO, pretraining, auxiliary BC.
- Modify `tests/test_agent_transformer_ppo_wrapper_integration.py`: episode lifecycle and frozen evaluation.
- Modify `tests/test_template_transformer_ppo_bc_15min.py`: new BC schema and Wave A templates.
- Modify `tests/test_run_experiment_runtime.py`: deterministic final episode remains update-free and exports final KPIs.

### Wave A Configurations

- Create `configs/recovery/tppo/wave_a/rbc_smart.yaml`.
- Create `configs/recovery/tppo/wave_a/rbc_community.yaml`.
- Create `configs/recovery/tppo/wave_a/tppo_plain.yaml`.
- Create `configs/recovery/tppo/wave_a/tppo_plain_conservative.yaml`.
- Create `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml`.
- Create `configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml`.
- Create `configs/recovery/tppo/wave_a/README.md`: run names, image/commit field, artifact checklist.

### Documentation

- Modify `docs/transformer_ppo_spec.md`: corrected on-policy and BC lifecycle.
- Include `docs/superpowers/specs/2026-08-02-tppo-recovery-campaign-design.md` and this plan in the Wave 0 commit.

## Task 1: Repair PPO Component Mathematics

**Files:**
- Modify: `algorithms/utils/ppo_components.py:142-357`
- Test: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing terminal-mask and Huber-gradient tests**

Add tests that use separate `terminated` and `truncated` flags. A truncated final step must include `gamma * last_value`; a terminated final step must not. Add a critic test with `values=0` and `returns=20` and assert a finite, nonzero gradient.

```python
def test_truncation_bootstraps_but_termination_does_not() -> None:
    truncated = RolloutBuffer(gamma=0.9, gae_lambda=1.0)
    terminated = RolloutBuffer(gamma=0.9, gae_lambda=1.0)
    kwargs = {
        "observation": torch.zeros(1),
        "action": torch.zeros(1, 1),
        "log_prob": torch.zeros(1),
        "reward": 1.0,
        "value": torch.zeros(1),
    }
    truncated.add(**kwargs, terminated=False, truncated=True)
    terminated.add(**kwargs, terminated=True, truncated=False)

    truncated.compute_returns_and_advantages(torch.tensor([2.0]))
    terminated.compute_returns_and_advantages(torch.tensor([2.0]))

    assert truncated.returns.item() == pytest.approx(2.8)
    assert terminated.returns.item() == pytest.approx(1.0)


def test_huber_value_loss_keeps_gradient_for_large_residual() -> None:
    values = torch.tensor([0.0], requires_grad=True)
    loss, _ = compute_ppo_loss(
        log_probs_new=torch.zeros(1, requires_grad=True),
        log_probs_old=torch.zeros(1),
        advantages=torch.ones(1),
        values=values,
        returns=torch.tensor([20.0]),
        clip_eps=0.2,
        value_coeff=0.5,
        entropy_coeff=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert values.grad is not None
    assert values.grad.abs().item() > 0.0
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
pytest tests/test_ppo_components.py -q
```

Expected: failures because `RolloutBuffer.add()` has only `done`, truncations do not bootstrap, and clamped MSE returns zero gradient.

- [ ] **Step 3: Store separate boundary flags**

Replace `dones` with `terminated` and `truncated` lists. Use only `terminated` to block bootstrap. Keep rollout flushes at every episode and topology boundary so GAE never crosses either boundary.

```python
self.terminated: List[bool] = []
self.truncated: List[bool] = []

def add(..., terminated: bool, truncated: bool) -> None:
    ...
    self.terminated.append(bool(terminated))
    self.truncated.append(bool(truncated))

for t in reversed(range(n)):
    bootstrap_mask = 1.0 - float(self.terminated[t])
    delta = self.rewards[t] + self.gamma * next_value * bootstrap_mask - values[t]
    gae = delta + self.gamma * self.gae_lambda * bootstrap_mask * gae
```

- [ ] **Step 4: Replace clamped MSE and add PPO metrics**

Use `torch.nn.functional.smooth_l1_loss(values, returns)`. Add:

```python
log_ratio = log_probs_new - log_probs_old
ratio = torch.exp(log_ratio)
approx_kl = ((ratio - 1.0) - log_ratio).mean()
explained_variance = 1.0 - torch.var(returns - values) / torch.var(returns).clamp_min(1e-8)
```

Expose `approx_kl`, `ratio_error_max`, and `explained_variance` in metrics. Compute `ratio_error_max` as `max(abs(ratio - 1))`.

- [ ] **Step 5: Add a scalar value normalizer**

Add `RunningValueNormalizer` to `ppo_components.py`. It stores `mean`, `variance`, and `count`, uses a parallel variance update, and provides `normalize()`, `denormalize()`, `state_dict()`, and `load_state_dict()`.

```python
class RunningValueNormalizer:
    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = torch.tensor(0.0)
        self.variance = torch.tensor(1.0)
        self.count = float(epsilon)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) / torch.sqrt(
            self.variance.to(values).clamp_min(1e-8)
        )

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * torch.sqrt(self.variance.to(values).clamp_min(1e-8)) + self.mean.to(values)
```

Update the normalizer from raw rollout returns before critic minibatches. Normalize critic targets; denormalize critic predictions used by GAE and bootstrap.

- [ ] **Step 6: Run component tests**

Run:

```bash
pytest tests/test_ppo_components.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the component repair**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "Wave 0: repair PPO returns and critic loss"
```

## Task 2: Preserve Exact Collection-Time Decisions

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py:59-335`
- Modify: `utils/config_schema.py:630-648`
- Test: `tests/test_agent_transformer_ppo.py`

- [ ] **Step 1: Write failing exact-ratio tests**

Add a test with configured Transformer dropout `0.1`. Call `predict()` once, then `update()` with the returned actions. Assert the buffer contains the exact cached action/log probability/value. Before optimization, assert `ratio_error_max <= 1e-5`.

Add a second test that passes actions different from the latest prediction and expects an actionable `ValueError` rather than silently training on mismatched data.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
pytest tests/test_agent_transformer_ppo.py -q
```

Expected: exact-decision tests fail because `update()` reconstructs values and log probabilities.

- [ ] **Step 3: Add per-building pending decisions**

Add a `_PendingDecision` dataclass:

```python
@dataclass
class _PendingDecision:
    observation: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
```

Store one pending decision per building in `predict()`. Actor actions, actor log probabilities, and denormalized critic values must come from the same forward pass. `update()` verifies the executed action matches the pending action and transfers this object into the rollout.

Clear pending decisions after transfer. A second `predict()` replaces stale deterministic-evaluation data. A topology change clears pending decisions for changed buildings.

- [ ] **Step 4: Make PPO representation deterministic**

Set TPPO dropout schema default to `0.0`. Reject any nonzero TPPO dropout during config validation with:

```text
AgentTransformerPPO requires transformer.dropout=0.0 because PPO old/new probability ratios must use the same representation.
```

Keep a unit test proving rejection. This is simpler and safer than toggling model mode around every collection/update path.

- [ ] **Step 5: Add actor variance configuration**

Add `actor_log_std_init: float = -0.5` to `TransformerPPOHyperparameters`, read it in TPPO, and pass it to `ActorHead`. This supports the conservative Wave A variant without code changes.

- [ ] **Step 6: Add CUDA device placement**

Reuse `_select_torch_device` and `_log_torch_runtime` from `maddpg_agent.py`. Add `require_cuda: bool = False` to TPPO hyperparameters. Move tokenizer, backbone, actor, critic, observations, rollout batches, bootstrap tensors, and normalizer tensors to `self.device`.

Wave A TPPO configurations set `require_cuda: true`; baseline policies do not.

- [ ] **Step 7: Run agent tests**

```bash
pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py -q
```

Expected: all tests pass, including exact decision identity and dropout rejection.

- [ ] **Step 8: Commit collection correctness**

```bash
git add algorithms/agents/agent_transformer_ppo.py utils/config_schema.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py
git commit -m "Wave 0: preserve TPPO on-policy decisions"
```

## Task 3: Make Rollout Flushing Lossless

**Files:**
- Modify: `algorithms/execution_unit.py:120-165`
- Modify: `algorithms/agents/base_agent.py:97-127`
- Modify: `algorithms/pipeline.py:170-190`
- Modify: `algorithms/agents/agent_transformer_ppo.py:279-335,605-695,782-906`
- Modify: `utils/wrapper_citylearn.py:1093-1605`
- Test: `tests/test_agent_transformer_ppo.py`
- Test: `tests/test_agent_transformer_ppo_wrapper_integration.py`
- Test: `tests/test_run_experiment_runtime.py`

- [ ] **Step 1: Write failing lifecycle tests**

Cover these behaviors:

- `update_step=True` with an undersized nonterminal rollout retains data.
- terminal episode flush trains a partial rollout with at least two samples.
- deterministic final episode invokes no `update()`.
- topology flush preserves unaffected building weights and logs a topology boundary.

- [ ] **Step 2: Run tests and confirm failure**

```bash
pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_run_experiment_runtime.py -q
```

Expected: failures for retained undersized rollouts and explicit episode lifecycle.

- [ ] **Step 3: Return update status instead of clearing unconditionally**

Change `_ppo_update()` and `_run_ppo_update_with_last_value()` to return `bool`. Clear PPO and aligned BC rollout buffers only after `True`.

Normal cadence requires `minibatch_size`. An episode-end flush permits a batch of two or more samples and uses `batch_size=min(minibatch_size, rollout_size)`.

- [ ] **Step 4: Add episode lifecycle hooks**

Add optional hooks to `ExecutionUnit` and delegate through `Pipeline` and `Ensemble`:

```python
def on_episode_start(self, *, episode: int, training: bool) -> None:
    return None

def on_episode_end(self, *, episode: int, training: bool) -> None:
    return None
```

The wrapper calls them after reset and after the step loop. `training` is `not deterministic`.

TPPO uses `on_episode_end(training=True)` to flush any valid partial rollout. It clears an invalid one-sample remainder with a warning because one normalized advantage cannot train PPO. It never carries an on-policy rollout across an optimizer update or episode reset.

- [ ] **Step 5: Correct terminal and topology bootstrapping**

Pass separate boundary flags to the rollout. A true termination gets zero bootstrap. A truncation computes the next-state critic value. A topology mutation closes the old layout with zero bootstrap and logs `rollout_boundary=topology_change`.

- [ ] **Step 6: Add config cadence validation**

In `ProjectConfig.validate_cross_constraints`, require:

```python
if isinstance(stage, TransformerPPOStageConfig):
    interval = self.training.steps_between_training_updates
    if interval <= 1 or interval < stage.hyperparameters.minibatch_size:
        raise ValueError(
            "AgentTransformerPPO requires training.steps_between_training_updates "
            ">= pipeline[].hyperparameters.minibatch_size."
        )
```

- [ ] **Step 7: Run lifecycle tests**

```bash
pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_run_experiment_runtime.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit rollout lifecycle**

```bash
git add algorithms/execution_unit.py algorithms/agents/base_agent.py algorithms/pipeline.py algorithms/agents/agent_transformer_ppo.py utils/wrapper_citylearn.py utils/config_schema.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_run_experiment_runtime.py
git commit -m "Wave 0: make TPPO rollout boundaries lossless"
```

## Task 4: Separate BC Demonstration And PPO Phases

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py`
- Modify: `algorithms/agents/agent_transformer_ppo.py`
- Modify: `utils/config_schema.py:650-683`
- Test: `tests/test_behavior_cloning_regularizer.py`
- Test: `tests/test_agent_transformer_ppo_behavior_cloning.py`

- [ ] **Step 1: Replace phaseout tests with separation tests**

Write tests proving:

- demonstration episodes execute only teacher actions and do not add PPO transitions;
- demonstration records include encoded observation, immutable layout snapshot, and teacher target;
- after demonstration collection, supervised pretraining changes actor parameters;
- PPO episodes execute actor actions only;
- auxiliary BC reads demonstrations but never changes executed actions;
- dynamic topology groups demonstrations by layout signature;
- the teacher is `RBCSmartPolicy` in campaign configs.

- [ ] **Step 2: Run BC tests and confirm failure**

```bash
pytest tests/test_behavior_cloning_regularizer.py tests/test_agent_transformer_ppo_behavior_cloning.py -q
```

Expected: failures because current BC blends or replaces environment actions.

- [ ] **Step 3: Define the new BC schema**

Replace phaseout semantics with:

```yaml
behavior_cloning:
  enabled: true
  demonstration_episodes: 1
  max_samples_per_building: 4096
  pretraining_epochs: 4
  batch_size: 64
  weight: 0.0
  min_weight: 0.0
  decay_start_step: 0
  decay_steps: 0
  ev_multiplier: 1.0
  storage_multiplier: 1.0
  teacher:
    policy: RBCSmartPolicy
    deterministic: true
    hyperparameters: {}
```

`weight` is auxiliary BC weight after pretraining. `weight=0` is pretraining-only. Remove `phaseout_steps`, `phaseout_mode`, and teacher noise from TPPO BC schema. Reject unknown legacy fields through `extra="forbid"` on TPPO BC models.

- [ ] **Step 4: Store bounded demonstrations**

Define a demonstration record with:

```python
@dataclass(frozen=True)
class Demonstration:
    observation: torch.Tensor
    action: torch.Tensor
    layout: BuildingTokenLayout
    layout_signature: Tuple[str, ...]
```

Use deterministic reservoir sampling per building, bounded by `max_samples_per_building`. Group minibatches by layout signature so tensors have compatible observation and action dimensions.

- [ ] **Step 5: Implement supervised pretraining**

During the configured demonstration episodes:

- `predict()` returns teacher actions;
- `update()` records demonstrations only;
- no PPO buffer receives a transition.

At the final demonstration episode boundary, train tokenizer, backbone, and actor against teacher actions for `pretraining_epochs`. Do not train the critic. Use weighted MSE on `tanh(actor mean)` and log sample count, raw BC loss, weighted CA loss, and epoch count.

- [ ] **Step 6: Implement actor-only PPO execution with optional auxiliary BC**

After pretraining, `predict()` always returns actor actions. During PPO minibatches, sample a demonstration minibatch independently and add the configured auxiliary BC loss. Auxiliary decay uses PPO learning steps, not environment action replacement progress.

- [ ] **Step 7: Run BC tests**

```bash
pytest tests/test_behavior_cloning_regularizer.py tests/test_agent_transformer_ppo_behavior_cloning.py -q
```

Expected: all pass and no test expects teacher blending.

- [ ] **Step 8: Commit BC separation**

```bash
git add algorithms/utils/behavior_cloning.py algorithms/agents/agent_transformer_ppo.py utils/config_schema.py tests/test_behavior_cloning_regularizer.py tests/test_agent_transformer_ppo_behavior_cloning.py
git commit -m "Wave 0: separate TPPO demonstrations from PPO"
```

## Task 5: Complete Checkpoint And Diagnostic State

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py:337-475,836-906`
- Modify: `algorithms/utils/behavior_cloning.py`
- Test: `tests/test_agent_transformer_ppo.py`
- Test: `tests/test_agent_transformer_ppo_behavior_cloning.py`

- [ ] **Step 1: Write failing checkpoint tests**

Train or mutate a small agent, save it, load into an attached fresh agent, and assert restoration of:

- model and optimizer state;
- value normalizer state;
- topology version and action names;
- global learning/update counters;
- BC demonstration/pretraining phase and decay progress.

- [ ] **Step 2: Run tests and confirm failure**

```bash
pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py -q
```

Expected: failures because current checkpoints omit normalizer, counters, topology version, and BC state.

- [ ] **Step 3: Extend checkpoint payload**

Add a checkpoint format version and all campaign-relevant state. Keep layout signature rejection. Load tensors with `map_location=self.device` and move optimizer tensors to the selected device.

Do not persist pending decisions. Reject checkpoint saves with an active nonempty rollout unless the save occurs at a completed optimizer/episode boundary.

- [ ] **Step 4: Add diagnostics**

Expose these metrics through `consume_latest_training_metrics()`:

```text
TPPO/update_count
TPPO/rollout_size
TPPO/policy_loss
TPPO/value_loss
TPPO/entropy
TPPO/clip_fraction
TPPO/approx_kl
TPPO/ratio_error_max
TPPO/explained_variance
TPPO/actor_grad_norm
TPPO/critic_grad_norm
TPPO/raw_reward_{mean,min,max}
TPPO/clipped_reward_{mean,min,max}
TPPO/value_residual_{p50,p90,p99}
TPPO/episode_training
TPPO/teacher_action_execution
```

Keep existing action diagnostics. Add per-building log lines and an explicit Building 15 line when present.

- [ ] **Step 5: Run checkpoint and diagnostic tests**

```bash
pytest tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit state completeness**

```bash
git add algorithms/agents/agent_transformer_ppo.py algorithms/utils/behavior_cloning.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py
git commit -m "Wave 0: persist and report TPPO training state"
```

## Task 6: Migrate Existing TPPO Templates And Documentation

**Files:**
- Modify: `configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml`
- Modify: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml`
- Modify: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_smoke.yaml`
- Modify: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_week.yaml`
- Modify: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_month.yaml`
- Modify: `configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_year.yaml`
- Modify: `tests/test_template_transformer_ppo_bc_entity_dynamic.py`
- Modify: `tests/test_template_transformer_ppo_bc_15min.py`
- Modify: `tests/test_template_transformer_ppo_entity_dynamic.py`
- Modify: `docs/transformer_ppo_spec.md`

- [ ] **Step 1: Update template tests first**

Assert every TPPO template has dropout `0.0`, valid update cadence, explicit `actor_log_std_init`, and no BC phaseout fields. Assert BC templates use the new demonstration schema and Smart teacher.

- [ ] **Step 2: Run tests and confirm failure**

```bash
pytest tests/test_template_transformer_ppo_bc_entity_dynamic.py tests/test_template_transformer_ppo_bc_15min.py tests/test_template_transformer_ppo_entity_dynamic.py -q
```

Expected: failures against legacy dropout and phaseout fields.

- [ ] **Step 3: Migrate templates**

Set `dropout: 0.0`, use `steps_between_training_updates: 256`, and migrate BC blocks. Preserve each template's intended horizon but remove names that imply teacher action blending.

- [ ] **Step 4: Update TPPO documentation**

Document exact pending-decision storage, Huber/value normalization, separate demonstration episodes, actor-only PPO execution, final deterministic evaluation, and required diagnostics.

- [ ] **Step 5: Run template and schema tests**

```bash
pytest tests/test_template_transformer_ppo_bc_entity_dynamic.py tests/test_template_transformer_ppo_bc_15min.py tests/test_template_transformer_ppo_entity_dynamic.py tests/test_config_validation.py -q
```

Expected: all pass.

- [ ] **Step 6: Run the focused Wave 0 suite**

```bash
pytest tests/test_ppo_components.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_behavior_cloning_regularizer.py tests/test_template_transformer_ppo_bc_entity_dynamic.py tests/test_template_transformer_ppo_bc_15min.py tests/test_template_transformer_ppo_entity_dynamic.py tests/test_run_experiment_runtime.py -q
```

Expected: zero failures.

- [ ] **Step 7: Commit the verified Wave 0 foundation**

Stage only TPPO recovery files, the approved spec, and this plan. Do not stage unrelated existing changes.

```bash
git add algorithms/agents/agent_transformer_ppo.py algorithms/agents/base_agent.py algorithms/execution_unit.py algorithms/pipeline.py algorithms/utils/behavior_cloning.py algorithms/utils/ppo_components.py utils/config_schema.py utils/wrapper_citylearn.py configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml configs/templates/dynamic/transformer_ppo_bc_entity_dynamic.yaml configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_smoke.yaml configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_week.yaml configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_month.yaml configs/templates/dynamic/transformer_ppo_bc_entity_dynamic_15min_year.yaml tests/test_ppo_components.py tests/test_agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_agent_transformer_ppo_wrapper_integration.py tests/test_behavior_cloning_regularizer.py tests/test_template_transformer_ppo_bc_entity_dynamic.py tests/test_template_transformer_ppo_bc_15min.py tests/test_template_transformer_ppo_entity_dynamic.py tests/test_run_experiment_runtime.py docs/transformer_ppo_spec.md docs/superpowers/specs/2026-08-02-tppo-recovery-campaign-design.md docs/superpowers/plans/2026-08-02-tppo-recovery-campaign-implementation.md
git commit -m "Wave 0: complete TPPO correctness foundation"
```

## Task 7: Create Wave A Remote Configurations

**Files:**
- Create: `configs/recovery/tppo/wave_a/rbc_smart.yaml`
- Create: `configs/recovery/tppo/wave_a/rbc_community.yaml`
- Create: `configs/recovery/tppo/wave_a/tppo_plain.yaml`
- Create: `configs/recovery/tppo/wave_a/tppo_plain_conservative.yaml`
- Create: `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml`
- Create: `configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml`
- Create: `configs/recovery/tppo/wave_a/README.md`
- Create: `tests/test_tppo_recovery_wave_a_configs.py`

- [ ] **Step 1: Write failing Wave A contract tests**

Load all six YAML files and validate them. Assert common scenario fields are equal:

```python
COMMON_SIMULATOR_FIELDS = (
    "dataset_name",
    "dataset_path",
    "interface",
    "topology_mode",
    "reward_function",
    "reward_function_kwargs",
    "simulation_start_time_step",
    "simulation_end_time_step",
    "episode_time_steps",
)
```

Assert:

- Smart and Community have one deterministic episode.
- Plain TPPO variants have two episodes and `deterministic_finish: true`.
- BC variants have three episodes: one demonstration, one PPO training, one deterministic evaluation.
- KPI export is final-episode-only.
- every TPPO config requires CUDA and uses dropout zero;
- only conservative plain changes `actor_log_std_init` and its declared run name;
- pretraining-only BC has auxiliary `weight: 0.0`;
- auxiliary BC has a positive decaying weight;
- all BC teachers are Smart;
- all six run names follow `tppo-recovery-wa-...-s7`.

- [ ] **Step 2: Run the test and confirm missing files**

```bash
pytest tests/test_tppo_recovery_wave_a_configs.py -q
```

Expected: failure because the Wave A directory does not exist.

- [ ] **Step 3: Create a common exact scenario**

Base every file on the dynamic 15-minute dataset used by the failed BC runs:

```yaml
simulator:
  dataset_name: citylearn_three_phase_dynamic_assets_only_demo_15min_parquet
  dataset_path: ./datasets/citylearn_three_phase_dynamic_assets_only_demo_15min_parquet/schema.json
  central_agent: false
  interface: entity
  topology_mode: dynamic
  entity_encoding:
    enabled: true
    normalization: minmax_space
    clip: true
  reward_function: CostHardConstraintReward
  reward_function_kwargs:
    export_credit_ratio: 0.0
    grid_violation_penalty: 60.0
    power_outage_penalty: 120.0
    ev_departure_window_hours: 1.0
    ev_departure_service_tolerance: 0.05
    ev_connected_deficit_penalty: 30.0
    ev_schedule_deficit_penalty: 120.0
    ev_departure_deficit_penalty: 120.0
    ev_departure_missed_penalty: 250.0
    battery_soc_min: 0.0
    battery_soc_max: 1.0
    use_observed_storage_soc_limits: true
    battery_soc_violation_penalty: 30.0
    battery_throughput_penalty: 0.2
    community_import_penalty: 0.01
    community_peak_import_penalty: 0.001
    community_penalty_divide_by_agents: true
    scale_state_penalties_by_time_step: true
    state_penalty_reference_seconds: 3600.0
```

Use the same community-market settings as the Smart baseline so cost and solar KPIs have identical semantics.

- [ ] **Step 4: Create baseline configs**

`rbc_smart.yaml` uses the tuned `RBCSmartPolicy` hyperparameters from `configs/templates/baselines/rbc_smart_15min_local.yaml`. `rbc_community.yaml` changes only the pipeline policy to `RBCCommunityPolicy` and uses its defaults unless Smart-derived parameters are accepted by that class. Both use seed 7, one episode, deterministic execution, and final-only KPI export.

- [ ] **Step 5: Create plain TPPO configs**

Both plain configs use two episodes with `deterministic_finish: true`. Episode 0 trains; episode 1 is frozen deterministic evaluation. Reference uses `actor_log_std_init: -0.5`. Conservative uses `-1.2`. Both use CUDA, 256-step rollouts, Huber loss, value normalization, and identical remaining hyperparameters.

- [ ] **Step 6: Create BC configs**

Both BC configs use three episodes with `deterministic_finish: true` and `demonstration_episodes: 1`. Episode 0 collects Smart demonstrations and pretrains. Episode 1 trains PPO with actor actions. Episode 2 evaluates.

Pretraining-only sets auxiliary weight zero. Auxiliary BC starts with a conservative nonzero weight, decays to zero during the PPO training episode, and uses equal EV/storage multipliers initially. Do not restore the previous `24.0` EV multiplier.

- [ ] **Step 7: Create the operator README**

Include a table with config path, UI run name, purpose, expected episode phases, and an initially blank `Required commit/image` field. State required exports: log, resolved YAML, KPI JSON/CSV, job ID, and image tag.

- [ ] **Step 8: Validate Wave A configs**

```bash
pytest tests/test_tppo_recovery_wave_a_configs.py -q
python3 - <<'PY'
from pathlib import Path
import yaml
from utils.config_schema import validate_config

for path in sorted(Path("configs/recovery/tppo/wave_a").glob("*.yaml")):
    validate_config(yaml.safe_load(path.read_text()))
    print(f"valid: {path}")
PY
```

Expected: all commands exit zero.

- [ ] **Step 9: Run the full test suite**

```bash
pytest -q
```

Expected: zero failures.

- [ ] **Step 10: Commit the runnable Wave A set**

Inspect `git status`, `git diff`, and recent log first. Stage only the six configs, README, test, and any focused fix required by validation.

```bash
git add configs/recovery/tppo/wave_a tests/test_tppo_recovery_wave_a_configs.py
git commit -m "Wave A: add TPPO recovery screening configs"
```

- [ ] **Step 11: Record the final Wave A commit for handoff**

Get the full SHA:

```bash
git rev-parse HEAD
```

The operator README states:

```text
Required commit/image: use the Wave A config commit reported in the final handoff.
```

Record the full and short SHA in the final response. The runnable image is the
commit created by Step 10. Do not create a metadata-only commit because a file
cannot contain the SHA of the commit that contains that file.

- [ ] **Step 12: Push the current branch**

Verify branch and upstream, then push without force:

```bash
git status --short
git log --oneline -10
git push
```

Expected: `gj/tppo_bclonning` is updated and CI begins building the commit-tagged image.

## Task 8: Final Operator Handoff

**Files:**
- Read: `configs/recovery/tppo/wave_a/README.md`

- [ ] **Step 1: Verify remote availability**

Confirm the pushed commit exists on the remote branch:

```bash
git rev-parse HEAD
git rev-parse @{upstream}
```

Expected: local and upstream SHAs match.

- [ ] **Step 2: Report only actionable run information**

Return:

- the six exact YAML paths;
- the required full commit SHA;
- the expected image tag;
- a note that all six can run in parallel;
- the four artifacts to export for each run.

Do not require timeseries unless Wave A aggregate evidence is insufficient.
