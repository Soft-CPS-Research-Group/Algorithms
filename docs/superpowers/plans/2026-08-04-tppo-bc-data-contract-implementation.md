# TPPO BC Data Contract And Setup Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the BC representation mismatch, expose the post-episode-1 setup stall, and validate both fixes locally before any server run.

**Architecture:** Store the encoded model observation and layout signature per demonstration. Pretrain each stored signature group with its stored layout. Fail fast on zero usable demonstrations. Bracket lifecycle callbacks and out-of-loop metadata attachment with watchdog phases. Ship a canary and a smoke config plus a runbook.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pydantic, PyYAML, pytest, CityLearn entity interface.

**Working branch:** `gj/tppo-recovery`. The Wave 0/A code that produced the failing log lives on this branch. Base every change on `gj/tppo-recovery`.

---

## File Map

### BC data contract

- Modify: `algorithms/utils/behavior_cloning.py:17-24` — add `encoded_length` to `Demonstration`.
- Modify: `algorithms/utils/behavior_cloning.py:183-210` — `record_demonstration` validates length and stores `encoded_length`.
- Modify: `algorithms/utils/behavior_cloning.py:280-293` — expose `rejected_at_record` in `snapshot_metrics()`.
- Modify: `algorithms/agents/agent_transformer_ppo.py:1500-1570` — replace `_run_bc_pretraining` grouping and shape check; add per-group logs and empty-sample failure.
- Modify: `algorithms/agents/agent_transformer_ppo.py:1615-1617` — remove `_infer_obs_dim` from BC validation path.

### Watchdog coverage

- Modify: `utils/wrapper_citylearn.py:447-448` — bracket the initial `_apply_entity_layout`/`_attach_model_environment_metadata` in `__init__` with `model_attach_start` / `model_attach_end`.
- Modify: `utils/wrapper_citylearn.py:1175-1177` — bracket `on_episode_start` with `episode_start_callback_start` / `episode_start_callback_end`.
- Modify: `utils/wrapper_citylearn.py:1394-1396` — bracket the mid-step `_attach_model_environment_metadata` with `model_attach_start` / `model_attach_end`.
- Modify: `utils/wrapper_citylearn.py:1634-1636` — bracket `on_episode_end` with `episode_end_callback_start` / `episode_end_callback_end`.

### Config reduction

- Modify: `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml` — reduce sample cap and epochs; require watchdog.
- Modify: `configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml` — same reduction and watchdog.

### Local validation harness

- Create: `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml`.
- Create: `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml`.
- Create: `configs/recovery/tppo/wave_a/local/README.md`.
- Create: `scripts/run_tppo_bc_local_checks.sh`.

### Tests

- Modify: `tests/test_behavior_cloning_regularizer.py` — `Demonstration.encoded_length`, shape rejection, grouping.
- Modify: `tests/test_agent_transformer_ppo_behavior_cloning.py` — encoded vector identity, multi-group pretraining, empty-sample failure, progress metrics.
- Modify: `tests/test_agent_transformer_ppo_wrapper_integration.py` — watchdog covers callbacks and out-of-loop attach.
- Modify: `tests/test_tppo_recovery_wave_a_configs.py` — reduced caps and canary/smoke presence.
- Create: `tests/test_tppo_bc_local_configs.py` — canary and smoke schema validation.

### Docs

- Modify: `docs/superpowers/plans/2026-08-02-tppo-recovery-campaign-implementation.md` — reference this plan.

---

## Task 1: Extend Demonstration record with encoded_length

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py:17-24`
- Test: `tests/test_behavior_cloning_regularizer.py`

- [ ] **Step 1: Write failing test for encoded_length**

Append to `tests/test_behavior_cloning_regularizer.py`:

```python
def test_demonstration_stores_encoded_length() -> None:
    regularizer = _regularizer(max_samples_per_building=4)
    layout = _layout(n_ca=1, observation_dim=6)
    observation = np.arange(6, dtype=np.float32)
    regularizer.record_demonstration(
        building_idx=0,
        observation=observation,
        layout=layout,
        target=[0.1],
    )
    demo = next(iter(regularizer.demonstrations_for_building_by_signature(0).values()))[0]
    assert demo.encoded_length == 6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_demonstration_stores_encoded_length -q
```

Expected: FAIL — `AttributeError: 'Demonstration' object has no attribute 'encoded_length'`.

- [ ] **Step 3: Add encoded_length field**

Change `Demonstration` in `algorithms/utils/behavior_cloning.py`:

```python
@dataclass(frozen=True)
class Demonstration:
    """One immutable encoded observation and its teacher action target."""

    observation: np.ndarray
    layout: BuildingTokenLayout
    layout_signature: Tuple[Any, ...]
    target: np.ndarray
    encoded_length: int
```

- [ ] **Step 4: Populate encoded_length in record_demonstration**

Change `record_demonstration` to derive and store the length:

```python
def record_demonstration(
    self,
    building_idx: int,
    observation: np.ndarray,
    layout: BuildingTokenLayout,
    target: List[float],
) -> None:
    copied_observation = np.asarray(observation, dtype=np.float32).copy()
    copied_target = np.asarray(target, dtype=np.float32).copy()
    if copied_target.shape != (layout.n_ca,) or not np.isfinite(copied_target).all():
        return
    copied_observation.setflags(write=False)
    copied_target.setflags(write=False)
    demo = Demonstration(
        observation=copied_observation,
        layout=deepcopy(layout),
        layout_signature=self.layout_signature(layout),
        target=copied_target,
        encoded_length=int(copied_observation.shape[0]),
    )
    demos = self._demonstrations.setdefault(building_idx, [])
    seen = self._seen_per_building.get(building_idx, 0) + 1
    self._seen_per_building[building_idx] = seen
    if len(demos) < self.max_samples_per_building:
        demos.append(demo)
        return
    replacement = self._rng.randrange(seen)
    if replacement < self.max_samples_per_building:
        demos[replacement] = demo
```

- [ ] **Step 5: Run the focused test**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_demonstration_stores_encoded_length -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add algorithms/utils/behavior_cloning.py tests/test_behavior_cloning_regularizer.py
git commit -m "feat(bc): record encoded_length per demonstration"
```

## Task 2: Reject shape-mismatched samples at record time

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py:183-210,280-293`
- Test: `tests/test_behavior_cloning_regularizer.py`

- [ ] **Step 1: Write failing rejection test**

Append to `tests/test_behavior_cloning_regularizer.py`:

```python
def test_record_demonstration_rejects_shape_mismatch() -> None:
    regularizer = _regularizer(max_samples_per_building=4)
    layout = _layout(n_ca=1, observation_dim=6)
    good = np.zeros(6, dtype=np.float32)
    bad = np.zeros(7, dtype=np.float32)
    regularizer.record_demonstration(0, good, layout, [0.1])
    regularizer.record_demonstration(0, bad, layout, [0.1])
    assert regularizer.demonstration_count(0) == 1
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_rejected_at_record"] == 1.0
```

`_layout(n_ca, observation_dim)` must return a `BuildingTokenLayout` whose
maximum `feature_indices` value equals `observation_dim - 1`. If the file
already exposes such a helper, reuse it. Otherwise add:

```python
from algorithms.utils.entity_token_layout import BuildingTokenLayout, Segment


def _layout(*, n_ca: int, observation_dim: int) -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id="Building_test",
        n_sro=0,
        n_ca=n_ca,
        ca_action_names=tuple(f"ca_action_{i}" for i in range(n_ca)),
        segments=(
            Segment(
                family="ca",
                type_name="storage",
                instance_id="storage_0",
                feature_indices=tuple(range(observation_dim)),
                feature_names=tuple(f"f_{i}" for i in range(observation_dim)),
            ),
        ),
    )
```

Adjust field names to whatever the current dataclass exposes on
`gj/tppo-recovery`. Import paths and field names take precedence over the
sketch above.

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_record_demonstration_rejects_shape_mismatch -q
```

Expected: FAIL — the regularizer accepts the mismatched sample.

- [ ] **Step 3: Add rejection counter and enforce shape check**

Add a counter and predicate in `BehaviorCloningRegularizer.__init__`:

```python
self._rejected_at_record: Dict[int, int] = {}
```

Add a helper method:

```python
@staticmethod
def _expected_encoded_length(layout: BuildingTokenLayout) -> int:
    return max(max(seg.feature_indices) for seg in layout.segments) + 1
```

Update `record_demonstration` to reject early:

```python
expected_length = self._expected_encoded_length(layout)
if copied_observation.shape != (expected_length,):
    self._rejected_at_record[building_idx] = (
        self._rejected_at_record.get(building_idx, 0) + 1
    )
    return
```

Add to `snapshot_metrics`:

```python
"behavior_cloning_rejected_at_record": float(
    sum(self._rejected_at_record.values())
),
```

- [ ] **Step 4: Run the rejection test**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_record_demonstration_rejects_shape_mismatch -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/behavior_cloning.py tests/test_behavior_cloning_regularizer.py
git commit -m "feat(bc): reject shape-mismatched demonstrations at record time"
```

## Task 3: Pretrain every stored signature group with its stored layout

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py:1500-1570`
- Test: `tests/test_agent_transformer_ppo_behavior_cloning.py`

- [ ] **Step 1: Write failing multi-signature test**

Append to `tests/test_agent_transformer_ppo_behavior_cloning.py`:

```python
def test_pretraining_trains_every_stored_signature_group() -> None:
    agent, dimension = _agent()
    state = agent._per_building[0]
    assert agent._bc is not None
    agent._bc.record_demonstration(
        0, np.ones(dimension, dtype=np.float32), state.layout, [0.5] * state.layout.n_ca
    )
    # Expand topology, then record on the new layout.
    _expand_charger_topology(agent, load_sample_observation_names_for_first_building())
    new_state = agent._per_building[0]
    new_dimension = max(max(seg.feature_indices) for seg in new_state.layout.segments) + 1
    agent._bc.record_demonstration(
        0, np.ones(new_dimension, dtype=np.float32), new_state.layout, [0.5] * new_state.layout.n_ca
    )
    trained_signatures: list = []
    original_loss = agent._bc.demonstration_loss

    def spy(**kwargs):
        trained_signatures.append(agent._bc.layout_signature(kwargs["layout"]))
        return original_loss(**kwargs)

    agent._bc.demonstration_loss = spy
    agent.on_episode_start(episode=0, training=True)
    agent.on_episode_end(episode=0, training=True)

    assert len(set(trained_signatures)) == 2
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_agent_transformer_ppo_behavior_cloning.py::test_pretraining_trains_every_stored_signature_group -q
```

Expected: FAIL — only the current-topology group trains, or none does.

- [ ] **Step 3: Rewrite `_run_bc_pretraining` to iterate every group**

Replace the body of `_run_bc_pretraining` in `algorithms/agents/agent_transformer_ppo.py`:

```python
def _run_bc_pretraining(self) -> None:
    """Fit representation and actor to every stored signature group."""
    assert self._bc is not None
    total_batches = 0
    empty_buildings: List[str] = []

    for building_idx, state in enumerate(self._per_building):
        grouped = self._bc.demonstrations_for_building_by_signature(building_idx)
        usable_samples = sum(len(demos) for demos in grouped.values())
        if usable_samples == 0:
            empty_buildings.append(state.building_id)
            continue

        planned_batches = sum(
            max(1, (len(demos) + self._bc.batch_size - 1) // self._bc.batch_size)
            for demos in grouped.values()
        ) * self._bc.pretraining_epochs
        logger.info(
            "BC pretraining building={} usable_samples={} planned_batches={} "
            "epochs={} groups={}",
            state.building_id,
            usable_samples,
            planned_batches,
            self._bc.pretraining_epochs,
            len(grouped),
        )

        for signature, demonstrations in grouped.items():
            group_layout = demonstrations[0].layout
            group_batches = 0
            for epoch in range(self._bc.pretraining_epochs):
                epoch_losses: List[float] = []
                for start in range(0, len(demonstrations), self._bc.batch_size):
                    batch = demonstrations[start : start + self._bc.batch_size]
                    observations = torch.as_tensor(
                        np.stack([demo.observation for demo in batch]),
                        dtype=torch.float,
                        device=self.device,
                    )
                    state.bc_optimizer.zero_grad()
                    tokenized = state.tokenizer(observations, group_layout)
                    ca_embeddings, _ = state.backbone(
                        tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
                    )
                    loss = self._bc.demonstration_loss(
                        layout=group_layout,
                        demonstrations=list(batch),
                        predicted_means=torch.tanh(state.actor.mlp(ca_embeddings)),
                        global_learning_step=0,
                        apply_weight=False,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(state.tokenizer.parameters())
                        + list(state.backbone.parameters())
                        + list(state.actor.parameters()),
                        self._max_grad_norm,
                    )
                    state.bc_optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu()))
                    group_batches += 1
                    total_batches += 1
                logger.info(
                    "BC pretraining building={} signature={} epoch={} batches={} "
                    "loss_mean={:.6f}",
                    state.building_id,
                    hash(signature),
                    epoch,
                    len(epoch_losses),
                    float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                )

    if empty_buildings:
        raise RuntimeError(
            "BC pretraining has zero compatible demonstrations for "
            f"{empty_buildings}. Check demonstration_episodes, teacher "
            "policy, and record_demonstration."
        )

    logger.info("BC pretraining complete total_batches={}", total_batches)
    self._bc.set_pretraining_epochs(self._bc.pretraining_epochs)
    self._bc.set_incompatible_demonstration_samples(0)
    self._latest_training_metrics.update(self._bc.snapshot_metrics())
```

Delete `set_incompatible_demonstration_samples` calls that reference historical
groups. The counter now reflects `rejected_at_record` samples only.

- [ ] **Step 4: Run BC pretraining tests**

```bash
pytest tests/test_agent_transformer_ppo_behavior_cloning.py -q
```

Expected: PASS. Any test that referenced the historical-topology warning is
updated in Task 4.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py
git commit -m "feat(bc): pretrain every stored signature group with its stored layout"
```

## Task 4: Fail fast when any building has zero usable demonstrations

**Files:**
- Modify: `algorithms/agents/agent_transformer_ppo.py:1500-1570`
- Test: `tests/test_agent_transformer_ppo_behavior_cloning.py`

- [ ] **Step 1: Write failing empty-sample failure test**

Append to `tests/test_agent_transformer_ppo_behavior_cloning.py`:

```python
def test_pretraining_raises_when_any_building_has_zero_usable_samples() -> None:
    agent, _ = _agent()
    agent.on_episode_start(episode=0, training=True)
    with pytest.raises(RuntimeError, match="zero compatible demonstrations"):
        agent.on_episode_end(episode=0, training=True)
```

- [ ] **Step 2: Run the test to confirm the raise behavior**

```bash
pytest tests/test_agent_transformer_ppo_behavior_cloning.py::test_pretraining_raises_when_any_building_has_zero_usable_samples -q
```

Expected: PASS if Task 3 is landed; else the test guides the raise.

- [ ] **Step 3: Update tests that previously asserted silent skipping**

In `tests/test_agent_transformer_ppo_behavior_cloning.py` and
`tests/test_agent_transformer_ppo_wrapper_integration.py`, remove or replace
assertions such as:

```python
assert pretraining_metrics[0]["behavior_cloning_incompatible_demonstration_samples"] == 2.0
```

Replace with:

```python
assert pretraining_metrics[0]["behavior_cloning_rejected_at_record"] == 0.0
```

- [ ] **Step 4: Run the touched suites**

```bash
pytest tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_agent_transformer_ppo_wrapper_integration.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_ppo.py tests/test_agent_transformer_ppo_behavior_cloning.py tests/test_agent_transformer_ppo_wrapper_integration.py
git commit -m "feat(bc): raise on zero usable demonstrations for any building"
```

## Task 5: Reject BC checkpoints without encoded_length

**Files:**
- Modify: `algorithms/utils/behavior_cloning.py:310-322`
- Test: `tests/test_behavior_cloning_regularizer.py`

- [ ] **Step 1: Write failing checkpoint-rejection test**

Append to `tests/test_behavior_cloning_regularizer.py`:

```python
def test_load_state_dict_rejects_pre_contract_demonstrations() -> None:
    regularizer = _regularizer(max_samples_per_building=4)
    layout = _layout(n_ca=1, observation_dim=4)
    legacy = Demonstration.__new__(Demonstration)
    object.__setattr__(legacy, "observation", np.zeros(4, dtype=np.float32))
    object.__setattr__(legacy, "layout", layout)
    object.__setattr__(legacy, "layout_signature", regularizer.layout_signature(layout))
    object.__setattr__(legacy, "target", np.zeros(1, dtype=np.float32))
    state = regularizer.state_dict()
    state["demonstrations"] = {0: [legacy]}
    with pytest.raises(RuntimeError, match="predates BC data contract"):
        regularizer.load_state_dict(state)
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_load_state_dict_rejects_pre_contract_demonstrations -q
```

Expected: FAIL — the loader accepts the legacy object.

- [ ] **Step 3: Add the guard in load_state_dict**

At the start of `BehaviorCloningRegularizer.load_state_dict`:

```python
demonstrations = state["demonstrations"]
for demos in demonstrations.values():
    for demo in demos:
        if not hasattr(demo, "encoded_length"):
            raise RuntimeError(
                "Checkpoint predates BC data contract. Re-collect "
                "demonstrations under the current representation before "
                "resuming."
            )
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_behavior_cloning_regularizer.py::test_load_state_dict_rejects_pre_contract_demonstrations -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/behavior_cloning.py tests/test_behavior_cloning_regularizer.py
git commit -m "feat(bc): reject checkpoints without encoded_length"
```

## Task 6: Bracket lifecycle callbacks with watchdog phases

**Files:**
- Modify: `utils/wrapper_citylearn.py:1175-1177,1634-1636`
- Test: `tests/test_agent_transformer_ppo_wrapper_integration.py`

- [ ] **Step 1: Write failing watchdog coverage test**

Append to `tests/test_agent_transformer_ppo_wrapper_integration.py`:

```python
def test_watchdog_arms_around_on_episode_start_and_end(monkeypatch) -> None:
    env = _DummyEntityEnvForPPO()
    wrapper = Wrapper_CityLearn(
        env=env,
        config=_entity_config(stall_watchdog_enabled=True, stall_watchdog_timeout_seconds=5.0),
        job_id="watchdog-callbacks",
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)
    phases: list[str] = []

    def record(phase, **_kwargs):
        phases.append(phase)

    monkeypatch.setattr(wrapper, "_arm_stall_watchdog", record)
    wrapper.learn(episodes=1, deterministic=True)
    assert "episode_start_callback_start" in phases
    assert "episode_end_callback_start" in phases
```

Extend `_entity_config` (or add a shim) to accept the two watchdog kwargs.

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_agent_transformer_ppo_wrapper_integration.py::test_watchdog_arms_around_on_episode_start_and_end -q
```

Expected: FAIL — neither phase appears.

- [ ] **Step 3: Add phase brackets in wrapper**

Around `on_episode_start` in `utils/wrapper_citylearn.py`:

```python
on_episode_start = getattr(self.model, "on_episode_start", None)
if callable(on_episode_start):
    self._write_phase_progress(
        phase="episode_start_callback_start",
        episode=episode,
        step=0,
        episode_total=episodes,
        step_total=episode_step_total,
        global_step_total=global_step_total,
    )
    on_episode_start(episode=episode, training=not deterministic)
    self._write_phase_progress(
        phase="episode_start_callback_end",
        episode=episode,
        step=0,
        episode_total=episodes,
        step_total=episode_step_total,
        global_step_total=global_step_total,
    )
```

Around `on_episode_end`:

```python
on_episode_end = getattr(self.model, "on_episode_end", None)
if callable(on_episode_end):
    self._write_phase_progress(
        phase="episode_end_callback_start",
        episode=episode,
        step=max(time_step - 1, 0),
        episode_total=episodes,
        step_total=episode_step_total,
        global_step_total=global_step_total,
    )
    on_episode_end(episode=episode, training=not deterministic)
    self._write_phase_progress(
        phase="episode_end_callback_end",
        episode=episode,
        step=max(time_step - 1, 0),
        episode_total=episodes,
        step_total=episode_step_total,
        global_step_total=global_step_total,
    )
```

- [ ] **Step 4: Run the watchdog test**

```bash
pytest tests/test_agent_transformer_ppo_wrapper_integration.py::test_watchdog_arms_around_on_episode_start_and_end -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/wrapper_citylearn.py tests/test_agent_transformer_ppo_wrapper_integration.py
git commit -m "feat(wrapper): bracket lifecycle callbacks with watchdog phases"
```

## Task 7: Bracket out-of-loop metadata attachment with watchdog phases

**Files:**
- Modify: `utils/wrapper_citylearn.py:447-448,1394-1396`
- Test: `tests/test_agent_transformer_ppo_wrapper_integration.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_agent_transformer_ppo_wrapper_integration.py`:

```python
def test_watchdog_arms_around_out_of_loop_model_attach(monkeypatch) -> None:
    env = _TerminalTopologyChangeEntityEnvForPPO(truncated=False)
    wrapper = Wrapper_CityLearn(
        env=env,
        config=_entity_config(stall_watchdog_enabled=True, stall_watchdog_timeout_seconds=5.0),
        job_id="watchdog-attach",
    )
    agent = AgentTransformerPPO(_ppo_full_config())
    wrapper.set_model(agent)
    phases: list[str] = []
    monkeypatch.setattr(wrapper, "_arm_stall_watchdog", lambda phase, **_k: phases.append(phase))
    wrapper.learn(episodes=1, deterministic=False)
    assert phases.count("model_attach_start") >= 1
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_agent_transformer_ppo_wrapper_integration.py::test_watchdog_arms_around_out_of_loop_model_attach -q
```

Expected: FAIL — no `model_attach_start` phase is recorded.

- [ ] **Step 3: Add phase brackets**

Wrap the two known out-of-loop `_attach_model_environment_metadata()` sites
in `utils/wrapper_citylearn.py`:

Initial reset in `__init__` (or the earliest point where the wrapper writes
phase progress). Introduce a helper if a bare progress write is not
available at construction time:

```python
def _attach_model_environment_metadata_with_watchdog(self, *, phase_hint: str) -> None:
    self._write_phase_progress(
        phase="model_attach_start",
        episode=getattr(self, "_current_episode", 0),
        step=0,
        episode_total=None,
        step_total=None,
        global_step_total=None,
        extra={"attach_source": phase_hint},
    )
    self._attach_model_environment_metadata()
    self._write_phase_progress(
        phase="model_attach_end",
        episode=getattr(self, "_current_episode", 0),
        step=0,
        episode_total=None,
        step_total=None,
        global_step_total=None,
        extra={"attach_source": phase_hint},
    )
```

Call the helper at every previous call site of
`_attach_model_environment_metadata()` except the ones already inside
`_apply_entity_layout`, which is already bracketed by `entity_layout_*`.

- [ ] **Step 4: Run watchdog tests**

```bash
pytest tests/test_agent_transformer_ppo_wrapper_integration.py -q -k watchdog
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/wrapper_citylearn.py tests/test_agent_transformer_ppo_wrapper_integration.py
git commit -m "feat(wrapper): bracket out-of-loop model attach with watchdog phases"
```

## Task 8: Reduce Wave A BC caps and require watchdog

**Files:**
- Modify: `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml`
- Modify: `configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml`
- Modify: `tests/test_tppo_recovery_wave_a_configs.py`

- [ ] **Step 1: Update the config test to expect the new caps**

In `tests/test_tppo_recovery_wave_a_configs.py`, change the auxiliary and
pretrain assertions:

```python
assert pretrain["pipeline"][0]["behavior_cloning"]["max_samples_per_building"] == 4096
assert pretrain["pipeline"][0]["behavior_cloning"]["pretraining_epochs"] == 2
assert auxiliary_bc["max_samples_per_building"] == 4096
assert auxiliary_bc["pretraining_epochs"] == 2
```

Add a watchdog assertion:

```python
tracking = pretrain.get("tracking", {})
assert tracking.get("stall_watchdog_enabled") is True
assert float(tracking.get("stall_watchdog_timeout_seconds") or 0) > 0
```

Repeat for `auxiliary`.

- [ ] **Step 2: Run the config test to confirm failure**

```bash
pytest tests/test_tppo_recovery_wave_a_configs.py -q
```

Expected: FAIL until the YAML files are updated.

- [ ] **Step 3: Reduce the BC caps and add watchdog to Wave A configs**

Change both `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml` and
`configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml`:

```yaml
    behavior_cloning:
      enabled: true
      demonstration_episodes: 1
      max_samples_per_building: 4096
      pretraining_epochs: 2
      batch_size: 64
```

Add or overwrite a `tracking` section at the top level:

```yaml
tracking:
  stall_watchdog_enabled: true
  stall_watchdog_timeout_seconds: 600.0
```

- [ ] **Step 4: Run the config test**

```bash
pytest tests/test_tppo_recovery_wave_a_configs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml configs/recovery/tppo/wave_a/tppo_bc_auxiliary.yaml tests/test_tppo_recovery_wave_a_configs.py
git commit -m "chore(wave-a): cap BC pretraining and require watchdog"
```

## Task 9: Add the canary config

**Files:**
- Create: `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml`
- Create: `tests/test_tppo_bc_local_configs.py`

- [ ] **Step 1: Write failing schema test**

Create `tests/test_tppo_bc_local_configs.py`:

```python
from pathlib import Path
import yaml
from utils.config_schema import validate_config


def test_canary_config_validates() -> None:
    path = Path("configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml")
    document = yaml.safe_load(path.read_text())
    validated = validate_config(document)
    assert validated.simulator.episodes == 3
    bc = validated.pipeline[0].behavior_cloning
    assert bc.demonstration_episodes == 1
    assert bc.max_samples_per_building == 16
    assert bc.pretraining_epochs == 1
    assert bc.batch_size == 4
```

- [ ] **Step 2: Run the test to confirm the file is missing**

```bash
pytest tests/test_tppo_bc_local_configs.py::test_canary_config_validates -q
```

Expected: FAIL — `FileNotFoundError`.

- [ ] **Step 3: Create the canary config**

Add `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml` with
the same shape as `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml`,
diverging only in:

```yaml
metadata:
  run_name: tppo-recovery-wa-local-canary
simulator:
  episodes: 3
  simulation_start_time_step: 0
  simulation_end_time_step: 15
  episode_time_steps: 16
tracking:
  stall_watchdog_enabled: true
  stall_watchdog_timeout_seconds: 60.0
training:
  seed: 7
  steps_between_training_updates: 8
pipeline:
  - algorithm: AgentTransformerPPO
    count: 1
    hyperparameters:
      require_cuda: false
      minibatch_size: 8
    behavior_cloning:
      demonstration_episodes: 1
      max_samples_per_building: 16
      pretraining_epochs: 1
      batch_size: 4
```

Keep all other tokenizer, transformer, and BC teacher fields identical to
the Wave A pretrain config so the runtime path is exercised end to end.

- [ ] **Step 4: Validate the config**

```bash
pytest tests/test_tppo_bc_local_configs.py::test_canary_config_validates -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml tests/test_tppo_bc_local_configs.py
git commit -m "feat(canary): add TPPO BC pretrain canary config"
```

## Task 10: Add the smoke config

**Files:**
- Create: `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml`
- Modify: `tests/test_tppo_bc_local_configs.py`

- [ ] **Step 1: Extend the schema test**

Append to `tests/test_tppo_bc_local_configs.py`:

```python
def test_smoke_config_validates() -> None:
    path = Path("configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml")
    document = yaml.safe_load(path.read_text())
    validated = validate_config(document)
    assert validated.simulator.episodes == 3
    assert validated.simulator.episode_time_steps == 192
    bc = validated.pipeline[0].behavior_cloning
    assert bc.max_samples_per_building == 128
    assert bc.pretraining_epochs == 1
    assert bc.batch_size == 16
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
pytest tests/test_tppo_bc_local_configs.py::test_smoke_config_validates -q
```

Expected: FAIL — file missing.

- [ ] **Step 3: Create the smoke config**

Add `configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml`,
based on `configs/recovery/tppo/wave_a/tppo_bc_pretrain.yaml` with these
overrides:

```yaml
metadata:
  run_name: tppo-recovery-wa-local-smoke
simulator:
  episodes: 3
  simulation_start_time_step: 0
  simulation_end_time_step: 191
  episode_time_steps: 192
tracking:
  stall_watchdog_enabled: true
  stall_watchdog_timeout_seconds: 120.0
pipeline:
  - algorithm: AgentTransformerPPO
    count: 1
    hyperparameters:
      require_cuda: false
      minibatch_size: 16
    behavior_cloning:
      demonstration_episodes: 1
      max_samples_per_building: 128
      pretraining_epochs: 1
      batch_size: 16
```

- [ ] **Step 4: Validate the smoke config**

```bash
pytest tests/test_tppo_bc_local_configs.py -q
```

Expected: PASS on both tests.

- [ ] **Step 5: Commit**

```bash
git add configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml tests/test_tppo_bc_local_configs.py
git commit -m "feat(smoke): add TPPO BC pretrain smoke config"
```

## Task 11: Add the runbook and the runner script

**Files:**
- Create: `configs/recovery/tppo/wave_a/local/README.md`
- Create: `scripts/run_tppo_bc_local_checks.sh`

- [ ] **Step 1: Write the README**

Create `configs/recovery/tppo/wave_a/local/README.md`:

```markdown
# Local TPPO BC Validation

Two configurations gate every server BC run.

## Canary

Synthetic entity payload, CPU only, three tiny episodes.

Purpose: prove pretraining, watchdog coverage, and episode-2 setup path
work without depending on the real dataset.

## Smoke

Real dynamic 15-minute dataset, first 192 steps per episode, three
episodes.

Purpose: exercise the true wrapper and adapter path against realistic
data at bounded cost.

## Pass criteria (both configs)

- `Completed episode 3/3` printed;
- pretraining log lines show usable samples greater than zero for every
  building;
- pretraining log lines show trained batches greater than zero for every
  building;
- no `Skipping behavior-cloning demonstrations` warning;
- no stall watchdog activation.

## Run

```bash
bash scripts/run_tppo_bc_local_checks.sh
```
```

- [ ] **Step 2: Write the runner script**

Create `scripts/run_tppo_bc_local_checks.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CANARY=configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml
SMOKE=configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml
LOG_DIR=${LOG_DIR:-runs/local_bc_checks}

mkdir -p "$LOG_DIR"

python3 -m algorithms.run_experiment --config "$CANARY" \
    --job-id local-bc-canary 2>&1 | tee "$LOG_DIR/canary.log"
python3 -m algorithms.run_experiment --config "$SMOKE" \
    --job-id local-bc-smoke 2>&1 | tee "$LOG_DIR/smoke.log"

for log in "$LOG_DIR/canary.log" "$LOG_DIR/smoke.log"; do
    grep -q "Completed episode 3/3" "$log" || {
        echo "FAIL: $log did not reach episode 3/3" >&2
        exit 1
    }
    if grep -q "Skipping behavior-cloning demonstrations" "$log"; then
        echo "FAIL: $log contains BC skip warnings" >&2
        exit 1
    fi
done

echo "Local BC checks passed."
```

Make it executable:

```bash
chmod +x scripts/run_tppo_bc_local_checks.sh
```

- [ ] **Step 3: Commit**

```bash
git add configs/recovery/tppo/wave_a/local/README.md scripts/run_tppo_bc_local_checks.sh
git commit -m "docs(canary): add local BC validation runbook and runner"
```

## Task 12: Run the full test suite and both local configs

**Files:** (none new)

- [ ] **Step 1: Run all tests**

```bash
pytest -q
```

Expected: 0 failures. Fix any regression before proceeding.

- [ ] **Step 2: Run the canary end to end**

```bash
bash scripts/run_tppo_bc_local_checks.sh
```

Expected: exits `0`. Log `runs/local_bc_checks/canary.log` contains:

- `BC pretraining building=` log lines with `usable_samples>=1`;
- `BC pretraining complete total_batches=` with a positive integer;
- `Completed episode 3/3`;
- no `Skipping behavior-cloning demonstrations` warning;
- no `stall_watchdog` traceback file.

- [ ] **Step 3: Inspect the smoke log**

Repeat the checks against `runs/local_bc_checks/smoke.log`.

- [ ] **Step 4: Commit any test-only follow-ups**

If the run surfaces a missing assertion or an obviously wrong log line,
land the smallest fix and commit it:

```bash
git add <files>
git commit -m "fix(bc-local): <specific correction>"
```

## Task 13: Wire this plan into the recovery plan

**Files:**
- Modify: `docs/superpowers/plans/2026-08-02-tppo-recovery-campaign-implementation.md`

- [ ] **Step 1: Add a preflight reference**

At the top of the recovery plan (below the header), add:

```markdown
> **Preflight:** Before any Wave A BC server submission, complete
> `docs/superpowers/plans/2026-08-04-tppo-bc-data-contract-implementation.md`
> and its canary and smoke runs. Do not submit a BC job until both pass.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-08-02-tppo-recovery-campaign-implementation.md
git commit -m "docs(recovery): require BC data contract preflight"
```

## Task 14: Push for the server run

**Files:** (none)

- [ ] **Step 1: Inspect final state**

```bash
git status --short
git log --oneline -20
```

Expected: clean tree, all commits reflect the tasks above.

- [ ] **Step 2: Push the recovery branch**

```bash
git push origin gj/tppo-recovery
```

Expected: remote updated. CI begins building the image if the pipeline
requires it.
