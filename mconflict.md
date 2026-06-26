# Merge Conflict Resolution Plan — `gj/wp06-template-and-e2e` ← `main`

> Authoritative context document for any execution agent finishing the merge.
> Operating constraints below apply to **every** edit made under this plan.

## Operating constraints

1. **Minimal code.** No bloat. No unnecessary abstractions. Prefer the smallest
   change that keeps both feature sets working.
2. **No comments unless they add real value.** Strip existing low-value
   comments encountered in the conflict regions, *especially* any that
   reference `docs/spec.md`, `spec v2`, section numbers, or merge history.
   Code should explain itself; only keep comments that document non-obvious
   invariants future readers will not infer from the code.
3. **Be explicit, not over-engineered.** Avoid clever indirection. A direct
   call is preferred over a new helper unless reuse is real.
4. **Don't break what already passes.** Run focused tests after each step and
   the full suite + smoke run at the end.

---

## Branch context

| Side | What it brings |
|---|---|
| `HEAD` (`gj/wp06-template-and-e2e`) | Single-algorithm shape (`algorithm:` block). Adds `AgentTransformerPPO` + `TransformerPPOAlgorithmConfig`. Adds `_topology_changed_during_step` flag that lets the wrapper skip `model.update` across a topology boundary. Adds registry-driven `supports_dynamic_topology` gate. |
| `MERGE_HEAD` (`main`, commit `3d9a926`) | Pipeline shape (`pipeline: List[stages]`), vertical hierarchies (CC, BuildingAgent, MASAC/MATD3/IPPO/MAPPO/HAPPO). Layout-rebuild block inside `_apply_entity_layout`. Phase-progress instrumentation around `model.update`. Gate replaced by `_algorithm_names & ENCODED_OBSERVATION_ALGORITHMS`. |

---

## The two unmerged files

### 1. `utils/config_schema.py` — three hunks

#### Conflict A (lines 758–775) — `ProjectConfig` body field
- HEAD: `algorithm: Union[...]` including `TransformerPPOAlgorithmConfig`.
- main: `pipeline: List[PipelineStageConfig]`.

Pipeline is what every other merged caller uses (registry, wrapper, every
template). `TransformerPPOAlgorithmConfig` is not in `PipelineStageConfig`.

#### Conflict B (lines 783–817) — `validate_cross_constraints`
- HEAD: registry-driven `supports_dynamic_topology` via `self.algorithm.name`.
- main: `stage_names & ENCODED_OBSERVATION_ALGORITHMS` over `self.pipeline`.

`AgentTransformerPPO._use_raw_observations` is `False`, so it IS in
`ENCODED_OBSERVATION_ALGORITHMS`. Main's check would reject it under dynamic
topology — kills the whole feature. The registry-driven flag is the right
axis (`AgentTransformerPPO.supports_dynamic_topology = True`).

#### Conflict C (lines 840–891) — `validate_config()`
- HEAD: loads + 5-rule validates the tokenizer JSON for `TransformerPPOAlgorithmConfig`.
- main: deprecation guard — raises if raw config has `algorithm` without `pipeline`.

Orthogonal. Both required.

---

### 2. `utils/wrapper_citylearn.py` — two hunks

#### Conflict D (lines 477–529) — inside `_apply_entity_layout`
- HEAD: sets `self._topology_changed_during_step`; runs `supports_dynamic_topology`
  gate; **references stale singular `self._algorithm_name` which no longer exists**
  (main replaced it with `self._algorithm_names: set[str]`).
- main: on `topology_changed`, rebuilds `observation_names`, `observation_space`,
  `action_space`, `action_names`, `encoders`, `_action_bounds_cache`. Then runs
  the `ENCODED_OBSERVATION_ALGORITHMS` gate.

Both needed. The layout-rebuild block is mandatory infrastructure for dynamic
topology. The flag is what tells the training loop to skip a cross-layout
`update`. The gate has the same fix as Conflict B.

#### Conflict E (lines 1329–1393) — `self.update(...)` in training loop
- HEAD: `if self._topology_changed_during_step: skip; else: self.update(...)`.
- main: instruments update with `_write_phase_progress("model_update_start/end")`
  and timing into `runtime_profile_metrics`.

Pure orthogonal — combine.

---

## Knock-on items (outside conflict markers)

1. `configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml` still uses
   `algorithm:` top-level. Migrate to `pipeline:` shape.
2. `tests/test_template_transformer_ppo_entity_dynamic.py` asserts
   `cfg["algorithm"]["name"]`. Switch to `cfg["pipeline"][0]["algorithm"]`.
3. `algorithms/registry.py::_stage_to_agent_view` propagates `hyperparameters`,
   `networks`, `replay_buffer`, `exploration`, `policy` — but **not**
   `tokenizer_config_path` nor `transformer`. The agent reads both from
   `config["algorithm"][...]`. Add them.
4. Unit tests `test_agent_transformer_ppo.py` /
   `test_agent_transformer_ppo_wrapper_integration.py` use the agent-view
   shape already and stay as-is once Step 3 lands.

---

## Resolution steps

### Step 1 — `utils/config_schema.py`

**1a.** Add a pipeline-stage variant for Transformer PPO. Replace the existing
`TransformerPPOAlgorithmConfig` (which uses `name:` and lives in the old
top-level union) with:

```python
class TransformerPPOStageConfig(BaseModel):
    algorithm: Literal["AgentTransformerPPO"]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    tokenizer_config_path: str = Field(min_length=1)
    transformer: TransformerPPOTransformerConfig
    hyperparameters: TransformerPPOHyperparameters
```

Add it to `PipelineStageConfig`:

```python
PipelineStageConfig = Union[
    BuildingAgentStageConfig,
    CCLevel1AlgorithmConfig,
    CCLevel2AlgorithmConfig,
    CommunityCoordinatorAlgorithmConfig,
    ActorCriticAlgorithmConfig,
    RuleBasedAlgorithmConfig,
    SingleAgentRLStageConfig,
    TransformerPPOStageConfig,
]
```

**1b.** Conflict A — keep only main's `pipeline:` field. Delete the
`algorithm:` Union line block entirely.

**1c.** Conflict B — single fused validator:

```python
if self.simulator.interface == "entity" and self.simulator.topology_mode == "dynamic":
    from algorithms.registry import ALGORITHM_REGISTRY
    for stage in self.pipeline:
        agent_cls = ALGORITHM_REGISTRY.get(stage.algorithm)
        if agent_cls is None:
            continue
        if not bool(getattr(agent_cls, "supports_dynamic_topology", False)):
            if stage.algorithm == "MADDPG":
                raise ValueError(
                    "algorithm.name='MADDPG' does not support simulator.interface='entity' "
                    "with simulator.topology_mode='dynamic'."
                )
            raise ValueError(
                f"algorithm={stage.algorithm!r} does not support simulator.topology_mode='dynamic' "
                "(supports_dynamic_topology=False)."
            )
```

MADDPG wording preserved; `AgentTransformerPPO` passes.

**1d.** Conflict C — fold both:

```python
def validate_config(raw_config: Dict[str, Any]) -> ProjectConfig:
    if isinstance(raw_config, dict) and "algorithm" in raw_config and "pipeline" not in raw_config:
        raise ValueError(
            "Configuration uses the deprecated top-level 'algorithm' key. "
            "Migrate to a 'pipeline' list, e.g.:\n\n"
            "  pipeline:\n"
            "    - algorithm: \"<name>\"\n"
            "      count: 1\n"
            "      hyperparameters: { ... }\n"
            "      networks: { ... }\n"
            "      replay_buffer: { ... }\n"
            "      exploration: { ... }\n"
        )
    project = ProjectConfig.model_validate(raw_config)

    for stage in project.pipeline:
        if isinstance(stage, TransformerPPOStageConfig):
            from utils.entity_tokenizer_schema import (
                _load_default_sample,
                load_entity_tokenizer_config,
                validate_against_payload,
            )
            tokenizer_cfg = load_entity_tokenizer_config(stage.tokenizer_config_path)
            sample = _load_default_sample()
            action_names_per_building = [
                [ca.action_field for ca in tokenizer_cfg.ca_types.values()]
            ]
            validate_against_payload(tokenizer_cfg, sample, action_names_per_building)
    return project
```

### Step 2 — `algorithms/registry.py::_stage_to_agent_view`

Add the two transformer-specific keys:

```python
for optional_key in (
    "networks",
    "replay_buffer",
    "exploration",
    "policy",
    "tokenizer_config_path",
    "transformer",
):
    if optional_key in stage_cfg and stage_cfg[optional_key] is not None:
        algorithm_block[optional_key] = stage_cfg[optional_key]
```

### Step 3 — `utils/wrapper_citylearn.py`

**3a.** Conflict D — fused block:

```python
self._topology_changed_during_step = (
    topology_changed and not force_attach and previous_version is not None
)
if topology_changed:
    self.observation_names = observation_names
    self.observation_space = observation_spaces
    self.action_space = list(getattr(self.env, "flat_action_space", []))
    self.action_names = [list(names) for names in getattr(self.env, "action_names", [])]
    if len(self.action_names) < len(self.action_space):
        self.action_names.extend([[] for _ in range(len(self.action_space) - len(self.action_names))])
    elif len(self.action_names) > len(self.action_space):
        self.action_names = self.action_names[: len(self.action_space)]
    self.encoders = self.set_encoders()
    self._action_bounds_cache = None

if topology_changed and self._entity_dynamic_mode and previous_version is not None:
    from algorithms.registry import ALGORITHM_REGISTRY
    offenders = [
        name for name in self._algorithm_names
        if (cls := ALGORITHM_REGISTRY.get(name)) is not None
        and not bool(getattr(cls, "supports_dynamic_topology", False))
    ]
    if offenders:
        if "MADDPG" in offenders:
            raise ValueError(
                "MADDPG supports entity interface only with topology_mode='static'. "
                "Detected topology change during runtime."
            )
        raise ValueError(
            f"{sorted(offenders)} do not support dynamic topology "
            "(supports_dynamic_topology=False). Detected topology change during runtime."
        )
```

Drop the `ENCODED_OBSERVATION_ALGORITHMS` import in this file if nothing else
uses it after the edit (grep first).

**3b.** Conflict E — fused block:

```python
if not deterministic:
    if self._topology_changed_during_step:
        logger.debug(
            "Skipping model.update at global step {} due to mid-step topology change.",
            self.global_step,
        )
        self._topology_changed_during_step = False
    else:
        self._write_phase_progress(
            phase="model_update_start",
            episode=episode, step=time_step,
            episode_total=episodes, step_total=episode_step_total,
            global_step_total=global_step_total, rewards=rewards,
        )
        phase_start_time = time.perf_counter() if should_profile_step else 0.0
        self.update(
            observations, actions, rewards, next_observations,
            terminated=terminated, truncated=truncated,
        )
        if should_profile_step:
            runtime_profile_metrics["Runtime/agent_update_seconds"] = (
                time.perf_counter() - phase_start_time
            )
            runtime_profile_metrics["Runtime/model_observation_encoding_seconds"] = float(
                getattr(self, "_last_model_observation_encoding_seconds", 0.0) or 0.0
            )
            runtime_profile_metrics["Runtime/model_update_seconds"] = float(
                getattr(self, "_last_model_update_seconds", 0.0) or 0.0
            )
        logger.debug("Model update executed at global step {}", self.global_step)
        self._write_phase_progress(
            phase="model_update_end",
            episode=episode, step=time_step,
            episode_total=episodes, step_total=episode_step_total,
            global_step_total=global_step_total, rewards=rewards,
        )
```

The downstream `_write_phase_progress("checkpoint_start", ...)` block (line
1395+) stays unchanged — checkpointing still runs every non-deterministic step.

### Step 4 — Template + tests

**4a.** Migrate `configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml`
to the pipeline shape:

```yaml
pipeline:
  - algorithm: "AgentTransformerPPO"
    count: 1
    tokenizer_config_path: "configs/tokenizers/entity_default.json"
    transformer:
      d_model: 64
      nhead: 4
      num_layers: 2
      dim_feedforward: 128
      dropout: 0.1
    hyperparameters:
      learning_rate: 2.0e-4
      gamma: 0.99
      gae_lambda: 0.95
      clip_eps: 0.2
      ppo_epochs: 4
      minibatch_size: 64
      entropy_coeff: 0.03
      value_coeff: 0.5
      max_grad_norm: 1.0
```

**4b.** `tests/test_template_transformer_ppo_entity_dynamic.py` — change two
assertions:
- `cfg["algorithm"]["name"] == "AgentTransformerPPO"` → `cfg["pipeline"][0]["algorithm"] == "AgentTransformerPPO"`
- `cfg["algorithm"]["tokenizer_config_path"]` → `cfg["pipeline"][0]["tokenizer_config_path"]`

---

## Verification

Run in this order, fail-fast:

```bash
# 1. No leftover markers / no stale singular name
grep -RE "^(<<<<<<<|=======|>>>>>>>)" utils/
grep -rn "_algorithm_name\b" --include="*.py" utils/ algorithms/ tests/

# 2. Focused tests
pytest tests/test_template_transformer_ppo_entity_dynamic.py \
       tests/test_agent_transformer_ppo.py \
       tests/test_agent_transformer_ppo_wrapper_integration.py \
       tests/test_entity_tokenizer_config_schema.py \
       tests/test_wrapper_entity_mode.py \
       tests/test_entity_adapter.py \
       tests/test_pipeline.py \
       tests/test_registry.py \
       tests/test_config_validation.py -x -v

# 3. Full suite
pytest -x

# 4. Smoke run — 1 episode, short horizon, Transformer PPO template
python - <<'PY'
import copy, yaml, pathlib, tempfile, subprocess, sys, os
src = pathlib.Path("configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml")
cfg = yaml.safe_load(src.read_text())
cfg["simulator"]["episodes"] = 1
cfg["simulator"]["episode_time_steps"] = 50
tmp = pathlib.Path(tempfile.mkstemp(suffix=".yaml")[1])
tmp.write_text(yaml.safe_dump(cfg))
print("smoke config:", tmp)
env = os.environ.copy()
env.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns_smoke")
r = subprocess.run([sys.executable, "run_experiment.py", "--config", str(tmp),
                    "--base-dir", "./runs_smoke"], env=env)
sys.exit(r.returncode)
PY
```

The smoke run must finish with exit code 0 and produce
`runs_smoke/jobs/<job_id>/results/result.json`.

---

## Risks

1. **Tokenizer sample file location.** `validate_config` imports
   `utils.entity_tokenizer_schema._load_default_sample` which reads
   `datasets/tmp_entity_obs_full_step2200_named.json`. Confirm it's still
   present post-merge before relying on Step 1d.
2. **`_algorithm_name` singular.** Must be 0 hits across the repo after
   editing — the only references currently live inside the conflict block.
3. **`tokenizer_config_path` / `transformer` propagation.** Without Step 2
   the agent crashes at construction. Don't skip.
4. **`ENCODED_OBSERVATION_ALGORITHMS` import in wrapper.** Drop only if grep
   confirms no other use.
5. **Multiple TransformerPPO stages.** Step 1d iterates all stages — fine.
   No per-build assumption.
