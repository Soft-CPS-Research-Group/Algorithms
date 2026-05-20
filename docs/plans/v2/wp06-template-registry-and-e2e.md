# WP06 — Algorithm Template, Registry Verification & End-to-End Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all v2 WPs:** every production-code task MUST follow `superpowers:test-driven-development`. The WP MUST end with `superpowers:requesting-code-review`.

**Goal:** Ship the operator-facing surface and prove the v2 stack runs end-to-end. Concretely: (1) the YAML template `configs/templates/transformer_ppo_entity_dynamic.yaml` that operators use to launch a transformer-PPO run on the entity-interface, dynamic-topology, assets-only demo dataset; (2) a registry/schema verification test confirming `AgentTransformerPPO` is wired into `validate_config(...)` end-to-end; (3) a real-binary E2E smoke test (`tests/test_e2e_transformer_ppo_entity_dynamic.py`) covering spec §16.7 — runs `run_experiment(...)` for a small horizon on `citylearn_three_phase_dynamic_assets_only_demo`, observes ≥1 topology mutation, asserts no crash, valid actions, KPI files, and a §14.2-conformant manifest with per-building ONNX whose filename encodes `topology_version`.

**Architecture:** Two artefacts and a test:

1. **Template YAML** — a thin clone of `configs/templates/rule_based_entity_dynamic_assets_only_local.yaml` with `algorithm.name = "AgentTransformerPPO"` and a populated `algorithm.hyperparameters` block matching the `TransformerPPOAlgorithmConfig` Pydantic model added in WP02.
2. **E2E test** — drives `run_experiment(config_path=..., job_id=..., base_dir=tmp_path)` against a downsized config (≤200 sim steps, 1 episode, MLflow disabled, checkpoints disabled, ONNX export enabled). Reads `runs/jobs/<id>/{results,artifact_manifest.json,onnx_models/}` from disk to assert the spec contract.
3. **Registry-and-schema sanity** — a focused unit test calling `validate_config(config_dict)` on the template (loaded via PyYAML) to assert it round-trips, the algorithm class resolves to `AgentTransformerPPO`, and the tokenizer JSON path resolves and validates.

This WP touches **no production code paths** — it only adds the template YAML and tests. Any failure points to gaps in WP02–WP05 that must be fixed there, not papered over here.

**Tech Stack:** Python 3.11, PyTorch (training), pytest, PyYAML, CityLearn (env). Consumes WP01–WP05.

**Branch:** `gj/wp06-template-and-e2e`
**Base branch:** `gj/wp05-agent-wrapper`

---

## Scope

**Files created:**

- `configs/templates/transformer_ppo_entity_dynamic.yaml` — operator-facing template.
- `tests/test_template_transformer_ppo_entity_dynamic.py` — schema/registry round-trip on the template.
- `tests/test_e2e_transformer_ppo_entity_dynamic.py` — spec §16.7 (6 rows).

**Files modified:**

- (None expected.) If a test reveals a missing piece (e.g. a schema field omitted in WP02, a manifest key missing in WP05), STOP-and-report rather than fix it in this WP. WP06 is a downstream verification WP; upstream gaps must be fixed in their owning WP and the chain re-merged.

**Out of scope:**

- Tuning hyperparameters for production. The template carries sane defaults that complete a smoke run; tuning is a separate downstream concern.
- Any `mlflow` / progress-tracking integration testing — disabled in the smoke config.
- GPU-only paths — the smoke test runs CPU.
- Cross-topology checkpoint/export portability (explicitly out per §14.3 and Decisions §17.12).

---

## File Structure

```
configs/
  templates/
    transformer_ppo_entity_dynamic.yaml         # NEW
tests/
  test_template_transformer_ppo_entity_dynamic.py  # NEW (~80 lines)
  test_e2e_transformer_ppo_entity_dynamic.py       # NEW (~250 lines)
docs/plans/v2/
  wp06-template-registry-and-e2e.md             # this plan
```

---

## Test Specification (RED first)

Per `superpowers:test-driven-development`, every task below is RED → GREEN → commit.

### Coverage map (spec §16.7, six rows)

| Spec row | Test name (this WP) |
|---|---|
| `test_smoke_run_completes` | `test_smoke_run_completes` |
| `test_actions_in_valid_range` | `test_actions_in_valid_range` |
| `test_topology_changes_observed_during_run` | `test_topology_changes_observed_during_run` |
| `test_kpi_files_generated` | `test_kpi_files_generated` |
| `test_artifact_manifest_includes_onnx_per_building` | `test_artifact_manifest_includes_onnx_per_building` |
| `test_buffer_flush_on_topology_change_does_not_crash` | `test_buffer_flush_on_topology_change_does_not_crash` |

Plus three template-validation tests not listed in §16.7 but required to make E2E meaningful:

| Test | Purpose |
|---|---|
| `test_template_passes_schema_validation` | `validate_config(yaml.safe_load(template))` returns without raising; schema model is `TransformerPPOAlgorithmConfig`. |
| `test_template_resolves_to_registered_agent` | `ALGORITHM_REGISTRY[cfg["algorithm"]["name"]] is AgentTransformerPPO`. |
| `test_template_tokenizer_path_validates` | The path under `algorithm.hyperparameters.tokenizer_config_path` exists and passes `EntityTokenizerConfig` Pydantic + 5-rule validation against `datasets/tmp_entity_obs_full_step2200_named.json`. |

---

## Tasks

### Task 1: Author the template YAML

Mirror `configs/templates/rule_based_entity_dynamic_assets_only_local.yaml` (every block stays except `algorithm.*`). The deviation is intentional: same dataset, same wrapper config, same simulation horizon — only the algorithm changes. This isolates template defects from environment defects.

The `algorithm.hyperparameters` block MUST match the field set declared by `TransformerPPOAlgorithmConfig` in `utils/config_schema.py` (added in WP02). If a field is missing or extra here, `extra="forbid"` will raise during schema validation in Task 5 — that's the desired RED.

- [ ] **Step 1: RED — write the schema-validation test first**

Create `tests/test_template_transformer_ppo_entity_dynamic.py`:

```python
"""Schema/registry sanity tests for the transformer-PPO entity-dynamic template."""
from __future__ import annotations
from pathlib import Path
import yaml
import pytest

TEMPLATE_PATH = Path(__file__).parents[1] / "configs/templates/transformer_ppo_entity_dynamic.yaml"


def _load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_template_passes_schema_validation():
    """The shipped template MUST validate against the v2 config schema."""
    from utils.config_schema import validate_config
    cfg = _load_template()
    # Should not raise. Captures schema mismatches early.
    validate_config(cfg)


def test_template_resolves_to_registered_agent():
    from algorithms.registry import ALGORITHM_REGISTRY
    from algorithms.agents.agent_transformer_ppo import AgentTransformerPPO
    cfg = _load_template()
    assert cfg["algorithm"]["name"] == "AgentTransformerPPO"
    assert ALGORITHM_REGISTRY["AgentTransformerPPO"] is AgentTransformerPPO


def test_template_tokenizer_path_validates():
    """Tokenizer JSON pointed to by the template MUST pass §13.4 5-rule validation against the bundled sample."""
    from utils.entity_tokenizer_schema import (
        load_tokenizer_config,
        EntityPayloadSample,
    )
    cfg = _load_template()
    tok_path = Path(cfg["algorithm"]["hyperparameters"]["tokenizer_config_path"])
    assert tok_path.exists(), tok_path
    tok = load_tokenizer_config(tok_path)
    sample_payload = Path(__file__).parents[1] / "datasets/tmp_entity_obs_full_step2200_named.json"
    sample = EntityPayloadSample.from_file(sample_payload)
    tok.validate_against_payload(sample)
```

```bash
pytest tests/test_template_transformer_ppo_entity_dynamic.py -v
```
Expected: `FileNotFoundError` for the template (it doesn't exist yet). RED.

- [ ] **Step 2: GREEN — author the template YAML**

Create `configs/templates/transformer_ppo_entity_dynamic.yaml`. Use the rule-based entity-dynamic-assets-only template as the structural base; replace only `metadata`, `algorithm.*`, and `tracking.mlflow_enabled` (disable for the smoke run):

```yaml
# Local AgentTransformerPPO template for entity interface with dynamic asset-only events.
# Smoke-tested by tests/test_e2e_transformer_ppo_entity_dynamic.py.

metadata:
  experiment_name: "transformer_ppo_entity_dynamic_assets_only_template"
  run_name: "Transformer PPO Entity Dynamic Assets-Only Local"
  community_name: "default_community"
  description: "Per-building Transformer + PPO over the entity interface; assets-only dynamic topology"

runtime:
  log_dir: null
  job_dir: null
  mlflow_uri: null
  job_id: null
  run_id: null
  run_name: null
  tracking_uri: null
  experiment_id: null
  mlflow_run_url: null

tracking:
  mlflow_enabled: false  # CPU smoke; flip to true for real runs
  log_level: "INFO"
  log_frequency: 1
  mlflow_step_sample_interval: 10
  mlflow_artifacts_profile: minimal
  progress_updates_enabled: true
  progress_update_interval: 5
  system_metrics_enabled: false
  system_metrics_interval: 10

checkpointing:
  resume_training: false
  checkpoint_run_id: null
  checkpoint_artifact: "transformer_ppo_checkpoint.pt"
  use_best_checkpoint_artifact: false
  reset_replay_buffer: false
  freeze_pretrained_layers: false
  fine_tune: false
  checkpoint_interval: null

bundle:
  bundle_version: null
  description: null
  alias_mapping_path: null
  require_observations_envelope: false
  artifact_config: {}
  per_agent_artifact_config: {}

simulator:
  dataset_name: citylearn_three_phase_dynamic_assets_only_demo
  dataset_path: ./datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json
  central_agent: false
  interface: entity
  topology_mode: dynamic
  entity_encoding:
    enabled: true
    normalization: minmax_space
    clip: true
  reward_function: RewardFunction
  reward_function_kwargs: {}
  episodes: 1
  simulation_start_time_step: 0
  simulation_end_time_step: 3400
  episode_time_steps: 3401
  export:
    mode: end
    export_kpis_on_episode_end: true
    session_name: null
  wrapper_reward:
    enabled: false
    profile: cost_limits_v1
    clip_enabled: true
    clip_min: -10.0
    clip_max: 10.0
    squash: none

training:
  seed: 42
  steps_between_training_updates: 1
  target_update_interval: 0  # PPO doesn't use a target net; preserved for schema parity

topology:
  num_agents: null
  observation_dimensions: null
  action_dimensions: null
  action_space: null

algorithm:
  name: "AgentTransformerPPO"
  hyperparameters:
    # ↓ Field names must match TransformerPPOAlgorithmConfig (utils/config_schema.py).
    # Values below are smoke-test defaults; tune separately for production.
    tokenizer_config_path: "configs/tokenizers/entity_default.json"
    d_model: 64
    n_heads: 4
    n_layers: 2
    d_ff: 128
    dropout: 0.0
    learning_rate: 3.0e-4
    rollout_length: 64
    n_epochs: 4
    minibatch_size: 32
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    value_coef: 0.5
    entropy_coef: 0.01
    max_grad_norm: 0.5
  networks: null
  replay_buffer: null
  exploration: null

execution: null
```

> **Field-set reconciliation:** the keys above are this plan's strawman. The authoritative set is whatever `TransformerPPOAlgorithmConfig` declares in `utils/config_schema.py` after WP02. If the schema uses different names (e.g. `n_steps` instead of `rollout_length`, `clip_coef` instead of `clip_range`), update this YAML to match the schema — do NOT modify the schema from this WP.

```bash
pytest tests/test_template_transformer_ppo_entity_dynamic.py -v
```
Expected: all 3 tests pass. If `test_template_passes_schema_validation` fails with "extra field" or "missing field", reconcile YAML keys against `TransformerPPOAlgorithmConfig` and re-run.

- [ ] **Step 3: Commit**

```bash
git add configs/templates/transformer_ppo_entity_dynamic.yaml tests/test_template_transformer_ppo_entity_dynamic.py
git commit -m "feat(wp06): add transformer-PPO entity-dynamic template + schema/registry sanity tests"
```

---

### Task 2: E2E test scaffolding — fixture that runs a downsized smoke

The E2E test runs `run_experiment(config_path=..., job_id=..., base_dir=tmp_path)` exactly as the CLI does (`run_experiment.py:43,62`). Instead of mutating the shipped template (which would race with other tests), we copy it to `tmp_path`, downsize the horizon, and pass that path.

The smoke must complete in well under a minute on CPU; we cut `simulation_end_time_step` to **300** and `rollout_length` to **32** so PPO updates fire at least twice. We also disable MLflow and checkpointing (already disabled in the template — assert this rather than re-set).

- [ ] **Step 1: Skeleton with module-level fixture**

Create `tests/test_e2e_transformer_ppo_entity_dynamic.py`:

```python
"""End-to-end smoke for v2 AgentTransformerPPO on the assets-only dynamic demo.

Spec §16.7. Runs the full run_experiment(...) pipeline for a small horizon
and asserts the post-run filesystem layout matches §14.2.

Marked `slow`: a single CPU run takes ~30-60s. Skipped if the demo dataset
is not present (e.g. minimal CI checkouts).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import yaml
import pytest

REPO_ROOT = Path(__file__).parents[1]
TEMPLATE = REPO_ROOT / "configs/templates/transformer_ppo_entity_dynamic.yaml"
DATASET = REPO_ROOT / "datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json"

pytestmark = pytest.mark.slow


def _require_dataset_or_skip():
    if not DATASET.exists():
        pytest.skip(f"Demo dataset not present: {DATASET}")


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Run a single downsized smoke and yield (job_dir, manifest_dict, results_dict)."""
    _require_dataset_or_skip()
    work = tmp_path_factory.mktemp("wp06_e2e")

    # Copy + downsize template
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["simulator"]["simulation_end_time_step"] = 300
    cfg["simulator"]["episode_time_steps"] = 301
    cfg["algorithm"]["hyperparameters"]["rollout_length"] = 32  # ensure ≥1 PPO update fires
    cfg["tracking"]["mlflow_enabled"] = False
    cfg["checkpointing"]["resume_training"] = False
    smoke_cfg_path = work / "smoke_config.yaml"
    smoke_cfg_path.write_text(yaml.safe_dump(cfg))

    job_id = "wp06_e2e_smoke"

    # Force CPU and a single thread for determinism
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from run_experiment import run_experiment
    run_experiment(config_path=str(smoke_cfg_path), job_id=job_id, base_dir=work)

    job_dir = work / "runs" / "jobs" / job_id
    assert job_dir.exists(), f"Job dir not created: {job_dir}"

    manifest_path = job_dir / "artifact_manifest.json"
    results_dir = job_dir / "results"

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {"job_dir": job_dir, "manifest": manifest, "results_dir": results_dir}
```

> **Note on `base_dir`:** `run_experiment(config_path, job_id, base_dir)` writes under `<base_dir>/runs/jobs/<job_id>/`. Verify against `run_experiment.py:62-91` if the path layout differs in this branch.

- [ ] **Step 2: Run the fixture in isolation to ensure the smoke completes at all**

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py -v --collect-only
# then a real run with a placeholder test:
```

Add a minimal test to force fixture execution:

```python
def test_smoke_run_completes(smoke_run):
    """Spec §16.7 row 1. Fixture having completed without exception is the assertion."""
    assert smoke_run["job_dir"].exists()
```

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_smoke_run_completes -v
```
Expected: PASS in <60s. If FAIL: triage and STOP — failures here indicate WP05/WP04/WP02 gaps. Do NOT add try/except in the fixture.

- [ ] **Step 3: Commit the scaffold**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): E2E smoke scaffold for AgentTransformerPPO on assets-only demo"
```

---

### Task 3: §16.7 row 2 — actions in valid range

The wrapper writes a per-step action log under `runs/jobs/<id>/results/simulation_data/`. We read the agent's action history from there. If the dataset doesn't carry one, fall back to inspecting the rollout buffer state via the saved checkpoint — but checkpoints are disabled in the smoke config, so prefer the simulation-data path.

- [ ] **Step 1: RED**

```python
def test_actions_in_valid_range(smoke_run):
    """Spec §16.7 row 2. All emitted actions must be in [-1, 1] and finite."""
    import math
    sim_dir = smoke_run["results_dir"] / "simulation_data"
    assert sim_dir.exists(), f"No simulation_data dir: {sim_dir}"
    # Find a per-building action log; the wrapper writes one of:
    #   actions.csv, action_history.json, or per-building action arrays under building_*.json
    action_files = list(sim_dir.glob("**/*action*"))
    assert action_files, f"No action artefacts in {sim_dir}"

    out_of_range = []
    for p in action_files:
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            # Walk; collect all leaf floats
            def _walk(x):
                if isinstance(x, (int, float)):
                    yield float(x)
                elif isinstance(x, list):
                    for y in x:
                        yield from _walk(y)
                elif isinstance(x, dict):
                    for y in x.values():
                        yield from _walk(y)
            for v in _walk(data):
                if not math.isfinite(v) or v < -1.0001 or v > 1.0001:
                    out_of_range.append((str(p), v))
        # CSV: skip; rare, can extend if needed
    assert not out_of_range, f"Out-of-range actions: {out_of_range[:10]}"
```

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_actions_in_valid_range -v
```
Expected: PASS. If the action log path differs, inspect `runs/jobs/<id>/results/` after a manual smoke run and adjust the glob. If actions exceed the range, the actor head is not bounding outputs (see WP01 ported `ActorHead`) — STOP and fix in WP05/WP01, not here.

- [ ] **Step 2: Commit**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): assert all emitted actions are finite and in [-1, 1]"
```

---

### Task 4: §16.7 row 3 — topology changes observed

The assets-only demo is documented (spec §17.2, decision 2) to inject at least one topology mutation. We assert the wrapper observed it by reading the `topology_version` trace the wrapper writes (or, fallback, the manifest's per-building `topology_version` ≥ 1).

- [ ] **Step 1: RED**

```python
def test_topology_changes_observed_during_run(smoke_run):
    """Spec §16.7 row 3. The assets-only demo guarantees ≥1 mutation in 200+ steps."""
    job_dir = smoke_run["job_dir"]
    # Path A: explicit topology trace, if the wrapper writes one
    trace_candidates = list((job_dir / "results" / "simulation_data").glob("**/topology*"))
    if trace_candidates:
        trace = json.loads(trace_candidates[0].read_text()) if trace_candidates[0].suffix == ".json" else None
        if isinstance(trace, list) and trace:
            versions = sorted(set(trace))
            assert len(versions) >= 2, f"No topology mutations observed; versions={versions}"
            return

    # Path B: fall back to manifest — if any per-building topology_version > 0, a mutation occurred
    manifest = smoke_run["manifest"]
    assert manifest is not None, "artifact_manifest.json missing"
    versions = [
        m["topology_version"]
        for m in manifest.get("agent_models", [])
        if "topology_version" in m
    ]
    assert versions, "manifest.agent_models entries missing topology_version"
    assert max(versions) >= 1, (
        f"Expected ≥1 topology mutation in the assets-only demo over 300 steps; "
        f"manifest topology_versions={versions}"
    )
```

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_topology_changes_observed_during_run -v
```

> If both paths return zero mutations, the demo dataset's mutation injector is misconfigured for the chosen horizon — bump `simulation_end_time_step` to 600 and re-run before changing the test.

- [ ] **Step 2: Commit**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): assert ≥1 topology_version increment observed in the smoke window"
```

---

### Task 5: §16.7 row 4 — KPI files generated

- [ ] **Step 1: RED**

```python
def test_kpi_files_generated(smoke_run):
    """Spec §16.7 row 4. result.json + summary.json must exist with non-empty content."""
    res = smoke_run["results_dir"]
    result_path = res / "result.json"
    summary_path = res / "summary.json"
    assert result_path.exists(), f"Missing {result_path}"
    assert summary_path.exists(), f"Missing {summary_path}"
    assert result_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    # Both must be valid JSON
    json.loads(result_path.read_text())
    json.loads(summary_path.read_text())
```

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_kpi_files_generated -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): assert result.json + summary.json are produced and parseable"
```

---

### Task 6: §16.7 row 5 — manifest includes per-building ONNX with topology version in filename

This is the §14.2 manifest contract test. Asserts: top-level `format`, non-empty `artifacts` list, one entry per building, each `path` exists on disk and ends in `.onnx`, filename matches `agent_<b>__topology_v<v>.onnx`. Also re-validates via `utils/bundle_validator.py` for belt-and-braces.

- [ ] **Step 1: RED**

```python
def test_artifact_manifest_includes_onnx_per_building(smoke_run):
    """Spec §16.7 row 5 + spec §14.2."""
    import re
    job_dir = smoke_run["job_dir"]
    manifest = smoke_run["manifest"]
    assert manifest is not None, "artifact_manifest.json missing"

    # §14.2 mandatory keys
    assert manifest.get("format") == "onnx", f"Expected format=onnx, got {manifest.get('format')!r}"
    artifacts = manifest.get("artifacts")
    assert isinstance(artifacts, list) and artifacts, "artifacts list missing or empty"

    # One entry per building (count via topology.num_agents written by run_experiment.py:691-697)
    resolved_cfg = yaml.safe_load((job_dir / "config.resolved.yaml").read_text())
    n_agents = resolved_cfg["topology"]["num_agents"]
    assert len(artifacts) == n_agents, (
        f"Expected {n_agents} artifact entries (matches topology.num_agents); got {len(artifacts)}"
    )

    pat = re.compile(r"agent_(\d+)__topology_v(\d+)\.onnx$")
    seen = set()
    for entry in artifacts:
        for k in ("agent_index", "path", "format", "config"):
            assert k in entry, f"Missing required artifact key {k!r}: {entry}"
        assert entry["format"] == "onnx"
        p = job_dir / entry["path"]
        assert p.exists(), f"ONNX file does not exist: {p}"
        m = pat.search(entry["path"])
        assert m, f"ONNX filename does not match agent_<b>__topology_v<v>.onnx: {entry['path']}"
        seen.add(int(m.group(1)))
    assert seen == set(range(n_agents)), f"Agent indices in filenames not contiguous 0..N-1: {seen}"

    # Belt-and-braces: bundle validator accepts the manifest
    from utils.bundle_validator import validate_manifest  # adjust if name differs
    validate_manifest(manifest, base_dir=job_dir)
```

> **`validate_manifest` import:** the actual symbol exported by `utils/bundle_validator.py` may differ (`validate`, `BundleValidator(...).validate(...)`, etc.). Inspect at the top of that file and adjust the import. If no public function exists, drop the belt-and-braces line — the prior assertions already cover §14.2.

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_artifact_manifest_includes_onnx_per_building -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): assert manifest §14.2 contract — per-building ONNX with topology_version filename"
```

---

### Task 7: §16.7 row 6 — buffer flush on topology change does not crash

This is implicit in the smoke completing (Task 2), but make it explicit by asserting the manifest shows training continued past the first topology mutation. Specifically: at least one building's `topology_version` in the manifest is ≥ 1 AND the smoke ran to completion (i.e. the fixture didn't raise). Combined, this proves the post-mutation PPO update + layout rebuild path executed successfully.

- [ ] **Step 1: RED**

```python
def test_buffer_flush_on_topology_change_does_not_crash(smoke_run):
    """Spec §16.7 row 6. If we got here, the fixture completed; combined with the
    topology-changed assertion, this proves the post-mutation update path
    (PPO step → buffer flush → layout rebuild → 5-rule re-validation) ran without crashing.
    """
    manifest = smoke_run["manifest"]
    assert manifest is not None
    versions = [m["topology_version"] for m in manifest.get("agent_models", [])]
    assert versions and max(versions) >= 1, (
        "No topology mutation in the smoke window — this test is meaningless without one. "
        "Bump simulation_end_time_step or investigate the demo dataset injector."
    )
    # Smoke fixture ran to completion → on_topology_change did not raise → contract upheld.
```

```bash
pytest tests/test_e2e_transformer_ppo_entity_dynamic.py::test_buffer_flush_on_topology_change_does_not_crash -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_e2e_transformer_ppo_entity_dynamic.py
git commit -m "test(wp06): assert post-mutation PPO update + buffer flush path completes without crash"
```

---

### Task 8: Full sweep + register `slow` marker

- [ ] **Step 1: Add the `slow` marker definition (skip if already present)**

Inspect `pyproject.toml` / `pytest.ini` / `setup.cfg`. If no `slow` marker is registered, register it to silence `PytestUnknownMarkWarning`:

```ini
# In pyproject.toml under [tool.pytest.ini_options] (preferred), or pytest.ini
markers = [
    "slow: marks tests that take >10s (deselect with '-m \"not slow\"')",
]
```

If a markers list already exists, append `"slow: …"` to it. If `slow` is already registered, skip this step.

- [ ] **Step 2: Run full repo test suite**

```bash
pytest -x -q
```
Expected: exit 0. The E2E tests will run; expect ~60s extra wall-clock.

```bash
pytest -x -q -m "not slow"
```
Expected: exit 0. Confirms developers can skip the slow E2E during inner-loop work.

- [ ] **Step 3: Commit any marker registration**

```bash
git add pyproject.toml  # or pytest.ini
git commit -m "chore(wp06): register pytest 'slow' marker for E2E suite"
```

---

## Self-Review Checklist (run before requesting code review)

- [ ] **Template parity:** `diff configs/templates/{rule_based_entity_dynamic_assets_only_local,transformer_ppo_entity_dynamic}.yaml` shows differences ONLY in `metadata`, `algorithm.*`, `tracking.mlflow_enabled`, and (intentionally) `checkpointing.checkpoint_artifact`. Every other block is identical so dataset/wrapper behaviour is comparable across the two templates.
- [ ] **Schema fields match exactly:** `algorithm.hyperparameters` keys in the YAML are the precise field set declared by `TransformerPPOAlgorithmConfig` in `utils/config_schema.py`. Verify with: `python -c "from utils.config_schema import TransformerPPOAlgorithmConfig; print(sorted(TransformerPPOAlgorithmConfig.model_fields))"` vs. `python -c "import yaml; print(sorted(yaml.safe_load(open('configs/templates/transformer_ppo_entity_dynamic.yaml'))['algorithm']['hyperparameters']))"`.
- [ ] **Spec §16.7 coverage:** all 6 rows have a corresponding `def test_…` in `tests/test_e2e_transformer_ppo_entity_dynamic.py`. `grep -c "^def test_" tests/test_e2e_transformer_ppo_entity_dynamic.py` ≥ 6.
- [ ] **Schema/registry coverage:** all 3 template-validation tests in `tests/test_template_transformer_ppo_entity_dynamic.py`. `grep -c "^def test_" tests/test_template_transformer_ppo_entity_dynamic.py` ≥ 3.
- [ ] **§14.2 manifest contract verified end-to-end:** `test_artifact_manifest_includes_onnx_per_building` checks `format`, `artifacts` keys, file existence, filename pattern, and bundle-validator round-trip (or documents why the latter was dropped).
- [ ] **No production code modified by this WP:** `git diff --stat gj/wp05-agent-wrapper..HEAD -- ':!docs' ':!tests' ':!configs/templates'` shows zero lines changed (except possibly `pyproject.toml` for the `slow` marker). If any production file changed, the change belongs in WP02–WP05 — STOP and re-route.
- [ ] **No skipped E2E tests in the final commit** — every §16.7 row has a real assertion. `pytest.skip` is only acceptable inside `_require_dataset_or_skip()` for environments without the demo dataset.
- [ ] **Smoke test runtime documented:** add a comment in the test module noting wall-clock on the dev machine (e.g. "≈45s on 8-core CPU"). Helps reviewers triage CI timeouts.
- [ ] **Full repo test suite passes:** `pytest -x -q` exits 0.
- [ ] **Inner-loop unaffected:** `pytest -x -q -m "not slow"` exits 0 in roughly the pre-WP06 wall-clock.

---

## Code Review

After self-review passes, invoke `superpowers:requesting-code-review`. Reviewer should check the diff against this plan and §14.2, §15.2 (WP9, WP10), §16.7, and §17 (decisions 4, 12, 13) of `docs/specv2.md`, with particular attention to:

1. **Template field set is the authoritative source paired with `TransformerPPOAlgorithmConfig`**, not a parallel schema. Any drift between YAML and Pydantic model is a bug.
2. **The E2E test exercises the real `run_experiment(...)` entry point**, not a hand-rolled training loop. Otherwise we are not validating the operator path.
3. **The smoke test does not silently mask production failures** (no broad `try/except`; no `pytest.skip` outside the dataset-missing guard).
4. **Manifest filename pattern matches §14.1 exactly:** `agent_<b>__topology_v<v>.onnx` (double underscore between `<b>` and `topology`). A mismatch here breaks downstream bundle deployment tooling.
5. **No production code touched.** This WP is verification-only. Failures triggered here MUST be fixed in their owning upstream WP and re-merged forward, not patched in WP06.

---

## PR Description

```markdown
## Summary
Ships the operator-facing template `configs/templates/transformer_ppo_entity_dynamic.yaml` and proves the v2 stack runs end-to-end via a real-binary smoke against `citylearn_three_phase_dynamic_assets_only_demo`. Drives the unmodified `run_experiment(...)` entry point for ~300 simulation steps and asserts every row of spec §16.7: smoke completion, finite/in-range actions, ≥1 observed topology mutation, KPI file generation, §14.2-conformant per-building ONNX manifest with `topology_version` in the filename, and post-mutation PPO update path completing without crash. Adds three template-validation unit tests confirming the YAML round-trips through `validate_config`, resolves to the registered `AgentTransformerPPO` class, and that its referenced tokenizer JSON passes the WP02 5-rule validation against the bundled sample payload.

## Key Changes
- Add `configs/templates/transformer_ppo_entity_dynamic.yaml`: structural clone of the rule-based assets-only template with `algorithm.name = "AgentTransformerPPO"` and a `hyperparameters` block matching `TransformerPPOAlgorithmConfig` exactly.
- Add `tests/test_template_transformer_ppo_entity_dynamic.py`: 3 unit tests — schema validation, registry resolution, tokenizer-JSON 5-rule validation against `datasets/tmp_entity_obs_full_step2200_named.json`.
- Add `tests/test_e2e_transformer_ppo_entity_dynamic.py`: 6 tests covering spec §16.7. Module-scoped fixture downsizes the smoke (300 steps, rollout_length=32, MLflow off, checkpointing off), runs `run_experiment(...)` once, and shares `(job_dir, manifest, results_dir)` across the row tests. Marked `slow`; auto-skips if the demo dataset is absent.
- [If applicable] Register `slow` pytest marker in `pyproject.toml` so `pytest -m "not slow"` cleanly excludes the E2E from inner-loop runs.
- No production code modified. Any failure surfaced by this WP indicates a gap in WP02–WP05 and must be fixed upstream, not patched here.
```
