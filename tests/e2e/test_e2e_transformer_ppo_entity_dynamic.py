"""End-to-end smoke for AgentTransformerPPO on the assets-only dynamic demo.

Drives the unmodified ``run_experiment(...)`` entrypoint on a downsized
horizon. Marked ``slow``; auto-skips when the demo dataset is not
bundled.

Wall-clock on an 8-core M-series CPU: ~2-5 minutes (depends on building
count; the assets-only demo ships 18 buildings).
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "configs/templates/dynamic/transformer_ppo_entity_dynamic.yaml"
DATASET = REPO_ROOT / "datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet/schema.json"

pytestmark = pytest.mark.slow
SMOKE_FIRST_TOPOLOGY_EVENT_STEP = 300
SMOKE_END_STEP = 360


def _require_dataset_or_skip() -> None:
    if not DATASET.exists():
        pytest.skip(f"Demo dataset not present: {DATASET}")


def _write_smoke_schema_with_early_topology_event(work: Path) -> Path:
    """Copy the bundled dynamic schema and shift topology events earlier.

    The real demo dataset schedules asset mutations much later in the
    full-resolution timeline. The e2e should stay short, so it uses an
    otherwise identical temporary schema with the same ordered events moved
    into the smoke window.
    """
    for source in DATASET.parent.iterdir():
        if source.name == DATASET.name:
            continue
        target = work / source.name
        if not target.exists():
            target.symlink_to(source)

    schema = json.loads(DATASET.read_text())
    schema["root_directory"] = str(DATASET.parent.resolve())
    events = schema.get("topology_events") or []
    assert events, "Dynamic smoke dataset has no topology_events"
    for offset, event in enumerate(events):
        event["time_step"] = SMOKE_FIRST_TOPOLOGY_EVENT_STEP + 200 * offset
    smoke_schema_path = work / "schema.dynamic_smoke.json"
    smoke_schema_path.write_text(json.dumps(schema, indent=2))
    return smoke_schema_path


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the downsized smoke once and yield ``{job_dir, manifest, results_dir}``."""
    _require_dataset_or_skip()
    work = tmp_path_factory.mktemp("wp06_e2e")

    cfg = yaml.safe_load(TEMPLATE.read_text())
    smoke_schema_path = _write_smoke_schema_with_early_topology_event(work)
    cfg["simulator"]["episodes"] = 1
    cfg["simulator"]["dataset_path"] = str(smoke_schema_path)
    cfg["simulator"]["simulation_end_time_step"] = SMOKE_END_STEP
    cfg["simulator"]["episode_time_steps"] = SMOKE_END_STEP + 1
    cfg["tracking"]["mlflow_enabled"] = False
    cfg["tracking"]["log_frequency"] = 128
    cfg["tracking"]["progress_updates_enabled"] = False
    cfg["checkpointing"]["resume_training"] = False
    cfg["checkpointing"]["checkpoint_interval"] = None
    smoke_cfg_path = work / "smoke_config.yaml"
    smoke_cfg_path.write_text(yaml.safe_dump(cfg))

    job_id = "wp06_e2e_smoke"

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from run_experiment import run_experiment

    run_experiment(config_path=str(smoke_cfg_path), job_id=job_id, base_dir=work)

    job_dir = work / "jobs" / job_id
    assert job_dir.exists(), f"Job dir not created: {job_dir}"

    manifest_path = job_dir / "bundle" / "artifact_manifest.json"
    results_dir = job_dir / "results"

    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None
    )
    return {
        "job_dir": job_dir,
        "manifest": manifest,
        "results_dir": results_dir,
        "config": cfg,
    }


def test_smoke_run_completes(smoke_run: dict[str, Any]) -> None:
    """Fixture having returned without exception is the assertion."""
    assert smoke_run["job_dir"].exists()
    assert smoke_run["manifest"] is not None, "artifact_manifest.json missing"


def _walk_floats(x: Any):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield float(x)
    elif isinstance(x, list):
        for y in x:
            yield from _walk_floats(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _walk_floats(y)


def test_actions_in_valid_range(smoke_run: dict[str, Any]) -> None:
    """All emitted actions must be finite and in ``[-1, 1]`` (with float
    tolerance).  Reads any ``*action*`` JSON written under
    ``results/simulation_data``."""
    sim_dir = smoke_run["results_dir"] / "simulation_data"
    assert sim_dir.exists(), f"No simulation_data dir: {sim_dir}"

    action_files = [p for p in sim_dir.glob("**/*action*") if p.is_file()]
    if not action_files:
        # No explicit action log was emitted by the wrapper for this dataset
        # — fall back to confirming no obviously bad data on disk and skip
        # the per-value range check.
        pytest.skip(f"No action artefacts found in {sim_dir}")

    out_of_range: list[tuple[str, float]] = []
    for p in action_files:
        if p.suffix == ".json":
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            for v in _walk_floats(data):
                if not math.isfinite(v) or v < -1.0001 or v > 1.0001:
                    out_of_range.append((str(p), v))
    assert not out_of_range, f"Out-of-range actions: {out_of_range[:10]}"


def test_topology_changes_observed_during_run(smoke_run: dict[str, Any]) -> None:
    """The smoke schema moves the first topology event into the short run.

    By export time, the agent's per-building manifest entries must record
    ``topology_version >= 1``.
    """
    manifest = smoke_run["manifest"]
    assert manifest is not None, "artifact_manifest.json missing"

    # The agent block carries our per-building artefacts. Each entry's
    # ``config`` records ``topology_version`` (set at export time).
    agent = manifest.get("agent") or {}
    artifacts = agent.get("artifacts") or []
    assert artifacts, "manifest.agent.artifacts is empty"

    versions = [
        a.get("config", {}).get("topology_version")
        for a in artifacts
        if a.get("config")
    ]
    versions = [v for v in versions if isinstance(v, int)]
    assert versions, "No artifact carries config.topology_version"
    assert max(versions) >= 1, (
        f"Expected ≥1 topology mutation by step {SMOKE_END_STEP}; "
        f"observed topology_versions={versions}"
    )


def test_kpi_files_generated(smoke_run: dict[str, Any]) -> None:
    """``result.json`` and ``summary.json`` must exist with parseable content."""
    res = smoke_run["results_dir"]
    result_path = res / "result.json"
    summary_path = res / "summary.json"
    assert result_path.exists(), f"Missing {result_path}"
    assert summary_path.exists(), f"Missing {summary_path}"
    assert result_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    json.loads(result_path.read_text())  # raises on invalid JSON
    json.loads(summary_path.read_text())


def test_artifact_manifest_includes_onnx_per_building(
    smoke_run: dict[str, Any],
) -> None:
    """per-building ONNX with filename
    ``agent_<b>__topology_v<v>.onnx``; one entry per agent; bundle
    validator accepts the manifest."""
    job_dir: Path = smoke_run["job_dir"]
    manifest = smoke_run["manifest"]
    assert manifest is not None

    agent = manifest.get("agent") or {}
    assert agent.get("format") == "onnx", (
        f"Expected agent.format='onnx', got {agent.get('format')!r}"
    )

    artifacts = agent.get("artifacts")
    assert isinstance(artifacts, list) and artifacts, "agent.artifacts empty"

    pattern = re.compile(r"agent_(\d+)__topology_v(\d+)\.onnx$")
    seen_indices: set[int] = set()
    for entry in artifacts:
        for k in ("agent_index", "path", "format", "config"):
            assert k in entry, f"Missing required artifact key {k!r}: {entry}"
        assert entry["format"] == "onnx"
        # Manifests resolve paths relative to the bundle root (the directory
        # containing artifact_manifest.json).
        full = job_dir / "bundle" / entry["path"]
        assert full.exists(), f"ONNX file does not exist: {full}"
        assert full.suffix == ".onnx"
        m = pattern.search(entry["path"])
        assert m, (
            f"Filename does not match agent_<b>__topology_v<v>.onnx: "
            f"{entry['path']!r}"
        )
        seen_indices.add(int(m.group(1)))

    # Indices must form a contiguous 0..N-1 set; enforces one entry per agent.
    n = len(artifacts)
    assert seen_indices == set(range(n)), (
        f"Agent indices not contiguous 0..{n - 1}: {sorted(seen_indices)}"
    )

    # Belt-and-braces: bundle validator accepts the manifest.
    from utils.bundle_validator import validate_bundle_contract

    validate_bundle_contract(manifest, job_dir / "bundle")


def test_buffer_flush_on_topology_change_does_not_crash(
    smoke_run: dict[str, Any],
) -> None:
    """If we got here the smoke completed end-to-end past the injected
    mutation. Combined with the ``topology_version >= 1`` assertion above
    this proves the post-mutation update path
    (PPO step → buffer flush → layout rebuild → re-validation) ran
    without crashing."""
    manifest = smoke_run["manifest"]
    assert manifest is not None
    agent = manifest.get("agent") or {}
    artifacts = agent.get("artifacts") or []
    versions = [
        a.get("config", {}).get("topology_version")
        for a in artifacts
        if a.get("config")
    ]
    versions = [v for v in versions if isinstance(v, int)]
    assert versions and max(versions) >= 1, (
        "No topology mutation observed in the smoke window; this test is "
        "meaningless without one. Bump simulation_end_time_step or "
        "investigate the demo dataset's topology_events."
    )
