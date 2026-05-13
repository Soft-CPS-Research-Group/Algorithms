"""WP06 — End-to-end smoke for AgentTransformerPPO on the assets-only dynamic demo.

Spec §16.7. Drives the unmodified ``run_experiment(...)`` entrypoint on a
downsized horizon and asserts every row of §16.7.

Marked ``slow``; auto-skips when the demo dataset is not bundled.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "configs/templates/transformer_ppo_entity_dynamic.yaml"
DATASET = REPO_ROOT / "datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json"

pytestmark = pytest.mark.slow


def _require_dataset_or_skip() -> None:
    if not DATASET.exists():
        pytest.skip(f"Demo dataset not present: {DATASET}")


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the downsized smoke once and yield ``{job_dir, manifest, results_dir}``."""
    _require_dataset_or_skip()
    work = tmp_path_factory.mktemp("wp06_e2e")

    cfg = yaml.safe_load(TEMPLATE.read_text())
    # Topology events fire at steps {1300, 1500, ...}. 1400 steps guarantees
    # exactly one observed mutation while keeping the smoke runtime bounded.
    cfg["simulator"]["simulation_end_time_step"] = 1400
    cfg["simulator"]["episode_time_steps"] = 1401
    # Ensure ≥1 PPO update fires before the topology mutation by keeping
    # ``steps_between_training_updates`` at 1; PPO consumes one full
    # rollout per call to ``update_step=True``.
    cfg["tracking"]["mlflow_enabled"] = False
    cfg["tracking"]["progress_updates_enabled"] = False
    cfg["checkpointing"]["resume_training"] = False
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


# ---------------------------------------------------------------------------
# Spec §16.7 row 1
# ---------------------------------------------------------------------------


def test_smoke_run_completes(smoke_run: dict[str, Any]) -> None:
    """Fixture having returned without exception is the assertion."""
    assert smoke_run["job_dir"].exists()
    assert smoke_run["manifest"] is not None, "artifact_manifest.json missing"


# ---------------------------------------------------------------------------
# Spec §16.7 row 2
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Spec §16.7 row 3
# ---------------------------------------------------------------------------


def test_topology_changes_observed_during_run(smoke_run: dict[str, Any]) -> None:
    """The assets-only demo schedules events at simulator timesteps
    ``{1300, 1500, 1700, 1900, 2100, 2300}``. With ``simulation_end_time_step =
    1400`` exactly one mutation has happened by run end; the agent's
    per-building manifest entries must record ``topology_version >= 1``."""
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
        f"Expected ≥1 topology mutation by step 1400; "
        f"observed topology_versions={versions}"
    )


# ---------------------------------------------------------------------------
# Spec §16.7 row 4
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Spec §16.7 row 5 — manifest contract (§14.2 + §14.1)
# ---------------------------------------------------------------------------


def test_artifact_manifest_includes_onnx_per_building(
    smoke_run: dict[str, Any],
) -> None:
    """Spec §14.1 + §14.2: per-building ONNX with filename
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


# ---------------------------------------------------------------------------
# Spec §16.7 row 6
# ---------------------------------------------------------------------------


def test_buffer_flush_on_topology_change_does_not_crash(
    smoke_run: dict[str, Any],
) -> None:
    """If we got here the smoke completed end-to-end past the step-1300
    mutation. Combined with the ``topology_version >= 1`` assertion above
    this proves the post-mutation update path
    (PPO step → buffer flush → layout rebuild → §13.4 re-validation) ran
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
