"""End-to-end Transformer MATD3 learning, topology, checkpoint, and export."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "configs/templates/dynamic/transformer_matd3_entity_dynamic.yaml"
DATASET = REPO_ROOT / "datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet/schema.json"
FIRST_TOPOLOGY_EVENT_STEP = 12
END_STEP = 24

pytestmark = pytest.mark.slow


def _dynamic_smoke_schema(work: Path) -> Path:
    for source in DATASET.parent.iterdir():
        if source.name != DATASET.name:
            (work / source.name).symlink_to(source)
    schema = json.loads(DATASET.read_text(encoding="utf-8"))
    schema["root_directory"] = str(DATASET.parent.resolve())
    events = schema.get("topology_events") or []
    assert events, "Dynamic smoke dataset has no topology events"
    for offset, event in enumerate(events):
        event["time_step"] = FIRST_TOPOLOGY_EVENT_STEP + 100 * offset
    path = work / "schema.dynamic_smoke.json"
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not DATASET.exists():
        pytest.skip(f"Demo dataset not present: {DATASET}")
    work = tmp_path_factory.mktemp("transformer_matd3_e2e")
    config = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    stage = config["pipeline"][0]
    config["simulator"]["dataset_path"] = str(_dynamic_smoke_schema(work))
    config["simulator"]["episodes"] = 1
    config["simulator"]["simulation_end_time_step"] = END_STEP
    config["simulator"]["episode_time_steps"] = END_STEP + 1
    config["tracking"]["progress_updates_enabled"] = False
    config["checkpointing"]["checkpoint_interval"] = 8
    config["checkpointing"]["require_initial_exploration_done"] = False
    config["training"]["steps_between_training_updates"] = 1
    config["training"]["target_update_interval"] = 2
    stage["transformer"] = {
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
    }
    stage["hyperparameters"].update(
        {
            "batch_size": 4,
            "buffer_capacity": 128,
            "random_exploration_steps": 0,
            "end_initial_exploration_time_step": 0,
        }
    )
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from run_experiment import run_experiment

    run_experiment(
        config_path=str(config_path),
        job_id="transformer_matd3_e2e",
        base_dir=work,
    )
    job_dir = work / "jobs" / "transformer_matd3_e2e"
    manifest_path = job_dir / "bundle" / "artifact_manifest.json"
    return {
        "job_dir": job_dir,
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
    }


def test_e2e_learning_mutation_checkpoint_and_export(
    smoke_run: dict[str, Any],
) -> None:
    job_dir: Path = smoke_run["job_dir"]
    manifest = smoke_run["manifest"]
    artifacts = manifest["agent"]["artifacts"]

    assert artifacts
    assert max(item["config"]["topology_version"] for item in artifacts) >= 1
    assert all((job_dir / "bundle" / item["path"]).exists() for item in artifacts)
    assert list((job_dir / "checkpoints").rglob("transformer_matd3_step*.pt"))
    assert (job_dir / "results" / "result.json").exists()
    assert (job_dir / "results" / "summary.json").exists()

    metrics_path = job_dir / "logs" / "metrics.jsonl"
    metric_lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    critic_losses = [
        (entry.get("metrics") or entry)["TransformerMATD3/critic_loss_mean"]
        for entry in metric_lines
        if "TransformerMATD3/critic_loss_mean" in (entry.get("metrics") or entry)
    ]
    assert critic_losses and all(math.isfinite(value) for value in critic_losses)

    from utils.bundle_validator import validate_bundle_contract

    validate_bundle_contract(manifest, job_dir / "bundle")
