"""Static contract for the local TPPO behavior-cloning validation runner."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "configs/recovery/tppo/wave_a/local/README.md"
RUNNER_PATH = REPO_ROOT / "scripts/run_tppo_bc_local_checks.sh"


def test_local_bc_runner_executes_and_validates_both_real_configurations() -> None:
    script = RUNNER_PATH.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert 'LOG_DIR="${LOG_DIR:-runs/local_bc_checks}"' in script
    assert "python run_experiment.py" in script
    assert "tppo_bc_pretrain_canary.yaml" in script
    assert "tppo_bc_pretrain_smoke.yaml" in script
    assert "--job_id" in script
    assert "Completed[[:space:]]+episode" in script
    assert "usable[ _-]?samples" in script
    assert "trained[ _-]?batches" in script
    assert "Skipping[[:space:]]+behavior-cloning" in script
    assert "watchdog" in script
    assert "Traceback" in script


def test_local_bc_readme_documents_real_dataset_gates_and_pass_criteria() -> None:
    readme = README_PATH.read_text(encoding="utf-8").lower()

    assert "16-step" in readme
    assert "real dynamic entity" in readme
    assert "not synthetic" in readme
    assert "192-step" in readme
    assert "real dataset" in readme
    assert "wave a server" in readme
    assert "3/3" in readme
    assert "usable samples" in readme
    assert "trained batches" in readme
