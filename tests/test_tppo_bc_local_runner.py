"""Static contract for the local TPPO behavior-cloning validation runner."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "configs/recovery/tppo/wave_a/local/README.md"
RUNNER_PATH = REPO_ROOT / "scripts/run_tppo_bc_local_checks.sh"


def test_local_bc_runner_declares_required_checks() -> None:
    script = RUNNER_PATH.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert 'LOG_DIR="${LOG_DIR:-runs/local_bc_checks}"' in script
    assert "python run_experiment.py" in script
    assert "tppo_bc_pretrain_canary.yaml" in script
    assert "tppo_bc_pretrain_smoke.yaml" in script
    assert "--job_id" in script
    assert "Completed[[:space:]]+episode" in script
    assert "behavior_cloning_building_" in script
    assert "_trained_batches" in script
    assert "behavior_cloning_pretraining_batches" in script
    assert "Skipping[[:space:]]+behavior-cloning" in script
    assert "watchdog" in script
    assert "_stall_watchdog.log" in script


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
    assert "behavior_cloning_building_<building>_usable_samples" in readme
    assert "behavior_cloning_pretraining_batches" in readme


def _write_fake_local_runner_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts/run_tppo_bc_local_checks.sh"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(RUNNER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    config_dir = repo_root / "configs/recovery/tppo/wave_a/local"
    config_dir.mkdir(parents=True)
    dataset_path = repo_root / "datasets/schema.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"buildings": {"Building_1": {}}}', encoding="utf-8")
    for name in ("canary", "smoke"):
        (config_dir / f"tppo_bc_pretrain_{name}.yaml").write_text(
            "simulator:\n  dataset_path: datasets/schema.json\n",
            encoding="utf-8",
        )

    fake_python = repo_root / "fake_python.py"
    fake_python.write_text(
        f"""#!{os.sys.executable}
import json
import os
import sys
from pathlib import Path

if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv
job_id = args[args.index('--job_id') + 1]
base_dir = Path(args[args.index('--base-dir') + 1])
config = args[args.index('--config') + 1]
logs = base_dir / 'jobs' / job_id / 'logs'
logs.mkdir(parents=True)
(logs / f'{{job_id}}.log').write_text('Completed episode 3/3\\n', encoding='utf-8')
metrics = {{
    'TPPO/behavior_cloning_building_Building_1_usable_samples': 1.0,
    'TPPO/behavior_cloning_building_Building_1_trained_batches': float(os.environ.get('FAKE_BUILDING_BATCHES', '2')),
    'TPPO/behavior_cloning_pretraining_batches': 2.0,
}}
(logs / 'metrics.jsonl').write_text(json.dumps({{'metrics': metrics}}) + '\\n', encoding='utf-8')
watchdog = logs / f'{{job_id}}_stall_watchdog.log'
watchdog.write_text(os.environ.get('FAKE_WATCHDOG', ''), encoding='utf-8')
with (base_dir / 'calls.log').open('a', encoding='utf-8') as handle:
    handle.write(config + '\\n')
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    return repo_root, script_path


def _run_fake_local_runner(
    tmp_path: Path, *, watchdog: str = "", building_batches: int = 2
) -> subprocess.CompletedProcess[str]:
    repo_root, script_path = _write_fake_local_runner_repo(tmp_path)
    environment = os.environ | {
        "LOG_DIR": str(tmp_path / "logs"),
        "PYTHON_BIN": str(repo_root / "fake_python.py"),
        "FAKE_WATCHDOG": watchdog,
        "FAKE_BUILDING_BATCHES": str(building_batches),
    }
    return subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_bc_runner_is_executable_and_bash_syntax_is_valid() -> None:
    assert RUNNER_PATH.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(RUNNER_PATH)], check=True)


def test_local_bc_runner_runs_canary_then_smoke_and_keeps_empty_watchdog_artifacts(tmp_path: Path) -> None:
    result = _run_fake_local_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "logs/calls.log").read_text(encoding="utf-8").splitlines()
    assert calls == [
        "configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_canary.yaml",
        "configs/recovery/tppo/wave_a/local/tppo_bc_pretrain_smoke.yaml",
    ]
    assert len(list((tmp_path / "logs").glob("*.stdout.log"))) == 2


@pytest.mark.parametrize("watchdog", ["Current thread 0x1\n", "Traceback (most recent call last)\n"])
def test_local_bc_runner_rejects_nonempty_watchdog_artifact(tmp_path: Path, watchdog: str) -> None:
    result = _run_fake_local_runner(tmp_path, watchdog=watchdog)

    assert result.returncode != 0
    assert "watchdog artifact" in result.stderr


def test_local_bc_runner_rejects_zero_trained_batches_for_a_building(tmp_path: Path) -> None:
    result = _run_fake_local_runner(tmp_path, building_batches=0)

    assert result.returncode != 0
    assert "trained batches for Building_1" in result.stderr
