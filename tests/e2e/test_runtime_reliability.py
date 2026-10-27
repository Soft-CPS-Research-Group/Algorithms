"""Independent-process repeatability, exact boundary resume and failure evidence."""

import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pytest

from tests.e2e.test_common_job_profiles import config_for


def job_process(config, job_id, root):
    path = root / f"{job_id}.json"
    path.write_text(json.dumps(config))
    code = """from pathlib import Path
import sys
from algorithms.runtime import job
from algorithms.runtime.evaluation import policy_digest
original = job.finalize_outputs
def check_evaluation(context, wrapper, agent, name):
    evaluated = getattr(wrapper, 'evaluated_policy', None)
    if evaluated:
        assert policy_digest(agent) == evaluated['sha256'], 'Evaluation changed learned weights/buffers'
    return original(context, wrapper, agent, name)
job.finalize_outputs = check_evaluation
job.run_experiment(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), job_id, str(root)],
        capture_output=True,
        text=True,
        env={**os.environ, "OPEVA_STARTUP_TRACE": "0"},
        timeout=90,
    )
    if result.returncode:
        logs = list((root / "jobs" / job_id / "logs").glob("*.log"))
        pytest.fail(result.stderr[-2000:] + "\n" + ("\n".join(p.read_text()[-4000:] for p in logs)))
    return json.loads((root / "jobs" / job_id / "results/result.json").read_text())


@pytest.mark.parametrize(
    "name", ["MADDPG", "AgentTransformerPPO", "AgentTransformerMATD3", "TIMARL"]
)
def test_repeatability_and_episode_resume_in_independent_processes(name, tmp_path):
    config = config_for(name, tmp_path)
    config["job"] = {"profile": "train"}
    config["training"] = {
        **config.get("training", {}),
        "episodes": 2,
        "seed": 0,
        "determinism": "strict",
    }
    config["evaluation"] = {"episodes": 1}
    config["simulator"]["episode_time_steps"] = 5
    config["outputs"] = {"model_artifacts": "checkpoint"}
    config["checkpointing"] = {"save_final": True, "checkpoint_interval": None}
    a = job_process(config, "full", tmp_path)
    b = job_process(config, "repeat", tmp_path)
    assert a["evaluated_policy"]["sha256"] == b["evaluated_policy"]["sha256"]
    assert [e["training"] for e in a["episode_execution"]] == [True, True, False]
    assert a["episode_execution"] == b["episode_execution"]
    changed = deepcopy(config)
    changed["training"]["seed"] = 1
    different = job_process(changed, "different_seed", tmp_path)
    assert a["evaluated_policy"]["sha256"] != different["evaluated_policy"]["sha256"]
    first = deepcopy(config)
    first["training"]["episodes"] = 1
    first["evaluation"]["episodes"] = 0
    job_process(first, "first", tmp_path)
    resumed = deepcopy(config)
    resumed["checkpointing"].update(
        resume_training=True,
        resume_mode="episode",
        checkpoint_local_path=str(tmp_path / "jobs/first/checkpoints"),
    )
    c = job_process(resumed, "resumed", tmp_path)
    assert a["evaluated_policy"]["sha256"] == c["evaluated_policy"]["sha256"]
    # The deliberately shorter first job exported its own final episode.
    # Preserve that truthful history while comparing all execution semantics.
    strip_export = lambda rows: [
        {k: v for k, v in row.items() if k != "export_enabled"} for row in rows
    ]
    assert strip_export(a["episode_execution"]) == strip_export(c["episode_execution"])


def test_invalid_config_creates_failure_and_retry_preserves_evidence(tmp_path):
    from algorithms.runtime.job import run_experiment

    with pytest.raises((ValueError, SystemExit)):
        run_experiment({"invalid": "config"}, "invalid", tmp_path)
    first = tmp_path / "jobs/invalid/results/result.json"
    assert json.loads(first.read_text())["status"] == "failed"
    config = config_for("NormalPolicy", tmp_path)
    config["outputs"] = {"model_artifacts": "none"}
    run_experiment(config, "invalid", tmp_path)
    assert json.loads(first.read_text())["status"] == "failed"
    result = next((tmp_path / "jobs/invalid/attempts").rglob("result.json"))
    assert json.loads(result.read_text())["status"] == "completed"


def test_sigterm_records_cancelled_and_closes_environment(tmp_path):
    config = config_for("NormalPolicy", tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    # Replace only the long-running loop; retain actual construction, lifecycle and outputs.
    code = """import json, sys, time
from pathlib import Path
from algorithms.runtime.job import run_experiment
from algorithms.runtime.controller import SimulationController
from citylearn.citylearn import CityLearnEnv
root=Path(sys.argv[2])
def learn(self, **kwargs):
    (root/'ready').write_text('ready')
    while True: time.sleep(.01)
def close(self): (root/'closed').write_text('closed')
SimulationController.learn=learn
CityLearnEnv.close=close
run_experiment(sys.argv[1], 'cancelled', root)
"""
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(tmp_path)], stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "process.log").read_text())
                time.sleep(0.05)
            assert (tmp_path / "ready").exists()
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert (
        json.loads((tmp_path / "jobs/cancelled/results/result.json").read_text())["status"]
        == "cancelled"
    )
    assert (tmp_path / "closed").exists()


def test_failed_export_never_commits_completed(tmp_path, monkeypatch):
    from algorithms.agents.baseline_policies import NormalPolicy
    from algorithms.runtime.job import run_experiment

    def fail(*args, **kwargs):
        raise OSError("simulated export write failure")

    monkeypatch.setattr(NormalPolicy, "export_artifacts", fail)
    config = config_for("NormalPolicy", tmp_path)
    config["outputs"] = {"model_artifacts": "bundle"}
    with pytest.raises(OSError, match="export write"):
        run_experiment(config, "export_failure", tmp_path)
    root = tmp_path / "jobs/export_failure"
    assert json.loads((root / "results/result.json").read_text())["status"] == "failed"
    assert json.loads((root / "progress/progress.json").read_text())["status"] == "failed"
    assert not (root / "bundle/artifact_manifest.json").exists()


@pytest.mark.parametrize(
    "name", ["RuleBasedPolicy", "AgentTransformerPPO", "AgentTransformerMATD3"]
)
def test_dynamic_member_and_asset_changes_use_common_runtime(name, tmp_path, monkeypatch):
    import yaml

    from algorithms.runtime.controller import SimulationController
    from algorithms.runtime.job import run_experiment

    dataset = (
        "citylearn_three_phase_dynamic_topology_demo_v1"
        if name == "RuleBasedPolicy"
        else "citylearn_three_phase_dynamic_assets_only_demo_15s_parquet"
    )
    source = Path("datasets") / dataset / "schema.json"
    source = source.resolve()
    for asset in source.parent.iterdir():
        if asset.name != "schema.json":
            (tmp_path / asset.name).symlink_to(asset)
    schema = json.loads(source.read_text())
    schema["root_directory"] = str(source.parent)
    for number, event in enumerate(schema["topology_events"], start=1):
        event["time_step"] = number
    target = tmp_path / "dynamic.json"
    target.write_text(json.dumps(schema))
    config = yaml.safe_load(
        Path("configs/templates/dynamic/rule_based_entity_dynamic_local.yaml").read_text()
    )
    if name != "RuleBasedPolicy":
        config["pipeline"] = config_for(name, tmp_path)["pipeline"]
    config["simulator"].update(
        dataset_path=str(target), simulation_end_time_step=10, episode_time_steps=11
    )
    config["training"]["episodes"] = 1
    config["evaluation"] = {"episodes": 1}
    config["outputs"] = {"model_artifacts": "none"}
    versions = []
    original = SimulationController._apply_entity_layout

    def observe(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        versions.append(self._entity_topology_version)
        return result

    monkeypatch.setattr(SimulationController, "_apply_entity_layout", observe)
    run_experiment(config, name, tmp_path)
    assert max(versions) >= len(schema["topology_events"])
    result = json.loads((tmp_path / "jobs" / name / "results/result.json").read_text())
    assert result["status"] == "completed"
    assert len(result["episode_execution"]) == 2
