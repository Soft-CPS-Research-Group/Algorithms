"""Small complete jobs exercising profiles and outputs through the public runner."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from algorithms.runtime.job import run_experiment
from algorithms.runtime.preflight import preflight_job


def config_for(name, tmp_path):
    config = yaml.safe_load(
        Path("configs/templates/experimental/transformer_maddpg_smoke.yaml").read_text()
    )
    config["job"] = {"profile": "smoke"}
    config["simulator"]["simulation_end_time_step"] = 16
    if name in {"NormalPolicy", "RBCBasicPolicy"}:
        stage = {"algorithm": name}
    elif name == "MADDPG":
        stage = deepcopy(yaml.safe_load(Path("configs/config.yaml").read_text())["pipeline"][0])
        stage["exploration"]["params"] = {
            "random_exploration_steps": 0,
            "end_initial_exploration_time_step": 0,
        }
        stage["networks"]["actor"]["layers"] = [8]
        stage["networks"]["critic"]["layers"] = [8]
        stage["replay_buffer"].update(capacity=16, batch_size=2)
    elif name in {"AgentTransformerPPO", "AgentTransformerMATD3"}:
        file = (
            "transformer_ppo_entity_dynamic"
            if name.endswith("PPO")
            else "transformer_matd3_entity_dynamic"
        )
        stage = deepcopy(
            yaml.safe_load(Path(f"configs/templates/dynamic/{file}.yaml").read_text())["pipeline"][
                0
            ]
        )
        stage["transformer"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        if name.endswith("PPO"):
            stage["hyperparameters"].update(ppo_epochs=1, minibatch_size=2)
        else:
            stage["hyperparameters"].update(buffer_capacity=16, batch_size=2)
    elif name == "TIMARL":
        from algorithms.configuration.project import validate_config
        from algorithms.runtime.simulator import create_environment
        from scripts.generate_typed_interfaces import (
            generate,
            generate_simulator_bindings,
            write_generated,
        )

        source = validate_config(config).to_dict()
        source["simulator"]["export"].update(mode="none", export_kpis_on_episode_end=False)
        env = create_environment(source, tmp_path / "discovery")
        try:
            payload, _ = env.reset()
            interfaces, coverage = generate(env.entity_specs, payload)
            bindings = generate_simulator_bindings(env.entity_specs, payload, interfaces)
            directory = tmp_path / "interfaces"
            write_generated(
                directory,
                interfaces,
                coverage,
                source="Simulator 1.9.2",
                simulator_bindings=bindings,
            )
        finally:
            env.close()
        stage = {
            "algorithm": name,
            "hyperparameters": {
                "contract_version": "ti_marl_v1",
                "typed_interfaces_dir": str(directory),
                "simulator_bindings_path": str(directory / "technology_bindings/simulator.yaml"),
                "backbone": {"name": "ppo"},
                "actor": {"d_model": 16, "attention_heads": 4, "relation_layers": 1},
                "critic": {"kind": "local"},
                "feasibility": {"kind": "analytic_projection"},
                "rollout_steps": 4,
                "ppo_epochs": 1,
            },
        }
    else:
        stage = config["pipeline"][0]
    config["pipeline"] = [stage]
    return config


@pytest.mark.parametrize(
    "name",
    [
        "NormalPolicy",
        "RBCBasicPolicy",
        "MADDPG",
        "TransformerMADDPG",
        "AgentTransformerPPO",
        "AgentTransformerMATD3",
        "TIMARL",
    ],
)
def test_representative_families_share_job_results_and_bundle(name, tmp_path):
    config = config_for(name, tmp_path)
    config["outputs"] = {"model_artifacts": "bundle"}
    preflight = preflight_job(config)
    assert preflight["simulation_steps"] == preflight["training_updates"] == 0
    run_experiment(config, name, tmp_path)
    job = tmp_path / "jobs" / name
    result = json.loads((job / "results/result.json").read_text())
    assert result["status"] == "completed"
    assert result["output_contract_version"] == "algorithms.outputs.v2"
    assert result["model_artifacts"] == "bundle"
    assert len(result["episode_execution"]) == 2
    assert result["episode_execution"][-1]["deterministic"] is True
    assert (job / "bundle/artifact_manifest.json").is_file()
    assert list((job / "results/simulation_data").rglob("exported_kpis.csv"))


@pytest.mark.parametrize("mode", ["none", "checkpoint", "bundle", "both"])
def test_requested_model_outputs_are_observed(mode, tmp_path):
    config = config_for("MADDPG", tmp_path)
    config["outputs"] = {"model_artifacts": mode}
    run_experiment(config, mode, tmp_path)
    job = tmp_path / "jobs" / mode
    result = json.loads((job / "results/result.json").read_text())
    assert result["status"] == "completed"
    assert (job / "bundle/artifact_manifest.json").exists() == (mode in {"bundle", "both"})
    assert bool(list((job / "checkpoints").rglob("*.pth"))) == (mode in {"checkpoint", "both"})
    assert bool(result["manifest_sha256"]) == (mode in {"bundle", "both"})


def test_evaluation_uses_deterministic_episode_without_online_updates(tmp_path, monkeypatch):
    config = config_for("MADDPG", tmp_path)
    config["outputs"] = {"model_artifacts": "checkpoint"}
    run_experiment(config, "trained", tmp_path)
    checkpoint = next((tmp_path / "jobs/trained/checkpoints").rglob("*.pth"))
    config["checkpointing"] = {"resume_training": True, "checkpoint_local_path": str(checkpoint)}
    from algorithms.agents.maddpg_agent import MADDPG

    def unexpected_update(*args, **kwargs):
        raise AssertionError("Evaluation must not update the trained model")

    monkeypatch.setattr(MADDPG, "update", unexpected_update)
    config["job"] = {"profile": "evaluate"}
    config["simulator"]["episode_time_steps"] = 5
    config["outputs"] = {"model_artifacts": "none"}
    run_experiment(config, "evaluation", tmp_path)
    result = json.loads((tmp_path / "jobs/evaluation/results/result.json").read_text())
    assert len(result["episode_execution"]) == 1
    assert result["episode_execution"][0]["deterministic"] is True
    assert result["episode_execution"][0]["training"] is False


def test_milp_resolves_each_step_and_exports_research_evidence(tmp_path):
    config = yaml.safe_load(Path("configs/templates/baselines/core30/milp.yaml").read_text())
    config["simulator"].update(
        dataset_path=str(
            Path(
                "datasets/citylearn_three_phase_electrical_service_demo_15min_parquet/schema.json"
            ).resolve()
        ),
        building_ids=["Building_1"],
        simulation_start_time_step=0,
        simulation_end_time_step=8,
        episode_time_steps=9,
    )
    config["evaluation"]["episodes"] = 1
    # This short fixture starts at the physical 10% minimum; a 50% reserve
    # cannot be reached with its conservative battery power in two hours.
    config["pipeline"][0]["hyperparameters"] = {"terminal_storage_soc": 0.1}
    config["tracking"].update(runtime_profiling_enabled=True, runtime_profiling_interval=1)
    run_experiment(config, "mpc", tmp_path)
    job = tmp_path / "jobs/mpc"
    result = json.loads((job / "results/result.json").read_text())
    assert result["status"] == "completed"
    rows = [
        json.loads(line)
        for line in next((job / "offline_training").rglob("milp_decisions.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 8
    assert [r["step"] for r in rows] == list(range(8))
    assert all(r["source_end_exclusive"] - r["source_start"] <= 96 for r in rows)
    assert all(not r["failed"] and r["solver"]["has_solution"] for r in rows)
    assert len({r["problem_sha256"] for r in rows}) >= 8
    assert (job / "bundle/artifact_manifest.json").exists()
    verification = json.loads((job / "bundle/verification.json").read_text())
    assert verification["status"] == "unverified"
    metrics = [json.loads(line) for line in (job / "logs/metrics.jsonl").read_text().splitlines()]
    assert sum("Runtime/predict_seconds" in row.get("metrics", {}) for row in metrics) == 8
