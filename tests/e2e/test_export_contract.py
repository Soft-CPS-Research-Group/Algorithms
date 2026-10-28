"""Real jobs must export decisions equivalent to the policy actually evaluated."""

import json

import pytest

from algorithms.runtime.job import run_experiment
from tests.e2e.test_common_job_profiles import config_for
from utils.bundle_validator import validate_bundle_contract


@pytest.mark.parametrize(
    "name",
    [
        "MADDPG",
        "MATD3",
        "MASAC",
        "PPO",
        "TD3",
        "IPPO",
        "MAPPO",
        "HAPPO",
        "NormalPolicy",
        "NormalNoBatteryPolicy",
        "RandomPolicy",
        "RBCBasicPolicy",
        "RBCCommunityPolicy",
        "RBCSmartPolicy",
        "RBCSmartLocalPolicy",
    ],
)
def test_policy_exports_match_final_evaluation(name, tmp_path):
    config = config_for(
        "MADDPG"
        if name in {"MADDPG", "MATD3", "MASAC", "PPO", "TD3", "IPPO", "MAPPO", "HAPPO"}
        else "NormalPolicy",
        tmp_path,
    )
    config["pipeline"][0]["algorithm"] = name
    if name in {"PPO", "TD3"}:
        config["simulator"]["building_ids"] = ["Building_2"]
    config["outputs"] = {"model_artifacts": "bundle"}
    config["bundle"] = {"require_deployable": True}
    run_experiment(config, name, tmp_path)
    job = tmp_path / "jobs" / name
    root = job / "bundle"
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    validate_bundle_contract(manifest, root)
    assert manifest["manifest_version"] == 2
    assert manifest["deployment"]["eligible"] is True
    from algorithms.export.consumer import verify_bundle

    assert verify_bundle(root)["status"] == "passed"
    verification = json.loads((root / "verification.json").read_text())
    assert verification["status"] == "passed" and verification["samples"] >= 2
    kpis = json.loads((job / "results/kpis.json").read_text())
    assert kpis["eligible_for_scorecard"] is True
    assert len(kpis["sources"]) == 1 and kpis["metrics"]
    assert kpis["evaluation"]["sha256"] == manifest["evaluation"]["sha256"]
    assert not list((job / "results/simulation_data").rglob("*ep0*.csv"))


def test_null_flags_default_to_final_and_diagnostics_are_separate(tmp_path):
    config = config_for("NormalPolicy", tmp_path)
    config["outputs"] = {"model_artifacts": "none"}
    config["simulator"]["export"].update(
        final_episode_only=True,
        kpis_final_episode_only=None,
        timeseries_final_episode_only=None,
        export_business_as_usual_timeseries=False,
    )
    run_experiment(config, "final", tmp_path)
    history = json.loads((tmp_path / "jobs/final/results/result.json").read_text())[
        "episode_execution"
    ]
    assert [item["export_enabled"] for item in history] == [False, True]
    config["simulator"]["export"].update(
        profile="diagnostic",
        final_episode_only=False,
        kpis_final_episode_only=False,
        diagnostic_max_episodes=1,
    )
    run_experiment(config, "diagnostic", tmp_path)
    files = list((tmp_path / "jobs/diagnostic/results/simulation_data").rglob("*.csv"))
    assert sum(path.name == "diagnostic_kpis_ep0.csv" for path in files) == 1
    assert sum(path.name == "exported_kpis.csv" for path in files) == 1
    assert not any("business_as_usual" in path.name for path in files)


def test_no_evaluation_never_claims_final_kpis(tmp_path):
    config = config_for("NormalPolicy", tmp_path)
    config["job"] = {"profile": "train"}
    config["training"]["episodes"] = 1
    config["evaluation"] = {"episodes": 0}
    config["outputs"] = {"model_artifacts": "none"}
    run_experiment(config, "training", tmp_path)
    result = json.loads((tmp_path / "jobs/training/results/kpis.json").read_text())
    assert result["missing_reason"] == "no_final_evaluation"
    assert result["eligible_for_scorecard"] is False
    assert not result["sources"]


@pytest.mark.parametrize(
    "name",
    [
        "PPO",
        "TD3",
        "SignalAwareRBC",
        "SignalAwareRBCSmartLocal",
        "SignalAwareRBCCommunity",
        "SignalDirectedRBC",
    ],
)
def test_compositions_replay_the_complete_graph(name, tmp_path):
    from algorithms.export.consumer import verify_bundle

    config = config_for("MADDPG" if name in {"PPO", "TD3"} else "NormalPolicy", tmp_path)
    config["pipeline"][0]["algorithm"] = name
    if name in {"PPO", "TD3"}:
        config["pipeline"][0]["count"] = 2
    else:
        config["pipeline"].insert(
            0,
            {
                "algorithm": "FixedPriceSignal",
                "hyperparameters": {
                    "schedule": [
                        {"start_step": 0, "multiplier": 0.7},
                        {"start_step": 2, "multiplier": 1.2},
                    ]
                },
            },
        )
    config["outputs"] = {"model_artifacts": "bundle"}
    config["bundle"] = {"require_deployable": True}
    run_experiment(config, name, tmp_path)
    root = tmp_path / "jobs" / name / "bundle"
    assert verify_bundle(root)["samples"] >= 2
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    assert manifest["agent"]["format"] == "composition"
    assert len(manifest["execution"]["nodes"]) == 3


@pytest.mark.parametrize(
    "manager", ["CausalPriceSignal", "CCLevel1", "CCLevel2", "CommunityCoordinator"]
)
def test_manager_export_preserves_complete_pipeline_and_declares_support(manager, tmp_path):
    config = config_for("NormalPolicy", tmp_path)
    parameters = (
        {}
        if manager == "CausalPriceSignal"
        else {"hidden_dims": [8], "num_steps": 4, "mini_batch_size": 2, "num_epochs": 1}
    )
    if manager == "CCLevel2":
        parameters.update(num_buildings=2, c_dim=28, reference_multipliers=[1.0, 1.0])
    if manager == "CommunityCoordinator":
        parameters.update(output_mode="signal", num_buildings=2)
    config["pipeline"].insert(0, {"algorithm": manager, "hyperparameters": parameters})
    config["outputs"] = {"model_artifacts": "bundle"}
    run_experiment(config, manager, tmp_path)
    root = tmp_path / "jobs" / manager / "bundle"
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    validate_bundle_contract(manifest, root)
    assert [node["algorithm"] for node in manifest["execution"]["nodes"][1:]] == [
        manager,
        "NormalPolicy",
    ]
    if manager == "CausalPriceSignal":
        from algorithms.export.consumer import verify_bundle

        assert verify_bundle(root)["status"] == "passed"
    else:
        assert manifest["deployment"]["eligible"] is False
        assert manifest["deployment"]["reasons"]


def test_building_agent_export_embeds_learned_normalisation_and_context(tmp_path):
    from algorithms.export.consumer import verify_bundle

    config = config_for("NormalPolicy", tmp_path)
    config["pipeline"] = [
        {"algorithm": "FixedPriceSignal", "hyperparameters": {"multipliers": [0.8, 1.2]}},
        {
            "algorithm": "BuildingAgent",
            "count": 2,
            "hyperparameters": {
                "hidden_dims": [8],
                "num_steps": 4,
                "mini_batch_size": 2,
                "num_epochs": 1,
            },
        },
    ]
    config["outputs"] = {"model_artifacts": "bundle"}
    config["bundle"] = {"require_deployable": True}
    run_experiment(config, "building", tmp_path)
    assert verify_bundle(tmp_path / "jobs/building/bundle")["status"] == "passed"
