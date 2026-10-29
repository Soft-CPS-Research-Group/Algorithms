"""Check actual appliance service with the pinned published simulator."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from citylearn.citylearn import CityLearnEnv

from algorithms.agents.baseline_policies import (
    NormalNoBatteryPolicy,
    NormalPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartLocalPolicy,
)


def schema_with_request(tmp_path):
    simulator = Path(
        os.environ.get("SIMULATOR_REPO", Path(__file__).resolve().parents[3] / "Simulator")
    )
    source = simulator / "tests/data/minute_ev_demo"
    assert source.exists(), "Pinned Simulator checkout is required for acceptance tests"
    target = tmp_path / "scenario"
    shutil.copytree(source, target)
    pd.DataFrame(
        [
            dict(
                profile_id="normal",
                duration_steps=2,
                total_energy_kwh=0.3,
                load_profile="[0.1,0.2]",
            )
        ]
    ).to_csv(target / "washer_profiles.csv", index=False)
    pd.DataFrame(
        [
            dict(
                cycle_id="requested",
                profile_id="normal",
                earliest_start_time_step=2,
                latest_start_time_step=5,
                deadline_time_step=6,
                priority=0.5,
                must_run=True,
            )
        ]
    ).to_csv(target / "washer_requests.csv", index=False)
    schema_path = target / "schema.json"
    schema = json.loads(schema_path.read_text())
    schema["root_directory"] = str(target)
    schema["actions"]["deferrable_appliance"] = {"active": True}
    for feature in (
        "pending",
        "running",
        "can_start",
        "deadline_missed",
        "must_run",
        "hours_until_latest_start",
        "slack_steps",
        "slack_ratio",
        "urgency_ratio",
    ):
        schema["observations"]["deferrable_appliance_" + feature] = {
            "active": True,
            "shared_in_central_agent": False,
        }
    schema["buildings"]["Building_1"]["deferrable_appliances"] = {
        "washer_1": {
            "type": "citylearn.energy_model.DeferrableAppliance",
            "cycle_profiles_file": "washer_profiles.csv",
            "flexibility_schedule_file": "washer_requests.csv",
        },
    }
    # No tariff incentive or PV: only the requested start and deadline distinguish policies.
    pricing = pd.read_csv(target / "pricing.csv")
    pricing.loc[:, :] = 0.2
    pricing.to_csv(target / "pricing.csv", index=False)
    schema_path.write_text(json.dumps(schema))
    return schema_path


@pytest.mark.parametrize(
    "cls,expected_start",
    [
        (NormalPolicy, 2),
        (NormalNoBatteryPolicy, 2),
        (RBCBasicPolicy, 4),
        (RBCSmartLocalPolicy, 4),
        (RBCCommunityPolicy, 4),
    ],
)
def test_washer_start_and_completed_service_in_simulator(cls, expected_start, tmp_path):
    schema_path = schema_with_request(tmp_path)
    env = CityLearnEnv(
        str(schema_path),
        interface="flat",
        central_agent=False,
        export_kpis_on_episode_end=False,
        offline=True,
    )
    agent = cls(
        {
            "simulator": {"dataset_path": str(schema_path)},
            "algorithm": {"name": cls.__name__, "hyperparameters": {}},
        }
    )
    try:
        observations, _ = env.reset(seed=0)
        agent.attach_environment(
            observation_names=env.observation_names,
            action_names=env.action_names,
            observation_space=env.observation_space,
            action_space=env.action_space,
            metadata={
                "building_names": [b.name for b in env.buildings],
                "seconds_per_time_step": env.seconds_per_time_step,
            },
        )
        appliance = env.buildings[0].deferrable_appliances[0]
        starts = []
        while not (env.terminated or env.truncated):
            step = env.time_step
            actions = agent.predict(observations, deterministic=True)
            if cls is NormalNoBatteryPolicy:
                index = env.action_names[0].index("electrical_storage")
                assert actions[0][index] == 0.0
            observations, *_ = env.step(actions)
            if appliance.action_feedback_last_start_applied[step]:
                starts.append(step)
        assert starts == [expected_start]
        assert appliance.cycle_state["requested"] == "completed"
        assert np.sum(appliance.electricity_consumption) == pytest.approx(0.3)
    finally:
        env.close()


@pytest.mark.parametrize(
    "cls",
    [NormalPolicy, NormalNoBatteryPolicy, RBCBasicPolicy, RBCSmartLocalPolicy, RBCCommunityPolicy],
)
def test_portuguese_dataset_complete_job_and_portable_policy(cls, tmp_path):
    from algorithms.export.consumer import verify_bundle
    from algorithms.runtime.job import run_experiment
    from tests.e2e.test_common_job_profiles import config_for

    simulator = Path(
        os.environ.get("SIMULATOR_REPO", Path(__file__).resolve().parents[3] / "Simulator")
    )
    schema = simulator / "data/datasets/rec_pt_2023_micro_4_v2/evaluation/core_4.json"
    assert schema.is_file(), "The current Portuguese dataset from Simulator 1.9.2 is required"
    config = config_for("NormalPolicy", tmp_path)
    config["job"] = {"profile": "evaluate"}
    config["training"]["episodes"] = 0
    config["evaluation"] = {"episodes": 1}
    config["simulator"].update(
        dataset_path=str(schema),
        dataset_name="CORE-4",
        interface="entity",
        central_agent=False,
        topology_mode="static",
        simulation_start_time_step=0,
        simulation_end_time_step=96,
        episode_time_steps=97,
        building_ids=None,
    )
    config["pipeline"] = [{"algorithm": cls.__name__}]
    config["outputs"] = {"model_artifacts": "bundle"}
    config["bundle"] = {"require_deployable": True}
    run_experiment(config, cls.__name__, tmp_path)
    job = tmp_path / "jobs" / cls.__name__
    assert verify_bundle(job / "bundle")["status"] == "passed"
    scorecard = json.loads((job / "results/kpis.json").read_text())
    assert scorecard["eligible_for_scorecard"] and scorecard["metrics"]
    assert scorecard["baseline_reference"]["equivalent_to_normal_policy"] is False
    assert len(list((job / "results/simulation_data").rglob("exported_kpis.manifest.json"))) == 1
