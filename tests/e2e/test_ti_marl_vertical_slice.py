"""Short real-Simulator TI-MARL dynamic topology vertical slice."""

from __future__ import annotations

import csv
import json
import gzip
from pathlib import Path

import citylearn
from packaging.version import Version
import pytest

from algorithms.registry import build_execution_unit
from utils.wrapper_citylearn import Wrapper_CityLearn


pytestmark = pytest.mark.skipif(
    Version(citylearn.__version__) < Version("1.7.0"),
    reason="TI-MARL vertical slice requires runtime_status_v1 from Simulator 1.7.0",
)


def test_ti_marl_trains_across_member_join_and_leave_without_resize(tmp_path):
    from citylearn.citylearn import CityLearnEnv

    dataset_root = Path("datasets/citylearn_three_phase_dynamic_topology_demo").resolve()
    if not dataset_root.exists():
        pytest.skip("dynamic topology demo dataset is not installed")
    schema = json.loads((dataset_root / "schema.json").read_text(encoding="utf-8"))
    schema["topology_events"] = [
        {**schema["topology_events"][0], "time_step": 2},
        {**schema["topology_events"][-1], "time_step": 4},
    ]
    events_path = tmp_path / "ti_marl_runtime_events.csv"
    event_fields = [
        "event_id",
        "module",
        "target_type",
        "target_id",
        "target_feature",
        "start_time_step",
        "end_time_step",
        "mode",
        "event_domain",
    ]
    events = [
        {
            "event_id": "building_sensor_stuck",
            "module": "observation",
            "target_type": "building",
            "target_id": "Building_1",
            "target_feature": "net_power_kw",
            "start_time_step": 0,
            "end_time_step": 4,
            "mode": "stuck",
            "event_domain": "SENSOR_CHANNEL",
        },
        {
            "event_id": "building_sensor_loss",
            "module": "observation",
            "target_type": "building",
            "target_id": "Building_2",
            "target_feature": "non_shiftable_load",
            "start_time_step": 1,
            "end_time_step": 2,
            "mode": "missing",
            "event_domain": "SENSOR_CHANNEL",
        },
        {
            "event_id": "storage_actuator_loss",
            "module": "action",
            "target_type": "storage",
            "target_id": "Building_1",
            "target_feature": "electrical_storage",
            "start_time_step": 1,
            "end_time_step": 2,
            "mode": "dropout",
            "event_domain": "ACTUATOR_CHANNEL",
        },
        {
            "event_id": "storage_unavailable",
            "module": "asset",
            "target_type": "storage",
            "target_id": "Building_2",
            "target_feature": "both",
            "start_time_step": 1,
            "end_time_step": 2,
            "mode": "unavailable",
            "event_domain": "ASSET_AVAILABILITY",
        },
        {
            "event_id": "community_link_loss",
            "module": "forecast",
            "target_type": "district",
            "target_id": "*",
            "target_feature": "electricity_pricing_predicted_1",
            "start_time_step": 1,
            "end_time_step": 2,
            "mode": "missing",
            "event_domain": "COMMUNICATION_LINK",
        },
    ]
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(events)
    schema.setdefault("observation_bundles", {})["entity_robustness"] = {
        "active": True
    }
    schema["robustness"] = {
        "enabled": True,
        "events_file": str(events_path),
        "random_seed": 11,
        "missing_replacement_value": -9999.0,
        "modules": {
            "observations": {"enabled": True},
            "forecasts": {"enabled": True},
            "actions": {"enabled": True},
            "assets": {"enabled": True},
        },
    }
    config = {
        "metadata": {"experiment_name": "ti_marl_e2e", "run_name": "join_leave"},
        "runtime": {"job_dir": str(tmp_path), "log_dir": str(tmp_path / "logs")},
        "tracking": {
            "mlflow_enabled": False,
            "progress_updates_enabled": False,
            "mlflow_step_sample_interval": 999,
        },
        "checkpointing": {},
        "simulator": {
            "dataset_name": "dynamic_topology_demo",
            "dataset_path": str(dataset_root / "schema.json"),
            "central_agent": False,
            "interface": "entity",
            "topology_mode": "dynamic",
            "reward_function": "CostHardConstraintReward",
            "episodes": 1,
            "episode_time_steps": 9,
            # Wrapper construction performs one metadata reset.  Replay that
            # exact scenario so absolute-time robustness fixtures remain
            # paired with the training episode.
            "repeat_episode_scenario": True,
            "entity_encoding": {"enabled": True, "profile": "minmax_space"},
        },
        "training": {
            "seed": 11,
            "steps_between_training_updates": 1,
            "target_update_interval": 0,
        },
        "topology": {},
        "pipeline": [
            {
                "algorithm": "TIMARL",
                "count": 1,
                "hyperparameters": {
                    "contract_version": "ti_marl_v1",
                    "typed_interface_path": "configs/ti_marl/typed_interface_v1.yaml",
                    "backbone": {"name": "mappo"},
                    "actor": {"d_model": 32, "attention_heads": 4, "relation_layers": 1},
                    "critic": {"kind": "set"},
                    "feasibility": {"kind": "analytic_projection"},
                    "rollout_steps": 64,
                    "ppo_epochs": 1,
                    "trace": {"enabled": True, "chunk_size": 8, "snapshot_interval": 2},
                },
            }
        ],
    }
    env = CityLearnEnv(
        schema,
        root_directory=dataset_root,
        interface="entity",
        topology_mode="dynamic",
        central_agent=False,
        episode_time_steps=9,
        random_seed=11,
    )
    try:
        model = build_execution_unit(config)
        initial_parameter_count = model._parameter_count
        wrapper = Wrapper_CityLearn(
            env,
            model=model,
            config=config,
            job_id="ti-marl-e2e",
            progress_path=tmp_path / "progress.json",
        )
        wrapper.learn(episodes=1, deterministic=False)

        assert wrapper.global_step == 8
        assert model._parameter_count == initial_parameter_count
        assert model._current_snapshot.topology_version == 2
        assert len(model._current_snapshot.agent_ids) == 17
        assert model.learner.update_count == 1
        assert model.trace_writer.transition_count == 8
        trace_files = sorted((tmp_path / "results" / "ti_marl_trace").glob("*.jsonl.gz"))
        assert trace_files
        records = []
        for trace_path in trace_files:
            with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle)
        transitions = [row["payload"] for row in records if row["kind"] == "transition"]
        snapshots = [row["payload"] for row in records if row["kind"] == "snapshot"]
        assert transitions
        assert all(
            row["execution"]["version"] == "entity_action_execution_v1"
            for row in transitions
        )
        assert any(row["topology_events"] for row in transitions)
        for row in transitions:
            requested_by_agent = {}
            for entry in row["execution"]["entries"]:
                requested_by_agent.setdefault(entry["agent_id"], []).append(
                    entry["requested_value"]
                )
            for agent_id, command in zip(row["agent_ids"], row["commands"]):
                assert sorted(requested_by_agent.get(agent_id, [])) == pytest.approx(
                    sorted(command)
                )

        fault_modes = {
            evidence["fault_mode"]
            for snapshot in snapshots
            for evidence in snapshot["fault_evidence"]
        }
        assert {"stuck", "missing", "dropout", "unavailable"} <= fault_modes
        stuck_states = {
            assessment["state"]
            for snapshot in snapshots
            for assessment in snapshot["health"]
            if assessment["subject_id"].startswith(
                "SENSOR_CHANNEL:building:Building_1:net_power_kw"
            )
        }
        assert {"DEGRADED", "STALE", "HEALTHY"} <= stuck_states
        consequences = {
            item["consequence"]
            for snapshot in snapshots
            for item in snapshot["closure_log"]
        }
        assert "invalidate_non_idle_ports" in consequences
        assert "disable_group" in consequences
        # A cloud/community failure never removes the local controller.
        assert all(row["commands"] for row in transitions)
    finally:
        env.close()
