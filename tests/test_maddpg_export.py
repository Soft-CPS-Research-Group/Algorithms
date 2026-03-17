from __future__ import annotations

import json

import torch

from algorithms.agents.maddpg_agent import MADDPG


def test_maddpg_export_artifacts_includes_per_artifact_format_and_config(tmp_path):
    agent = MADDPG.__new__(MADDPG)
    agent.device = torch.device("cpu")
    agent.actors = [
        torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Tanh()),
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Tanh()),
    ]
    agent.observation_dimension = [3, 2]
    agent.action_dimension = [1, 2]

    metadata = agent.export_artifacts(
        output_dir=str(tmp_path),
        context={
            "config": {
                "bundle": {
                    "require_observations_envelope": True,
                    "artifact_config": {"input_site_key": "site_a", "community_optimization_enabled": False},
                    "per_agent_artifact_config": {
                        "1": {
                            "input_site_key": "site_b",
                            "community_optimization_enabled": True,
                            "require_observations_envelope": False,
                        }
                    },
                }
            }
        },
    )

    assert metadata["format"] == "onnx"
    assert len(metadata["artifacts"]) == 2

    for i, artifact in enumerate(metadata["artifacts"]):
        assert artifact["agent_index"] == i
        assert artifact["format"] == "onnx"
        assert artifact["config"]["require_observations_envelope"] is True
        if i == 0:
            assert artifact["config"]["input_site_key"] == "site_a"
            assert artifact["config"]["community_optimization_enabled"] is False
        else:
            assert artifact["config"]["input_site_key"] == "site_b"
            assert artifact["config"]["community_optimization_enabled"] is True
        assert (tmp_path / artifact["path"]).exists()


def test_maddpg_export_artifacts_auto_populates_phase_config_from_schema(tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "buildings": {
                    "Building_1": {
                        "chargers": {
                            "AC0001": {
                                "attributes": {
                                    "max_charging_power": 4.6,
                                    "min_charging_power": 0.0,
                                    "phase_connection": "L1",
                                }
                            },
                            "BB0001": {
                                "attributes": {
                                    "max_charging_power": 9.0,
                                    "min_charging_power": 0.0,
                                    "phase_connection": "all_phases",
                                }
                            },
                        },
                        "electrical_service": {
                            "limits": {
                                "total": {"import_kw": 12.0},
                                "per_phase": {
                                    "L1": {"import_kw": 6.0},
                                    "L2": {"import_kw": 6.0},
                                },
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    agent = MADDPG.__new__(MADDPG)
    agent.device = torch.device("cpu")
    agent.actors = [torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Tanh())]
    agent.observation_dimension = [2]
    agent.action_dimension = [2]

    metadata = agent.export_artifacts(
        output_dir=str(tmp_path),
        context={
            "config": {
                "simulator": {"dataset_path": str(schema_path)},
                "bundle": {
                    "artifact_config": {"input_site_key": "boavista"},
                    "per_agent_artifact_config": {"0": {"max_board_kw": 15.0}},
                },
            },
            "environment": {
                "building_names": ["Building_1"],
                "action_names_by_agent": {"0": ["AC0001", "BB0001"]},
            },
        },
    )

    artifact = metadata["artifacts"][0]
    assert artifact["config"]["input_site_key"] == "boavista"
    assert artifact["config"]["max_board_kw"] == 15.0
    assert artifact["config"]["chargers"]["AC0001"]["line"] == "L1"
    assert artifact["config"]["chargers"]["BB0001"]["phases"] == ["L1", "L2"]
    assert artifact["config"]["line_limits"]["L1"]["limit_kw"] == 6.0
    assert artifact["config"]["action_order"] == ["AC0001", "BB0001"]
