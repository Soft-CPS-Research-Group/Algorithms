from __future__ import annotations

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
                    "artifact_config": {"input_site_key": "site_a"},
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
        assert artifact["config"]["input_site_key"] == "site_a"
        assert (tmp_path / artifact["path"]).exists()
