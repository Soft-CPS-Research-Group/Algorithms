from utils.artifact_manifest import build_manifest


def test_manifest_contains_core_sections_and_normalized_artifacts():
    config = {
        "metadata": {"experiment_name": "test", "run_name": "run", "community_name": "porto_cluster_a"},
        "bundle": {
            "bundle_version": "2026-03-10-v1",
            "description": "Bundle emitted from unit test",
            "alias_mapping_path": "aliases.json",
        },
        "simulator": {
            "dataset_name": "ds",
            "dataset_path": "path",
            "central_agent": False,
            "reward_function": "Reward",
        },
        "training": {"seed": 1},
        "topology": {"num_agents": 1},
        "algorithm": {"name": "MADDPG", "hyperparameters": {"gamma": 0.99}},
    }
    env_meta = {
        "observation_names": [["feat"]],
        "encoders": [[{"type": "NoNormalization", "params": {}}]],
        "action_bounds": [[{"low": [0.0], "high": [1.0]}]],
        "action_names": ["a0"],
        "action_names_by_agent": {"0": ["a0"]},
        "reward_function": {"name": "Reward", "params": {}},
    }
    agent_meta = {
        "format": "onnx",
        "artifacts": [
            {
                "agent_index": 0,
                "path": "onnx_models/agent_0.onnx",
                "observation_dimension": 1,
                "action_dimension": 1,
            }
        ],
    }

    manifest = build_manifest(config, env_meta, agent_meta)

    assert manifest["manifest_version"] == 1
    assert manifest["metadata"]["experiment_name"] == "test"
    assert manifest["metadata"]["community_name"] == "porto_cluster_a"
    assert manifest["metadata"]["bundle_version"] == "2026-03-10-v1"
    assert manifest["metadata"]["description"] == "Bundle emitted from unit test"
    assert manifest["metadata"]["alias_mapping_path"] == "aliases.json"
    assert manifest["environment"] == env_meta
    assert manifest["agent"]["format"] == "onnx"
    assert manifest["agent"]["artifacts"][0]["format"] == "onnx"
    assert manifest["agent"]["artifacts"][0]["config"] == {}
