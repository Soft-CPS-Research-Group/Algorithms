from utils.artifact_manifest import build_manifest


def test_manifest_contains_core_sections():
    config = {
        "metadata": {"experiment_name": "test", "run_name": "run"},
        "simulator": {"dataset_name": "ds", "dataset_path": "path", "central_agent": False, "reward_function": "Reward"},
        "training": {"seed": 1},
        "topology": {"num_agents": 4},
        "algorithm": {"name": "MADDPG", "hyperparameters": {"gamma": 0.99}},
    }
    env_meta = {"observation_names": [], "encoders": [], "action_bounds": [], "reward_function": {"name": "Reward", "params": {}}}
    agent_meta = {"format": "onnx", "artifacts": []}

    manifest = build_manifest(config, env_meta, agent_meta)

    assert manifest["manifest_version"] == 1
    assert manifest["metadata"]["experiment_name"] == "test"
    assert manifest["environment"] == env_meta
    assert manifest["agent"] == agent_meta
