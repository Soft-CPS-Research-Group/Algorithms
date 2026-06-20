import json

import pytest

from utils.bundle_validator import BundleValidationError, validate_bundle_contract


def _base_manifest() -> dict:
    return {
        "manifest_version": 1,
        "metadata": {
            "experiment_name": "demo",
            "run_name": "run-1",
        },
        "simulator": {},
        "training": {},
        "topology": {"num_agents": 1},
        "pipeline": [
            {"stage_index": 0, "algorithm": "MADDPG", "count": 1, "hyperparameters": {}}
        ],
        "environment": {
            "observation_names": [["feat"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [0], "high": [1]}]],
            "action_names": ["action"],
            "action_names_by_agent": {"0": ["action"]},
            "reward_function": {"name": "RewardFunction", "params": {}},
        },
        "agent": {
            "format": "onnx",
            "artifacts": [
                {
                    "agent_index": 0,
                    "path": "onnx_models/agent_0.onnx",
                    "format": "onnx",
                    "config": {},
                }
            ],
        },
    }


def test_validate_bundle_contract_accepts_onnx_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "onnx_models").mkdir(parents=True)
    (bundle_dir / "onnx_models" / "agent_0.onnx").write_bytes(b"placeholder")
    (bundle_dir / "aliases.json").write_text("{}", encoding="utf-8")

    manifest = _base_manifest()
    manifest["metadata"]["alias_mapping_path"] = "aliases.json"

    validate_bundle_contract(manifest, bundle_dir)


def test_validate_bundle_contract_accepts_rule_based_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "policy_agent_0.json").write_text(
        json.dumps({"default_actions": {"a": 0.0}, "rules": []}),
        encoding="utf-8",
    )

    manifest = _base_manifest()
    manifest["agent"] = {
        "format": "rule_based",
        "artifacts": [
            {
                "agent_index": 0,
                "path": "policy_agent_0.json",
                "format": "rule_based",
                "config": {"require_observations_envelope": True},
            }
        ],
    }

    validate_bundle_contract(manifest, bundle_dir)


def test_validate_bundle_contract_rejects_wrong_rule_based_filename(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "rbc_policy.json").write_text("{}", encoding="utf-8")

    manifest = _base_manifest()
    manifest["agent"] = {
        "format": "rule_based",
        "artifacts": [
            {
                "agent_index": 0,
                "path": "rbc_policy.json",
                "format": "rule_based",
                "config": {},
            }
        ],
    }

    with pytest.raises(BundleValidationError):
        validate_bundle_contract(manifest, bundle_dir)
