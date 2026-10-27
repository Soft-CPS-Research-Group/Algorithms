"""Generate small synthetic producer/consumer fixtures; never production model evidence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from algorithms import __version__
from algorithms.export.bundle import export_tree
from algorithms.export.references import begin_capture
from algorithms.runtime.atomic import atomic_json
from algorithms.runtime.interaction import decide
from utils.artifact_manifest import build_manifest, write_manifest
from utils.bundle_validator import validate_bundle_contract
from utils.training_provenance import file_digest, seal_manifest


class FixtureLinear:
    agent_name = "SyntheticLinearFixture"
    capabilities = SimpleNamespace(
        learning="inference", observations="encoded", input_signal="none", output_signal="actions"
    )

    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.model.weight.copy_(torch.tensor([[0.25, -0.5]]))
            self.model.bias.zero_()

    def predict(self, observations, deterministic=None, context=None):
        return (
            self.model(torch.tensor(np.asarray(observations), dtype=torch.float32))
            .detach()
            .numpy()
            .tolist()
        )

    def export_artifacts(self, output_dir, context=None):
        path = Path(output_dir) / "model.onnx"
        torch.onnx.export(
            self.model,
            torch.zeros(1, 2),
            str(path),
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=13,
        )
        return {
            "format": "onnx",
            "artifacts": [{"agent_index": 0, "path": path.name, "format": "onnx"}],
        }


def generate():
    from algorithms.registry import build_execution_unit

    base = Path("contracts/bundle/v2/fixtures")
    for kind in ("onnx", "policy"):
        root = base / f"valid_{kind}"
        root.mkdir(parents=True, exist_ok=True)
        algorithm = "RandomPolicy" if kind == "policy" else "SyntheticLinearFixture"
        config = {
            "metadata": {"experiment_name": "synthetic-contract-fixture", "run_name": kind},
            "training": {"seed": 31},
            "pipeline": [{"algorithm": algorithm}],
            "simulator": {},
            "topology": {"num_agents": 1, "observation_dimensions": [2], "action_dimensions": [1]},
        }
        environment = {
            "observation_names": [["a", "b"]],
            "encoders": [[{"type": "NoNormalization", "params": {}}] * 2],
            "action_bounds": [{"low": [-1.0], "high": [1.0]}],
            "action_names_by_agent": {"0": ["action"]},
            "reward_function": {"name": "fixture", "params": {}},
            "building_names": ["SyntheticBuilding"],
        }
        if kind == "policy":
            unit = build_execution_unit(config)
            unit.attach_environment(
                observation_names=environment["observation_names"],
                action_names=[["action"]],
                action_space=[Box(-1, 1, (1,))],
                observation_space=[Box(-1, 1, (2,))],
                metadata={},
            )
        else:
            unit = FixtureLinear()
        begin_capture(unit, config)
        for observation in ([[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]):
            decide(unit, [np.array(row) for row in observation], True)
        agent = export_tree(unit, root, {"config": config, "environment": environment})
        manifest = build_manifest(config, environment, agent)
        fixture_path = root / "fixture.json"
        atomic_json(
            fixture_path,
            {"synthetic": True, "purpose": "bundle compatibility, not trained model quality"},
        )
        manifest.update(
            manifest_version=2,
            contract="algorithms.bundle.v2",
            algorithm={"name": algorithm},
            execution=agent["composition"],
            deployment=agent["deployment"],
            evaluation={"fixture": True},
            build={"fixture": True, "versions": {"energaize-algorithms": __version__}},
            evidence={"fixture": {"path": "fixture.json", "sha256": file_digest(fixture_path)}},
        )
        seal_manifest(manifest, root, fixture_path, f"synthetic-{kind}")
        validate_bundle_contract(manifest, root)
        write_manifest(manifest, root)
    atomic_json(
        base / "invalid_cases.json",
        [
            {"name": "missing_model", "operation": "delete", "path": "model.onnx"},
            {"name": "wrong_hash", "operation": "corrupt", "path": "model.onnx"},
            {"name": "absolute_path", "operation": "artifact_path", "value": "/tmp/model.onnx"},
            {"name": "traversal", "operation": "artifact_path", "value": "../model.onnx"},
            {"name": "unsupported_version", "operation": "version", "value": 99},
            {"name": "unknown_format", "operation": "format", "value": "unrecognised_policy"},
            {"name": "duplicate_node", "operation": "duplicate_node"},
            {"name": "unsealed_file", "operation": "extra_file"},
            {"name": "symlink", "operation": "symlink", "path": "model.onnx"},
        ],
    )


if __name__ == "__main__":
    generate()
