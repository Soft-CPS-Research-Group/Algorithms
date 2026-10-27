"""Offline deterministic actors with feature order and learned standardisation embedded."""

from pathlib import Path

import torch

from algorithms.constants import DEFAULT_ONNX_OPSET


class StandardisedPolicy(torch.nn.Module):
    def __init__(self, policy, standardiser, observation_order, action_order):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.as_tensor(standardiser.mean))
        self.register_buffer("std", torch.as_tensor(standardiser.std))
        self.observation_order = observation_order
        self.action_order = action_order

    def forward(self, observations):
        observations = torch.nan_to_num(observations, nan=0.0)
        standardized = (observations[:, self.observation_order] - self.mean) / self.std
        return self.policy.predict_deterministic(standardized)[:, self.action_order]


def export_entity_policies(unit, output_dir):
    root = Path(output_dir)
    directory = root / "onnx_models"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for index, dimension in enumerate(unit._obs_dims_per_agent):
        order = unit._observation_orders[index]
        action_order = unit._action_orders[index]
        model = (
            StandardisedPolicy(
                unit._policy_map[dimension], unit._standardiser_map[dimension], order, action_order
            )
            .to(unit._device)
            .eval()
        )
        path = directory / f"agent_{index}.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, dimension, device=unit._device),
            str(path),
            opset_version=DEFAULT_ONNX_OPSET,
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
        )
        artifacts.append(
            {
                "agent_index": index,
                "path": path.relative_to(root).as_posix(),
                "format": "onnx",
                "observation_dimension": dimension,
                "action_dimension": len(action_order),
                "config": {"standardisation": "embedded", "feature_order": "embedded"},
            }
        )
    return {"format": "onnx", "artifacts": artifacts}
