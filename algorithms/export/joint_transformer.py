"""Static-layout historical MADDPG export, including its learned tokenizer."""

from pathlib import Path

import torch

from algorithms.constants import TRANSFORMER_ONNX_OPSET


class JointTransformerPolicy(torch.nn.Module):
    def __init__(self, unit):
        super().__init__()
        self.tokenizer = unit.tokenizer
        self.actor = unit.actor

    def forward(self, *observations):
        tokenizer = self.tokenizer
        ca = []
        for index, observation in enumerate(observations):
            indices = tokenizer.ca_indices[index]
            features = observation[:, indices] if indices else observation[:, :1] * 0
            ca.append(tokenizer.ca_embeddings[index](features))
        first = observations[0]
        sro = first[:, tokenizer.sro_indices] if tokenizer.sro_indices else first[:, :1] * 0
        nfc = first[:, tokenizer.nfc_indices] if tokenizer.nfc_indices else first[:, :1] * 0
        return self.actor(
            torch.stack(ca, dim=1),
            tokenizer.sro_embedding(sro).unsqueeze(1),
            tokenizer.nfc_embedding(nfc).unsqueeze(1),
        )


def export_joint_transformer(unit, output_dir):
    root = Path(output_dir)
    path = root / "onnx_models/joint_policy.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = JointTransformerPolicy(unit).to(unit.device).eval()
    inputs = tuple(
        torch.zeros(1, len(names), device=unit.device) for names in unit.observation_names
    )
    input_names = [f"observation_agent_{index}" for index in range(len(inputs))]
    axes = {name: {0: "batch_size"} for name in input_names + ["actions"]}
    # Disable the fused inference-only operator unsupported by legacy ONNX.
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(
            model,
            inputs,
            str(path),
            opset_version=TRANSFORMER_ONNX_OPSET,
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes=axes,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    return {
        "format": "onnx",
        "architecture": "TransformerMADDPG",
        "artifacts": [
            {
                "agent_index": 0,
                "path": path.relative_to(root).as_posix(),
                "format": "onnx",
                "config": {
                    "scope": "joint",
                    "topology": "static",
                    "tokenizer": "embedded",
                    "input_names": input_names,
                    "output_order": unit.action_names,
                },
            }
        ],
    }
