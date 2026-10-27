"""ONNX export utilities for Transformer networks.

This module provides ONNX export with dynamic axes support for
variable-cardinality Transformer models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from algorithms.constants import TRANSFORMER_ONNX_OPSET


def export_transformer_actor_to_onnx(
    actor: torch.nn.Module,
    output_path: str,
    example_ca_tokens: torch.Tensor,
    example_sro_tokens: torch.Tensor,
    example_nfc_token: torch.Tensor,
    opset_version: int = TRANSFORMER_ONNX_OPSET,
) -> None:
    """Export TransformerActor to ONNX with dynamic axes.

    The exported model supports variable batch size and sequence lengths
    for CA and SRO tokens at runtime.

    Args:
        actor: TransformerActor model to export.
        output_path: Path for the exported ONNX file.
        example_ca_tokens: Example CA tokens [B, N_ca, d_model].
        example_sro_tokens: Example SRO tokens [B, N_sro, d_model].
        example_nfc_token: Example NFC token [B, 1, d_model].
        opset_version: ONNX opset version (default: 14).
    """
    # Ensure model is in eval mode
    actor.eval()

    # Move to same device as examples
    device = example_ca_tokens.device
    actor = actor.to(device)

    # Create wrapper to handle optional masks
    class ActorWrapper(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(
            self,
            ca_tokens: torch.Tensor,
            sro_tokens: torch.Tensor,
            nfc_token: torch.Tensor,
        ) -> torch.Tensor:
            # The legacy MultiheadAttention exporter freezes the trace sequence
            # length in its reshape nodes. Express the same evaluation math with
            # input-derived dimensions so every advertised dynamic axis works.
            x = torch.cat([
                ca_tokens + self.actor.type_emb.weight[self.actor.TYPE_CA],
                sro_tokens + self.actor.type_emb.weight[self.actor.TYPE_SRO],
                nfc_token + self.actor.type_emb.weight[self.actor.TYPE_NFC],
            ], dim=1)
            for layer in self.actor.encoder.layers:
                normalized = layer.norm1(x) if layer.norm_first else x
                attn = layer.self_attn
                batch, length, width = normalized.shape
                heads = attn.num_heads
                head_dim = width // heads
                qkv = torch.nn.functional.linear(normalized, attn.in_proj_weight, attn.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.reshape(batch, length, heads, head_dim).transpose(1, 2)
                k = k.reshape(batch, length, heads, head_dim).transpose(1, 2)
                v = v.reshape(batch, length, heads, head_dim).transpose(1, 2)
                weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
                attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
                x = x + attn.out_proj(attended)
                if not layer.norm_first:
                    x = layer.norm1(x)
                normalized = layer.norm2(x) if layer.norm_first else x
                x = x + layer.linear2(layer.activation(layer.linear1(normalized)))
                if not layer.norm_first:
                    x = layer.norm2(x)
            if self.actor.encoder.norm is not None:
                x = self.actor.encoder.norm(x)
            return torch.tanh(self.actor.head(x[:, :ca_tokens.shape[1], :]))

    wrapped = ActorWrapper(actor)
    wrapped.eval()

    # Dynamic axes for variable cardinality
    dynamic_axes = {
        "ca_tokens": {0: "batch_size", 1: "n_ca"},
        "sro_tokens": {0: "batch_size", 1: "n_sro"},
        "nfc_token": {0: "batch_size"},
        "actions": {0: "batch_size", 1: "n_ca"},
    }

    # Export
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        (example_ca_tokens, example_sro_tokens, example_nfc_token),
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["ca_tokens", "sro_tokens", "nfc_token"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
    )


def verify_onnx_model(
    onnx_path: str,
    test_inputs: list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> bool:
    """Verify ONNX model with onnxruntime.

    Args:
        onnx_path: Path to ONNX model file.
        test_inputs: List of (ca_tokens, sro_tokens, nfc_token) tuples.

    Returns:
        True if all tests pass.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])

    for ca_tokens, sro_tokens, nfc_token in test_inputs:
        inputs = {
            "ca_tokens": ca_tokens.cpu().numpy(),
            "sro_tokens": sro_tokens.cpu().numpy(),
            "nfc_token": nfc_token.cpu().numpy(),
        }

        outputs = session.run(None, inputs)

        # Check output shape matches N_ca
        B, N_ca, _ = ca_tokens.shape
        assert outputs[0].shape[1] == N_ca, (
            f"Output N_ca mismatch: {outputs[0].shape[1]} vs {N_ca}"
        )

    return True
