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
            return self.actor(ca_tokens, sro_tokens, nfc_token)
    
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
    
    session = ort.InferenceSession(onnx_path)
    
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
