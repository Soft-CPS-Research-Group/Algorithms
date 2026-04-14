"""Shared constants used across algorithm implementations."""

DEFAULT_ONNX_OPSET = 13
"""Opset version used for ONNX exports to stay compatible with inference runtime."""

TRANSFORMER_ONNX_OPSET = 14
"""Opset version for Transformer models (requires scaled_dot_product_attention)."""
