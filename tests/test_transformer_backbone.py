"""Tests for TransformerBackbone."""

import pytest
import torch

from algorithms.utils.transformer_backbone import TransformerOutput


class TestTransformerOutput:
    """Tests for TransformerOutput dataclass."""

    def test_transformer_output_creation(self) -> None:
        """TransformerOutput should store embeddings and metadata."""
        batch_size = 2
        n_total = 5
        n_ca = 2
        d_model = 64

        result = TransformerOutput(
            all_embeddings=torch.randn(batch_size, n_total, d_model),
            ca_embeddings=torch.randn(batch_size, n_ca, d_model),
            pooled=torch.randn(batch_size, d_model),
            n_ca=n_ca,
        )

        assert result.all_embeddings.shape == (batch_size, n_total, d_model)
        assert result.ca_embeddings.shape == (batch_size, n_ca, d_model)
        assert result.pooled.shape == (batch_size, d_model)
        assert result.n_ca == n_ca
