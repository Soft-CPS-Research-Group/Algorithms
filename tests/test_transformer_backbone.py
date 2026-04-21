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


class TestTransformerBackbone:
    """Tests for TransformerBackbone class."""

    def test_backbone_creation(self) -> None:
        """Backbone should create with specified architecture."""
        from algorithms.utils.transformer_backbone import TransformerBackbone

        backbone = TransformerBackbone(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )

        assert backbone.d_model == 64
        assert backbone.type_embedding is not None
        assert backbone.encoder is not None

    def test_backbone_forward_shape(self) -> None:
        """Forward pass should produce correctly shaped outputs."""
        from algorithms.utils.transformer_backbone import TransformerBackbone

        d_model = 64
        backbone = TransformerBackbone(
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
        )

        batch_size = 2
        n_ca = 3
        n_sro = 2

        ca_tokens = torch.randn(batch_size, n_ca, d_model)
        sro_tokens = torch.randn(batch_size, n_sro, d_model)
        nfc_token = torch.randn(batch_size, 1, d_model)

        result = backbone(ca_tokens, sro_tokens, nfc_token)

        n_total = n_ca + n_sro + 1
        assert result.all_embeddings.shape == (batch_size, n_total, d_model)
        assert result.ca_embeddings.shape == (batch_size, n_ca, d_model)
        assert result.pooled.shape == (batch_size, d_model)
        assert result.n_ca == n_ca

    def test_backbone_type_embeddings(self) -> None:
        """Type embeddings should differentiate CA, SRO, NFC."""
        from algorithms.utils.transformer_backbone import TransformerBackbone

        d_model = 64
        backbone = TransformerBackbone(d_model=d_model, nhead=4, num_layers=1)

        # Type embeddings: 0=CA, 1=SRO, 2=NFC
        assert backbone.type_embedding.num_embeddings == 3
        assert backbone.type_embedding.embedding_dim == d_model

    def test_backbone_variable_token_count(self) -> None:
        """Same backbone should handle different token counts."""
        from algorithms.utils.transformer_backbone import TransformerBackbone

        d_model = 64
        backbone = TransformerBackbone(d_model=d_model, nhead=4, num_layers=2)

        # First call: 2 CAs, 2 SROs
        result1 = backbone(
            torch.randn(1, 2, d_model),
            torch.randn(1, 2, d_model),
            torch.randn(1, 1, d_model),
        )
        assert result1.n_ca == 2

        # Second call: 5 CAs, 3 SROs
        result2 = backbone(
            torch.randn(1, 5, d_model),
            torch.randn(1, 3, d_model),
            torch.randn(1, 1, d_model),
        )
        assert result2.n_ca == 5

    def test_backbone_gradient_flow(self) -> None:
        """Gradients should flow through attention to all token types."""
        from algorithms.utils.transformer_backbone import TransformerBackbone

        d_model = 64
        backbone = TransformerBackbone(d_model=d_model, nhead=4, num_layers=2)

        ca_tokens = torch.randn(1, 2, d_model, requires_grad=True)
        sro_tokens = torch.randn(1, 2, d_model, requires_grad=True)
        nfc_token = torch.randn(1, 1, d_model, requires_grad=True)

        result = backbone(ca_tokens, sro_tokens, nfc_token)
        loss = result.pooled.sum()
        loss.backward()

        # All inputs should have gradients
        assert ca_tokens.grad is not None
        assert sro_tokens.grad is not None
        assert nfc_token.grad is not None
