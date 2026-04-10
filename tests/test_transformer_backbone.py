"""Tests for the Transformer backbone (Phase 2).

Verifies forward shapes, variable cardinality handling, CA output extraction,
and gradient flow through attention to all token types.
"""

from __future__ import annotations

import pytest
import torch

from algorithms.utils.observation_tokenizer import TokenizedObservation
from algorithms.utils.transformer_backbone import TransformerBackbone, TransformerOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 32
BATCH = 4


def _make_tokenized(
    batch: int,
    n_ca: int,
    n_sro: int,
    d_model: int,
    ca_types: list[str] | None = None,
) -> TokenizedObservation:
    """Create a synthetic TokenizedObservation for testing."""
    return TokenizedObservation(
        ca_tokens=torch.randn(batch, n_ca, d_model),
        sro_tokens=torch.randn(batch, n_sro, d_model),
        rl_token=torch.randn(batch, 1, d_model),
        ca_types=ca_types or [f"type_{i}" for i in range(n_ca)],
        n_ca=n_ca,
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestForwardShape:
    def test_basic_shape(self):
        """Given N_ca=3, N_sro=2 → output shape [batch, 6, d_model]."""
        backbone = TransformerBackbone(
            d_model=D_MODEL, nhead=4, num_layers=2,
            dim_feedforward=64, dropout=0.0,
        )
        tok = _make_tokenized(BATCH, n_ca=3, n_sro=2, d_model=D_MODEL)
        out = backbone(tok)

        n_total = 3 + 2 + 1  # CA + SRO + RL
        assert out.all_embeddings.shape == (BATCH, n_total, D_MODEL)
        assert out.ca_embeddings.shape == (BATCH, 3, D_MODEL)
        assert out.pooled.shape == (BATCH, D_MODEL)
        assert out.n_ca == 3

    def test_no_ca_tokens(self):
        """N_ca=0 → ca_embeddings is [batch, 0, d_model]."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(BATCH, n_ca=0, n_sro=2, d_model=D_MODEL)
        out = backbone(tok)

        n_total = 0 + 2 + 1
        assert out.all_embeddings.shape == (BATCH, n_total, D_MODEL)
        assert out.ca_embeddings.shape == (BATCH, 0, D_MODEL)
        assert out.n_ca == 0

    def test_no_sro_tokens(self):
        """N_sro=0 → only CA + RL tokens."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(BATCH, n_ca=2, n_sro=0, d_model=D_MODEL)
        out = backbone(tok)

        n_total = 2 + 0 + 1
        assert out.all_embeddings.shape == (BATCH, n_total, D_MODEL)
        assert out.ca_embeddings.shape == (BATCH, 2, D_MODEL)

    def test_only_rl_token(self):
        """N_ca=0, N_sro=0 → only the RL token."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(BATCH, n_ca=0, n_sro=0, d_model=D_MODEL)
        out = backbone(tok)

        assert out.all_embeddings.shape == (BATCH, 1, D_MODEL)
        assert out.pooled.shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# Variable cardinality test
# ---------------------------------------------------------------------------


class TestVariableCardinality:
    def test_same_model_different_n_ca(self):
        """Same backbone processes inputs with different N_ca."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)

        tok_3ca = _make_tokenized(BATCH, n_ca=3, n_sro=2, d_model=D_MODEL)
        tok_1ca = _make_tokenized(BATCH, n_ca=1, n_sro=2, d_model=D_MODEL)

        out_3 = backbone(tok_3ca)
        out_1 = backbone(tok_1ca)

        assert out_3.ca_embeddings.shape == (BATCH, 3, D_MODEL)
        assert out_1.ca_embeddings.shape == (BATCH, 1, D_MODEL)
        assert out_3.all_embeddings.shape[1] == 6  # 3+2+1
        assert out_1.all_embeddings.shape[1] == 4  # 1+2+1

    def test_large_ca_count(self):
        """Stress test: 21 CA tokens (i-charging scenario)."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(2, n_ca=21, n_sro=3, d_model=D_MODEL)
        out = backbone(tok)

        assert out.ca_embeddings.shape == (2, 21, D_MODEL)
        assert out.all_embeddings.shape[1] == 25  # 21+3+1


# ---------------------------------------------------------------------------
# CA output extraction test
# ---------------------------------------------------------------------------


class TestCAOutputExtraction:
    def test_ca_positions_are_first(self):
        """First N_ca positions in all_embeddings match ca_embeddings."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(BATCH, n_ca=3, n_sro=2, d_model=D_MODEL)
        out = backbone(tok)

        torch.testing.assert_close(
            out.ca_embeddings, out.all_embeddings[:, :3, :],
        )


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_to_all_token_types(self):
        """Ensure gradients propagate through attention to all token types.

        Loss is computed on CA embeddings only, but through self-attention
        gradients should flow back to SRO and RL tokens too.
        """
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)

        ca_tokens = torch.randn(BATCH, 2, D_MODEL, requires_grad=True)
        sro_tokens = torch.randn(BATCH, 2, D_MODEL, requires_grad=True)
        rl_token = torch.randn(BATCH, 1, D_MODEL, requires_grad=True)

        tok = TokenizedObservation(
            ca_tokens=ca_tokens,
            sro_tokens=sro_tokens,
            rl_token=rl_token,
            ca_types=["a", "b"],
            n_ca=2,
        )

        out = backbone(tok)
        loss = out.ca_embeddings.sum()
        loss.backward()

        # Gradients should reach all input token types through attention
        assert ca_tokens.grad is not None, "No gradient for CA tokens"
        assert sro_tokens.grad is not None, "No gradient for SRO tokens"
        assert rl_token.grad is not None, "No gradient for RL token"

    def test_gradients_from_pooled(self):
        """Gradients from pooled output reach type embeddings."""
        backbone = TransformerBackbone(d_model=D_MODEL, nhead=4, num_layers=1)
        tok = _make_tokenized(BATCH, n_ca=2, n_sro=1, d_model=D_MODEL)
        out = backbone(tok)
        loss = out.pooled.sum()
        loss.backward()

        assert backbone.type_embedding.weight.grad is not None


# ---------------------------------------------------------------------------
# Type embedding test
# ---------------------------------------------------------------------------


class TestTypeEmbedding:
    def test_type_embedding_has_3_types(self):
        backbone = TransformerBackbone(d_model=D_MODEL)
        assert backbone.type_embedding.num_embeddings == 3
        assert backbone.type_embedding.embedding_dim == D_MODEL
