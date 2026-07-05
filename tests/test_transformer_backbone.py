"""Tests for TransformerBackbone (entity-mode contract)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


D_MODEL = 16


@pytest.fixture
def backbone():
    from algorithms.utils.transformer_backbone import TransformerBackbone

    return TransformerBackbone(
        d_model=D_MODEL,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def test_type_embedding_table_size_3(backbone) -> None:
    """nn.Embedding(3, d_model) for {SRO=0, NFC=1, CA=2}."""
    embeddings = [
        m for m in backbone.modules() if isinstance(m, nn.Embedding)
    ]
    matching = [
        e for e in embeddings
        if e.num_embeddings == 3 and e.embedding_dim == D_MODEL
    ]
    assert matching, (
        f"Expected nn.Embedding(3, {D_MODEL}); got "
        f"{[(e.num_embeddings, e.embedding_dim) for e in embeddings]}"
    )


def test_forward_returns_ca_and_pooled_with_correct_shapes(backbone) -> None:
    """forward(sros, nfc, cas) → (ca_embeddings[B,N_ca,D], pooled[B,D])."""
    n_sro, n_ca = 3, 2
    sros = torch.randn(1, n_sro, D_MODEL)
    nfc = torch.randn(1, 1, D_MODEL)
    cas = torch.randn(1, n_ca, D_MODEL)

    ca_emb, pooled = backbone(sros, nfc, cas)

    assert ca_emb.shape == (1, n_ca, D_MODEL)
    assert pooled.shape == (1, D_MODEL)


def test_ca_embeddings_sliced_at_correct_offset(backbone) -> None:
    """CA slice positions are ``[N_sro+1 : N_sro+1+N_ca]`` in the concat
    sequence. Bypass attention mixing by replacing the encoder with identity
    and zeroing the type embedding so the output equals the input concat."""
    n_sro, n_ca = 4, 3
    sros = torch.zeros(1, n_sro, D_MODEL)
    nfc = torch.zeros(1, 1, D_MODEL)
    cas = (
        torch.arange(n_ca * D_MODEL, dtype=torch.float)
        .view(1, n_ca, D_MODEL)
    )

    backbone.encoder = nn.Identity()
    with torch.no_grad():
        backbone.type_embedding.weight.zero_()

    ca_emb, _ = backbone(sros, nfc, cas)

    assert torch.allclose(ca_emb, cas), (
        f"CA slice misaligned. Expected {cas}, got {ca_emb}."
    )


def test_pooled_is_mean_over_all_tokens(backbone) -> None:
    """Pooled = mean over every token (SRO + NFC + CA). Bypass attention
    mixing as in the slice test."""
    n_sro, n_ca = 2, 2
    sros = torch.ones(1, n_sro, D_MODEL) * 1.0
    nfc = torch.ones(1, 1, D_MODEL) * 5.0
    cas = torch.ones(1, n_ca, D_MODEL) * 2.0

    backbone.encoder = nn.Identity()
    with torch.no_grad():
        backbone.type_embedding.weight.zero_()

    _, pooled = backbone(sros, nfc, cas)
    expected = (n_sro * 1.0 + 1 * 5.0 + n_ca * 2.0) / (n_sro + 1 + n_ca)
    assert torch.allclose(pooled, torch.full_like(pooled, expected))


def test_gradient_flows_through_sro_and_nfc(backbone) -> None:
    """Critic-side gradient must reach SRO and NFC inputs (they only contribute
    via the pooled mean, so we backprop a pooled-only objective)."""
    n_sro, n_ca = 2, 1
    sros = torch.randn(1, n_sro, D_MODEL, requires_grad=True)
    nfc = torch.randn(1, 1, D_MODEL, requires_grad=True)
    cas = torch.randn(1, n_ca, D_MODEL, requires_grad=True)

    _, pooled = backbone(sros, nfc, cas)
    pooled.sum().backward()

    assert sros.grad is not None and sros.grad.abs().sum() > 0
    assert nfc.grad is not None and nfc.grad.abs().sum() > 0
    assert cas.grad is not None and cas.grad.abs().sum() > 0


def test_variable_token_count_supported(backbone) -> None:
    """Same backbone instance handles different (N_sro, N_ca) on successive
    calls — required for dynamic topology."""
    out_a = backbone(
        torch.randn(1, 2, D_MODEL),
        torch.randn(1, 1, D_MODEL),
        torch.randn(1, 2, D_MODEL),
    )
    assert out_a[0].shape == (1, 2, D_MODEL)

    out_b = backbone(
        torch.randn(1, 5, D_MODEL),
        torch.randn(1, 1, D_MODEL),
        torch.randn(1, 4, D_MODEL),
    )
    assert out_b[0].shape == (1, 4, D_MODEL)
