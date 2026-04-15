"""Tests for ObservationTokenizer."""

import pytest
import torch

from algorithms.utils.observation_tokenizer import TokenizedObservation


class TestTokenizedObservation:
    """Tests for TokenizedObservation dataclass."""

    def test_tokenized_observation_creation(self) -> None:
        """TokenizedObservation should store token tensors and metadata."""
        batch_size = 2
        n_ca = 3
        n_sro = 2
        d_model = 64

        result = TokenizedObservation(
            ca_tokens=torch.randn(batch_size, n_ca, d_model),
            sro_tokens=torch.randn(batch_size, n_sro, d_model),
            nfc_token=torch.randn(batch_size, 1, d_model),
            ca_types=["battery", "ev_charger", "ev_charger"],
            n_ca=n_ca,
        )

        assert result.ca_tokens.shape == (batch_size, n_ca, d_model)
        assert result.sro_tokens.shape == (batch_size, n_sro, d_model)
        assert result.nfc_token.shape == (batch_size, 1, d_model)
        assert result.n_ca == 3
        assert len(result.ca_types) == 3
