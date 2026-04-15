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


class TestMarkerScanning:
    """Tests for scanning marker values in encoded tensors."""

    def test_find_marker_positions_ca(self) -> None:
        """Should find CA marker positions (1001-1999 range)."""
        from algorithms.utils.observation_tokenizer import _find_marker_positions

        # Encoded tensor: [marker_1001, val, val, marker_1002, val]
        encoded = torch.tensor([[1001.0, 0.5, 0.3, 1002.0, 0.8]])

        ca_positions, sro_positions, nfc_position = _find_marker_positions(
            encoded, ca_base=1000, sro_base=2000, nfc_marker=3001
        )

        assert ca_positions == [[0, 3]]  # batch 0: positions 0 and 3
        assert sro_positions == [[]]
        assert nfc_position == [None]

    def test_find_marker_positions_all_types(self) -> None:
        """Should find markers for CA, SRO, and NFC."""
        from algorithms.utils.observation_tokenizer import _find_marker_positions

        # [CA_1001, val, SRO_2001, val, val, NFC_3001, val]
        encoded = torch.tensor([[1001.0, 0.5, 2001.0, 0.1, 0.2, 3001.0, 0.9]])

        ca_positions, sro_positions, nfc_position = _find_marker_positions(
            encoded, ca_base=1000, sro_base=2000, nfc_marker=3001
        )

        assert ca_positions == [[0]]
        assert sro_positions == [[2]]
        assert nfc_position == [5]

    def test_find_marker_positions_batch(self) -> None:
        """Should handle batched inputs."""
        from algorithms.utils.observation_tokenizer import _find_marker_positions

        encoded = torch.tensor([
            [1001.0, 0.5, 2001.0, 0.1],  # batch 0: 1 CA, 1 SRO
            [1001.0, 1002.0, 2001.0, 0.1],  # batch 1: 2 CAs, 1 SRO
        ])

        ca_positions, sro_positions, nfc_position = _find_marker_positions(
            encoded, ca_base=1000, sro_base=2000, nfc_marker=3001
        )

        assert ca_positions[0] == [0]
        assert ca_positions[1] == [0, 1]
        assert sro_positions[0] == [2]
        assert sro_positions[1] == [2]
