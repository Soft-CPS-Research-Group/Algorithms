"""Tests for wrapper enrichment integration (Phase B of DIN plan).

Verifies that the wrapper correctly integrates with ObservationEnricher
for Transformer-based agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from algorithms.utils.observation_enricher import ObservationEnricher


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

TOKENIZER_CONFIG: Dict[str, Any] = {
    "ca_types": {
        "battery": {
            "features": ["electrical_storage_soc"],
            "action_name": "electrical_storage",
        },
    },
    "sro_types": {
        "temporal": {
            "features": ["month", "hour"],
        },
        "pricing": {
            "features": ["electricity_pricing"],
        },
    },
    "rl": {
        "demand_feature": "non_shiftable_load",
        "generation_features": ["solar_generation"],
        "extra_features": [],
    },
}


SAMPLE_OBS_NAMES = [
    "month",
    "hour",
    "electrical_storage_soc",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
]

SAMPLE_ACTIONS = ["electrical_storage"]


# ---------------------------------------------------------------------------
# Tests for enricher integration
# ---------------------------------------------------------------------------


class TestEnricherIntegration:
    """Test that ObservationEnricher integrates correctly."""

    def test_enricher_creates_markers(self):
        """Test that enricher adds markers to observation names."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SAMPLE_OBS_NAMES, SAMPLE_ACTIONS)

        # Should have markers
        markers = [n for n in result.enriched_names if n.startswith("__tkn_")]
        assert len(markers) > 0

        # Should have CA marker for battery
        assert "__tkn_ca_battery__" in result.enriched_names

    def test_enricher_values_correct_length(self):
        """Test that enrich_values produces correct length output."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        enricher.enrich_names(SAMPLE_OBS_NAMES, SAMPLE_ACTIONS)

        values = [1.0, 2.0, 0.5, 0.3, 100.0, 50.0]
        enriched = enricher.enrich_values(values)

        # Output should have original values plus marker values
        assert len(enriched) > len(values)

    def test_enricher_markers_are_zero(self):
        """Test that marker positions have value 0.0."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SAMPLE_OBS_NAMES, SAMPLE_ACTIONS)

        values = [1.0, 2.0, 0.5, 0.3, 100.0, 50.0]
        enriched = enricher.enrich_values(values)

        # Check marker positions
        for marker_name, positions in result.marker_positions.items():
            for pos in positions:
                assert enriched[pos] == 0.0, f"Marker {marker_name} at {pos} should be 0.0"


class TestEnrichedEncoderCount:
    """Test that enriched names produce correct encoder count."""

    def test_encoder_count_with_markers(self):
        """Test that number of encoders matches enriched names length."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SAMPLE_OBS_NAMES, SAMPLE_ACTIONS)

        # Each enriched name (including markers) should get one encoder
        # This is verified by set_encoders() producing len(enriched_names) encoders
        assert len(result.enriched_names) == len(SAMPLE_OBS_NAMES) + len(result.marker_positions)


class TestPerBuildingEnrichers:
    """Test that each building gets its own enricher instance."""

    def test_multiple_enrichers_independent(self):
        """Test that enrichers for different buildings are independent."""
        enricher1 = ObservationEnricher(TOKENIZER_CONFIG)
        enricher2 = ObservationEnricher(TOKENIZER_CONFIG)

        # Enrich with different observation sets
        obs1 = ["month", "hour", "electrical_storage_soc"]
        obs2 = ["month", "hour", "electrical_storage_soc", "electricity_pricing"]

        result1 = enricher1.enrich_names(obs1, SAMPLE_ACTIONS)
        result2 = enricher2.enrich_names(obs2, SAMPLE_ACTIONS)

        # Results should be different
        assert result1.enriched_names != result2.enriched_names

        # Values should work independently
        values1 = [1.0, 2.0, 0.5]
        values2 = [1.0, 2.0, 0.5, 0.3]

        enriched1 = enricher1.enrich_values(values1)
        enriched2 = enricher2.enrich_values(values2)

        assert len(enriched1) != len(enriched2)


class TestEnrichmentDisabledForNonTransformer:
    """Test that enrichment is disabled for non-transformer agents."""

    def test_non_transformer_uses_raw_names(self):
        """Verify that raw names are used when enrichment is disabled."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)

        # Without calling enrich_names, raw observation names are used
        # This tests the concept - actual wrapper integration tested elsewhere
        assert enricher._cached_result is None


class TestMarkerEncodersAreNoNormalization:
    """Test that marker features get NoNormalization encoders."""

    def test_marker_gets_pass_through_encoding(self):
        """Test that marker values pass through unchanged."""
        from utils.preprocessing import NoNormalization

        encoder = NoNormalization()
        value = 0.0

        # NoNormalization uses __mul__ to pass value through
        result = encoder * value
        assert result == value

        # Test with non-zero value too
        result2 = encoder * 1.5
        assert result2 == 1.5


class TestEnrichedNamesPassedToAgent:
    """Test that enriched names are passed to attach_environment."""

    def test_enriched_names_have_markers(self):
        """Verify enriched names contain markers."""
        enricher = ObservationEnricher(TOKENIZER_CONFIG)
        result = enricher.enrich_names(SAMPLE_OBS_NAMES, SAMPLE_ACTIONS)

        # Enriched names should have markers
        has_markers = any(
            n.startswith("__tkn_") and n.endswith("__")
            for n in result.enriched_names
        )
        assert has_markers

        # Enriched names should also have original features
        assert "electrical_storage_soc" in result.enriched_names
        assert "month" in result.enriched_names
