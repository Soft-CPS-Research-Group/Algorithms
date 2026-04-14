"""Tests for ObservationEnricher."""

import pytest

from algorithms.utils.observation_enricher import (
    EnrichmentResult,
    ObservationEnricher,
)


class TestEnrichmentResult:
    """Tests for EnrichmentResult dataclass."""

    def test_enrichment_result_creation(self) -> None:
        """EnrichmentResult should store enriched names and positions."""
        result = EnrichmentResult(
            enriched_names=["__marker__", "feature1", "feature2"],
            marker_positions={"__marker__": [0]},
            marker_to_type={"__marker__": ("ca", "battery", None)},
        )
        
        assert result.enriched_names == ["__marker__", "feature1", "feature2"]
        assert result.marker_positions == {"__marker__": [0]}
        assert result.marker_to_type["__marker__"] == ("ca", "battery", None)
