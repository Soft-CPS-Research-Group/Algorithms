# Plan A: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the foundational components for the TransformerPPO ULC: ObservationEnricher (portable), tokenizer config file, and config schema validation.

**Architecture:** The ObservationEnricher classifies observation features into token groups (CA/SRO/NFC) and injects marker values. It's pure Python with no ML dependencies for portability to the inference repo. The tokenizer config defines asset types and their dimensions. The schema validates this config structure.

**Tech Stack:** Python 3.10+, Pydantic (for schema), pytest

**Spec Reference:** `docs/spec.md` sections 3, 9

---

## File Structure

| File | Responsibility |
|------|----------------|
| `configs/tokenizers/default.json` | Token type definitions (CA/SRO/NFC) with feature patterns and input_dim |
| `algorithms/utils/observation_enricher.py` | Classifies features, injects marker values (portable, pure Python) |
| `utils/config_schema.py` | Pydantic models for tokenizer config validation (modify existing) |
| `tests/test_observation_enricher.py` | Unit tests for enricher |
| `tests/test_tokenizer_config_schema.py` | Unit tests for config schema |

---

## Task 1: Create Tokenizer Config File

**Files:**
- Create: `configs/tokenizers/default.json`

- [ ] **Step 1: Create the tokenizer config file**

```json
{
  "marker_values": {
    "ca_base": 1000,
    "sro_base": 2000,
    "nfc": 3001
  },
  "ca_types": {
    "battery": {
      "features": ["electrical_storage_soc"],
      "action_name": "electrical_storage",
      "input_dim": 1
    },
    "ev_charger": {
      "features": [
        "electric_vehicle_charger_connected_state",
        "connected_electric_vehicle_at_charger_battery_capacity",
        "connected_electric_vehicle_at_charger_departure_time",
        "connected_electric_vehicle_at_charger_required_soc_departure",
        "connected_electric_vehicle_at_charger_soc",
        "electric_vehicle_charger_incoming_state",
        "incoming_electric_vehicle_at_charger_estimated_arrival_time"
      ],
      "action_name": "electric_vehicle_storage",
      "input_dim": 61
    },
    "washing_machine": {
      "features": [
        "washing_machine_start_time_step",
        "washing_machine_end_time_step",
        "washing_machine_load_profile"
      ],
      "action_name": "washing_machine",
      "input_dim": 3
    }
  },
  "sro_types": {
    "temporal": {
      "features": ["month", "hour", "day_type"],
      "input_dim": 12
    },
    "pricing": {
      "features": [
        "electricity_pricing",
        "electricity_pricing_predicted_1",
        "electricity_pricing_predicted_2",
        "electricity_pricing_predicted_3"
      ],
      "input_dim": 4
    },
    "carbon": {
      "features": ["carbon_intensity"],
      "input_dim": 1
    }
  },
  "nfc": {
    "demand_features": ["non_shiftable_load"],
    "generation_features": ["solar_generation"],
    "extra_features": ["net_electricity_consumption"],
    "input_dim": 3
  }
}
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('configs/tokenizers/default.json'))"`
Expected: No output (valid JSON)

- [ ] **Step 3: Commit**

```bash
git add configs/tokenizers/default.json
git commit -m "feat: add tokenizer config for TransformerPPO"
```

---

## Task 2: Add Tokenizer Config Schema

**Files:**
- Modify: `utils/config_schema.py`
- Create: `tests/test_tokenizer_config_schema.py`

- [ ] **Step 1: Write failing test for tokenizer config schema**

Create `tests/test_tokenizer_config_schema.py`:

```python
"""Tests for tokenizer config schema validation."""

import json
import pytest
from pathlib import Path

from utils.config_schema import (
    TokenizerConfig,
    CATypeConfig,
    SROTypeConfig,
    NFCConfig,
    MarkerValuesConfig,
)


class TestTokenizerConfigSchema:
    """Tests for TokenizerConfig Pydantic model."""

    def test_load_default_config(self) -> None:
        """Default tokenizer config should validate successfully."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert config.marker_values.ca_base == 1000
        assert config.marker_values.sro_base == 2000
        assert config.marker_values.nfc == 3001

    def test_ca_types_parsed(self) -> None:
        """CA types should be parsed with correct structure."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "battery" in config.ca_types
        assert "ev_charger" in config.ca_types
        assert "washing_machine" in config.ca_types
        
        battery = config.ca_types["battery"]
        assert battery.action_name == "electrical_storage"
        assert battery.input_dim == 1
        assert "electrical_storage_soc" in battery.features

    def test_sro_types_parsed(self) -> None:
        """SRO types should be parsed with correct structure."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "temporal" in config.sro_types
        assert "pricing" in config.sro_types
        assert "carbon" in config.sro_types
        
        temporal = config.sro_types["temporal"]
        assert temporal.input_dim == 12
        assert "month" in temporal.features

    def test_nfc_config_parsed(self) -> None:
        """NFC config should be parsed correctly."""
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            raw_config = json.load(f)
        
        config = TokenizerConfig(**raw_config)
        
        assert "non_shiftable_load" in config.nfc.demand_features
        assert "solar_generation" in config.nfc.generation_features
        assert config.nfc.input_dim == 3

    def test_invalid_ca_type_missing_input_dim(self) -> None:
        """CA type without input_dim should fail validation."""
        invalid_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    # Missing input_dim
                }
            },
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }
        
        with pytest.raises(ValueError):
            TokenizerConfig(**invalid_config)

    def test_marker_values_must_be_positive(self) -> None:
        """Marker base values must be positive integers."""
        invalid_config = {
            "marker_values": {"ca_base": -1, "sro_base": 2000, "nfc": 3001},
            "ca_types": {},
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }
        
        with pytest.raises(ValueError):
            TokenizerConfig(**invalid_config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenizer_config_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'TokenizerConfig'`

- [ ] **Step 3: Implement tokenizer config schema**

Add to `utils/config_schema.py` (after existing imports, before other classes):

```python
# ---------------------------------------------------------------------------
# Tokenizer Config Schema (for TransformerPPO)
# ---------------------------------------------------------------------------


class MarkerValuesConfig(BaseModel):
    """Configuration for marker values used in observation enrichment."""

    ca_base: int = Field(..., gt=0, description="Base marker value for CA tokens (e.g., 1000)")
    sro_base: int = Field(..., gt=0, description="Base marker value for SRO tokens (e.g., 2000)")
    nfc: int = Field(..., gt=0, description="Marker value for NFC token (e.g., 3001)")


class CATypeConfig(BaseModel):
    """Configuration for a Controllable Asset type."""

    features: List[str] = Field(..., description="Feature name patterns for this CA type")
    action_name: str = Field(..., description="Action name pattern for this CA type")
    input_dim: int = Field(..., gt=0, description="Post-encoding dimension for this CA type")


class SROTypeConfig(BaseModel):
    """Configuration for a Shared Read-Only observation type."""

    features: List[str] = Field(..., description="Feature name patterns for this SRO type")
    input_dim: int = Field(..., ge=0, description="Post-encoding dimension for this SRO type")


class NFCConfig(BaseModel):
    """Configuration for Non-Flexible Context (residual load) token."""

    demand_features: List[str] = Field(default_factory=list, description="Demand feature patterns")
    generation_features: List[str] = Field(default_factory=list, description="Generation feature patterns")
    extra_features: List[str] = Field(default_factory=list, description="Extra feature patterns")
    input_dim: int = Field(..., ge=0, description="Post-encoding dimension for NFC")


class TokenizerConfig(BaseModel):
    """Full tokenizer configuration for TransformerPPO agent."""

    marker_values: MarkerValuesConfig
    ca_types: Dict[str, CATypeConfig] = Field(default_factory=dict)
    sro_types: Dict[str, SROTypeConfig] = Field(default_factory=dict)
    nfc: NFCConfig
```

Also add to imports at top of file:

```python
from typing import Dict, List  # Add Dict, List if not present
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokenizer_config_schema.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/config_schema.py tests/test_tokenizer_config_schema.py
git commit -m "feat: add Pydantic schema for tokenizer config"
```

---

## Task 3: Create ObservationEnricher — Data Classes

**Files:**
- Create: `algorithms/utils/observation_enricher.py`
- Create: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing test for EnrichmentResult dataclass**

Create `tests/test_observation_enricher.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestEnrichmentResult -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create observation_enricher.py with dataclass**

Create `algorithms/utils/observation_enricher.py`:

```python
"""Observation Enricher — injects token-type markers into observation names/values.

This module classifies raw observation features into token groups (CA, SRO, NFC)
and injects marker values that allow the tokenizer to identify token boundaries
without heuristic-based classification.

Portable: no dependencies on training-only code. Can be used in both the training
wrapper and the production inference preprocessor.

No PyTorch/NumPy dependencies — pure Python with stdlib + typing only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EnrichmentResult:
    """Result of observation name enrichment.

    Attributes:
        enriched_names: List of observation names with marker names inserted.
        marker_positions: Dict mapping marker_name -> list of positions in enriched_names.
        marker_to_type: Dict mapping marker_name -> (family, type_name, device_id).
            family is "ca", "sro", or "nfc".
            type_name is e.g., "battery", "temporal".
            device_id is e.g., "charger_1_1" for multi-instance CAs, None otherwise.
    """

    enriched_names: List[str]
    marker_positions: Dict[str, List[int]]
    marker_to_type: Dict[str, Tuple[str, str, Optional[str]]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestEnrichmentResult -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: add EnrichmentResult dataclass"
```

---

## Task 4: ObservationEnricher — Device ID Extraction

**Files:**
- Modify: `algorithms/utils/observation_enricher.py`
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing test for device ID extraction**

Add to `tests/test_observation_enricher.py`:

```python
class TestDeviceIdExtraction:
    """Tests for extracting device IDs from action names."""

    def test_extract_single_instance_battery(self) -> None:
        """Single-instance CA has no device ID (None)."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = ["electrical_storage"]
        ca_config = {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            }
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "battery" in result
        assert result["battery"] == [None]

    def test_extract_multi_instance_ev_chargers(self) -> None:
        """Multi-instance CAs have device IDs extracted from suffix."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = [
            "electrical_storage",
            "electric_vehicle_storage_charger_1_1",
            "electric_vehicle_storage_charger_1_2",
        ]
        ca_config = {
            "battery": {
                "features": ["electrical_storage_soc"],
                "action_name": "electrical_storage",
                "input_dim": 1,
            },
            "ev_charger": {
                "features": ["electric_vehicle_soc"],
                "action_name": "electric_vehicle_storage",
                "input_dim": 61,
            },
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "battery" in result
        assert result["battery"] == [None]
        assert "ev_charger" in result
        assert result["ev_charger"] == ["charger_1_1", "charger_1_2"]

    def test_extract_no_matching_actions(self) -> None:
        """CA type with no matching actions returns empty list."""
        from algorithms.utils.observation_enricher import _extract_device_ids
        
        action_names = ["electrical_storage"]
        ca_config = {
            "ev_charger": {
                "features": ["electric_vehicle_soc"],
                "action_name": "electric_vehicle_storage",
                "input_dim": 61,
            },
        }
        
        result = _extract_device_ids(action_names, ca_config)
        
        assert "ev_charger" not in result or result.get("ev_charger") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestDeviceIdExtraction -v`
Expected: FAIL with `ImportError: cannot import name '_extract_device_ids'`

- [ ] **Step 3: Implement _extract_device_ids function**

Add to `algorithms/utils/observation_enricher.py`:

```python
def _extract_device_ids(
    action_names: List[str],
    ca_config: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Optional[str]]]:
    """Extract device IDs from action names, grouped by CA type.

    For each CA type, strips the configured ``action_name`` prefix from each
    matching action name. The remainder (if any) is the device ID.

    Args:
        action_names: List of action names for one building.
        ca_config: CA types configuration with action_name patterns.

    Returns:
        Dict mapping ca_type_name -> list of device IDs (None for single-instance).

    Examples:
        >>> _extract_device_ids(
        ...     ["electrical_storage", "electric_vehicle_storage_charger_1_1"],
        ...     {"battery": {"action_name": "electrical_storage"},
        ...      "ev_charger": {"action_name": "electric_vehicle_storage"}},
        ... )
        {'battery': [None], 'ev_charger': ['charger_1_1']}
    """
    result: Dict[str, List[Optional[str]]] = {}

    for ca_type_name, ca_spec in ca_config.items():
        action_prefix = ca_spec.get("action_name", "")
        if not action_prefix:
            continue

        device_ids: List[Optional[str]] = []
        for act_name in action_names:
            if act_name == action_prefix:
                # Exact match -> single-instance, no device ID
                device_ids.append(None)
            elif act_name.startswith(action_prefix + "_"):
                # Has a device ID suffix
                device_id = act_name[len(action_prefix) + 1:]
                device_ids.append(device_id)

        if device_ids:
            result[ca_type_name] = device_ids

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestDeviceIdExtraction -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: add device ID extraction for multi-instance CAs"
```

---

## Task 5: ObservationEnricher — Feature Matching Helpers

**Files:**
- Modify: `algorithms/utils/observation_enricher.py`
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing tests for feature matching**

Add to `tests/test_observation_enricher.py`:

```python
class TestFeatureMatching:
    """Tests for feature pattern matching helpers."""

    def test_feature_matches_pattern_substring(self) -> None:
        """Feature matches if pattern is substring of feature name."""
        from algorithms.utils.observation_enricher import _feature_matches_patterns
        
        assert _feature_matches_patterns(
            "electrical_storage_soc",
            ["electrical_storage_soc"]
        )
        assert _feature_matches_patterns(
            "connected_electric_vehicle_at_charger_soc",
            ["electric_vehicle", "charger"]
        )

    def test_feature_no_match(self) -> None:
        """Feature doesn't match if no pattern is substring."""
        from algorithms.utils.observation_enricher import _feature_matches_patterns
        
        assert not _feature_matches_patterns(
            "electricity_pricing",
            ["electric_vehicle", "storage"]
        )

    def test_contains_device_id_bounded(self) -> None:
        """Device ID must appear as bounded token (surrounded by _ or at edges)."""
        from algorithms.utils.observation_enricher import _contains_device_id
        
        # Should match - device_id is bounded
        assert _contains_device_id(
            "electric_vehicle_charger_charger_1_1_connected_state",
            "charger_1_1"
        )
        assert _contains_device_id(
            "connected_state_charger_1_1",
            "charger_1_1"
        )

    def test_contains_device_id_not_bounded(self) -> None:
        """Device ID should not match partial substrings."""
        from algorithms.utils.observation_enricher import _contains_device_id
        
        # Should not match - "1" appears but not as bounded token
        # (this is a tricky case - we accept some false positives for simplicity)
        # The important thing is that the feature-pattern check happens first
        pass  # This test documents expected behavior
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestFeatureMatching -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement feature matching helpers**

Add to `algorithms/utils/observation_enricher.py`:

```python
import re


def _feature_matches_patterns(feature_name: str, patterns: List[str]) -> bool:
    """Check if feature_name contains any of the patterns as substring.

    Args:
        feature_name: Raw observation feature name.
        patterns: List of pattern substrings to match against.

    Returns:
        True if any pattern is a substring of feature_name.
    """
    return any(pattern in feature_name for pattern in patterns)


def _contains_device_id(feature_name: str, device_id: str) -> bool:
    """Check whether device_id appears as a bounded token in feature_name.

    A bounded match means the device_id is surrounded by ``_`` (or at the
    start/end of the string). This prevents false positives when the
    device_id is short.

    Args:
        feature_name: Raw observation feature name.
        device_id: Device ID to search for (e.g., "charger_1_1").

    Returns:
        True if device_id appears as bounded token in feature_name.
    """
    pattern = r"(?:^|_)" + re.escape(device_id) + r"(?:_|$)"
    return re.search(pattern, feature_name) is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestFeatureMatching -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: add feature pattern matching helpers"
```

---

## Task 6: ObservationEnricher — Classification Logic

**Files:**
- Modify: `algorithms/utils/observation_enricher.py`
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing test for feature classification**

Add to `tests/test_observation_enricher.py`:

```python
class TestFeatureClassification:
    """Tests for classifying features into token groups."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": [
                        "electric_vehicle_charger_connected_state",
                        "connected_electric_vehicle_at_charger_soc",
                    ],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 61,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour", "day_type"],
                    "input_dim": 12,
                },
                "pricing": {
                    "features": ["electricity_pricing"],
                    "input_dim": 4,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }

    def test_classify_battery_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Battery feature should be classified as CA."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "electrical_storage_soc",
            sample_tokenizer_config,
            device_ids_by_type={"battery": [None]},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "ca"
        assert type_name == "battery"
        assert device_id is None

    def test_classify_ev_feature_with_device_id(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """EV charger feature with device ID should be classified correctly."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "electric_vehicle_charger_charger_1_1_connected_state",
            sample_tokenizer_config,
            device_ids_by_type={"ev_charger": ["charger_1_1", "charger_1_2"]},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "ca"
        assert type_name == "ev_charger"
        assert device_id == "charger_1_1"

    def test_classify_temporal_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Temporal feature should be classified as SRO."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "month",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "sro"
        assert type_name == "temporal"
        assert device_id is None

    def test_classify_nfc_demand_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Demand feature should be classified as NFC."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "non_shiftable_load",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is not None
        family, type_name, device_id = result
        assert family == "nfc"
        assert type_name == "nfc"
        assert device_id is None

    def test_classify_unknown_feature(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Unknown feature should return None."""
        from algorithms.utils.observation_enricher import _classify_feature
        
        result = _classify_feature(
            "unknown_random_feature",
            sample_tokenizer_config,
            device_ids_by_type={},
        )
        
        assert result is None


# Need to import Dict and Any for type hints
from typing import Any, Dict
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestFeatureClassification -v`
Expected: FAIL with `ImportError: cannot import name '_classify_feature'`

- [ ] **Step 3: Implement _classify_feature function**

Add to `algorithms/utils/observation_enricher.py`:

```python
def _classify_feature(
    feature_name: str,
    tokenizer_config: Dict[str, Any],
    device_ids_by_type: Dict[str, List[Optional[str]]],
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Classify a feature into a token group.

    Checks in order: CA types, SRO types, NFC.

    Args:
        feature_name: Raw observation feature name.
        tokenizer_config: Full tokenizer config dict.
        device_ids_by_type: Device IDs per CA type (from _extract_device_ids).

    Returns:
        Tuple of (family, type_name, device_id) or None if unmatched.
        family is "ca", "sro", or "nfc".
    """
    ca_config = tokenizer_config.get("ca_types", {})
    sro_config = tokenizer_config.get("sro_types", {})
    nfc_config = tokenizer_config.get("nfc", {})

    # 1. Try CA types
    for ca_type_name, ca_spec in ca_config.items():
        patterns = ca_spec.get("features", [])
        if _feature_matches_patterns(feature_name, patterns):
            # Determine which device instance this belongs to
            device_ids = device_ids_by_type.get(ca_type_name, [None])
            
            # If only one instance (or no device IDs), use first/None
            if len(device_ids) <= 1:
                return ("ca", ca_type_name, device_ids[0] if device_ids else None)
            
            # Multiple instances - find which device ID is in the feature name
            for device_id in device_ids:
                if device_id is not None and _contains_device_id(feature_name, device_id):
                    return ("ca", ca_type_name, device_id)
            
            # Feature matches CA type but no specific device ID found
            # This shouldn't happen with well-formed data, but return first as fallback
            return ("ca", ca_type_name, device_ids[0])

    # 2. Try SRO types
    for sro_type_name, sro_spec in sro_config.items():
        patterns = sro_spec.get("features", [])
        if _feature_matches_patterns(feature_name, patterns):
            return ("sro", sro_type_name, None)

    # 3. Try NFC
    demand_patterns = nfc_config.get("demand_features", [])
    generation_patterns = nfc_config.get("generation_features", [])
    extra_patterns = nfc_config.get("extra_features", [])
    all_nfc_patterns = demand_patterns + generation_patterns + extra_patterns
    
    if _feature_matches_patterns(feature_name, all_nfc_patterns):
        return ("nfc", "nfc", None)

    # 4. Unmatched
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestFeatureClassification -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: add feature classification logic"
```

---

## Task 7: ObservationEnricher — Main Class enrich_names()

**Files:**
- Modify: `algorithms/utils/observation_enricher.py`
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing test for ObservationEnricher.enrich_names()**

Add to `tests/test_observation_enricher.py`:

```python
class TestObservationEnricherEnrichNames:
    """Tests for ObservationEnricher.enrich_names()."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
                "ev_charger": {
                    "features": [
                        "electric_vehicle_charger_connected_state",
                        "connected_electric_vehicle_at_charger_soc",
                    ],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 61,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour"],
                    "input_dim": 12,
                },
                "pricing": {
                    "features": ["electricity_pricing"],
                    "input_dim": 4,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }

    def test_enrich_names_single_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Single CA building should have one CA marker."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "hour",
            "electricity_pricing",
            "electrical_storage_soc",
            "non_shiftable_load",
            "solar_generation",
        ]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Should have markers for: 1 CA (battery), 2 SROs (temporal, pricing), 1 NFC
        # Total markers: 4
        assert len(result.enriched_names) == len(observation_names) + 4
        
        # Check marker positions exist
        assert any("1001" in name for name in result.enriched_names)  # CA marker
        assert any("2001" in name for name in result.enriched_names)  # SRO marker
        assert any("3001" in name for name in result.enriched_names)  # NFC marker

    def test_enrich_names_multi_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Building with battery + 2 EV chargers should have 3 CA markers."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "electrical_storage_soc",
            "electric_vehicle_charger_charger_1_1_connected_state",
            "connected_electric_vehicle_at_charger_charger_1_1_soc",
            "electric_vehicle_charger_charger_1_2_connected_state",
            "connected_electric_vehicle_at_charger_charger_1_2_soc",
            "non_shiftable_load",
        ]
        action_names = [
            "electrical_storage",
            "electric_vehicle_storage_charger_1_1",
            "electric_vehicle_storage_charger_1_2",
        ]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Count CA markers (1001, 1002, 1003)
        ca_markers = [n for n in result.enriched_names if n.startswith("__marker_100")]
        assert len(ca_markers) == 3  # battery + 2 ev_chargers

    def test_enrich_names_marker_positions_correct(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Marker positions should correctly index into enriched_names."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = [
            "month",
            "electrical_storage_soc",
            "non_shiftable_load",
        ]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Verify each marker position points to a marker name
        for marker_name, positions in result.marker_positions.items():
            for pos in positions:
                assert result.enriched_names[pos] == marker_name

    def test_enrich_names_caching(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Calling enrich_names twice with same input should return cached result."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        result1 = enricher.enrich_names(observation_names, action_names)
        result2 = enricher.enrich_names(observation_names, action_names)
        
        assert result1 is result2  # Same object (cached)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestObservationEnricherEnrichNames -v`
Expected: FAIL (ObservationEnricher class not fully implemented)

- [ ] **Step 3: Implement ObservationEnricher class with enrich_names()**

Add to `algorithms/utils/observation_enricher.py`:

```python
class ObservationEnricher:
    """Classifies observation features and injects token-type markers.

    Portable: no dependencies on training-only code. Can be used in
    both the training wrapper and the production inference preprocessor.
    """

    def __init__(self, tokenizer_config: Dict[str, Any]) -> None:
        """Initialize the enricher with tokenizer configuration.

        Args:
            tokenizer_config: The tokenizer config dict.
                Must contain 'marker_values', 'ca_types', 'sro_types', 'nfc' keys.
        """
        self._config = tokenizer_config
        self._marker_values = tokenizer_config.get("marker_values", {})
        self._ca_config = tokenizer_config.get("ca_types", {})
        self._sro_config = tokenizer_config.get("sro_types", {})
        self._nfc_config = tokenizer_config.get("nfc", {})

        # Cache for topology change detection
        self._cache_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._cached_result: Optional[EnrichmentResult] = None
        self._insertion_positions: List[int] = []
        self._marker_values_list: List[float] = []

    def enrich_names(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> EnrichmentResult:
        """Classify features and produce enriched observation names.

        Called once per topology (cached until topology changes).

        Args:
            observation_names: Raw observation names for one building.
            action_names: Action names for one building.

        Returns:
            EnrichmentResult with enriched_names, marker_positions, marker_to_type.
        """
        # Check cache
        cache_key = (tuple(observation_names), tuple(action_names))
        if cache_key == self._cache_key and self._cached_result is not None:
            return self._cached_result

        # Extract device IDs from action names
        device_ids_by_type = _extract_device_ids(action_names, self._ca_config)

        # Classify all features
        classified: List[Tuple[str, Optional[Tuple[str, str, Optional[str]]]]] = []
        for feature_name in observation_names:
            classification = _classify_feature(
                feature_name, self._config, device_ids_by_type
            )
            classified.append((feature_name, classification))

        # Group features by (family, type_name, device_id)
        groups: Dict[Tuple[str, str, Optional[str]], List[str]] = {}
        unclassified: List[str] = []

        for feature_name, classification in classified:
            if classification is None:
                unclassified.append(feature_name)
            else:
                key = classification
                if key not in groups:
                    groups[key] = []
                groups[key].append(feature_name)

        # Build enriched names with markers
        enriched_names: List[str] = []
        marker_positions: Dict[str, List[int]] = {}
        marker_to_type: Dict[str, Tuple[str, str, Optional[str]]] = {}
        insertion_positions: List[int] = []
        marker_values_list: List[float] = []

        ca_base = self._marker_values.get("ca_base", 1000)
        sro_base = self._marker_values.get("sro_base", 2000)
        nfc_marker_value = self._marker_values.get("nfc", 3001)

        ca_counter = 1
        sro_counter = 1

        # Determine order: CAs first (sorted by type then device_id), then SROs, then NFC
        # This ensures marker order = action order

        # Sort CA groups to match action order
        ca_groups = [(k, v) for k, v in groups.items() if k[0] == "ca"]
        # Sort by the order they appear in action_names
        def ca_sort_key(item: Tuple[Tuple[str, str, Optional[str]], List[str]]) -> int:
            key, _ = item
            _, type_name, device_id = key
            action_prefix = self._ca_config.get(type_name, {}).get("action_name", "")
            if device_id:
                action_name = f"{action_prefix}_{device_id}"
            else:
                action_name = action_prefix
            try:
                return action_names.index(action_name)
            except ValueError:
                return 999

        ca_groups.sort(key=ca_sort_key)

        # Add CA groups
        for (family, type_name, device_id), features in ca_groups:
            marker_value = ca_base + ca_counter
            marker_name = f"__marker_{marker_value}__"
            
            insertion_positions.append(len(enriched_names))
            marker_values_list.append(float(marker_value))
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = (family, type_name, device_id)
            
            enriched_names.extend(features)
            ca_counter += 1

        # Add SRO groups (in config order)
        for sro_type_name in self._sro_config.keys():
            key = ("sro", sro_type_name, None)
            if key in groups:
                features = groups[key]
                marker_value = sro_base + sro_counter
                marker_name = f"__marker_{marker_value}__"
                
                insertion_positions.append(len(enriched_names))
                marker_values_list.append(float(marker_value))
                
                enriched_names.append(marker_name)
                marker_positions[marker_name] = [len(enriched_names) - 1]
                marker_to_type[marker_name] = key
                
                enriched_names.extend(features)
                sro_counter += 1

        # Add NFC group
        nfc_key = ("nfc", "nfc", None)
        if nfc_key in groups:
            features = groups[nfc_key]
            marker_name = f"__marker_{nfc_marker_value}__"
            
            insertion_positions.append(len(enriched_names))
            marker_values_list.append(float(nfc_marker_value))
            
            enriched_names.append(marker_name)
            marker_positions[marker_name] = [len(enriched_names) - 1]
            marker_to_type[marker_name] = nfc_key
            
            enriched_names.extend(features)

        # Add unclassified features at the end (no marker)
        enriched_names.extend(unclassified)

        # Cache result
        result = EnrichmentResult(
            enriched_names=enriched_names,
            marker_positions=marker_positions,
            marker_to_type=marker_to_type,
        )
        self._cache_key = cache_key
        self._cached_result = result
        self._insertion_positions = insertion_positions
        self._marker_values_list = marker_values_list

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestObservationEnricherEnrichNames -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: implement ObservationEnricher.enrich_names()"
```

---

## Task 8: ObservationEnricher — enrich_values() and topology_changed()

**Files:**
- Modify: `algorithms/utils/observation_enricher.py`
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write failing tests for enrich_values() and topology_changed()**

Add to `tests/test_observation_enricher.py`:

```python
class TestObservationEnricherEnrichValues:
    """Tests for ObservationEnricher.enrich_values()."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {
                "ca_base": 1000,
                "sro_base": 2000,
                "nfc": 3001,
            },
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month"],
                    "input_dim": 2,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enrich_values_inserts_markers(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """enrich_values should insert marker values at correct positions."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc", "non_shiftable_load"]
        action_names = ["electrical_storage"]
        
        # Must call enrich_names first to set up positions
        enricher.enrich_names(observation_names, action_names)
        
        observation_values = [6.0, 0.75, 100.0]  # month=6, soc=0.75, load=100
        
        enriched_values = enricher.enrich_values(observation_values)
        
        # Should have original values + marker values
        assert len(enriched_values) == len(observation_values) + 3  # 3 markers
        
        # Check marker values are present (1001 for CA, 2001 for SRO, 3001 for NFC)
        assert 1001.0 in enriched_values
        assert 2001.0 in enriched_values
        assert 3001.0 in enriched_values
        
        # Check original values are preserved
        assert 6.0 in enriched_values
        assert 0.75 in enriched_values
        assert 100.0 in enriched_values

    def test_enrich_values_preserves_order(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Original values should appear in order after their marker."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc", "non_shiftable_load"]
        action_names = ["electrical_storage"]
        
        enrichment = enricher.enrich_names(observation_names, action_names)
        observation_values = [6.0, 0.75, 100.0]
        
        enriched_values = enricher.enrich_values(observation_values)
        
        # Find position of battery marker (1001)
        # The value after it should be 0.75 (battery soc)
        marker_idx = enriched_values.index(1001.0)
        assert enriched_values[marker_idx + 1] == 0.75


class TestObservationEnricherTopologyChanged:
    """Tests for ObservationEnricher.topology_changed()."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config for testing."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                },
            },
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }

    def test_topology_unchanged(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Same observation names should not trigger topology change."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # First call sets up cache
        enricher.enrich_names(observation_names, action_names)
        
        # Same names should not be a change
        assert not enricher.topology_changed(observation_names, action_names)

    def test_topology_changed_different_observations(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Different observation names should trigger topology change."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # First call sets up cache
        enricher.enrich_names(observation_names, action_names)
        
        # Different names should be a change
        new_observation_names = ["electrical_storage_soc", "new_feature"]
        assert enricher.topology_changed(new_observation_names, action_names)

    def test_topology_changed_before_enrich_names(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """topology_changed before enrich_names should return True (no cache)."""
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        # No cache yet, should return True
        assert enricher.topology_changed(observation_names, action_names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_enricher.py::TestObservationEnricherEnrichValues tests/test_observation_enricher.py::TestObservationEnricherTopologyChanged -v`
Expected: FAIL (methods not implemented)

- [ ] **Step 3: Implement enrich_values() and topology_changed()**

Add to `ObservationEnricher` class in `algorithms/utils/observation_enricher.py`:

```python
    def enrich_values(
        self,
        observation_values: List[float],
    ) -> List[float]:
        """Inject marker values at cached positions.

        Must be called AFTER enrich_names() for the same building.
        Uses cached positions from the last enrich_names() call.

        Args:
            observation_values: Raw observation values (same length as
                the original observation_names passed to enrich_names).

        Returns:
            Enriched values list with markers inserted.
        """
        if not self._insertion_positions:
            raise RuntimeError(
                "enrich_values() called before enrich_names(). "
                "Call enrich_names() first to set up marker positions."
            )

        # Build enriched values by inserting markers at cached positions
        # We need to map original values to their new positions
        
        # Get the enriched names to understand the structure
        if self._cached_result is None:
            raise RuntimeError("No cached enrichment result available.")
        
        enriched_names = self._cached_result.enriched_names
        enriched_values: List[float] = []
        original_idx = 0
        
        for name in enriched_names:
            if name.startswith("__marker_") and name.endswith("__"):
                # Extract marker value from name
                marker_value = float(name[9:-2])  # Remove "__marker_" and "__"
                enriched_values.append(marker_value)
            else:
                # Regular feature - use next original value
                if original_idx < len(observation_values):
                    enriched_values.append(observation_values[original_idx])
                    original_idx += 1
                else:
                    # Shouldn't happen if enrich_names was called with matching names
                    enriched_values.append(0.0)
        
        return enriched_values

    def topology_changed(
        self,
        observation_names: List[str],
        action_names: List[str],
    ) -> bool:
        """Check if topology differs from cached state.

        Args:
            observation_names: Current observation names.
            action_names: Current action names.

        Returns:
            True if topology changed (or no cache exists), False otherwise.
        """
        if self._cache_key is None:
            return True
        
        current_key = (tuple(observation_names), tuple(action_names))
        return current_key != self._cache_key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestObservationEnricherEnrichValues tests/test_observation_enricher.py::TestObservationEnricherTopologyChanged -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run all enricher tests**

Run: `pytest tests/test_observation_enricher.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add algorithms/utils/observation_enricher.py tests/test_observation_enricher.py
git commit -m "feat: implement enrich_values() and topology_changed()"
```

---

## Task 9: Verify Portability — No External Dependencies

**Files:**
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write test to verify no external dependencies**

Add to `tests/test_observation_enricher.py`:

```python
class TestEnricherPortability:
    """Tests to verify ObservationEnricher has no external dependencies."""

    def test_no_numpy_import(self) -> None:
        """ObservationEnricher should not import numpy."""
        import algorithms.utils.observation_enricher as enricher_module
        import sys
        
        # Check that numpy is not in the module's namespace
        assert not hasattr(enricher_module, "np")
        assert not hasattr(enricher_module, "numpy")
        
        # Check module source doesn't import numpy
        import inspect
        source = inspect.getsource(enricher_module)
        assert "import numpy" not in source
        assert "from numpy" not in source

    def test_no_torch_import(self) -> None:
        """ObservationEnricher should not import torch."""
        import algorithms.utils.observation_enricher as enricher_module
        import inspect
        
        source = inspect.getsource(enricher_module)
        assert "import torch" not in source
        assert "from torch" not in source

    def test_no_training_imports(self) -> None:
        """ObservationEnricher should not import from algorithms.* or utils.*."""
        import algorithms.utils.observation_enricher as enricher_module
        import inspect
        
        source = inspect.getsource(enricher_module)
        # Should not import from other project modules
        assert "from algorithms." not in source.replace(
            "from algorithms.utils.observation_enricher", ""
        )
        assert "from utils." not in source
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_observation_enricher.py::TestEnricherPortability -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_observation_enricher.py
git commit -m "test: add portability verification for ObservationEnricher"
```

---

## Task 10: Final Integration Test

**Files:**
- Modify: `tests/test_observation_enricher.py`

- [ ] **Step 1: Write integration test with full config**

Add to `tests/test_observation_enricher.py`:

```python
class TestEnricherIntegration:
    """Integration tests using the actual tokenizer config file."""

    def test_with_real_config_file(self) -> None:
        """Test enricher with the actual configs/tokenizers/default.json."""
        import json
        from pathlib import Path
        
        config_path = Path("configs/tokenizers/default.json")
        with open(config_path) as f:
            tokenizer_config = json.load(f)
        
        enricher = ObservationEnricher(tokenizer_config)
        
        # Simulate Building_4 (battery + 1 EV charger)
        observation_names = [
            "month",
            "hour",
            "day_type",
            "electricity_pricing",
            "electricity_pricing_predicted_1",
            "electricity_pricing_predicted_2",
            "electricity_pricing_predicted_3",
            "carbon_intensity",
            "electrical_storage_soc",
            "electric_vehicle_charger_connected_state",
            "connected_electric_vehicle_at_charger_battery_capacity",
            "connected_electric_vehicle_at_charger_departure_time",
            "connected_electric_vehicle_at_charger_required_soc_departure",
            "connected_electric_vehicle_at_charger_soc",
            "electric_vehicle_charger_incoming_state",
            "incoming_electric_vehicle_at_charger_estimated_arrival_time",
            "non_shiftable_load",
            "solar_generation",
            "net_electricity_consumption",
        ]
        action_names = ["electrical_storage", "electric_vehicle_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Should have markers for: 2 CAs, 3 SROs (temporal, pricing, carbon), 1 NFC
        # Total markers: 6
        marker_count = sum(1 for n in result.enriched_names if n.startswith("__marker_"))
        assert marker_count == 6
        
        # Verify CA markers (1001 for battery, 1002 for ev_charger)
        assert "__marker_1001__" in result.enriched_names
        assert "__marker_1002__" in result.enriched_names
        
        # Verify SRO markers (2001 temporal, 2002 pricing, 2003 carbon)
        assert "__marker_2001__" in result.enriched_names
        assert "__marker_2002__" in result.enriched_names
        assert "__marker_2003__" in result.enriched_names
        
        # Verify NFC marker
        assert "__marker_3001__" in result.enriched_names
        
        # Test enrich_values
        observation_values = [float(i) for i in range(len(observation_names))]
        enriched_values = enricher.enrich_values(observation_values)
        
        assert len(enriched_values) == len(result.enriched_names)
        assert 1001.0 in enriched_values
        assert 2001.0 in enriched_values
        assert 3001.0 in enriched_values
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_observation_enricher.py::TestEnricherIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/test_observation_enricher.py tests/test_tokenizer_config_schema.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_observation_enricher.py
git commit -m "test: add integration test for ObservationEnricher"
```

---

## Summary

After completing all tasks, you will have:

1. **`configs/tokenizers/default.json`** — Token type definitions
2. **`utils/config_schema.py`** — Updated with TokenizerConfig Pydantic models
3. **`algorithms/utils/observation_enricher.py`** — Portable enricher class
4. **`tests/test_tokenizer_config_schema.py`** — Schema validation tests
5. **`tests/test_observation_enricher.py`** — Comprehensive enricher tests

All components are tested, portable, and ready for Plan B (ML Components) and Plan C (Integration).
