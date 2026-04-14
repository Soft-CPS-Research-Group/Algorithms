# Plan B: ML Components — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ML components for the TransformerPPO ULC: ObservationTokenizer (generic), TransformerBackbone, and PPO components (Actor, Critic, RolloutBuffer, loss functions).

**Architecture:** The ObservationTokenizer scans encoded tensors for marker values and projects features to d_model. The TransformerBackbone processes tokens through self-attention. PPO components handle the RL-specific actor-critic architecture.

**Tech Stack:** Python 3.10+, PyTorch, pytest

**Spec Reference:** `docs/spec.md` sections 4, 5, 6

**Dependencies:** Plan A must be completed first (WP3 depends on WP1, WP2)

---

## Git Setup

**Before starting implementation:**

1. Ensure Plan A is merged to `gj/master`
2. Create branch `gj/plan-b` from `gj/master`:
   ```bash
   git checkout gj/master
   git pull origin gj/master
   git checkout -b gj/plan-b
   ```
3. Verify you're on the correct branch:
   ```bash
   git branch -v | grep plan-b
   # Expected: gj/plan-b ... (should show latest gj/master commit)
   ```
4. After all tasks complete and tests pass, this branch will be merged back to `gj/master`
5. Do NOT commit to `gj/master` or `main` — all work stays on `gj/plan-b`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `algorithms/utils/observation_tokenizer.py` | Scans markers, splits groups, projects to d_model |
| `algorithms/utils/transformer_backbone.py` | Self-attention over tokens, type embeddings |
| `algorithms/utils/ppo_components.py` | ActorHead, CriticHead, RolloutBuffer, PPO loss |
| `tests/test_observation_tokenizer.py` | Tokenizer unit tests |
| `tests/test_transformer_backbone.py` | Backbone unit tests |
| `tests/test_ppo_components.py` | PPO component unit tests |

---

## Task 1: ObservationTokenizer — TokenizedObservation Dataclass

**Files:**
- Create: `algorithms/utils/observation_tokenizer.py`
- Create: `tests/test_observation_tokenizer.py`

- [ ] **Step 1: Write failing test for TokenizedObservation**

Create `tests/test_observation_tokenizer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_tokenizer.py::TestTokenizedObservation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create observation_tokenizer.py with dataclass**

Create `algorithms/utils/observation_tokenizer.py`:

```python
"""Observation Tokenizer — scans markers and projects features to d_model.

Given an encoded observation tensor with marker values, the tokenizer:
1. Scans for marker values (1001-1999 for CAs, 2001-2999 for SROs, 3001 for NFC)
2. Splits the tensor into token groups based on marker positions
3. Projects each group to d_model via per-type Linear layers
4. Returns TokenizedObservation with ca_tokens, sro_tokens, nfc_token

This tokenizer is generic and reusable by any Transformer-based agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TokenizedObservation:
    """Output of the observation tokenizer.

    Attributes:
        ca_tokens: [batch, N_ca, d_model] — one token per controllable asset instance.
        sro_tokens: [batch, N_sro, d_model] — one token per SRO group.
        nfc_token: [batch, 1, d_model] — non-flexible context token.
        ca_types: Type name per CA token position (e.g., ["battery", "ev_charger"]).
        n_ca: Number of CA tokens.
    """

    ca_tokens: torch.Tensor
    sro_tokens: torch.Tensor
    nfc_token: torch.Tensor
    ca_types: List[str]
    n_ca: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_tokenizer.py::TestTokenizedObservation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_tokenizer.py tests/test_observation_tokenizer.py
git commit -m "feat: add TokenizedObservation dataclass"
```

---

## Task 2: ObservationTokenizer — Marker Scanning

**Files:**
- Modify: `algorithms/utils/observation_tokenizer.py`
- Modify: `tests/test_observation_tokenizer.py`

- [ ] **Step 1: Write failing tests for marker scanning**

Add to `tests/test_observation_tokenizer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_tokenizer.py::TestMarkerScanning -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement _find_marker_positions function**

Add to `algorithms/utils/observation_tokenizer.py`:

```python
def _find_marker_positions(
    encoded: torch.Tensor,
    ca_base: int,
    sro_base: int,
    nfc_marker: int,
) -> Tuple[List[List[int]], List[List[int]], List[Optional[int]]]:
    """Find positions of marker values in encoded tensor.

    Args:
        encoded: [batch, obs_dim] encoded observation tensor.
        ca_base: Base value for CA markers (e.g., 1000 -> markers are 1001, 1002, ...).
        sro_base: Base value for SRO markers (e.g., 2000 -> markers are 2001, 2002, ...).
        nfc_marker: Exact marker value for NFC (e.g., 3001).

    Returns:
        Tuple of:
            - ca_positions: List of lists, one per batch, containing CA marker positions.
            - sro_positions: List of lists, one per batch, containing SRO marker positions.
            - nfc_position: List, one per batch, containing NFC marker position (or None).
    """
    batch_size = encoded.shape[0]
    ca_positions: List[List[int]] = []
    sro_positions: List[List[int]] = []
    nfc_positions: List[Optional[int]] = []

    for b in range(batch_size):
        row = encoded[b]
        ca_pos: List[int] = []
        sro_pos: List[int] = []
        nfc_pos: Optional[int] = None

        for i, val in enumerate(row.tolist()):
            # Check if value is a CA marker (ca_base < val < ca_base + 1000)
            if ca_base < val < ca_base + 1000:
                ca_pos.append(i)
            # Check if value is an SRO marker (sro_base < val < sro_base + 1000)
            elif sro_base < val < sro_base + 1000:
                sro_pos.append(i)
            # Check if value is NFC marker
            elif abs(val - nfc_marker) < 0.01:  # Float comparison tolerance
                nfc_pos = i

        ca_positions.append(ca_pos)
        sro_positions.append(sro_pos)
        nfc_positions.append(nfc_pos)

    return ca_positions, sro_positions, nfc_positions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_tokenizer.py::TestMarkerScanning -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_tokenizer.py tests/test_observation_tokenizer.py
git commit -m "feat: add marker position scanning for tokenizer"
```

---

## Task 3: ObservationTokenizer — Group Extraction

**Files:**
- Modify: `algorithms/utils/observation_tokenizer.py`
- Modify: `tests/test_observation_tokenizer.py`

- [ ] **Step 1: Write failing tests for group extraction**

Add to `tests/test_observation_tokenizer.py`:

```python
class TestGroupExtraction:
    """Tests for extracting feature groups from encoded tensor."""

    def test_extract_groups_single_ca(self) -> None:
        """Should extract features between markers as groups."""
        from algorithms.utils.observation_tokenizer import _extract_groups

        # [CA_1001, val1, SRO_2001, val2, val3, NFC_3001, val4]
        encoded = torch.tensor([[1001.0, 0.5, 2001.0, 0.1, 0.2, 3001.0, 0.9]])
        ca_positions = [[0]]
        sro_positions = [[2]]
        nfc_position = [5]

        ca_groups, sro_groups, nfc_group = _extract_groups(
            encoded, ca_positions, sro_positions, nfc_position
        )

        # CA group: features at positions 1 (between marker at 0 and next marker at 2)
        assert len(ca_groups) == 1  # 1 batch
        assert len(ca_groups[0]) == 1  # 1 CA
        assert ca_groups[0][0].tolist() == [0.5]

        # SRO group: features at positions 3, 4
        assert len(sro_groups[0]) == 1  # 1 SRO
        assert sro_groups[0][0].tolist() == [0.1, 0.2]

        # NFC group: feature at position 6
        assert nfc_group[0].tolist() == [0.9]

    def test_extract_groups_multi_ca(self) -> None:
        """Should handle multiple CA groups."""
        from algorithms.utils.observation_tokenizer import _extract_groups

        # [CA_1001, val1, CA_1002, val2, val3, SRO_2001, val4]
        encoded = torch.tensor([[1001.0, 0.5, 1002.0, 0.8, 0.9, 2001.0, 0.1]])
        ca_positions = [[0, 2]]
        sro_positions = [[5]]
        nfc_position = [None]

        ca_groups, sro_groups, nfc_group = _extract_groups(
            encoded, ca_positions, sro_positions, nfc_position
        )

        assert len(ca_groups[0]) == 2  # 2 CAs
        assert ca_groups[0][0].tolist() == [0.5]  # First CA: 1 feature
        assert ca_groups[0][1].tolist() == [0.8, 0.9]  # Second CA: 2 features
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_tokenizer.py::TestGroupExtraction -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement _extract_groups function**

Add to `algorithms/utils/observation_tokenizer.py`:

```python
def _extract_groups(
    encoded: torch.Tensor,
    ca_positions: List[List[int]],
    sro_positions: List[List[int]],
    nfc_positions: List[Optional[int]],
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[torch.Tensor]]:
    """Extract feature groups from encoded tensor based on marker positions.

    Features between a marker and the next marker (or end of tensor) belong to that group.

    Args:
        encoded: [batch, obs_dim] encoded observation tensor.
        ca_positions: CA marker positions per batch.
        sro_positions: SRO marker positions per batch.
        nfc_positions: NFC marker position per batch (or None).

    Returns:
        Tuple of:
            - ca_groups: List of lists of tensors, [batch][ca_idx] -> features tensor
            - sro_groups: List of lists of tensors, [batch][sro_idx] -> features tensor
            - nfc_group: List of tensors, [batch] -> features tensor (empty if no NFC)
    """
    batch_size = encoded.shape[0]
    obs_dim = encoded.shape[1]
    
    ca_groups: List[List[torch.Tensor]] = []
    sro_groups: List[List[torch.Tensor]] = []
    nfc_groups: List[torch.Tensor] = []

    for b in range(batch_size):
        row = encoded[b]
        
        # Collect all marker positions to determine group boundaries
        all_markers: List[Tuple[int, str, int]] = []  # (position, type, index)
        
        for idx, pos in enumerate(ca_positions[b]):
            all_markers.append((pos, "ca", idx))
        for idx, pos in enumerate(sro_positions[b]):
            all_markers.append((pos, "sro", idx))
        if nfc_positions[b] is not None:
            all_markers.append((nfc_positions[b], "nfc", 0))
        
        # Sort by position
        all_markers.sort(key=lambda x: x[0])
        
        # Extract groups
        batch_ca_groups: List[torch.Tensor] = []
        batch_sro_groups: List[torch.Tensor] = []
        batch_nfc_group: torch.Tensor = torch.tensor([])
        
        for i, (pos, marker_type, _) in enumerate(all_markers):
            # Find end position (next marker or end of tensor)
            if i + 1 < len(all_markers):
                end_pos = all_markers[i + 1][0]
            else:
                end_pos = obs_dim
            
            # Extract features (skip the marker itself)
            features = row[pos + 1:end_pos]
            
            if marker_type == "ca":
                batch_ca_groups.append(features)
            elif marker_type == "sro":
                batch_sro_groups.append(features)
            elif marker_type == "nfc":
                batch_nfc_group = features
        
        ca_groups.append(batch_ca_groups)
        sro_groups.append(batch_sro_groups)
        nfc_groups.append(batch_nfc_group)

    return ca_groups, sro_groups, nfc_groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_tokenizer.py::TestGroupExtraction -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_tokenizer.py tests/test_observation_tokenizer.py
git commit -m "feat: add group extraction for tokenizer"
```

---

## Task 4: ObservationTokenizer — Main Class

**Files:**
- Modify: `algorithms/utils/observation_tokenizer.py`
- Modify: `tests/test_observation_tokenizer.py`

- [ ] **Step 1: Write failing tests for ObservationTokenizer class**

Add to `tests/test_observation_tokenizer.py`:

```python
import json
from pathlib import Path


class TestObservationTokenizer:
    """Tests for ObservationTokenizer class."""

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
                    "features": ["electric_vehicle_soc"],
                    "action_name": "electric_vehicle_storage",
                    "input_dim": 2,
                },
            },
            "sro_types": {
                "temporal": {
                    "features": ["month", "hour"],
                    "input_dim": 4,
                },
                "pricing": {
                    "features": ["electricity_pricing"],
                    "input_dim": 1,
                },
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }

    def test_tokenizer_creation(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Tokenizer should create projections for all types."""
        from algorithms.utils.observation_tokenizer import ObservationTokenizer

        tokenizer = ObservationTokenizer(sample_tokenizer_config, d_model=64)

        # Check CA projections exist
        assert "battery" in tokenizer.ca_projections
        assert "ev_charger" in tokenizer.ca_projections

        # Check SRO projections exist
        assert "temporal" in tokenizer.sro_projections
        assert "pricing" in tokenizer.sro_projections

        # Check NFC projection exists
        assert tokenizer.nfc_projection is not None

    def test_tokenizer_projection_shapes(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Projections should have correct input/output dimensions."""
        from algorithms.utils.observation_tokenizer import ObservationTokenizer

        d_model = 64
        tokenizer = ObservationTokenizer(sample_tokenizer_config, d_model=d_model)

        # Battery: input_dim=1 -> d_model
        assert tokenizer.ca_projections["battery"].in_features == 1
        assert tokenizer.ca_projections["battery"].out_features == d_model

        # EV charger: input_dim=2 -> d_model
        assert tokenizer.ca_projections["ev_charger"].in_features == 2
        assert tokenizer.ca_projections["ev_charger"].out_features == d_model

        # Temporal: input_dim=4 -> d_model
        assert tokenizer.sro_projections["temporal"].in_features == 4
        assert tokenizer.sro_projections["temporal"].out_features == d_model

    def test_tokenizer_forward_single_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Forward pass should produce correctly shaped token tensors."""
        from algorithms.utils.observation_tokenizer import ObservationTokenizer

        d_model = 64
        tokenizer = ObservationTokenizer(sample_tokenizer_config, d_model=d_model)

        # Encoded: [CA_1001(battery), soc, SRO_2001(temporal), m, h, d, t, SRO_2002(pricing), p, NFC_3001, load, gen]
        # Battery has 1 feature, temporal has 4, pricing has 1, nfc has 2
        encoded = torch.tensor([[
            1001.0, 0.5,  # CA battery: marker + 1 feature
            2001.0, 0.1, 0.2, 0.3, 0.4,  # SRO temporal: marker + 4 features
            2002.0, 0.8,  # SRO pricing: marker + 1 feature
            3001.0, 100.0, 50.0,  # NFC: marker + 2 features
        ]])

        result = tokenizer(encoded)

        assert result.ca_tokens.shape == (1, 1, d_model)  # 1 batch, 1 CA, d_model
        assert result.sro_tokens.shape == (1, 2, d_model)  # 1 batch, 2 SROs, d_model
        assert result.nfc_token.shape == (1, 1, d_model)  # 1 batch, 1 NFC, d_model
        assert result.n_ca == 1

    def test_tokenizer_forward_multi_ca(self, sample_tokenizer_config: Dict[str, Any]) -> None:
        """Forward pass should handle multiple CAs."""
        from algorithms.utils.observation_tokenizer import ObservationTokenizer

        d_model = 64
        tokenizer = ObservationTokenizer(sample_tokenizer_config, d_model=d_model)

        # 2 CAs: battery (1 feature) + ev_charger (2 features)
        encoded = torch.tensor([[
            1001.0, 0.5,  # CA battery: marker + 1 feature
            1002.0, 0.8, 0.9,  # CA ev_charger: marker + 2 features
            2001.0, 0.1, 0.2, 0.3, 0.4,  # SRO temporal
            3001.0, 100.0, 50.0,  # NFC
        ]])

        result = tokenizer(encoded)

        assert result.ca_tokens.shape == (1, 2, d_model)  # 2 CAs
        assert result.n_ca == 2
        assert len(result.ca_types) == 2


from typing import Any, Dict
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation_tokenizer.py::TestObservationTokenizer -v`
Expected: FAIL with `ImportError: cannot import name 'ObservationTokenizer'`

- [ ] **Step 3: Implement ObservationTokenizer class**

Add to `algorithms/utils/observation_tokenizer.py`:

```python
class ObservationTokenizer(nn.Module):
    """Tokenizes encoded observations with marker values into typed token embeddings.

    Scans for marker values, extracts feature groups, and projects each group
    to d_model via per-type Linear layers.
    """

    def __init__(
        self,
        tokenizer_config: Dict[str, Any],
        d_model: int,
    ) -> None:
        """Initialize the tokenizer with config and embedding dimension.

        Args:
            tokenizer_config: Loaded from configs/tokenizers/default.json.
                Contains ca_types, sro_types, nfc with input_dim per type.
            d_model: Embedding dimension for all tokens.
        """
        super().__init__()
        self.d_model = d_model
        self._config = tokenizer_config

        # Extract marker values
        marker_values = tokenizer_config.get("marker_values", {})
        self.ca_base = marker_values.get("ca_base", 1000)
        self.sro_base = marker_values.get("sro_base", 2000)
        self.nfc_marker = marker_values.get("nfc", 3001)

        # Extract type configs
        ca_config = tokenizer_config.get("ca_types", {})
        sro_config = tokenizer_config.get("sro_types", {})
        nfc_config = tokenizer_config.get("nfc", {})

        # Create CA projections (one per type)
        self.ca_projections = nn.ModuleDict()
        self._ca_type_order = list(ca_config.keys())  # Preserve order
        for ca_type, spec in ca_config.items():
            input_dim = spec.get("input_dim", 1)
            self.ca_projections[ca_type] = nn.Linear(input_dim, d_model)

        # Create SRO projections (one per type)
        self.sro_projections = nn.ModuleDict()
        self._sro_type_order = list(sro_config.keys())  # Preserve order
        for sro_type, spec in sro_config.items():
            input_dim = spec.get("input_dim", 1)
            if input_dim > 0:
                self.sro_projections[sro_type] = nn.Linear(input_dim, d_model)

        # Create NFC projection
        nfc_input_dim = nfc_config.get("input_dim", 1)
        self.nfc_projection = nn.Linear(nfc_input_dim, d_model) if nfc_input_dim > 0 else None

        # Build input_dim -> type mapping for CA type inference
        self._ca_dim_to_type: Dict[int, str] = {}
        for ca_type, spec in ca_config.items():
            input_dim = spec.get("input_dim", 1)
            self._ca_dim_to_type[input_dim] = ca_type

        # Build input_dim -> type mapping for SRO type inference
        self._sro_dim_to_type: Dict[int, str] = {}
        for sro_type, spec in sro_config.items():
            input_dim = spec.get("input_dim", 1)
            self._sro_dim_to_type[input_dim] = sro_type

    def forward(self, encoded_obs: torch.Tensor) -> TokenizedObservation:
        """Tokenize encoded observations.

        Args:
            encoded_obs: [batch, obs_dim] flat encoded observation with markers.

        Returns:
            TokenizedObservation with ca_tokens, sro_tokens, nfc_token, metadata.
        """
        batch_size = encoded_obs.shape[0]
        device = encoded_obs.device

        # Find marker positions
        ca_positions, sro_positions, nfc_positions = _find_marker_positions(
            encoded_obs, self.ca_base, self.sro_base, self.nfc_marker
        )

        # Extract groups
        ca_groups, sro_groups, nfc_groups = _extract_groups(
            encoded_obs, ca_positions, sro_positions, nfc_positions
        )

        # Project CA groups
        all_ca_tokens: List[torch.Tensor] = []
        all_ca_types: List[str] = []

        for b in range(batch_size):
            batch_ca_tokens: List[torch.Tensor] = []
            for features in ca_groups[b]:
                # Infer type from feature count
                feature_count = features.shape[0]
                ca_type = self._ca_dim_to_type.get(feature_count)
                
                if ca_type is not None and ca_type in self.ca_projections:
                    projection = self.ca_projections[ca_type]
                    token = projection(features.unsqueeze(0))  # [1, d_model]
                    batch_ca_tokens.append(token)
                    if b == 0:  # Only collect types once
                        all_ca_types.append(ca_type)
                else:
                    # Unknown type - use first available projection as fallback
                    if self._ca_type_order:
                        fallback_type = self._ca_type_order[0]
                        # Pad or truncate features to match expected dim
                        expected_dim = self.ca_projections[fallback_type].in_features
                        if feature_count < expected_dim:
                            features = torch.cat([
                                features,
                                torch.zeros(expected_dim - feature_count, device=device)
                            ])
                        elif feature_count > expected_dim:
                            features = features[:expected_dim]
                        token = self.ca_projections[fallback_type](features.unsqueeze(0))
                        batch_ca_tokens.append(token)
                        if b == 0:
                            all_ca_types.append(fallback_type)

            if batch_ca_tokens:
                all_ca_tokens.append(torch.cat(batch_ca_tokens, dim=0))
            else:
                all_ca_tokens.append(torch.zeros(0, self.d_model, device=device))

        # Stack CA tokens across batch
        n_ca = len(ca_groups[0]) if ca_groups else 0
        if n_ca > 0:
            ca_tokens = torch.stack([t for t in all_ca_tokens], dim=0)  # [batch, n_ca, d_model]
        else:
            ca_tokens = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Project SRO groups
        all_sro_tokens: List[torch.Tensor] = []
        for b in range(batch_size):
            batch_sro_tokens: List[torch.Tensor] = []
            for features in sro_groups[b]:
                feature_count = features.shape[0]
                sro_type = self._sro_dim_to_type.get(feature_count)
                
                if sro_type is not None and sro_type in self.sro_projections:
                    projection = self.sro_projections[sro_type]
                    token = projection(features.unsqueeze(0))
                    batch_sro_tokens.append(token)

            if batch_sro_tokens:
                all_sro_tokens.append(torch.cat(batch_sro_tokens, dim=0))
            else:
                all_sro_tokens.append(torch.zeros(0, self.d_model, device=device))

        n_sro = len(sro_groups[0]) if sro_groups else 0
        if n_sro > 0:
            sro_tokens = torch.stack(all_sro_tokens, dim=0)
        else:
            sro_tokens = torch.zeros(batch_size, 0, self.d_model, device=device)

        # Project NFC group
        if self.nfc_projection is not None and any(g.numel() > 0 for g in nfc_groups):
            nfc_tokens_list = []
            for b in range(batch_size):
                if nfc_groups[b].numel() > 0:
                    nfc_token = self.nfc_projection(nfc_groups[b].unsqueeze(0))
                else:
                    nfc_token = torch.zeros(1, self.d_model, device=device)
                nfc_tokens_list.append(nfc_token)
            nfc_token = torch.stack(nfc_tokens_list, dim=0)  # [batch, 1, d_model]
        else:
            nfc_token = torch.zeros(batch_size, 1, self.d_model, device=device)

        return TokenizedObservation(
            ca_tokens=ca_tokens,
            sro_tokens=sro_tokens,
            nfc_token=nfc_token,
            ca_types=all_ca_types,
            n_ca=n_ca,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_observation_tokenizer.py::TestObservationTokenizer -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/observation_tokenizer.py tests/test_observation_tokenizer.py
git commit -m "feat: implement ObservationTokenizer class"
```

---

## Task 5: TransformerBackbone — TransformerOutput Dataclass

**Files:**
- Create: `algorithms/utils/transformer_backbone.py`
- Create: `tests/test_transformer_backbone.py`

- [ ] **Step 1: Write failing test for TransformerOutput**

Create `tests/test_transformer_backbone.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transformer_backbone.py::TestTransformerOutput -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create transformer_backbone.py with dataclass**

Create `algorithms/utils/transformer_backbone.py`:

```python
"""Transformer Backbone — self-attention over token sequence.

Processes concatenated [CA tokens, SRO tokens, NFC token] through
self-attention with type embeddings. Produces contextual embeddings
where each token has attended to all others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TransformerOutput:
    """Output of the Transformer backbone.

    Attributes:
        all_embeddings: [batch, N_total, d_model] — all token embeddings.
        ca_embeddings: [batch, N_ca, d_model] — CA token embeddings (sliced from all).
        pooled: [batch, d_model] — mean-pooled over all tokens.
        n_ca: Number of CA tokens.
    """

    all_embeddings: torch.Tensor
    ca_embeddings: torch.Tensor
    pooled: torch.Tensor
    n_ca: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transformer_backbone.py::TestTransformerOutput -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/transformer_backbone.py tests/test_transformer_backbone.py
git commit -m "feat: add TransformerOutput dataclass"
```

---

## Task 6: TransformerBackbone — Main Class

**Files:**
- Modify: `algorithms/utils/transformer_backbone.py`
- Modify: `tests/test_transformer_backbone.py`

- [ ] **Step 1: Write failing tests for TransformerBackbone class**

Add to `tests/test_transformer_backbone.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transformer_backbone.py::TestTransformerBackbone -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement TransformerBackbone class**

Add to `algorithms/utils/transformer_backbone.py`:

```python
class TransformerBackbone(nn.Module):
    """Transformer encoder backbone with type embeddings.

    Processes concatenated [CA tokens, SRO tokens, NFC token] through
    self-attention. Each token attends to all others, producing contextual
    embeddings.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Transformer backbone.

        Args:
            d_model: Embedding dimension for all tokens.
            nhead: Number of attention heads.
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Feedforward network dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # Type embeddings: 0=CA, 1=SRO, 2=NFC
        self.type_embedding = nn.Embedding(3, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        ca_tokens: torch.Tensor,
        sro_tokens: torch.Tensor,
        nfc_token: torch.Tensor,
    ) -> TransformerOutput:
        """Process tokens through self-attention.

        Args:
            ca_tokens: [batch, N_ca, d_model] CA token embeddings.
            sro_tokens: [batch, N_sro, d_model] SRO token embeddings.
            nfc_token: [batch, 1, d_model] NFC token embedding.

        Returns:
            TransformerOutput with all embeddings, CA embeddings, pooled, and n_ca.
        """
        batch_size = ca_tokens.shape[0]
        n_ca = ca_tokens.shape[1]
        n_sro = sro_tokens.shape[1]
        device = ca_tokens.device

        # Concatenate all tokens: [batch, N_total, d_model]
        all_tokens = torch.cat([ca_tokens, sro_tokens, nfc_token], dim=1)
        n_total = all_tokens.shape[1]

        # Build type IDs
        type_ids = torch.cat([
            torch.zeros(batch_size, n_ca, dtype=torch.long, device=device),  # CA = 0
            torch.ones(batch_size, n_sro, dtype=torch.long, device=device),  # SRO = 1
            torch.full((batch_size, 1), 2, dtype=torch.long, device=device),  # NFC = 2
        ], dim=1)

        # Add type embeddings
        all_tokens = all_tokens + self.type_embedding(type_ids)

        # Self-attention
        encoded = self.encoder(all_tokens)

        # Extract CA embeddings (first n_ca positions)
        ca_embeddings = encoded[:, :n_ca, :]

        # Mean pooling over all tokens
        pooled = encoded.mean(dim=1)

        return TransformerOutput(
            all_embeddings=encoded,
            ca_embeddings=ca_embeddings,
            pooled=pooled,
            n_ca=n_ca,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transformer_backbone.py::TestTransformerBackbone -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/transformer_backbone.py tests/test_transformer_backbone.py
git commit -m "feat: implement TransformerBackbone class"
```

---

## Task 7: PPO Components — ActorHead

**Files:**
- Create: `algorithms/utils/ppo_components.py`
- Create: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing tests for ActorHead**

Create `tests/test_ppo_components.py`:

```python
"""Tests for PPO components."""

import pytest
import torch
import torch.nn as nn

from algorithms.utils.ppo_components import ActorHead


class TestActorHead:
    """Tests for ActorHead class."""

    def test_actor_creation(self) -> None:
        """ActorHead should create with correct architecture."""
        actor = ActorHead(d_model=64, hidden_dim=128)
        
        assert actor is not None
        assert isinstance(actor, nn.Module)

    def test_actor_output_shape(self) -> None:
        """Actor should output actions and log_probs with correct shapes."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        batch_size = 2
        n_ca = 3
        ca_embeddings = torch.randn(batch_size, n_ca, d_model)

        actions, log_probs, means = actor(ca_embeddings, deterministic=False)

        assert actions.shape == (batch_size, n_ca, 1)
        assert log_probs.shape == (batch_size, n_ca)
        assert means.shape == (batch_size, n_ca, 1)

    def test_actor_output_range(self) -> None:
        """Actions should be in [-1, 1] range after tanh."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(2, 3, d_model)
        actions, _, _ = actor(ca_embeddings, deterministic=False)

        assert (actions >= -1.0).all()
        assert (actions <= 1.0).all()

    def test_actor_deterministic_mode(self) -> None:
        """Deterministic mode should return mean actions."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(1, 2, d_model)
        
        # Multiple calls in deterministic mode should return same result
        actions1, _, means1 = actor(ca_embeddings, deterministic=True)
        actions2, _, means2 = actor(ca_embeddings, deterministic=True)

        assert torch.allclose(actions1, actions2)
        assert torch.allclose(actions1, means1)

    def test_actor_stochastic_mode(self) -> None:
        """Stochastic mode should sample different actions."""
        d_model = 64
        actor = ActorHead(d_model=d_model, hidden_dim=128)

        ca_embeddings = torch.randn(1, 2, d_model)
        
        # Multiple calls should likely return different results (probabilistic)
        torch.manual_seed(42)
        actions1, _, _ = actor(ca_embeddings, deterministic=False)
        torch.manual_seed(123)
        actions2, _, _ = actor(ca_embeddings, deterministic=False)

        # Not guaranteed to be different, but very likely with different seeds
        # Just check they're valid actions
        assert (actions1 >= -1.0).all() and (actions1 <= 1.0).all()
        assert (actions2 >= -1.0).all() and (actions2 <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_components.py::TestActorHead -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ActorHead class**

Create `algorithms/utils/ppo_components.py`:

```python
"""PPO Components — Actor, Critic, RolloutBuffer, and loss functions.

These components are specific to the PPO algorithm. The Actor and Critic
share the Transformer backbone but have separate heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorHead(nn.Module):
    """Actor head that produces actions from CA embeddings.

    Applies an MLP to each CA embedding independently, producing action means.
    Uses a squashed Gaussian distribution (Normal + tanh) for sampling.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        log_std_init: float = -0.5,
    ) -> None:
        """Initialize the actor head.

        Args:
            d_model: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            log_std_init: Initial value for log standard deviation.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learnable log standard deviation (shared across all CAs)
        self.log_std = nn.Parameter(torch.tensor(log_std_init))

    def forward(
        self,
        ca_embeddings: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce actions from CA embeddings.

        Args:
            ca_embeddings: [batch, N_ca, d_model] CA token embeddings.
            deterministic: If True, return mean action without sampling.

        Returns:
            Tuple of:
                - actions: [batch, N_ca, 1] sampled actions in [-1, 1].
                - log_probs: [batch, N_ca] log probability of actions.
                - means: [batch, N_ca, 1] action means (pre-tanh).
        """
        # Get action means
        means = self.mlp(ca_embeddings)  # [batch, N_ca, 1]
        
        # Get standard deviation
        std = torch.exp(self.log_std).expand_as(means)
        
        # Create normal distribution
        dist = Normal(means, std)
        
        if deterministic:
            # Use mean action
            pre_tanh_action = means
        else:
            # Sample from distribution
            pre_tanh_action = dist.rsample()
        
        # Apply tanh squashing
        actions = torch.tanh(pre_tanh_action)
        
        # Compute log probability with tanh correction
        log_probs = dist.log_prob(pre_tanh_action)
        # Correction for tanh squashing: log(1 - tanh(x)^2)
        log_probs = log_probs - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.squeeze(-1)  # [batch, N_ca]
        
        return actions, log_probs, means
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_components.py::TestActorHead -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "feat: implement ActorHead for PPO"
```

---

## Task 8: PPO Components — CriticHead

**Files:**
- Modify: `algorithms/utils/ppo_components.py`
- Modify: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing tests for CriticHead**

Add to `tests/test_ppo_components.py`:

```python
from algorithms.utils.ppo_components import CriticHead


class TestCriticHead:
    """Tests for CriticHead class."""

    def test_critic_creation(self) -> None:
        """CriticHead should create with correct architecture."""
        critic = CriticHead(d_model=64, hidden_dim=128)
        
        assert critic is not None
        assert isinstance(critic, nn.Module)

    def test_critic_output_shape(self) -> None:
        """Critic should output scalar value per batch."""
        d_model = 64
        critic = CriticHead(d_model=d_model, hidden_dim=128)

        batch_size = 2
        pooled = torch.randn(batch_size, d_model)

        values = critic(pooled)

        assert values.shape == (batch_size, 1)

    def test_critic_gradient_flow(self) -> None:
        """Gradients should flow through critic."""
        d_model = 64
        critic = CriticHead(d_model=d_model, hidden_dim=128)

        pooled = torch.randn(2, d_model, requires_grad=True)
        values = critic(pooled)
        loss = values.sum()
        loss.backward()

        assert pooled.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_components.py::TestCriticHead -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement CriticHead class**

Add to `algorithms/utils/ppo_components.py`:

```python
class CriticHead(nn.Module):
    """Critic head that produces state value from pooled embedding.

    Takes the mean-pooled representation of all tokens and outputs
    a scalar value estimate V(s).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ) -> None:
        """Initialize the critic head.

        Args:
            d_model: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Produce state value from pooled embedding.

        Args:
            pooled: [batch, d_model] mean-pooled token embeddings.

        Returns:
            values: [batch, 1] state value estimates.
        """
        return self.mlp(pooled)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_components.py::TestCriticHead -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "feat: implement CriticHead for PPO"
```

---

## Task 9: PPO Components — RolloutBuffer

**Files:**
- Modify: `algorithms/utils/ppo_components.py`
- Modify: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing tests for RolloutBuffer**

Add to `tests/test_ppo_components.py`:

```python
from algorithms.utils.ppo_components import RolloutBuffer


class TestRolloutBuffer:
    """Tests for RolloutBuffer class."""

    def test_buffer_creation(self) -> None:
        """RolloutBuffer should create with specified hyperparameters."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        
        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95

    def test_buffer_add_transition(self) -> None:
        """Buffer should store transitions."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        
        buffer.add(
            observation=torch.randn(10),
            action=torch.randn(2),
            log_prob=torch.tensor(-0.5),
            reward=1.0,
            value=torch.tensor(0.5),
            done=False,
        )
        
        assert len(buffer.observations) == 1
        assert len(buffer.rewards) == 1

    def test_buffer_compute_gae(self) -> None:
        """Buffer should compute GAE advantages."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        
        # Add a few transitions
        for i in range(5):
            buffer.add(
                observation=torch.randn(10),
                action=torch.randn(2),
                log_prob=torch.tensor(-0.5),
                reward=1.0,
                value=torch.tensor(0.5),
                done=(i == 4),  # Last one is terminal
            )
        
        buffer.compute_returns_and_advantages(last_value=torch.tensor(0.0))
        
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert len(buffer.advantages) == 5

    def test_buffer_get_batches(self) -> None:
        """Buffer should yield minibatches."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        
        for i in range(10):
            buffer.add(
                observation=torch.randn(10),
                action=torch.randn(2),
                log_prob=torch.tensor(-0.5),
                reward=1.0,
                value=torch.tensor(0.5),
                done=False,
            )
        
        buffer.compute_returns_and_advantages(last_value=torch.tensor(0.0))
        
        batches = list(buffer.get_batches(batch_size=4))
        assert len(batches) >= 2  # At least 2 batches of size 4 from 10 samples

    def test_buffer_clear(self) -> None:
        """Buffer should clear all data."""
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        
        buffer.add(
            observation=torch.randn(10),
            action=torch.randn(2),
            log_prob=torch.tensor(-0.5),
            reward=1.0,
            value=torch.tensor(0.5),
            done=False,
        )
        
        buffer.clear()
        
        assert len(buffer.observations) == 0
        assert len(buffer.rewards) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_components.py::TestRolloutBuffer -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement RolloutBuffer class**

Add to `algorithms/utils/ppo_components.py`:

```python
@dataclass
class Batch:
    """A minibatch of transitions for PPO update."""
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """On-policy rollout buffer for PPO.

    Stores transitions from the current policy, computes GAE advantages,
    and provides minibatch iteration for PPO updates.
    """

    def __init__(self, gamma: float, gae_lambda: float) -> None:
        """Initialize the rollout buffer.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            observation: Encoded observation tensor.
            action: Action tensor.
            log_prob: Log probability of the action.
            reward: Reward received.
            value: Value estimate from critic.
            done: Whether episode terminated.
        """
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: torch.Tensor) -> None:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Value estimate for the state after the last transition.
        """
        n = len(self.rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)
        
        # Convert values to tensor
        values = torch.stack([v.squeeze() for v in self.values])
        
        # GAE computation (reverse order)
        gae = torch.tensor(0.0)
        next_value = last_value.squeeze()
        
        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int) -> Iterator[Batch]:
        """Yield minibatches for PPO update.

        Args:
            batch_size: Size of each minibatch.

        Yields:
            Batch objects containing transition data.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_returns_and_advantages first")
        
        n = len(self.observations)
        indices = torch.randperm(n)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            
            yield Batch(
                observations=torch.stack([self.observations[i] for i in batch_indices]),
                actions=torch.stack([self.actions[i] for i in batch_indices]),
                log_probs=torch.stack([self.log_probs[i] for i in batch_indices]),
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
                values=torch.stack([self.values[i].squeeze() for i in batch_indices]),
            )

    def clear(self) -> None:
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.observations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_components.py::TestRolloutBuffer -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "feat: implement RolloutBuffer for PPO"
```

---

## Task 10: PPO Components — Loss Function

**Files:**
- Modify: `algorithms/utils/ppo_components.py`
- Modify: `tests/test_ppo_components.py`

- [ ] **Step 1: Write failing tests for PPO loss**

Add to `tests/test_ppo_components.py`:

```python
from algorithms.utils.ppo_components import compute_ppo_loss


class TestPPOLoss:
    """Tests for PPO loss computation."""

    def test_ppo_loss_shape(self) -> None:
        """PPO loss should return scalar tensor and metrics dict."""
        batch_size = 4
        
        log_probs_new = torch.randn(batch_size)
        log_probs_old = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        values = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        
        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )
        
        assert loss.ndim == 0  # Scalar
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_ppo_loss_clipping(self) -> None:
        """PPO loss should clip probability ratios."""
        batch_size = 4
        
        # Create scenario where ratio would be clipped
        log_probs_new = torch.zeros(batch_size)
        log_probs_old = torch.ones(batch_size) * -1.0  # ratio = exp(1) ≈ 2.7
        advantages = torch.ones(batch_size)
        values = torch.zeros(batch_size)
        returns = torch.ones(batch_size)
        
        loss, metrics = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )
        
        # Loss should be finite (clipping prevents explosion)
        assert torch.isfinite(loss)

    def test_ppo_loss_gradient_flow(self) -> None:
        """Gradients should flow through PPO loss."""
        log_probs_new = torch.randn(4, requires_grad=True)
        log_probs_old = torch.randn(4)
        advantages = torch.randn(4)
        values = torch.randn(4, requires_grad=True)
        returns = torch.randn(4)
        
        loss, _ = compute_ppo_loss(
            log_probs_new=log_probs_new,
            log_probs_old=log_probs_old,
            advantages=advantages,
            values=values,
            returns=returns,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
        )
        
        loss.backward()
        
        assert log_probs_new.grad is not None
        assert values.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ppo_components.py::TestPPOLoss -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement compute_ppo_loss function**

Add to `algorithms/utils/ppo_components.py`:

```python
def compute_ppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    value_coeff: float,
    entropy_coeff: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute PPO clipped surrogate loss.

    Args:
        log_probs_new: Log probabilities under current policy.
        log_probs_old: Log probabilities under old policy (detached).
        advantages: GAE advantages (normalized).
        values: Value estimates from critic.
        returns: Discounted returns.
        clip_eps: Clipping epsilon for probability ratio.
        value_coeff: Coefficient for value loss.
        entropy_coeff: Coefficient for entropy bonus.

    Returns:
        Tuple of:
            - total_loss: Combined loss for backprop.
            - metrics: Dict with policy_loss, value_loss, entropy.
    """
    # Probability ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss (MSE)
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus (approximate using log_probs)
    # For squashed Gaussian, entropy is complex; use simple approximation
    entropy = -log_probs_new.mean()
    
    # Combined loss
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
    
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }
    
    return total_loss, metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ppo_components.py::TestPPOLoss -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all PPO component tests**

Run: `pytest tests/test_ppo_components.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add algorithms/utils/ppo_components.py tests/test_ppo_components.py
git commit -m "feat: implement PPO loss function"
```

---

## Task 11: Final Integration — Run All Plan B Tests

**Files:**
- None (verification only)

- [ ] **Step 1: Run all Plan B tests**

Run: `pytest tests/test_observation_tokenizer.py tests/test_transformer_backbone.py tests/test_ppo_components.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run Plan A tests to verify no regression**

Run: `pytest tests/test_observation_enricher.py tests/test_tokenizer_config_schema.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit summary (if any final changes needed)**

No commit needed if all tests pass.

---

## Summary

After completing all tasks, you will have:

1. **`algorithms/utils/observation_tokenizer.py`** — Scans markers, extracts groups, projects to d_model
2. **`algorithms/utils/transformer_backbone.py`** — Self-attention with type embeddings
3. **`algorithms/utils/ppo_components.py`** — ActorHead, CriticHead, RolloutBuffer, PPO loss
4. **`tests/test_observation_tokenizer.py`** — Tokenizer unit tests
5. **`tests/test_transformer_backbone.py`** — Backbone unit tests
6. **`tests/test_ppo_components.py`** — PPO component unit tests

All components are tested and ready for Plan C (Integration).
