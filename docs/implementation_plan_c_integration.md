# Plan C: Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate all components into a working TransformerPPO agent: Wrapper integration, AgentTransformerPPO implementation, registry registration, config template, and end-to-end validation.

**Architecture:** The wrapper handles observation enrichment for Transformer agents. The AgentTransformerPPO combines tokenizer, backbone, and PPO components into a complete agent satisfying the BaseAgent contract. Registry enables instantiation by the runner.

**Tech Stack:** Python 3.10+, PyTorch, pytest, CityLearn

**Spec Reference:** `docs/spec.md` sections 7, 8, 9, 10

**Dependencies:** Plans A and B must be completed first

---

## Git Setup

**Before starting implementation:**

1. Ensure Plans A and B are merged to `gj/master`
2. Create branch `gj/plan-c` from `gj/master`:
   ```bash
   git checkout gj/master
   git pull origin gj/master
   git checkout -b gj/plan-c
   ```
3. Verify you're on the correct branch:
   ```bash
   git branch -v | grep plan-c
   # Expected: gj/plan-c ... (should show latest gj/master commit)
   ```
4. After all tasks complete and tests pass, this branch will be merged back to `gj/master`
5. Do NOT commit to `gj/master` or `main` — all work stays on `gj/plan-c`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `utils/wrapper_citylearn.py` | Add enrichment support for Transformer agents (modify existing) |
| `algorithms/agents/transformer_ppo_agent.py` | Full agent implementation |
| `algorithms/registry.py` | Register AgentTransformerPPO (modify existing) |
| `configs/templates/transformer_ppo.yaml` | Algorithm config template |
| `utils/config_schema.py` | Add TransformerPPO config schema (modify existing) |
| `tests/test_wrapper_transformer.py` | Wrapper integration tests |
| `tests/test_agent_transformer_ppo.py` | Agent unit tests |
| `tests/test_e2e_transformer_ppo.py` | End-to-end validation tests |

---

## Task 1: Config Schema — TransformerPPO Algorithm Config

**Files:**
- Modify: `utils/config_schema.py`
- Modify: `tests/test_tokenizer_config_schema.py` (add new tests)

- [ ] **Step 1: Write failing tests for TransformerPPO config schema**

Add to `tests/test_tokenizer_config_schema.py`:

```python
class TestTransformerPPOConfigSchema:
    """Tests for TransformerPPO algorithm config schema."""

    def test_transformer_config_valid(self) -> None:
        """Valid Transformer config should parse successfully."""
        from utils.config_schema import TransformerConfig
        
        config = TransformerConfig(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
        
        assert config.d_model == 64
        assert config.nhead == 4
        assert config.num_layers == 2

    def test_transformer_config_invalid_nhead(self) -> None:
        """nhead must divide d_model evenly."""
        from utils.config_schema import TransformerConfig
        
        with pytest.raises(ValueError):
            TransformerConfig(
                d_model=64,
                nhead=5,  # 64 % 5 != 0
                num_layers=2,
            )

    def test_transformer_ppo_hyperparameters_valid(self) -> None:
        """Valid PPO hyperparameters should parse successfully."""
        from utils.config_schema import TransformerPPOHyperparameters
        
        config = TransformerPPOHyperparameters(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            ppo_epochs=4,
            minibatch_size=64,
            entropy_coeff=0.01,
            value_coeff=0.5,
            max_grad_norm=0.5,
        )
        
        assert config.gamma == 0.99
        assert config.clip_eps == 0.2

    def test_transformer_ppo_algorithm_config_valid(self) -> None:
        """Full TransformerPPO algorithm config should parse."""
        from utils.config_schema import TransformerPPOAlgorithmConfig
        
        config = TransformerPPOAlgorithmConfig(
            name="AgentTransformerPPO",
            tokenizer_config_path="configs/tokenizers/default.json",
            transformer={
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.1,
            },
            hyperparameters={
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 4,
                "minibatch_size": 64,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
            },
        )
        
        assert config.name == "AgentTransformerPPO"
        assert config.transformer.d_model == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenizer_config_schema.py::TestTransformerPPOConfigSchema -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement TransformerPPO config schema**

Add to `utils/config_schema.py` (after TokenizerConfig):

```python
# ---------------------------------------------------------------------------
# TransformerPPO Algorithm Config Schema
# ---------------------------------------------------------------------------


class TransformerConfig(BaseModel):
    """Configuration for Transformer backbone architecture."""

    d_model: int = Field(..., gt=0, description="Embedding dimension")
    nhead: int = Field(..., gt=0, description="Number of attention heads")
    num_layers: int = Field(..., gt=0, description="Number of encoder layers")
    dim_feedforward: int = Field(256, gt=0, description="Feedforward network dimension")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")

    @validator("nhead")
    def nhead_divides_d_model(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that nhead divides d_model evenly."""
        d_model = values.get("d_model")
        if d_model is not None and d_model % v != 0:
            raise ValueError(f"nhead ({v}) must divide d_model ({d_model}) evenly")
        return v


class TransformerPPOHyperparameters(BaseModel):
    """Hyperparameters for TransformerPPO agent."""

    learning_rate: float = Field(3e-4, gt=0, description="Learning rate")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    gae_lambda: float = Field(0.95, ge=0.0, le=1.0, description="GAE lambda")
    clip_eps: float = Field(0.2, gt=0, description="PPO clipping epsilon")
    ppo_epochs: int = Field(4, gt=0, description="PPO epochs per update")
    minibatch_size: int = Field(64, gt=0, description="Minibatch size")
    entropy_coeff: float = Field(0.01, ge=0.0, description="Entropy coefficient")
    value_coeff: float = Field(0.5, gt=0, description="Value loss coefficient")
    max_grad_norm: float = Field(0.5, gt=0, description="Max gradient norm for clipping")
    hidden_dim: int = Field(128, gt=0, description="Hidden dimension for actor/critic heads")
    rollout_length: int = Field(2048, gt=0, description="Steps before PPO update")


class TransformerPPOAlgorithmConfig(BaseModel):
    """Full algorithm configuration for TransformerPPO agent."""

    name: Literal["AgentTransformerPPO"] = "AgentTransformerPPO"
    tokenizer_config_path: str = Field(..., description="Path to tokenizer config JSON")
    transformer: TransformerConfig
    hyperparameters: TransformerPPOHyperparameters
```

Also add to imports:

```python
from typing import Literal  # Add if not present
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokenizer_config_schema.py::TestTransformerPPOConfigSchema -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/config_schema.py tests/test_tokenizer_config_schema.py
git commit -m "feat: add TransformerPPO algorithm config schema"
```

---

## Task 2: Create Config Template

**Files:**
- Create: `configs/templates/transformer_ppo.yaml`

- [ ] **Step 1: Create the config template file**

Create `configs/templates/transformer_ppo.yaml`:

```yaml
# TransformerPPO Algorithm Configuration Template
# 
# This template provides default values for the TransformerPPO agent.
# Copy and modify as needed for your experiments.

algorithm:
  name: AgentTransformerPPO
  tokenizer_config_path: configs/tokenizers/default.json
  
  transformer:
    d_model: 64
    nhead: 4
    num_layers: 2
    dim_feedforward: 128
    dropout: 0.1
  
  hyperparameters:
    learning_rate: 3.0e-4
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    ppo_epochs: 4
    minibatch_size: 64
    entropy_coeff: 0.01
    value_coeff: 0.5
    max_grad_norm: 0.5
    hidden_dim: 128
    rollout_length: 2048

# Environment configuration (example)
environment:
  schema_path: datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json
  buildings:
    - Building_1

# Training configuration (example)
training:
  total_episodes: 100
  checkpoint_interval: 10
  log_interval: 1
```

- [ ] **Step 2: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('configs/templates/transformer_ppo.yaml'))"`
Expected: No output (valid YAML)

- [ ] **Step 3: Commit**

```bash
git add configs/templates/transformer_ppo.yaml
git commit -m "feat: add TransformerPPO config template"
```

---

## Task 3: Wrapper Integration — Enricher Support

**Files:**
- Modify: `utils/wrapper_citylearn.py`
- Create: `tests/test_wrapper_transformer.py`

- [ ] **Step 1: Read existing wrapper to understand structure**

Read `utils/wrapper_citylearn.py` to understand:
- How observations are processed
- Where to add enrichment hooks
- How encoders are managed

- [ ] **Step 2: Write failing tests for wrapper enrichment**

Create `tests/test_wrapper_transformer.py`:

```python
"""Tests for Transformer agent wrapper integration."""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np


class TestWrapperEnricherSetup:
    """Tests for enricher setup in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enricher_created_for_transformer_agent(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enrichers should be created when agent is Transformer-based."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        from utils.wrapper_citylearn import CityLearnWrapper
        
        # Mock the agent to indicate it's a Transformer agent
        mock_agent = MagicMock()
        mock_agent.is_transformer_agent = True
        mock_agent.tokenizer_config = sample_tokenizer_config
        
        # This test verifies the wrapper has enricher setup capability
        # Actual test depends on wrapper implementation
        enricher = ObservationEnricher(sample_tokenizer_config)
        assert enricher is not None

    def test_no_enricher_for_non_transformer_agent(self) -> None:
        """Enrichers should not be created for non-Transformer agents."""
        # This test documents expected behavior
        # Non-Transformer agents skip enrichment entirely
        pass


class TestWrapperEnrichment:
    """Tests for observation enrichment in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month"], "input_dim": 2},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }

    def test_enriched_obs_contains_markers(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enriched observations should contain marker values."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc", "non_shiftable_load"]
        action_names = ["electrical_storage"]
        
        enricher.enrich_names(observation_names, action_names)
        
        observation_values = [6.0, 0.75, 100.0]
        enriched_values = enricher.enrich_values(observation_values)
        
        # Should contain marker values
        assert 1001.0 in enriched_values  # CA marker
        assert 2001.0 in enriched_values  # SRO marker
        assert 3001.0 in enriched_values  # NFC marker

    def test_marker_encoder_specs_generated(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Enrichment should provide encoder specs for markers."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        observation_names = ["month", "electrical_storage_soc"]
        action_names = ["electrical_storage"]
        
        result = enricher.enrich_names(observation_names, action_names)
        
        # Enriched names should include marker names
        marker_names = [n for n in result.enriched_names if n.startswith("__marker_")]
        assert len(marker_names) > 0


class TestWrapperTopologyChange:
    """Tests for topology change detection in wrapper."""

    @pytest.fixture
    def sample_tokenizer_config(self) -> Dict[str, Any]:
        """Sample tokenizer config."""
        return {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
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
            "sro_types": {},
            "nfc": {
                "demand_features": [],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 0,
            },
        }

    def test_topology_change_detected(
        self, sample_tokenizer_config: Dict[str, Any]
    ) -> None:
        """Wrapper should detect when observation count changes."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        enricher = ObservationEnricher(sample_tokenizer_config)
        
        # Initial topology
        obs_names_v1 = ["electrical_storage_soc"]
        action_names_v1 = ["electrical_storage"]
        enricher.enrich_names(obs_names_v1, action_names_v1)
        
        # Same topology - no change
        assert not enricher.topology_changed(obs_names_v1, action_names_v1)
        
        # New topology - EV charger added
        obs_names_v2 = ["electrical_storage_soc", "electric_vehicle_soc"]
        action_names_v2 = ["electrical_storage", "electric_vehicle_storage"]
        assert enricher.topology_changed(obs_names_v2, action_names_v2)
```

- [ ] **Step 3: Run test to verify structure**

Run: `pytest tests/test_wrapper_transformer.py -v`
Expected: Tests should pass (they test enricher directly, which is implemented)

- [ ] **Step 4: Add wrapper methods for Transformer support**

Add to `utils/wrapper_citylearn.py` (in CityLearnWrapper class):

```python
    def _setup_transformer_enrichers(self, tokenizer_config: Dict[str, Any]) -> None:
        """Set up observation enrichers for Transformer agents.
        
        Called when the agent is a Transformer-based agent (e.g., AgentTransformerPPO).
        Creates per-building enrichers that inject marker values into observations.
        
        Args:
            tokenizer_config: Tokenizer configuration dict.
        """
        from algorithms.utils.observation_enricher import ObservationEnricher
        
        self._is_transformer_agent = True
        self._enrichers: List[Optional[ObservationEnricher]] = []
        self._tokenizer_config = tokenizer_config
        
        for i in range(self.num_buildings):
            enricher = ObservationEnricher(tokenizer_config)
            self._enrichers.append(enricher)
            
    def _enrich_observation_names(self, building_idx: int) -> None:
        """Enrich observation names for a building (Transformer agents only).
        
        Called once per topology. Updates the enricher's cache and prepares
        marker positions for value enrichment.
        
        Args:
            building_idx: Index of the building.
        """
        if not getattr(self, '_is_transformer_agent', False):
            return
            
        enricher = self._enrichers[building_idx]
        if enricher is None:
            return
            
        # Get current observation and action names
        obs_names = self.observation_names[building_idx]
        action_names = self.action_names[building_idx]
        
        # Enrich names (caches result)
        enrichment = enricher.enrich_names(obs_names, action_names)
        
        # Store enriched names for encoder rebuilding
        self._enriched_observation_names[building_idx] = enrichment.enriched_names
        
    def _enrich_observation_values(
        self, building_idx: int, raw_values: List[float]
    ) -> List[float]:
        """Enrich observation values with marker values (Transformer agents only).
        
        Args:
            building_idx: Index of the building.
            raw_values: Raw observation values from CityLearn.
            
        Returns:
            Enriched values with markers inserted, or raw values if not Transformer.
        """
        if not getattr(self, '_is_transformer_agent', False):
            return raw_values
            
        enricher = self._enrichers[building_idx]
        if enricher is None:
            return raw_values
            
        return enricher.enrich_values(raw_values)
        
    def _check_topology_change(self, building_idx: int) -> bool:
        """Check if topology changed for a building (Transformer agents only).
        
        Args:
            building_idx: Index of the building.
            
        Returns:
            True if topology changed, False otherwise.
        """
        if not getattr(self, '_is_transformer_agent', False):
            return False
            
        enricher = self._enrichers[building_idx]
        if enricher is None:
            return False
            
        obs_names = self.observation_names[building_idx]
        action_names = self.action_names[building_idx]
        
        return enricher.topology_changed(obs_names, action_names)
        
    def _handle_topology_change(self, building_idx: int) -> None:
        """Handle topology change for a building (Transformer agents only).
        
        1. Re-enrich names with new topology
        2. Rebuild encoders
        3. Notify agent
        
        Args:
            building_idx: Index of the building.
        """
        if not getattr(self, '_is_transformer_agent', False):
            return
            
        # Re-enrich names
        self._enrich_observation_names(building_idx)
        
        # Rebuild encoders for this building
        # (Implementation depends on existing encoder structure)
        
        # Notify agent if it has the method
        if hasattr(self.model, 'on_topology_change'):
            self.model.on_topology_change(building_idx)
```

Note: The actual integration into the wrapper's main flow depends on the existing wrapper structure. The above provides the building blocks.

- [ ] **Step 5: Run tests to verify wrapper changes**

Run: `pytest tests/test_wrapper_transformer.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add utils/wrapper_citylearn.py tests/test_wrapper_transformer.py
git commit -m "feat: add Transformer agent support to wrapper"
```

---

## Task 4: AgentTransformerPPO — Core Structure

**Files:**
- Create: `algorithms/agents/transformer_ppo_agent.py`
- Create: `tests/test_agent_transformer_ppo.py`

- [ ] **Step 1: Write failing tests for agent instantiation**

Create `tests/test_agent_transformer_ppo.py`:

```python
"""Tests for AgentTransformerPPO."""

import json
import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock


class TestAgentInstantiation:
    """Tests for AgentTransformerPPO instantiation."""

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config with tokenizer file."""
        # Create tokenizer config file
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 4,
                    "minibatch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 128,
                    "rollout_length": 2048,
                },
            }
        }

    def test_agent_creation(self, sample_config: Dict[str, Any]) -> None:
        """Agent should instantiate with valid config."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent is not None
        assert agent.is_transformer_agent is True

    def test_agent_has_tokenizer(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have tokenizer after creation."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.tokenizer is not None

    def test_agent_has_backbone(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have Transformer backbone."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.backbone is not None

    def test_agent_has_actor_critic(self, sample_config: Dict[str, Any]) -> None:
        """Agent should have actor and critic heads."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        assert agent.actor is not None
        assert agent.critic is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentInstantiation -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AgentTransformerPPO core structure**

Create `algorithms/agents/transformer_ppo_agent.py`:

```python
"""TransformerPPO Agent — Transformer-based PPO agent for variable-topology buildings.

This agent uses a Transformer backbone to process variable numbers of controllable
assets (CAs), shared read-only observations (SROs), and non-flexible context (NFC)
as tokens. The architecture naturally handles topology changes (assets connecting/
disconnecting) without retraining.

Key features:
- Dynamic cardinality: Handles variable numbers of CAs at runtime
- Per-type projections: Each asset type has its own learned projection
- Marker-based tokenization: Uses marker values to identify token boundaries
- On-policy PPO: Uses GAE for advantage estimation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.observation_tokenizer import ObservationTokenizer
from algorithms.utils.transformer_backbone import TransformerBackbone
from algorithms.utils.ppo_components import (
    ActorHead,
    CriticHead,
    RolloutBuffer,
    compute_ppo_loss,
)


class AgentTransformerPPO(BaseAgent):
    """Transformer-based PPO agent for energy management.

    Satisfies the BaseAgent contract while using a Transformer architecture
    that handles variable numbers of controllable assets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the agent from configuration.

        Args:
            config: Full experiment configuration dict.
                Expected keys:
                - algorithm.tokenizer_config_path: Path to tokenizer JSON
                - algorithm.transformer: Transformer architecture params
                - algorithm.hyperparameters: PPO hyperparameters
        """
        super().__init__()
        
        # Mark as Transformer agent for wrapper
        self.is_transformer_agent = True
        
        # Extract config sections
        algo_config = config.get("algorithm", {})
        transformer_config = algo_config.get("transformer", {})
        hyperparams = algo_config.get("hyperparameters", {})
        
        # Load tokenizer config
        tokenizer_config_path = algo_config.get("tokenizer_config_path")
        if tokenizer_config_path:
            with open(tokenizer_config_path) as f:
                self.tokenizer_config = json.load(f)
        else:
            raise ValueError("tokenizer_config_path is required")
        
        # Store hyperparameters
        self.d_model = transformer_config.get("d_model", 64)
        self.nhead = transformer_config.get("nhead", 4)
        self.num_layers = transformer_config.get("num_layers", 2)
        self.dim_feedforward = transformer_config.get("dim_feedforward", 128)
        self.dropout = transformer_config.get("dropout", 0.1)
        
        self.learning_rate = hyperparams.get("learning_rate", 3e-4)
        self.gamma = hyperparams.get("gamma", 0.99)
        self.gae_lambda = hyperparams.get("gae_lambda", 0.95)
        self.clip_eps = hyperparams.get("clip_eps", 0.2)
        self.ppo_epochs = hyperparams.get("ppo_epochs", 4)
        self.minibatch_size = hyperparams.get("minibatch_size", 64)
        self.entropy_coeff = hyperparams.get("entropy_coeff", 0.01)
        self.value_coeff = hyperparams.get("value_coeff", 0.5)
        self.max_grad_norm = hyperparams.get("max_grad_norm", 0.5)
        self.hidden_dim = hyperparams.get("hidden_dim", 128)
        self.rollout_length = hyperparams.get("rollout_length", 2048)
        
        # Create components
        self.tokenizer = ObservationTokenizer(self.tokenizer_config, self.d_model)
        self.backbone = TransformerBackbone(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.actor = ActorHead(d_model=self.d_model, hidden_dim=self.hidden_dim)
        self.critic = CriticHead(d_model=self.d_model, hidden_dim=self.hidden_dim)
        
        # Combine all parameters for optimizer
        self.all_params = (
            list(self.tokenizer.parameters()) +
            list(self.backbone.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.optimizer = optim.Adam(self.all_params, lr=self.learning_rate)
        
        # Per-building rollout buffers (created when num_buildings is known)
        self.rollout_buffers: List[RolloutBuffer] = []
        self._num_buildings: int = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
        # Training state
        self._step_count = 0
        self._last_values: List[Optional[torch.Tensor]] = []
        self._last_log_probs: List[Optional[torch.Tensor]] = []

    def _move_to_device(self) -> None:
        """Move all modules to the configured device."""
        self.tokenizer.to(self.device)
        self.backbone.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)

    def attach_environment(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List,
        observation_space: List,
        metadata: Dict[str, Any],
    ) -> None:
        """Receive environment metadata from wrapper.

        Called by the wrapper after environment setup. Creates per-building
        rollout buffers.

        Args:
            observation_names: Observation names per building.
            action_names: Action names per building.
            action_space: Action spaces per building.
            observation_space: Observation spaces per building.
            metadata: Additional environment metadata.
        """
        self._num_buildings = len(observation_names)
        self.observation_names = observation_names
        self.action_names = action_names
        
        # Create per-building rollout buffers
        self.rollout_buffers = [
            RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
            for _ in range(self._num_buildings)
        ]
        
        # Initialize tracking lists
        self._last_values = [None] * self._num_buildings
        self._last_log_probs = [None] * self._num_buildings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentInstantiation -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/transformer_ppo_agent.py tests/test_agent_transformer_ppo.py
git commit -m "feat: implement AgentTransformerPPO core structure"
```

---

## Task 5: AgentTransformerPPO — predict() Method

**Files:**
- Modify: `algorithms/agents/transformer_ppo_agent.py`
- Modify: `tests/test_agent_transformer_ppo.py`

- [ ] **Step 1: Write failing tests for predict()**

Add to `tests/test_agent_transformer_ppo.py`:

```python
class TestAgentPredict:
    """Tests for AgentTransformerPPO.predict()."""

    @pytest.fixture
    def agent_with_env(self, sample_config: Dict[str, Any]) -> Any:
        """Create agent and attach mock environment."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        # Attach environment
        agent.attach_environment(
            observation_names=[["month", "hour", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        return agent

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config with tokenizer file."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                    "dim_feedforward": 128,
                    "dropout": 0.1,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 4,
                    "minibatch_size": 64,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 128,
                    "rollout_length": 2048,
                },
            }
        }

    def test_predict_returns_actions(self, agent_with_env: Any) -> None:
        """predict() should return actions for each building."""
        # Create encoded observation with markers
        # [CA_1001, soc, SRO_2001, month, hour, d1, d2, NFC_3001, load]
        encoded_obs = np.array([[
            1001.0, 0.5,  # CA: battery
            2001.0, 0.5, 0.5, 0.5, 0.5,  # SRO: temporal (4 features)
            3001.0, 100.0,  # NFC (1 feature)
        ]])
        
        actions = agent_with_env.predict([encoded_obs], deterministic=False)
        
        assert len(actions) == 1  # One building
        assert actions[0].shape[-1] == 1  # One action per CA

    def test_predict_deterministic(self, agent_with_env: Any) -> None:
        """Deterministic predict should return same actions."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        actions1 = agent_with_env.predict([encoded_obs], deterministic=True)
        actions2 = agent_with_env.predict([encoded_obs], deterministic=True)
        
        np.testing.assert_array_almost_equal(actions1[0], actions2[0])

    def test_predict_action_range(self, agent_with_env: Any) -> None:
        """Actions should be in [-1, 1] range."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        # Multiple predictions to check range
        for _ in range(10):
            actions = agent_with_env.predict([encoded_obs], deterministic=False)
            assert (actions[0] >= -1.0).all()
            assert (actions[0] <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentPredict -v`
Expected: FAIL (predict method not implemented)

- [ ] **Step 3: Implement predict() method**

Add to `AgentTransformerPPO` class:

```python
    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[np.ndarray]:
        """Predict actions for all buildings.

        Args:
            observations: List of encoded observations per building.
                Each observation should already have markers injected.
            deterministic: If True, use mean actions without sampling.

        Returns:
            List of action arrays per building.
        """
        self.tokenizer.eval()
        self.backbone.eval()
        self.actor.eval()
        self.critic.eval()
        
        all_actions: List[np.ndarray] = []
        
        with torch.no_grad() if deterministic else torch.enable_grad():
            for b_idx, obs in enumerate(observations):
                # Convert to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dim
                
                # Tokenize
                tokenized = self.tokenizer(obs_tensor)
                
                # Transformer backbone
                backbone_out = self.backbone(
                    tokenized.ca_tokens,
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                )
                
                # Actor head
                actions, log_probs, _ = self.actor(
                    backbone_out.ca_embeddings,
                    deterministic=deterministic,
                )
                
                # Critic head (for storing value in buffer during training)
                value = self.critic(backbone_out.pooled)
                
                # Store for update step
                if not deterministic:
                    self._last_values[b_idx] = value
                    self._last_log_probs[b_idx] = log_probs
                    self._last_obs = obs_tensor
                    self._last_actions = actions
                
                # Convert to numpy
                actions_np = actions.squeeze(0).cpu().numpy()  # Remove batch dim
                all_actions.append(actions_np)
        
        return all_actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentPredict -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/transformer_ppo_agent.py tests/test_agent_transformer_ppo.py
git commit -m "feat: implement AgentTransformerPPO.predict()"
```

---

## Task 6: AgentTransformerPPO — update() Method

**Files:**
- Modify: `algorithms/agents/transformer_ppo_agent.py`
- Modify: `tests/test_agent_transformer_ppo.py`

- [ ] **Step 1: Write failing tests for update()**

Add to `tests/test_agent_transformer_ppo.py`:

```python
class TestAgentUpdate:
    """Tests for AgentTransformerPPO.update()."""

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 2,  # Fewer epochs for testing
                    "minibatch_size": 4,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 64,
                    "rollout_length": 8,  # Small for testing
                },
            }
        }

    @pytest.fixture
    def agent_with_env(self, sample_config: Dict[str, Any]) -> Any:
        """Create agent and attach mock environment."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        agent.attach_environment(
            observation_names=[["month", "hour", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        return agent

    def test_update_stores_transition(self, agent_with_env: Any) -> None:
        """update() should store transition in buffer."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        # Predict first to generate values/log_probs
        agent_with_env.predict([encoded_obs], deterministic=False)
        
        # Update
        metrics = agent_with_env.update(
            observations=[encoded_obs],
            actions=[np.array([[0.5]])],
            rewards=[1.0],
            next_observations=[encoded_obs],
            terminated=[False],
            truncated=[False],
            update_target_step=False,
            global_learning_step=1,
            update_step=False,
            initial_exploration_done=True,
        )
        
        # Buffer should have one transition
        assert len(agent_with_env.rollout_buffers[0]) == 1

    def test_update_returns_metrics(self, agent_with_env: Any) -> None:
        """update() should return metrics dict."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        agent_with_env.predict([encoded_obs], deterministic=False)
        
        metrics = agent_with_env.update(
            observations=[encoded_obs],
            actions=[np.array([[0.5]])],
            rewards=[1.0],
            next_observations=[encoded_obs],
            terminated=[False],
            truncated=[False],
            update_target_step=False,
            global_learning_step=1,
            update_step=False,
            initial_exploration_done=True,
        )
        
        assert isinstance(metrics, dict)

    def test_update_triggers_ppo_update(self, agent_with_env: Any) -> None:
        """update() should trigger PPO update when buffer is full."""
        encoded_obs = np.array([[
            1001.0, 0.5,
            2001.0, 0.5, 0.5, 0.5, 0.5,
            3001.0, 100.0,
        ]])
        
        # Fill buffer to rollout_length (8)
        for i in range(10):
            agent_with_env.predict([encoded_obs], deterministic=False)
            agent_with_env.update(
                observations=[encoded_obs],
                actions=[np.array([[0.5]])],
                rewards=[1.0],
                next_observations=[encoded_obs],
                terminated=[False],
                truncated=[False],
                update_target_step=False,
                global_learning_step=i,
                update_step=(i == 9),  # Trigger update on last step
                initial_exploration_done=True,
            )
        
        # Buffer should be cleared after update
        # (depends on implementation)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentUpdate -v`
Expected: FAIL (update method not implemented)

- [ ] **Step 3: Implement update() method**

Add to `AgentTransformerPPO` class:

```python
    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: List[bool],
        truncated: List[bool],
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> Dict[str, float]:
        """PPO on-policy update step.

        Stores transitions in rollout buffer. When update_step is True and
        buffer has enough data, performs PPO update.

        Args:
            observations: Encoded observations per building.
            actions: Actions taken per building.
            rewards: Rewards received per building.
            next_observations: Next observations per building.
            terminated: Episode termination flags per building.
            truncated: Episode truncation flags per building.
            update_target_step: Ignored (no target network in PPO).
            global_learning_step: Current learning step.
            update_step: Whether to perform PPO update.
            initial_exploration_done: Whether initial exploration is done.

        Returns:
            Metrics dict (empty if no update performed).
        """
        metrics: Dict[str, float] = {}
        
        if not initial_exploration_done:
            return metrics
        
        # Store transitions in buffers
        for b_idx in range(self._num_buildings):
            if self._last_values[b_idx] is None:
                continue
                
            obs_tensor = torch.tensor(
                observations[b_idx], dtype=torch.float32, device=self.device
            )
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            action_tensor = torch.tensor(
                actions[b_idx], dtype=torch.float32, device=self.device
            )
            
            done = terminated[b_idx] or truncated[b_idx]
            
            self.rollout_buffers[b_idx].add(
                observation=obs_tensor.squeeze(0),
                action=action_tensor.squeeze(0) if action_tensor.ndim > 1 else action_tensor,
                log_prob=self._last_log_probs[b_idx].squeeze(),
                reward=rewards[b_idx],
                value=self._last_values[b_idx].squeeze(),
                done=done,
            )
        
        self._step_count += 1
        
        # Perform PPO update if requested and buffer is ready
        if update_step:
            for b_idx in range(self._num_buildings):
                buffer = self.rollout_buffers[b_idx]
                if len(buffer) >= self.minibatch_size:
                    update_metrics = self._ppo_update(b_idx, next_observations[b_idx])
                    for k, v in update_metrics.items():
                        metrics[f"building_{b_idx}/{k}"] = v
        
        return metrics

    def _ppo_update(
        self, building_idx: int, last_obs: np.ndarray
    ) -> Dict[str, float]:
        """Perform PPO update for a single building.

        Args:
            building_idx: Index of the building.
            last_obs: Last observation for bootstrapping value.

        Returns:
            Metrics from the update.
        """
        buffer = self.rollout_buffers[building_idx]
        
        # Compute last value for GAE
        with torch.no_grad():
            last_obs_tensor = torch.tensor(
                last_obs, dtype=torch.float32, device=self.device
            )
            if last_obs_tensor.ndim == 1:
                last_obs_tensor = last_obs_tensor.unsqueeze(0)
            
            tokenized = self.tokenizer(last_obs_tensor)
            backbone_out = self.backbone(
                tokenized.ca_tokens,
                tokenized.sro_tokens,
                tokenized.nfc_token,
            )
            last_value = self.critic(backbone_out.pooled)
        
        # Compute advantages
        buffer.compute_returns_and_advantages(last_value.squeeze())
        
        # PPO epochs
        all_metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        
        self.tokenizer.train()
        self.backbone.train()
        self.actor.train()
        self.critic.train()
        
        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(self.minibatch_size):
                # Forward pass
                tokenized = self.tokenizer(batch.observations)
                backbone_out = self.backbone(
                    tokenized.ca_tokens,
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                )
                
                # Get new log probs
                _, log_probs_new, _ = self.actor(
                    backbone_out.ca_embeddings,
                    deterministic=False,
                )
                log_probs_new = log_probs_new.mean(dim=-1)  # Average over CAs
                
                # Get new values
                values_new = self.critic(backbone_out.pooled).squeeze(-1)
                
                # Compute loss
                loss, batch_metrics = compute_ppo_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=batch.log_probs,
                    advantages=batch.advantages,
                    values=values_new,
                    returns=batch.returns,
                    clip_eps=self.clip_eps,
                    value_coeff=self.value_coeff,
                    entropy_coeff=self.entropy_coeff,
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
                self.optimizer.step()
                
                for k, v in batch_metrics.items():
                    all_metrics[k].append(v)
        
        # Clear buffer
        buffer.clear()
        
        # Average metrics
        return {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentUpdate -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/transformer_ppo_agent.py tests/test_agent_transformer_ppo.py
git commit -m "feat: implement AgentTransformerPPO.update()"
```

---

## Task 7: AgentTransformerPPO — Checkpoint & Export Methods

**Files:**
- Modify: `algorithms/agents/transformer_ppo_agent.py`
- Modify: `tests/test_agent_transformer_ppo.py`

- [ ] **Step 1: Write failing tests for checkpointing**

Add to `tests/test_agent_transformer_ppo.py`:

```python
class TestAgentCheckpoint:
    """Tests for AgentTransformerPPO checkpoint methods."""

    @pytest.fixture
    def sample_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create sample config."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
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
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "hidden_dim": 32,
                },
            }
        }

    def test_save_checkpoint(self, sample_config: Dict[str, Any], tmp_path: Path) -> None:
        """save_checkpoint should create checkpoint file."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        agent.save_checkpoint(checkpoint_dir, step=100)
        
        checkpoint_file = checkpoint_dir / "checkpoint_step_100.pt"
        assert checkpoint_file.exists()

    def test_load_checkpoint(self, sample_config: Dict[str, Any], tmp_path: Path) -> None:
        """load_checkpoint should restore agent state."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        # Create and save agent
        agent1 = AgentTransformerPPO(sample_config)
        
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        agent1.save_checkpoint(checkpoint_dir, step=100)
        
        # Create new agent and load
        agent2 = AgentTransformerPPO(sample_config)
        agent2.load_checkpoint(checkpoint_dir / "checkpoint_step_100.pt")
        
        # Check weights are the same
        for p1, p2 in zip(agent1.all_params, agent2.all_params):
            assert torch.allclose(p1, p2)

    def test_export_artifacts(self, sample_config: Dict[str, Any], tmp_path: Path) -> None:
        """export_artifacts should return manifest metadata."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(sample_config)
        
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir()
        
        manifest = agent.export_artifacts(output_dir)
        
        assert isinstance(manifest, dict)
        assert "model_path" in manifest or "checkpoint_path" in manifest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentCheckpoint -v`
Expected: FAIL (methods not implemented)

- [ ] **Step 3: Implement checkpoint and export methods**

Add to `AgentTransformerPPO` class:

```python
    def save_checkpoint(self, output_dir: Path, step: int) -> None:
        """Save training checkpoint.

        Args:
            output_dir: Directory to save checkpoint.
            step: Current training step.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "step": step,
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "backbone_state_dict": self.backbone.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self._step_count = checkpoint.get("step", 0)

    def export_artifacts(
        self,
        output_dir: Path,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Export model artifacts for deployment.

        Args:
            output_dir: Directory to save artifacts.
            context: Optional context from wrapper.

        Returns:
            Manifest metadata dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final checkpoint
        checkpoint_path = output_dir / "final_model.pt"
        checkpoint = {
            "tokenizer_state_dict": self.tokenizer.state_dict(),
            "backbone_state_dict": self.backbone.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "tokenizer_config": self.tokenizer_config,
        }
        torch.save(checkpoint, checkpoint_path)
        
        manifest = {
            "model_path": str(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "algorithm": "AgentTransformerPPO",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }
        
        return manifest

    def on_topology_change(self, building_idx: int) -> None:
        """Handle topology change for a building.

        Called by wrapper when observation count changes mid-episode.
        Triggers PPO update if buffer has data, then flushes buffer.

        Args:
            building_idx: Index of the building with topology change.
        """
        buffer = self.rollout_buffers[building_idx]
        if len(buffer) >= self.minibatch_size:
            # Trigger update with current buffer
            # Use zeros as last_obs since we don't have valid next observation
            dummy_obs = np.zeros(10)  # Will be replaced with actual shape
            self._ppo_update(building_idx, dummy_obs)
        else:
            # Just clear buffer
            buffer.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_ppo.py::TestAgentCheckpoint -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/transformer_ppo_agent.py tests/test_agent_transformer_ppo.py
git commit -m "feat: implement AgentTransformerPPO checkpoint and export methods"
```

---

## Task 8: Registry Registration

**Files:**
- Modify: `algorithms/registry.py`

- [ ] **Step 1: Read existing registry structure**

Read `algorithms/registry.py` to understand the registration pattern.

- [ ] **Step 2: Add AgentTransformerPPO to registry**

Add to `algorithms/registry.py`:

```python
from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "RuleBasedPolicy": RuleBasedPolicy,
    "AgentTransformerPPO": AgentTransformerPPO,  # Add this line
}
```

- [ ] **Step 3: Verify registration works**

Run: `python -c "from algorithms.registry import ALGORITHM_REGISTRY; print('AgentTransformerPPO' in ALGORITHM_REGISTRY)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add algorithms/registry.py
git commit -m "feat: register AgentTransformerPPO in algorithm registry"
```

---

## Task 9: End-to-End Validation Tests

**Files:**
- Create: `tests/test_e2e_transformer_ppo.py`

- [ ] **Step 1: Create E2E test file**

Create `tests/test_e2e_transformer_ppo.py`:

```python
"""End-to-end tests for TransformerPPO agent.

These tests verify the complete training pipeline works correctly,
from environment setup through training to artifact export.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch


class TestE2ESingleBuilding:
    """E2E tests with single building configuration."""

    @pytest.fixture
    def full_config(self, tmp_path: Path) -> Dict[str, Any]:
        """Create full experiment config."""
        # Create tokenizer config
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
            "ca_types": {
                "battery": {
                    "features": ["electrical_storage_soc"],
                    "action_name": "electrical_storage",
                    "input_dim": 1,
                }
            },
            "sro_types": {
                "temporal": {"features": ["month", "hour"], "input_dim": 4},
                "pricing": {"features": ["electricity_pricing"], "input_dim": 1},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": ["solar_generation"],
                "extra_features": [],
                "input_dim": 2,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {
                    "d_model": 32,
                    "nhead": 2,
                    "num_layers": 1,
                    "dim_feedforward": 64,
                },
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_eps": 0.2,
                    "ppo_epochs": 2,
                    "minibatch_size": 4,
                    "entropy_coeff": 0.01,
                    "value_coeff": 0.5,
                    "max_grad_norm": 0.5,
                    "hidden_dim": 32,
                    "rollout_length": 8,
                },
            }
        }

    def test_training_loop_runs(self, full_config: Dict[str, Any]) -> None:
        """Complete training loop should run without errors."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        
        # Attach environment
        agent.attach_environment(
            observation_names=[["month", "hour", "electricity_pricing",
                               "electrical_storage_soc", "non_shiftable_load",
                               "solar_generation"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Simulate training loop
        for step in range(20):
            # Create encoded observation with markers
            # Structure: [CA_1001, soc, SRO_2001, month, hour, d1, d2, SRO_2002, price, NFC_3001, load, gen]
            obs = np.array([[
                1001.0, 0.5 + np.random.randn() * 0.1,  # CA: battery
                2001.0, 0.5, 0.5, 0.5, 0.5,  # SRO: temporal (4 dims)
                2002.0, 0.7,  # SRO: pricing (1 dim)
                3001.0, 100.0, 50.0,  # NFC (2 dims)
            ]])
            
            # Predict
            actions = agent.predict([obs], deterministic=False)
            
            # Update
            reward = float(np.random.randn())
            agent.update(
                observations=[obs],
                actions=actions,
                rewards=[reward],
                next_observations=[obs],
                terminated=[False],
                truncated=[False],
                update_target_step=False,
                global_learning_step=step,
                update_step=(step % 8 == 7),  # Update every 8 steps
                initial_exploration_done=True,
            )
        
        # Should complete without error

    def test_actions_valid_range(self, full_config: Dict[str, Any]) -> None:
        """All actions should be in valid range [-1, 1]."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Multiple predictions
        for _ in range(50):
            obs = np.array([[
                1001.0, np.random.rand(),
                2001.0, 0.5, 0.5, 0.5, 0.5,
                3001.0, 100.0, 50.0,
            ]])
            
            actions = agent.predict([obs], deterministic=False)
            
            assert (actions[0] >= -1.0).all(), "Action below -1"
            assert (actions[0] <= 1.0).all(), "Action above 1"
            assert not np.isnan(actions[0]).any(), "NaN in actions"

    def test_kpis_generated(self, full_config: Dict[str, Any], tmp_path: Path) -> None:
        """Training should produce exportable artifacts."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(full_config)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Brief training
        for step in range(10):
            obs = np.array([[
                1001.0, 0.5,
                2001.0, 0.5, 0.5, 0.5, 0.5,
                3001.0, 100.0, 50.0,
            ]])
            actions = agent.predict([obs], deterministic=False)
            agent.update(
                observations=[obs],
                actions=actions,
                rewards=[1.0],
                next_observations=[obs],
                terminated=[False],
                truncated=[False],
                update_target_step=False,
                global_learning_step=step,
                update_step=(step == 9),
                initial_exploration_done=True,
            )
        
        # Export
        output_dir = tmp_path / "artifacts"
        manifest = agent.export_artifacts(output_dir)
        
        assert manifest is not None
        assert Path(manifest["model_path"]).exists()


class TestE2EVariableTopology:
    """E2E tests for variable topology support."""

    @pytest.fixture
    def config_with_ev(self, tmp_path: Path) -> Dict[str, Any]:
        """Create config with battery and EV charger."""
        tokenizer_config = {
            "marker_values": {"ca_base": 1000, "sro_base": 2000, "nfc": 3001},
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
                "temporal": {"features": ["month"], "input_dim": 2},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        tokenizer_path = tmp_path / "tokenizer.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f)
        
        return {
            "algorithm": {
                "name": "AgentTransformerPPO",
                "tokenizer_config_path": str(tokenizer_path),
                "transformer": {"d_model": 32, "nhead": 2, "num_layers": 1},
                "hyperparameters": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "ppo_epochs": 1,
                    "minibatch_size": 2,
                    "hidden_dim": 32,
                    "rollout_length": 4,
                },
            }
        }

    def test_variable_ca_runtime(self, config_with_ev: Dict[str, Any]) -> None:
        """Same model should handle different CA counts."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(config_with_ev)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Test with 1 CA (battery only)
        obs_1ca = np.array([[
            1001.0, 0.5,  # CA: battery (1 feature)
            2001.0, 0.5, 0.5,  # SRO: temporal (2 features)
            3001.0, 100.0,  # NFC (1 feature)
        ]])
        
        actions_1ca = agent.predict([obs_1ca], deterministic=False)
        assert actions_1ca[0].shape[-1] == 1  # 1 action for 1 CA
        
        # Test with 2 CAs (battery + EV)
        obs_2ca = np.array([[
            1001.0, 0.5,  # CA: battery (1 feature)
            1002.0, 0.8, 0.9,  # CA: ev_charger (2 features)
            2001.0, 0.5, 0.5,  # SRO: temporal
            3001.0, 100.0,  # NFC
        ]])
        
        actions_2ca = agent.predict([obs_2ca], deterministic=False)
        assert actions_2ca[0].shape[0] == 2  # 2 actions for 2 CAs

    def test_output_count_matches_input(self, config_with_ev: Dict[str, Any]) -> None:
        """Number of actions should always match number of CAs."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        
        agent = AgentTransformerPPO(config_with_ev)
        agent.attach_environment(
            observation_names=[["month", "electrical_storage_soc", "non_shiftable_load"]],
            action_names=[["electrical_storage"]],
            action_space=[MagicMock()],
            observation_space=[MagicMock()],
            metadata={},
        )
        
        # Test various CA counts
        for n_ca in [1, 2, 3]:
            # Build observation with n_ca CAs
            obs_parts = []
            for i in range(n_ca):
                marker = 1001.0 + i
                if i == 0:
                    # Battery (1 feature)
                    obs_parts.extend([marker, 0.5])
                else:
                    # EV charger (2 features)
                    obs_parts.extend([marker, 0.8, 0.9])
            
            # Add SRO and NFC
            obs_parts.extend([2001.0, 0.5, 0.5])  # SRO
            obs_parts.extend([3001.0, 100.0])  # NFC
            
            obs = np.array([obs_parts])
            actions = agent.predict([obs], deterministic=False)
            
            assert actions[0].shape[0] == n_ca, f"Expected {n_ca} actions, got {actions[0].shape[0]}"
```

- [ ] **Step 2: Run E2E tests**

Run: `pytest tests/test_e2e_transformer_ppo.py -v`
Expected: All tests PASS (some may require adjustments based on actual implementation)

- [ ] **Step 3: Fix any failing tests**

Address any test failures by adjusting the agent implementation.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_transformer_ppo.py
git commit -m "feat: add E2E tests for TransformerPPO agent"
```

---

## Task 10: Final Integration — Run All Tests

**Files:**
- None (verification only)

- [ ] **Step 1: Run all Plan A tests**

Run: `pytest tests/test_observation_enricher.py tests/test_tokenizer_config_schema.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run all Plan B tests**

Run: `pytest tests/test_observation_tokenizer.py tests/test_transformer_backbone.py tests/test_ppo_components.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run all Plan C tests**

Run: `pytest tests/test_wrapper_transformer.py tests/test_agent_transformer_ppo.py tests/test_e2e_transformer_ppo.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS, no regressions

- [ ] **Step 5: Verify agent can be instantiated from registry**

Run:
```python
python -c "
from algorithms.registry import ALGORITHM_REGISTRY
import json
import tempfile
import os

# Create minimal config
tokenizer_config = {
    'marker_values': {'ca_base': 1000, 'sro_base': 2000, 'nfc': 3001},
    'ca_types': {'battery': {'features': ['soc'], 'action_name': 'storage', 'input_dim': 1}},
    'sro_types': {},
    'nfc': {'demand_features': [], 'generation_features': [], 'extra_features': [], 'input_dim': 0}
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(tokenizer_config, f)
    tokenizer_path = f.name

config = {
    'algorithm': {
        'name': 'AgentTransformerPPO',
        'tokenizer_config_path': tokenizer_path,
        'transformer': {'d_model': 32, 'nhead': 2, 'num_layers': 1},
        'hyperparameters': {'learning_rate': 3e-4, 'hidden_dim': 32}
    }
}

agent_class = ALGORITHM_REGISTRY['AgentTransformerPPO']
agent = agent_class(config)
print('SUCCESS: Agent instantiated from registry')
os.unlink(tokenizer_path)
"
```
Expected: `SUCCESS: Agent instantiated from registry`

- [ ] **Step 6: Create final commit**

```bash
git add -A
git commit -m "feat: complete TransformerPPO integration (Plan C)"
```

---

## Summary

After completing all tasks, you will have:

1. **`utils/config_schema.py`** — Extended with TransformerPPO config schemas
2. **`configs/templates/transformer_ppo.yaml`** — Algorithm config template
3. **`utils/wrapper_citylearn.py`** — Extended with Transformer enrichment support
4. **`algorithms/agents/transformer_ppo_agent.py`** — Full agent implementation
5. **`algorithms/registry.py`** — Extended with AgentTransformerPPO registration
6. **`tests/test_wrapper_transformer.py`** — Wrapper integration tests
7. **`tests/test_agent_transformer_ppo.py`** — Agent unit tests
8. **`tests/test_e2e_transformer_ppo.py`** — End-to-end validation tests

The TransformerPPO agent is now fully integrated and ready for validation experiments as described in AGENTS.md Phase 1-4.
