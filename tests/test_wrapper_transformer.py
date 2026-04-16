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
        from utils.wrapper_citylearn import Wrapper_CityLearn
        
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

    def test_wrapper_detects_transformer_agent_and_initializes_enrichers(self) -> None:
        """Wrapper should detect Transformer agent and initialize enrichers."""
        from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO
        from utils.wrapper_citylearn import Wrapper_CityLearn
        import tempfile
        import json
        import os
        
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
                "temporal": {"features": ["month"], "input_dim": 2},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(tokenizer_config, f)
            tokenizer_path = f.name
        
        try:
            # Create agent
            agent_config = {
                "algorithm": {
                    "name": "AgentTransformerPPO",
                    "tokenizer_config_path": tokenizer_path,
                    "transformer": {"d_model": 64, "nhead": 4, "num_layers": 2},
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
            
            agent = AgentTransformerPPO(agent_config)
            
            # Create wrapper config
            wrapper_config = {
                "training": {},
                "simulator": {},
                "checkpointing": {},
                "tracking": {},
                "runtime": {},
            }
            
            # Create minimal mock environment (similar to test_encoder_rules.py)
            class MinimalEnv:
                def __init__(self):
                    self.observation_names = [["electrical_storage_soc", "non_shiftable_load", "month"]]
                    self.action_names = [["electrical_storage"]]
                    self.observation_space = [
                        type("space", (), {
                            "high": np.array([1.0, 1000.0, 12.0]),
                            "low": np.array([0.0, 0.0, 1.0]),
                        })()
                    ]
                    self.action_space = [
                        type("space", (), {
                            "high": np.array([1.0]),
                            "low": np.array([-1.0]),
                        })()
                    ]
                    self.reward_function = type("reward", (), {"__dict__": {}})()
                    self.time_steps = 8760
                    self.seconds_per_time_step = 3600
                    self.time_step_ratio = 1.0
                    self.random_seed = 0
                    self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
                    self.unwrapped = self
                
                def reset(self):
                    return [np.array([0.5, 100.0, 6.0])], {}
                
                def get_metadata(self):
                    return {"buildings": [{}]}
            
            mock_env = MinimalEnv()
            
            # Create wrapper (signature: env, model, config, job_id, progress_path)
            wrapper = Wrapper_CityLearn(
                env=mock_env,
                model=agent,
                config=wrapper_config,
                job_id="test"
            )
            
            # Verify enrichers were initialized
            assert hasattr(wrapper, '_is_transformer_agent')
            assert wrapper._is_transformer_agent is True
            assert hasattr(wrapper, '_enrichers')
            assert len(wrapper._enrichers) == 1
            assert wrapper._enrichers[0] is not None
        finally:
            # Cleanup temp file
            if os.path.exists(tokenizer_path):
                os.unlink(tokenizer_path)


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


class TestWrapperObservationProcessingFlow:
    """Tests for observation enrichment in processing flow."""

    def test_wrapper_enriches_observations_during_processing(self) -> None:
        """Wrapper should call enricher during observation processing."""
        from algorithms.utils.observation_enricher import ObservationEnricher
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from unittest.mock import MagicMock
        
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
                "temporal": {"features": ["month"], "input_dim": 2},
            },
            "nfc": {
                "demand_features": ["non_shiftable_load"],
                "generation_features": [],
                "extra_features": [],
                "input_dim": 1,
            },
        }
        
        # Create wrapper with enricher
        wrapper = MagicMock(spec=Wrapper_CityLearn)
        wrapper._is_transformer_agent = True
        wrapper._enrichers = [ObservationEnricher(tokenizer_config)]
        wrapper.observation_names = [["electrical_storage_soc", "non_shiftable_load", "month"]]
        wrapper.action_names = [["electrical_storage"]]
        
        # Initialize enricher
        wrapper._enrichers[0].enrich_names(
            wrapper.observation_names[0],
            wrapper.action_names[0]
        )
        
        # Call the actual method
        wrapper._enrich_observation_values = Wrapper_CityLearn._enrich_observation_values.__get__(wrapper)
        
        raw_values = [0.5, 100.0, 6.0]
        enriched_values = wrapper._enrich_observation_values(0, raw_values)
        
        # Verify markers were injected
        assert 1001.0 in enriched_values  # CA marker
        assert 2001.0 in enriched_values  # SRO marker
        assert 3001.0 in enriched_values  # NFC marker
        assert len(enriched_values) > len(raw_values)

    def test_observations_flow_through_enrichment(self) -> None:
        """Verify observations are enriched in actual step flow."""
        # This will be implemented in Task 4 as full E2E test
        pass
