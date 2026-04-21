"""Transformer wrapper coordination helpers.

This module keeps Transformer-specific wrapper orchestration isolated from
the generic CityLearn wrapper runtime loop.

Responsibilities:
- Lifecycle: Initialize/clear transformer state on wrapper
- Enrichment: Inject markers into observation names and values  
- Registry Sync: Keep model's marker registry in sync with wrapper
- Topology: Detect and handle observation topology changes
- Encoders: Rebuild encoders when topology changes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from algorithms.utils.observation_enricher import ObservationEnricher


def parse_marker_value(name: str) -> Optional[float]:
    """Parse a marker feature name (e.g. ``__marker_1001__``) into a float."""
    if name.startswith("__marker_") and name.endswith("__"):
        try:
            return float(name[9:-2])
        except ValueError:
            return None
    return None


class TransformerObservationCoordinator:
    """Namespace for Transformer wrapper helper functions.
    
    All methods are static and operate on wrapper instances. This keeps
    transformer-specific logic isolated from the main wrapper class.
    """

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    @staticmethod
    def configure_model_transformer_state(wrapper: Any) -> None:
        """Configure wrapper for transformer agent, or clear state if not transformer.
        
        This is the main entry point called from wrapper.set_model().
        """
        if not getattr(wrapper.model, "is_transformer_agent", False):
            TransformerObservationCoordinator._clear_transformer_state(wrapper)
            logger.debug("Model is not a transformer agent; cleared transformer state.")
            return

        tokenizer_config = getattr(wrapper.model, "tokenizer_config", None)
        if tokenizer_config is None:
            raise ValueError(
                "Transformer agent requires tokenizer_config; "
                "set model.tokenizer_config before attaching it to the wrapper."
            )

        TransformerObservationCoordinator._initialize_enrichers(wrapper, tokenizer_config)
        
        for building_idx in range(len(wrapper.observation_names)):
            TransformerObservationCoordinator._enrich_and_rebuild(wrapper, building_idx)
        
        logger.info(
            "Configured transformer state for {} building(s).",
            len(wrapper.observation_names),
        )

    @staticmethod
    def _clear_transformer_state(wrapper: Any) -> None:
        """Reset all transformer-related state on wrapper."""
        wrapper._is_transformer_agent = False
        wrapper._enrichers = []
        wrapper._tokenizer_config = None
        wrapper._enriched_observation_names = {}
        wrapper._marker_registry_by_building = {}
        wrapper.encoders = wrapper.set_encoders()

    @staticmethod
    def _initialize_enrichers(wrapper: Any, tokenizer_config: Dict[str, Any]) -> None:
        """Initialize enrichers for all buildings."""
        wrapper._is_transformer_agent = True
        wrapper._enrichers = []
        wrapper._tokenizer_config = tokenizer_config
        wrapper._marker_registry_by_building = {}
        wrapper._enriched_observation_names = {}

        for _ in range(len(wrapper.observation_names)):
            wrapper._enrichers.append(ObservationEnricher(tokenizer_config))

        logger.debug(
            "Initialized {} observation enricher(s).",
            len(wrapper._enrichers),
        )

    # -------------------------------------------------------------------------
    # Enrichment & Registry Sync
    # -------------------------------------------------------------------------

    @staticmethod
    def _enrich_and_rebuild(wrapper: Any, building_idx: int) -> None:
        """Enrich observation names and rebuild encoders for a building."""
        TransformerObservationCoordinator._enrich_observation_names(wrapper, building_idx)
        TransformerObservationCoordinator._rebuild_encoders(wrapper, building_idx)

    @staticmethod
    def _enrich_observation_names(wrapper: Any, building_idx: int) -> None:
        """Enrich observation names and sync marker registry with model."""
        enricher = wrapper._enrichers[building_idx]
        if enricher is None:
            return

        obs_names = wrapper.observation_names[building_idx]
        action_names = wrapper.action_names[building_idx]
        enrichment_result = enricher.enrich_names(obs_names, action_names)

        # Store enriched names
        wrapper._enriched_observation_names[building_idx] = enrichment_result.enriched_names

        # Build marker registry (marker_value -> (family, type, device_id))
        marker_registry = TransformerObservationCoordinator._build_marker_registry(
            enrichment_result.marker_to_type
        )
        wrapper._marker_registry_by_building[building_idx] = marker_registry

        # Sync with model if it supports registry updates
        TransformerObservationCoordinator._sync_model_registry(
            wrapper.model, building_idx, marker_registry
        )

        logger.debug(
            "Enriched building {} (raw={}, enriched={}, markers={}).",
            building_idx,
            len(obs_names),
            len(enrichment_result.enriched_names),
            len(marker_registry),
        )

    @staticmethod
    def _build_marker_registry(
        marker_to_type: Dict[str, Tuple[str, str, Optional[str]]]
    ) -> Dict[float, Tuple[str, str, Optional[str]]]:
        """Convert marker names to numeric registry."""
        registry: Dict[float, Tuple[str, str, Optional[str]]] = {}
        for marker_name, marker_type in marker_to_type.items():
            marker_value = parse_marker_value(marker_name)
            if marker_value is not None:
                registry[marker_value] = marker_type
        return registry

    @staticmethod
    def _sync_model_registry(
        model: Any,
        building_idx: int,
        marker_registry: Dict[float, Tuple[str, str, Optional[str]]],
    ) -> None:
        """Sync marker registry with model if it supports updates."""
        update_fn = getattr(model, "update_marker_registry", None)
        if callable(update_fn):
            update_fn(building_idx, marker_registry)

    @staticmethod
    def enrich_observation_values(
        wrapper: Any,
        building_idx: int,
        raw_values: List[float],
    ) -> List[float]:
        """Inject marker values into raw observation values.
        
        Called during observation encoding to add marker tokens.
        """
        if not getattr(wrapper, "_is_transformer_agent", False):
            return raw_values

        enricher = wrapper._enrichers[building_idx]
        if enricher is None:
            return raw_values

        return enricher.enrich_values(raw_values)

    # -------------------------------------------------------------------------
    # Topology Change Handling
    # -------------------------------------------------------------------------

    @staticmethod
    def check_topology_change(wrapper: Any, building_idx: int) -> bool:
        """Check if observation topology has changed for a building."""
        if not getattr(wrapper, "_is_transformer_agent", False):
            return False

        enricher = wrapper._enrichers[building_idx]
        if enricher is None:
            return False

        obs_names = wrapper.observation_names[building_idx]
        action_names = wrapper.action_names[building_idx]
        return enricher.topology_changed(obs_names, action_names)

    @staticmethod
    def handle_topology_change(wrapper: Any, building_idx: int) -> None:
        """Handle topology change by re-enriching and rebuilding encoders."""
        if not getattr(wrapper, "_is_transformer_agent", False):
            return

        logger.info("Handling topology change for building {}.", building_idx)
        TransformerObservationCoordinator._enrich_and_rebuild(wrapper, building_idx)

        # Notify model if it has a topology change handler
        if hasattr(wrapper.model, "on_topology_change"):
            wrapper.model.on_topology_change(building_idx)
            logger.debug("Notified model about topology change for building {}.", building_idx)

    # -------------------------------------------------------------------------
    # Encoder Rebuilding
    # -------------------------------------------------------------------------

    @staticmethod
    def _rebuild_encoders(wrapper: Any, building_idx: int) -> None:
        """Rebuild encoders for enriched observation names."""
        enriched_names = wrapper._enriched_observation_names.get(building_idx)
        if not enriched_names:
            return

        enriched_space = TransformerObservationCoordinator._build_enriched_space(
            wrapper, building_idx, enriched_names
        )
        wrapper.encoders[building_idx] = wrapper._build_encoder_group(
            enriched_names, enriched_space
        )

        logger.debug(
            "Rebuilt encoders for building {} (count={}).",
            building_idx,
            len(wrapper.encoders[building_idx]),
        )

    @staticmethod
    def _build_enriched_space(
        wrapper: Any,
        building_idx: int,
        enriched_names: List[str],
    ) -> Any:
        """Build observation space for enriched names (including markers)."""
        raw_names = list(wrapper.observation_names[building_idx])
        raw_space = wrapper.observation_space[building_idx]
        raw_high = np.asarray(raw_space.high, dtype=np.float64)
        raw_low = np.asarray(raw_space.low, dtype=np.float64)

        # Map names to their indices (handles duplicates)
        name_to_indices: Dict[str, List[int]] = {}
        for index, name in enumerate(raw_names):
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(index)

        enriched_high: List[float] = []
        enriched_low: List[float] = []

        for name in enriched_names:
            marker_value = parse_marker_value(name)
            if marker_value is not None:
                # Marker: use marker value as both high and low
                enriched_high.append(marker_value)
                enriched_low.append(marker_value)
            else:
                # Regular feature: look up bounds from raw space
                indices = name_to_indices.get(name)
                if not indices:
                    raise ValueError(
                        f"Cannot find bounds for enriched feature '{name}' "
                        f"in building {building_idx}"
                    )
                raw_index = indices.pop(0)
                enriched_high.append(float(raw_high[raw_index]))
                enriched_low.append(float(raw_low[raw_index]))

        return type(
            "space",
            (),
            {
                "high": np.asarray(enriched_high, dtype=np.float64),
                "low": np.asarray(enriched_low, dtype=np.float64),
            },
        )()


# Backwards compatibility aliases
TransformerObservationCoordinator.clear_transformer_state = (
    TransformerObservationCoordinator._clear_transformer_state
)
TransformerObservationCoordinator.setup_transformer_enrichers = (
    TransformerObservationCoordinator._initialize_enrichers
)
TransformerObservationCoordinator.enrich_observation_names = (
    TransformerObservationCoordinator._enrich_observation_names
)
TransformerObservationCoordinator.rebuild_encoders_for_enriched_names = (
    TransformerObservationCoordinator._rebuild_encoders
)
