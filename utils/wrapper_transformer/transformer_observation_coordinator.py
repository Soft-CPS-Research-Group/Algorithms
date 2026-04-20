"""Transformer wrapper coordination helpers.

This module keeps Transformer-specific wrapper orchestration isolated from
the generic CityLearn wrapper runtime loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def parse_marker_value(name: str) -> Optional[float]:
    """Parse a marker feature name (e.g. ``__marker_1001__``) into a float."""
    if name.startswith("__marker_") and name.endswith("__"):
        try:
            return float(name[9:-2])
        except ValueError:
            return None
    return None


class TransformerObservationCoordinator:
    """Namespace for Transformer wrapper helper functions."""

    @staticmethod
    def clear_transformer_state(wrapper: Any) -> None:
        wrapper._is_transformer_agent = False
        wrapper._enrichers = []
        wrapper._tokenizer_config = None
        wrapper._enriched_observation_names = {}
        wrapper._marker_registry_by_building = {}
        wrapper.encoders = wrapper.set_encoders()

    @staticmethod
    def configure_model_transformer_state(wrapper: Any) -> None:
        if not getattr(wrapper.model, "is_transformer_agent", False):
            TransformerObservationCoordinator.clear_transformer_state(wrapper)
            return

        tokenizer_config = getattr(wrapper.model, "tokenizer_config", None)
        if tokenizer_config is None:
            raise ValueError(
                "Transformer agent requires tokenizer_config; "
                "set model.tokenizer_config before attaching it to the wrapper."
            )

        TransformerObservationCoordinator.setup_transformer_enrichers(wrapper, tokenizer_config)
        for building_idx in range(len(wrapper.observation_names)):
            TransformerObservationCoordinator.enrich_observation_names(wrapper, building_idx)
            TransformerObservationCoordinator.rebuild_encoders_for_enriched_names(
                wrapper,
                building_idx,
            )

    @staticmethod
    def setup_transformer_enrichers(wrapper: Any, tokenizer_config: Dict[str, Any]) -> None:
        from algorithms.utils.observation_enricher import ObservationEnricher

        wrapper._is_transformer_agent = True
        wrapper._enrichers = []
        wrapper._tokenizer_config = tokenizer_config
        wrapper._marker_registry_by_building = {}

        for _ in range(len(wrapper.observation_names)):
            wrapper._enrichers.append(ObservationEnricher(tokenizer_config))

        logger.info(
            "Initialized Transformer observation enrichers for {} building(s).",
            len(wrapper._enrichers),
        )

    @staticmethod
    def enrich_observation_names(wrapper: Any, building_idx: int) -> None:
        if not getattr(wrapper, "_is_transformer_agent", False):
            return

        enricher = wrapper._enrichers[building_idx]
        if enricher is None:
            return

        obs_names = wrapper.observation_names[building_idx]
        action_names = wrapper.action_names[building_idx]
        enrichment = enricher.enrich_names(obs_names, action_names)

        if not hasattr(wrapper, "_enriched_observation_names"):
            wrapper._enriched_observation_names = {}
        wrapper._enriched_observation_names[building_idx] = enrichment.enriched_names

        marker_registry: Dict[float, Tuple[str, str, Optional[str]]] = {}
        for marker_name, marker_type in enrichment.marker_to_type.items():
            marker_value = parse_marker_value(marker_name)
            if marker_value is None:
                continue
            marker_registry[marker_value] = marker_type

        if not hasattr(wrapper, "_marker_registry_by_building"):
            wrapper._marker_registry_by_building = {}
        wrapper._marker_registry_by_building[building_idx] = marker_registry

        update_marker_registry = getattr(wrapper.model, "update_marker_registry", None)
        if callable(update_marker_registry):
            update_marker_registry(building_idx, marker_registry)
            logger.debug(
                "Updated marker registry for building {} (entries={}).",
                building_idx,
                len(marker_registry),
            )

        logger.debug(
            "Enriched observation names for building {} (raw={}, enriched={}).",
            building_idx,
            len(obs_names),
            len(enrichment.enriched_names),
        )

    @staticmethod
    def enrich_observation_values(
        wrapper: Any,
        building_idx: int,
        raw_values: List[float],
    ) -> List[float]:
        if not getattr(wrapper, "_is_transformer_agent", False):
            return raw_values

        enricher = wrapper._enrichers[building_idx]
        if enricher is None:
            return raw_values

        return enricher.enrich_values(raw_values)

    @staticmethod
    def check_topology_change(wrapper: Any, building_idx: int) -> bool:
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
        if not getattr(wrapper, "_is_transformer_agent", False):
            return

        logger.info("Handling topology change for building {}.", building_idx)
        TransformerObservationCoordinator.enrich_observation_names(wrapper, building_idx)
        TransformerObservationCoordinator.rebuild_encoders_for_enriched_names(wrapper, building_idx)

        logger.debug(
            "Rebuilt encoders for building {} after topology change (encoder_count={}).",
            building_idx,
            len(wrapper.encoders[building_idx]),
        )

        if hasattr(wrapper.model, "on_topology_change"):
            wrapper.model.on_topology_change(building_idx)
            logger.info("Notified model about topology change for building {}.", building_idx)

    @staticmethod
    def rebuild_encoders_for_enriched_names(wrapper: Any, building_idx: int) -> None:
        enriched_names = wrapper._enriched_observation_names.get(building_idx)
        if not enriched_names:
            return

        raw_names = list(wrapper.observation_names[building_idx])
        raw_space = wrapper.observation_space[building_idx]
        raw_high = np.asarray(raw_space.high, dtype=np.float64)
        raw_low = np.asarray(raw_space.low, dtype=np.float64)

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
                enriched_high.append(marker_value)
                enriched_low.append(marker_value)
                continue

            indices = name_to_indices.get(name)
            if not indices:
                raise ValueError(
                    f"Cannot rebuild encoder space for enriched feature '{name}' in building {building_idx}"
                )

            raw_index = indices.pop(0)
            enriched_high.append(float(raw_high[raw_index]))
            enriched_low.append(float(raw_low[raw_index]))

        enriched_space = type(
            "space",
            (),
            {
                "high": np.asarray(enriched_high, dtype=np.float64),
                "low": np.asarray(enriched_low, dtype=np.float64),
            },
        )()

        wrapper.encoders[building_idx] = wrapper._build_encoder_group(enriched_names, enriched_space)
