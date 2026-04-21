"""State-management helpers for AgentTransformerPPO."""

from __future__ import annotations

from typing import Dict, Optional

from algorithms.utils.ppo_components import RolloutBuffer


class TransformerPPOStateHelper:
    """Helper namespace for per-building agent state."""

    @staticmethod
    def initialize_environment_state(
        agent,
        observation_names,
        action_names,
    ) -> None:
        agent._num_buildings = len(observation_names)
        agent.observation_names = observation_names
        agent.action_names = action_names

        agent.rollout_buffers = [
            RolloutBuffer(gamma=agent.gamma, gae_lambda=agent.gae_lambda)
            for _ in range(agent._num_buildings)
        ]

        agent._last_values = [None] * agent._num_buildings
        agent._last_log_probs = [None] * agent._num_buildings
        agent._last_obs = [None] * agent._num_buildings
        agent._last_actions = [None] * agent._num_buildings

        previous_marker_registry = list(agent._marker_registry_by_building)
        agent._marker_registry_by_building = [dict() for _ in range(agent._num_buildings)]
        for idx in range(min(len(previous_marker_registry), agent._num_buildings)):
            if previous_marker_registry[idx]:
                agent._marker_registry_by_building[idx] = dict(previous_marker_registry[idx])

    @staticmethod
    def update_marker_registry(
        agent,
        building_idx: int,
        marker_registry: Dict[float, tuple[str, str, Optional[str]]],
    ) -> None:
        if building_idx < 0:
            return

        if building_idx >= len(agent._marker_registry_by_building):
            agent._marker_registry_by_building.extend(
                {} for _ in range(building_idx + 1 - len(agent._marker_registry_by_building))
            )

        agent._marker_registry_by_building[building_idx] = dict(marker_registry)

    @staticmethod
    def marker_registry_for_building(
        agent,
        building_idx: int,
    ) -> Optional[Dict[float, tuple[str, str, Optional[str]]]]:
        if 0 <= building_idx < len(agent._marker_registry_by_building):
            marker_registry = agent._marker_registry_by_building[building_idx]
            if marker_registry:
                return marker_registry
        return None
