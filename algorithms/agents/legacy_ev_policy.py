"""Private compatibility controller for historical offline datasets.

Not a registered agent or a supported baseline for new jobs. Retained because
historical BC/IQL companions depend on these exact EV decisions.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithms.agents.rbc_agent import ChargerInfo, RulePolicyBase


class LegacyEVPolicy(RulePolicyBase):
    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        _ = context
        actions: List[List[float]] = []

        for agent_idx, obs in enumerate(observations):
            obs = np.asarray(obs, dtype=float)
            obs_map = self._obs_index[agent_idx] if agent_idx < len(self._obs_index) else {}
            action_names = (
                self._action_labels[agent_idx] if agent_idx < len(self._action_labels) else []
            )
            agent_actions: List[float] = []
            residual_headroom: Dict[Tuple[int, str, bool], float] = {}

            for action_position, action_name in enumerate(action_names):
                bounds = self._get_action_bounds(agent_idx, action_position)

                if self.control_storage and self._is_storage_action_name(action_name):
                    value = self._compute_storage_action(
                        agent_idx, obs, obs_map, action_name, bounds
                    )
                    value = self._apply_storage_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        self._get_storage_info(agent_idx),
                        bounds,
                        agent_idx,
                        residual_headroom,
                    )
                    value = self._apply_storage_soc_limit(value, obs, obs_map, bounds)
                elif "electric_vehicle" in action_name and self.control_evs:
                    charger_info = self._get_charger_info(agent_idx, action_position)
                    normalised = self._compute_ev_action(
                        agent_idx,
                        obs,
                        obs_map,
                        charger_info,
                        bounds,
                        action_name,
                    )
                    high = bounds[1]
                    value = normalised * high if high > 1.0 else normalised
                    value = self._apply_ev_dynamic_headroom_limit(
                        value,
                        obs,
                        obs_map,
                        charger_info,
                        bounds,
                        agent_idx,
                        residual_headroom,
                    )
                elif self.control_deferrables and self._is_deferrable_action_name(
                    action_name, obs_map
                ):
                    value = self._compute_deferrable_action(obs, obs_map, action_name, bounds)
                else:
                    value = 0.0

                value = max(bounds[0], min(bounds[1], value))
                agent_actions.append(value)

            actions.append(agent_actions)

        self.actions = actions
        return actions

    def _policy_type(self) -> str:
        if self.control_evs and self.control_deferrables:
            return "rule_based_ev_deferrable"
        if self.control_evs:
            return "rule_based_ev_only"
        if self.control_deferrables:
            return "rule_based_deferrable_only"
        return "zero_action"

    def _compute_storage_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        del agent_idx, obs, obs_map, action_name
        low, high = bounds
        return float(np.clip(0.0, low, high))

    def _compute_ev_action(
        self,
        agent_idx: int,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        charger_info: Optional[ChargerInfo],
        bounds: Sequence[float],
        action_name: str = "",
    ) -> float:
        charger_id = self._resolve_ev_charger_id(action_name, charger_info)
        building_name = self._agent_buildings.get(agent_idx)
        if charger_info is None:
            capacity = self.default_capacity
            max_power = 7.4
        else:
            capacity = charger_info.capacity or self.default_capacity
            max_power = charger_info.max_power or 7.4

        capacity = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="battery_capacity",
            generic_names=["connected_electric_vehicle_at_charger_battery_capacity"],
            default=capacity,
        )

        current_soc = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="soc",
            generic_names=["electric_vehicle_soc", "connected_electric_vehicle_at_charger_soc"],
            default=0.0,
        )
        required_soc = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="required_soc_departure",
            generic_names=[
                "electric_vehicle_required_soc_departure",
                "connected_electric_vehicle_at_charger_required_soc_departure",
            ],
            default=current_soc,
        )
        charger_state = self._get_ev_value(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            feature="connected_state",
            generic_names=[
                "electric_vehicle_charger_state",
                "electric_vehicle_charger_connected_state",
            ],
            default=0.0,
        )

        if charger_state <= 0.0:
            return 0.0

        soc_gap_fraction = self._soc_gap_fraction(current_soc, required_soc)
        energy_needed = soc_gap_fraction * capacity
        if energy_needed <= self.energy_epsilon:
            return 0.0

        time_to_departure = self._get_ev_departure_hours(
            obs,
            obs_map,
            charger_id=charger_id,
            building_name=building_name,
            default=self.flexibility_hours,
        )
        time_to_departure = max(time_to_departure, self.step_hours)

        flex_flag = self._get_value(obs, obs_map, "electric_vehicle_is_flexible", default=1.0)
        is_flexible = flex_flag > 0.5 and (
            not charger_info or charger_info.charger_id not in self.non_flexible_chargers
        )

        solar_generation = self._get_value(obs, obs_map, "pv_power_kw", default=float("nan"))
        if math.isnan(solar_generation):
            solar_generation = self._get_value(
                obs, obs_map, "solar_generation", default=float("nan")
            )
        if math.isnan(solar_generation):
            solar_generation = self._get_value(obs, obs_map, "electricity_generation", default=0.0)
        solar_generation = max(0.0, solar_generation)

        required_power = energy_needed / time_to_departure
        if max_power <= 0:
            max_power = 7.4

        normalised = min(1.0, required_power / max_power)

        if solar_generation >= self.pv_charge_threshold:
            normalised = max(normalised, self.pv_preferred_charge_rate)

        if time_to_departure <= self.emergency_hours or not is_flexible:
            normalised = max(normalised, self.emergency_charge_rate)

        elif (
            is_flexible
            and solar_generation < self.pv_charge_threshold
            and time_to_departure > self.flexibility_hours
        ):
            normalised = max(self.flex_trickle_charge, min(normalised, self.flex_trickle_charge))

        if normalised > 0:
            normalised = max(normalised, self.min_charge_rate)

        low, high = bounds
        normalised = float(
            np.clip(normalised, 0.0 if low <= 0 else low, 1.0 if high >= 1 else high)
        )
        return normalised

    def _compute_deferrable_action(
        self,
        obs: np.ndarray,
        obs_map: Dict[str, int],
        action_name: str,
        bounds: Sequence[float],
    ) -> float:
        prefix = self._resolve_deferrable_observation_prefix(action_name, obs_map)
        if prefix is None:
            return 0.0

        pending = self._get_deferrable_value(obs, obs_map, prefix, "pending", default=0.0)
        running = self._get_deferrable_value(obs, obs_map, prefix, "running", default=0.0)
        can_start = self._get_deferrable_value(obs, obs_map, prefix, "can_start", default=0.0)
        deadline_missed = self._get_deferrable_value(
            obs, obs_map, prefix, "deadline_missed", default=0.0
        )

        if pending <= 0.5 or running > 0.5 or can_start <= 0.5 or deadline_missed > 0.5:
            return 0.0

        urgency = self._get_deferrable_value(obs, obs_map, prefix, "urgency_ratio", default=0.0)
        slack = self._get_deferrable_value(obs, obs_map, prefix, "slack_ratio", default=1.0)
        priority = self._get_deferrable_value(obs, obs_map, prefix, "priority", default=0.0)

        should_start = (
            urgency >= self.deferrable_urgency_threshold
            or slack <= self.deferrable_slack_threshold
            or priority >= self.deferrable_priority_threshold
        )
        if not should_start:
            return 0.0

        low, high = bounds
        return float(np.clip(self.deferrable_start_action, low, high))
