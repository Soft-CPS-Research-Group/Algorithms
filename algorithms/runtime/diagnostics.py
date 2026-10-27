"""Shared diagnostics service for the simulation controller; no agent-specific training."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


class Diagnostics:
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed

    @staticmethod
    def _metric_safe_name(value: Any) -> str:
        text = str(value or "unknown")
        safe = [char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text]
        return "".join(safe).strip("_") or "unknown"

    @staticmethod
    def _is_deferrable_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "deferrable_appliance" in raw or raw.endswith("::start") or raw == "start"

    @staticmethod
    def _is_ev_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "electric_vehicle_storage" in raw or "ev_storage" in raw

    @staticmethod
    def _charger_token_from_action_name(action_name: str) -> str | None:
        match = re.search(r"charger_\d+_\d+", str(action_name or "").lower())
        return match.group(0) if match else None

    @staticmethod
    def _is_storage_action_name(action_name: str) -> bool:
        raw = str(action_name or "").lower()
        return "electrical_storage" in raw or raw in {"battery", "storage"}

    def _action_bounds_for_agent(
        self, agent_index: int, action_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if agent_index < len(getattr(self, "action_space", [])):
            action_space = self.action_space[agent_index]
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
                high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
                if low.shape[0] == action_count and high.shape[0] == action_count:
                    return low, high
        return (
            np.full(action_count, -1.0, dtype=np.float64),
            np.full(action_count, 1.0, dtype=np.float64),
        )

    def _ev_connected_for_action(
        self,
        *,
        agent_index: int,
        observation: np.ndarray,
        action_name: str,
    ) -> Optional[bool]:
        charger_token = self._charger_token_from_action_name(action_name)
        if not charger_token:
            return None
        names = (
            [str(name).lower() for name in self.observation_names[agent_index]]
            if agent_index < len(getattr(self, "observation_names", []))
            else []
        )
        if not names:
            return None
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        for obs_index, obs_name in enumerate(names[: values.shape[0]]):
            if charger_token not in obs_name:
                continue
            if obs_name.endswith("::connected_state") or obs_name.endswith("__connected_state"):
                return bool(values[obs_index] > 0.5)
        return None

    def _build_action_diagnostic_metrics(
        self,
        actions: List[List[float]],
        observations: List[np.ndarray],
    ) -> Dict[str, float]:
        if not getattr(self, "action_diagnostics_enabled", False):
            return {}

        rows: List[Dict[str, Any]] = []
        deferrable_start_delays: List[float] = []
        deferrable_pending_can_start_count = 0
        deferrable_start_command_count = 0
        deferrable_start_when_available_count = 0

        for agent_index, agent_actions in enumerate(actions):
            action_values = np.asarray(agent_actions, dtype=np.float64).reshape(-1)
            low, high = self._action_bounds_for_agent(agent_index, action_values.shape[0])
            span = np.maximum(high - low, 1.0e-9)
            names = (
                [str(name) for name in self.action_names[agent_index]]
                if agent_index < len(getattr(self, "action_names", []))
                else []
            )

            for action_index, value in enumerate(action_values):
                action_name = (
                    names[action_index] if action_index < len(names) else f"action_{action_index}"
                )
                action_low = float(low[action_index])
                action_high = float(high[action_index])
                action_span = float(span[action_index])
                threshold = action_low + 0.5 * action_span
                category = "other"
                if self._is_deferrable_action_name(action_name):
                    category = "deferrable"
                    threshold = action_low + 0.5 * action_span
                    if getattr(self.model, "deferrable_trigger_threshold", None) is not None:
                        threshold = (
                            action_low
                            + float(self.model.deferrable_trigger_threshold) * action_span
                        )
                    if value > threshold:
                        deferrable_start_command_count += 1
                    pending, can_start = self._deferrable_pending_can_start(
                        agent_index=agent_index,
                        observation=observations[agent_index]
                        if agent_index < len(observations)
                        else np.array([]),
                        action_name=action_name,
                    )
                    if pending and can_start:
                        deferrable_pending_can_start_count += 1
                        if value > threshold:
                            deferrable_start_when_available_count += 1
                            key = (agent_index, action_name)
                            deferrable_start_delays.append(
                                float(self._deferrable_wait_steps.get(key, 0))
                            )
                            self._deferrable_wait_steps.pop(key, None)
                        else:
                            key = (agent_index, action_name)
                            self._deferrable_wait_steps[key] = (
                                self._deferrable_wait_steps.get(key, 0) + 1
                            )
                    else:
                        self._deferrable_wait_steps.pop((agent_index, action_name), None)
                elif self._is_ev_action_name(action_name):
                    category = "ev"
                elif self._is_storage_action_name(action_name):
                    category = "storage"
                ev_connected = (
                    self._ev_connected_for_action(
                        agent_index=agent_index,
                        observation=observations[agent_index]
                        if agent_index < len(observations)
                        else np.array([]),
                        action_name=action_name,
                    )
                    if category == "ev"
                    else None
                )

                rows.append(
                    {
                        "agent_index": agent_index,
                        "action_index": action_index,
                        "action_name": action_name,
                        "category": category,
                        "ev_connected": ev_connected,
                        "value": float(value),
                        "low": action_low,
                        "high": action_high,
                        "span": action_span,
                        "threshold": threshold,
                    }
                )

        if not rows:
            return {}

        values = np.asarray([row["value"] for row in rows], dtype=np.float64)
        lows = np.asarray([row["low"] for row in rows], dtype=np.float64)
        highs = np.asarray([row["high"] for row in rows], dtype=np.float64)
        spans = np.maximum(highs - lows, 1.0e-9)
        saturation_tolerance = np.maximum(self.action_saturation_tolerance * spans, 1.0e-9)

        metrics: Dict[str, float] = {
            "Action/all_count": float(values.shape[0]),
            "Action/all_mean": float(np.mean(values)),
            "Action/all_std": float(np.std(values)),
            "Action/all_min": float(np.min(values)),
            "Action/all_max": float(np.max(values)),
            "Action/near_low_fraction": float(np.mean(values <= lows + saturation_tolerance)),
            "Action/near_high_fraction": float(np.mean(values >= highs - saturation_tolerance)),
            "Action/near_zero_fraction": float(
                np.mean(np.abs(values) <= self.action_idle_tolerance)
            ),
        }

        for category in ("storage", "ev", "deferrable"):
            category_rows = [row for row in rows if row["category"] == category]
            if not category_rows:
                continue
            category_values = np.asarray([row["value"] for row in category_rows], dtype=np.float64)
            metrics[f"Action/{category}_count"] = float(category_values.shape[0])
            metrics[f"Action/{category}_mean"] = float(np.mean(category_values))
            metrics[f"Action/{category}_std"] = float(np.std(category_values))
            if category in {"storage", "ev"}:
                metrics[f"Action/{category}_positive_fraction"] = float(
                    np.mean(category_values > self.action_idle_tolerance)
                )
                metrics[f"Action/{category}_negative_fraction"] = float(
                    np.mean(category_values < -self.action_idle_tolerance)
                )
                metrics[f"Action/{category}_idle_fraction"] = float(
                    np.mean(np.abs(category_values) <= self.action_idle_tolerance)
                )
                if category == "ev":
                    connected_rows = [
                        row for row in category_rows if row.get("ev_connected") is True
                    ]
                    disconnected_rows = [
                        row for row in category_rows if row.get("ev_connected") is False
                    ]
                    for prefix, subset_rows in (
                        ("ev_connected", connected_rows),
                        ("ev_disconnected", disconnected_rows),
                    ):
                        if not subset_rows:
                            continue
                        subset_values = np.asarray(
                            [row["value"] for row in subset_rows], dtype=np.float64
                        )
                        metrics[f"Action/{prefix}_count"] = float(subset_values.shape[0])
                        metrics[f"Action/{prefix}_mean"] = float(np.mean(subset_values))
                        metrics[f"Action/{prefix}_std"] = float(np.std(subset_values))
                        metrics[f"Action/{prefix}_positive_fraction"] = float(
                            np.mean(subset_values > self.action_idle_tolerance)
                        )
                        metrics[f"Action/{prefix}_negative_fraction"] = float(
                            np.mean(subset_values < -self.action_idle_tolerance)
                        )
                        metrics[f"Action/{prefix}_idle_fraction"] = float(
                            np.mean(np.abs(subset_values) <= self.action_idle_tolerance)
                        )
            elif category == "deferrable":
                thresholds = np.asarray(
                    [row["threshold"] for row in category_rows], dtype=np.float64
                )
                metrics["Action/deferrable_on_fraction"] = float(
                    np.mean(category_values > thresholds)
                )
                metrics["Action/deferrable_off_fraction"] = float(
                    np.mean(category_values <= thresholds)
                )

        metrics["Deferrable/pending_can_start_count"] = float(deferrable_pending_can_start_count)
        metrics["Deferrable/start_command_count"] = float(deferrable_start_command_count)
        metrics["Deferrable/start_when_available_count"] = float(
            deferrable_start_when_available_count
        )
        if deferrable_start_delays:
            delays = np.asarray(deferrable_start_delays, dtype=np.float64)
            metrics["Deferrable/start_delay_steps_mean"] = float(np.mean(delays))
            metrics["Deferrable/start_delay_steps_max"] = float(np.max(delays))

        if self.action_diagnostics_detail == "per_action":
            for row in rows:
                prefix = (
                    "Action/"
                    f"agent_{row['agent_index']}/"
                    f"{row['action_index']}_{self._metric_safe_name(row['action_name'])}"
                )
                span = max(float(row["span"]), 1.0e-9)
                tolerance = max(self.action_saturation_tolerance * span, 1.0e-9)
                metrics[f"{prefix}/value"] = float(row["value"])
                metrics[f"{prefix}/normalized"] = float((row["value"] - row["low"]) / span)
                metrics[f"{prefix}/near_low"] = float(row["value"] <= row["low"] + tolerance)
                metrics[f"{prefix}/near_high"] = float(row["value"] >= row["high"] - tolerance)

        return metrics

    def _deferrable_pending_can_start(
        self,
        *,
        agent_index: int,
        observation: np.ndarray,
        action_name: str,
    ) -> tuple[bool, bool]:
        asset_id = self._deferrable_asset_id(action_name)
        if not asset_id:
            return False, False

        lookup = self._build_observation_lookup(agent_index, observation)
        pending = self._find_deferrable_observation(lookup, asset_id, "pending")
        can_start = self._find_deferrable_observation(lookup, asset_id, "can_start")
        return pending > 0.5, can_start > 0.5

    @staticmethod
    def _deferrable_asset_id(action_name: str) -> Optional[str]:
        raw = str(action_name or "")
        if raw.startswith("deferrable_appliance_"):
            return raw[len("deferrable_appliance_") :]
        if "::" in raw:
            parts = raw.split("::")
            if len(parts) >= 2:
                asset = parts[-2].split("/")[-1]
                return asset or None
        return None

    @staticmethod
    def _find_deferrable_observation(
        observation_lookup: Dict[str, float],
        asset_id: str,
        signal: str,
    ) -> float:
        for name, value in observation_lookup.items():
            if "deferrable_appliance" not in name:
                continue
            if asset_id not in name:
                continue
            if name.endswith(f"::{signal}") or name.endswith(signal):
                return value
        return 0.0

    def _collect_model_status_metrics(self) -> Dict[str, float]:
        hook = getattr(self.model, "get_diagnostic_metrics", None)
        if callable(hook):
            return {str(key): self._safe_float(value) for key, value in hook().items()}
        return {}

    def _consume_model_training_metrics(self) -> Dict[str, float]:
        hook = getattr(self.model, "consume_latest_training_metrics", None)
        if callable(hook):
            return {str(key): self._safe_float(value) for key, value in hook().items()}
        return {}

    def _build_reward_component_metrics(self) -> Dict[str, float]:
        if not getattr(self, "reward_diagnostics_enabled", True):
            return {}

        reward_fn = getattr(self.env, "reward_function", None)
        if reward_fn is None and hasattr(self.env, "unwrapped"):
            reward_fn = getattr(self.env.unwrapped, "reward_function", None)
        if reward_fn is None:
            return {}

        components_hook = getattr(reward_fn, "get_last_components", None)
        if callable(components_hook):
            components = components_hook()
        else:
            components = {
                "per_agent": getattr(reward_fn, "last_components_by_agent", []),
                "community": getattr(reward_fn, "last_community_components", {}),
            }
        if not isinstance(components, Mapping):
            return {}

        metrics: Dict[str, float] = {}
        per_agent = components.get("per_agent", [])
        if isinstance(per_agent, list) and per_agent:
            numeric_keys = sorted(
                {
                    str(key)
                    for row in per_agent
                    if isinstance(row, Mapping)
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                }
            )
            for key in numeric_keys:
                values = [
                    self._safe_float(row.get(key), default=0.0)
                    for row in per_agent
                    if isinstance(row, Mapping)
                ]
                if not values:
                    continue
                safe_key = self._metric_safe_name(key)
                array = np.asarray(values, dtype=np.float64)
                metrics[f"RewardComponent/{safe_key}_mean"] = float(np.mean(array))
                metrics[f"RewardComponent/{safe_key}_sum"] = float(np.sum(array))
                if self.reward_diagnostics_detail == "per_agent":
                    for agent_index, value in enumerate(array):
                        metrics[f"RewardComponent/agent_{agent_index}/{safe_key}"] = float(value)

        community = components.get("community", {})
        if isinstance(community, Mapping):
            for key, value in community.items():
                if not isinstance(value, (int, float)):
                    continue
                metrics[f"RewardComponent/community/{self._metric_safe_name(key)}"] = (
                    self._safe_float(value)
                )

        return metrics

    def _get_reward_components_payload(self) -> Mapping[str, Any]:
        """Return the unflattened reward evidence for typed transition traces."""

        reward_fn = getattr(self.env, "reward_function", None)
        if reward_fn is None and hasattr(self.env, "unwrapped"):
            reward_fn = getattr(self.env.unwrapped, "reward_function", None)
        if reward_fn is None:
            return {}
        components_hook = getattr(reward_fn, "get_last_components", None)
        if callable(components_hook):
            components = components_hook()
        else:
            components = {
                "per_agent": getattr(reward_fn, "last_components_by_agent", []),
                "community": getattr(reward_fn, "last_community_components", {}),
            }
        return deepcopy(components) if isinstance(components, Mapping) else {}

    def _build_observation_lookup(
        self, agent_index: int, observation: List[float]
    ) -> Dict[str, float]:
        names: List[str] = []
        if agent_index < len(self.observation_names):
            names = [str(name) for name in self.observation_names[agent_index]]
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        return {name: self._safe_float(value) for name, value in zip(names, values)}

    @staticmethod
    def _extract_signal(
        observation_lookup: Dict[str, float], candidates: List[str], default: float = 0.0
    ) -> float:
        for key in candidates:
            if key in observation_lookup:
                return observation_lookup[key]
        return default

    def _shape_rewards(self, rewards: List[float], observations: List[List[float]]) -> List[float]:
        if not self.wrapper_reward_enabled:
            return [self._safe_float(reward) for reward in rewards]

        profile = self.wrapper_reward_profile_config
        enabled_terms = profile.get("enabled_terms", {})
        weights = profile.get("weights", {})
        params = profile.get("params", {})

        observation_lookup = [
            self._build_observation_lookup(agent_index=i, observation=observation)
            for i, observation in enumerate(observations)
        ]

        community_net_consumption = sum(
            self._extract_signal(
                values,
                ["net_electricity_consumption", "net_electricity_consumption_without_storage"],
                default=0.0,
            )
            for values in observation_lookup
        )
        community_export_bonus_ratio = self._safe_float(
            params.get("community_export_bonus_ratio"), default=0.2
        )
        community_term = -abs(community_net_consumption) + (
            max(-community_net_consumption, 0.0) * community_export_bonus_ratio
        )

        shaped_rewards: List[float] = []
        export_credit_ratio = self._safe_float(params.get("export_credit_ratio"), default=0.8)
        ev_soc_tolerance = max(self._safe_float(params.get("ev_soc_tolerance"), default=0.1), 1e-6)

        for i, reward in enumerate(rewards):
            base_reward = self._safe_float(reward)
            obs_values = observation_lookup[i] if i < len(observation_lookup) else {}

            net_consumption = self._extract_signal(
                obs_values,
                ["net_electricity_consumption", "net_electricity_consumption_without_storage"],
                default=0.0,
            )
            electricity_price = max(
                self._extract_signal(
                    obs_values,
                    ["electricity_pricing", "electricity_price", "electricity_tariff"],
                    default=0.0,
                ),
                0.0,
            )
            import_cost = max(net_consumption, 0.0) * electricity_price
            export_credit = max(-net_consumption, 0.0) * electricity_price * export_credit_ratio
            energy_cost_term = -(import_cost - export_credit)

            grid_violation = self._extract_signal(
                obs_values,
                [
                    "electrical_service_violation",
                    "electrical_service_violation_kwh",
                    "service_violation",
                    "service_violation_kwh",
                ],
                default=0.0,
            )
            grid_violation_term = -max(grid_violation, 0.0)

            ev_success_signal = self._extract_signal(
                obs_values,
                [
                    "ev_departure_success",
                    "ev_departure_success_rate",
                    "ev_departure_status",
                    "departure_success",
                ],
                default=np.nan,
            )
            if not np.isnan(ev_success_signal):
                clipped_signal = float(np.clip(ev_success_signal, 0.0, 1.0))
                ev_term = 2.0 * clipped_signal - 1.0
            else:
                soc = self._extract_signal(obs_values, ["ev_soc", "soc"], default=np.nan)
                required_soc = self._extract_signal(
                    obs_values,
                    ["ev_required_soc_departure", "required_soc_departure", "required_soc"],
                    default=np.nan,
                )
                if np.isnan(soc) or np.isnan(required_soc):
                    ev_term = 0.0
                else:
                    distance = abs(soc - required_soc)
                    ev_term = float(np.clip(1.0 - (distance / ev_soc_tolerance), -1.0, 1.0))

            shaped_reward = base_reward
            if enabled_terms.get("energy_cost", True):
                shaped_reward += (
                    self._safe_float(weights.get("energy_cost"), default=1.0) * energy_cost_term
                )
            if enabled_terms.get("grid_violation", True):
                shaped_reward += (
                    self._safe_float(weights.get("grid_violation"), default=1.0)
                    * grid_violation_term
                )
            if enabled_terms.get("ev_success", True):
                shaped_reward += self._safe_float(weights.get("ev_success"), default=1.0) * ev_term
            if enabled_terms.get("community", True):
                shaped_reward += (
                    self._safe_float(weights.get("community"), default=1.0) * community_term
                )

            if self.wrapper_reward_clip_enabled:
                shaped_reward = float(
                    np.clip(
                        shaped_reward, self.wrapper_reward_clip_min, self.wrapper_reward_clip_max
                    )
                )

            if self.wrapper_reward_squash == "tanh":
                shaped_reward = float(np.tanh(shaped_reward))

            shaped_rewards.append(shaped_reward)

        return shaped_rewards
