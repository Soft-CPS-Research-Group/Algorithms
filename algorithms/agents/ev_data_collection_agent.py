"""Hour-based RBC behaviour policy for offline RL data collection.

Ports the ``BasicElectricVehicleRBC_ReferenceController`` action map into the
platform's :class:`BaseAgent` contract and records every transition for a single
target building into an in-memory buffer that is flushed to CSV on export.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from loguru import logger

from algorithms.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Hour → action maps from BasicElectricVehicleRBC_ReferenceController
# ---------------------------------------------------------------------------
# CityLearn uses hours 1–24 (via Building.get_periodic_observation_metadata).
_HOURS_1_24 = list(range(1, 25))


def _build_action_map_for_name(action_name: str) -> Dict[int, float]:
    """Return an ``{hour: action_value}`` map for a single action name."""
    hour_map: Dict[int, float] = {}

    if action_name == "electrical_storage":
        for h in _HOURS_1_24:
            if 9 <= h <= 21:
                hour_map[h] = -0.08
            elif (1 <= h <= 8) or (22 <= h <= 24):
                hour_map[h] = 0.091
            else:
                hour_map[h] = 0.0

    elif action_name in ("cooling_device",):
        for h in _HOURS_1_24:
            if 9 <= h <= 21:
                hour_map[h] = 0.8
            elif (1 <= h <= 8) or (22 <= h <= 24):
                hour_map[h] = 0.4
            else:
                hour_map[h] = 0.0

    elif action_name in ("heating_device",):
        for h in _HOURS_1_24:
            if 9 <= h <= 21:
                hour_map[h] = 0.4
            elif (1 <= h <= 8) or (22 <= h <= 24):
                hour_map[h] = 0.8
            else:
                hour_map[h] = 0.0

    elif action_name in ("cooling_or_heating_device",):
        for h in _HOURS_1_24:
            if h < 7:
                hour_map[h] = 0.4
            elif h < 21:
                hour_map[h] = -0.4
            else:
                hour_map[h] = 0.8

    elif "electric_vehicle" in action_name:
        for h in _HOURS_1_24:
            if h < 7:
                hour_map[h] = 0.4
            elif h < 10:
                hour_map[h] = 1.0
            elif h < 15:
                hour_map[h] = -1.0
            elif h < 20:
                hour_map[h] = -0.6
            else:
                hour_map[h] = 0.8

    elif "dhw_storage" in action_name:
        for h in _HOURS_1_24:
            hour_map[h] = 1.0

    elif "washing_machine" in action_name:
        for h in _HOURS_1_24:
            hour_map[h] = 1.0

    else:
        # Fallback: zero action for any unknown device.
        for h in _HOURS_1_24:
            hour_map[h] = 0.0
        logger.warning("Unknown action name '{}' — defaulting to 0.0", action_name)

    return hour_map


class EVDataCollectionRBC(BaseAgent):
    """Hour-based RBC that collects offline RL transitions for a target building.

    The policy is a faithful port of
    ``BasicElectricVehicleRBC_ReferenceController``: every hour is mapped to a
    fixed action value for each controllable device.  Transition tuples
    ``(obs, action, reward, next_obs, done)`` for the **target building** are
    accumulated in memory and written to a CSV file on
    :meth:`export_artifacts`.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        self._target_idx: int = int(hyper.get("target_building_index", 4))

        # Populated by attach_environment ----------------------------------
        self._all_observation_names: List[List[str]] = []
        self._all_action_names: List[List[str]] = []
        self._target_obs_names: List[str] = []
        self._target_action_names: List[str] = []
        self._hour_indices: List[int] = []  # per-agent hour index cache
        self._action_maps: List[List[Dict[int, float]]] = []  # per-agent, per-action

        # Transition buffer ------------------------------------------------
        self._transitions: List[Dict[str, Any]] = []
        self._episode: int = 0
        self._timestep: int = 0

    # -----------------------------------------------------------------
    # BaseAgent hooks
    # -----------------------------------------------------------------
    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._all_observation_names = [list(n) for n in observation_names]
        self._all_action_names = [list(n) for n in action_names]

        # Cache target building metadata for CSV headers -------------------
        if self._target_idx < len(observation_names):
            self._target_obs_names = list(observation_names[self._target_idx])
        else:
            logger.warning(
                "target_building_index={} exceeds agent count {}; falling back to agent 0",
                self._target_idx,
                len(observation_names),
            )
            self._target_idx = 0
            self._target_obs_names = list(observation_names[0])

        self._target_action_names = list(action_names[self._target_idx])

        # Pre-compute per-agent helpers ------------------------------------
        self._hour_indices = []
        self._action_maps = []
        for agent_obs_names, agent_action_names in zip(observation_names, action_names):
            # hour index in raw observations
            try:
                hour_idx = list(agent_obs_names).index("hour")
            except ValueError:
                hour_idx = 2  # conventional fallback
            self._hour_indices.append(hour_idx)

            # build action map per device
            agent_maps: List[Dict[int, float]] = []
            for a_name in agent_action_names:
                agent_maps.append(_build_action_map_for_name(a_name))
            self._action_maps.append(agent_maps)

        num_agents = len(observation_names)
        target_obs = len(self._target_obs_names)
        target_act = len(self._target_action_names)
        logger.info(
            "EVDataCollectionRBC attached: {} agents, target building index={}, "
            "{} observations, {} actions",
            num_agents,
            self._target_idx,
            target_obs,
            target_act,
        )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        """Return hour-mapped actions for all agents."""
        all_actions: List[List[float]] = []

        for agent_idx, obs in enumerate(observations):
            obs = np.asarray(obs, dtype=float)

            # Resolve current hour -----------------------------------------
            hour_idx = (
                self._hour_indices[agent_idx]
                if agent_idx < len(self._hour_indices)
                else 2
            )
            raw_hour = float(obs[hour_idx]) if hour_idx < len(obs) else 0.0
            hour = int(round(raw_hour))

            # Build candidate hours (handle 0–23 vs 1–24) ------------------
            hour_candidates = []
            for candidate in (hour, hour % 24, ((hour - 1) % 24) + 1):
                if candidate not in hour_candidates:
                    hour_candidates.append(candidate)

            # Look up action for each device --------------------------------
            agent_maps = (
                self._action_maps[agent_idx]
                if agent_idx < len(self._action_maps)
                else []
            )
            agent_action_names = (
                self._all_action_names[agent_idx]
                if agent_idx < len(self._all_action_names)
                else []
            )
            agent_actions: List[float] = []
            for action_pos, a_name in enumerate(agent_action_names):
                hour_map = agent_maps[action_pos] if action_pos < len(agent_maps) else {}
                value = 0.0
                for candidate in hour_candidates:
                    if candidate in hour_map:
                        value = hour_map[candidate]
                        break
                agent_actions.append(value)

            all_actions.append(agent_actions)

        return all_actions

    def update(
        self,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float64]],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        """Record the target building's transition."""
        idx = self._target_idx

        obs = np.asarray(observations[idx], dtype=float).tolist()
        act = np.asarray(actions[idx], dtype=float).tolist()
        reward = float(rewards[idx])
        next_obs = np.asarray(next_observations[idx], dtype=float).tolist()
        done = terminated or truncated

        row: Dict[str, Any] = {
            "episode": self._episode,
            "timestep": self._timestep,
        }

        # Observations
        for i, val in enumerate(obs):
            col = self._target_obs_names[i] if i < len(self._target_obs_names) else f"obs_{i}"
            row[f"obs_{col}"] = val

        # Actions
        for i, val in enumerate(act):
            col = self._target_action_names[i] if i < len(self._target_action_names) else f"action_{i}"
            row[f"action_{col}"] = val

        # Reward
        row["reward"] = reward

        # Next observations
        for i, val in enumerate(next_obs):
            col = self._target_obs_names[i] if i < len(self._target_obs_names) else f"obs_{i}"
            row[f"next_obs_{col}"] = val

        # Done flag
        row["done"] = int(done)

        self._transitions.append(row)
        self._timestep += 1

        # Track episode boundaries
        if done:
            logger.info(
                "Episode {} finished after {} steps ({} transitions total)",
                self._episode,
                self._timestep,
                len(self._transitions),
            )
            self._episode += 1
            self._timestep = 0

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write collected transitions to CSV and return manifest metadata."""
        export_root = Path(output_dir)
        export_root.mkdir(parents=True, exist_ok=True)
        csv_path = export_root / "offline_dataset.csv"

        if not self._transitions:
            logger.warning("No transitions collected — writing empty CSV")

        # Derive ordered column names from first row (or build from metadata)
        if self._transitions:
            fieldnames = list(self._transitions[0].keys())
        else:
            fieldnames = self._build_empty_fieldnames()

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._transitions)

        logger.info(
            "Offline dataset written: {} transitions → {}",
            len(self._transitions),
            csv_path,
        )

        metadata: Dict[str, Any] = {
            "format": "offline_dataset",
            "parameters": {
                "target_building_index": self._target_idx,
                "num_transitions": len(self._transitions),
                "num_episodes": self._episode,
                "observation_names": self._target_obs_names,
                "action_names": self._target_action_names,
            },
            "artifacts": [
                {
                    "agent_index": self._target_idx,
                    "path": csv_path.name,
                    "format": "offline_dataset",
                }
            ],
        }
        return metadata

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _build_empty_fieldnames(self) -> List[str]:
        """Build CSV column names when no transitions were collected."""
        fields = ["episode", "timestep"]
        for name in self._target_obs_names:
            fields.append(f"obs_{name}")
        for name in self._target_action_names:
            fields.append(f"action_{name}")
        fields.append("reward")
        for name in self._target_obs_names:
            fields.append(f"next_obs_{name}")
        fields.append("done")
        return fields
