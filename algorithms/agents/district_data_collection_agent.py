"""District-wide RBC behaviour policy with per-building offline-RL data collection.

This agent is the Milestone-2 successor to :class:`EVDataCollectionRBC`:

* It uses the existing :class:`RuleBasedPolicy` as the **clean** behaviour
  policy (so the same EV-aware reference controller drives both data sets).
* It runs across the **whole district**, not a single target building.
* It supports **noise injection on selected episodes** to widen state-action
  coverage — a follow-up to the M1 conclusions doc point A.1.
* It records every transition for every active building into per-building
  in-memory buffers and flushes them to one CSV per building on
  :meth:`export_artifacts`.

Per-row CSV schema (one CSV per building):

    episode, timestep, topology_version, policy_mode, noise_sigma_applied,
    obs_<name>...,
    action_<name>...,           # actually executed by the env (post-noise+clip)
    action_clean_<name>...,     # the RBC's intended (clean) action
    reward,
    next_obs_<name>...,
    terminated, truncated

`terminated` and `truncated` are stored separately (M1 follow-up B.5).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.rbc_agent import RuleBasedPolicy


class DistrictDataCollectionRBC(BaseAgent):
    """RBC-backed multi-building offline-RL data collector with optional noise."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        # We delegate predict() to RuleBasedPolicy, which uses raw observations.
        self.use_raw_observations = True

        self._config = config
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})

        self._noise_sigma: float = float(hyper.get("noise_sigma", 0.1))
        raw_indices = hyper.get("noisy_episode_indices", [7, 8, 9])
        self._noisy_episode_indices: set[int] = {int(i) for i in raw_indices}
        self._seed: int = int(hyper.get("seed", 22))
        self._reward_function_name: str = str(
            (config.get("simulator", {}) or {}).get("reward_function", "RewardFunction")
        )
        self._dataset_path = str(
            (config.get("simulator", {}) or {}).get("dataset_path", "")
        )

        # Inner clean-action behaviour policy. We pass the *same* config so it
        # picks up its own hyperparameters from `algorithm.rbc_hyperparameters`
        # if provided; otherwise it uses its own defaults.
        rbc_config = self._build_inner_rbc_config(config)
        self._rbc = RuleBasedPolicy(rbc_config)

        # Populated by attach_environment ---------------------------------
        self._observation_names: List[List[str]] = []
        self._action_names: List[List[str]] = []
        self._building_names: List[str] = []
        self._topology_version: int = 0
        self._topology_versions_seen: set[int] = set()
        self._reward_probed: bool = False

        # Per-building transition buffers ---------------------------------
        # building_index -> list[dict]
        self._transitions: Dict[int, List[Dict[str, Any]]] = {}

        # Episode/timestep tracking ---------------------------------------
        self._episode: int = 0
        self._timestep: int = 0
        self._episode_rng: Optional[np.random.Generator] = None
        self._current_episode_is_noisy: bool = False

        # Per-step staging used between predict() and update():
        # holds the *clean* per-agent actions chosen by the RBC, so update()
        # can log them alongside the (possibly noisy) executed actions that
        # the env actually saw.
        self._last_clean_actions: List[List[float]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_inner_rbc_config(outer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build a sub-config for the inner RuleBasedPolicy.

        We forward the simulator block (so RBC can load schema/charger info)
        and synthesise an algorithm block with name='RuleBasedPolicy' and any
        explicit RBC hyperparameters provided under
        `algorithm.rbc_hyperparameters` in the outer config.
        """
        outer_algo = outer_config.get("algorithm", {}) or {}
        rbc_hyper = outer_algo.get("rbc_hyperparameters", {}) or {}
        return {
            "simulator": dict(outer_config.get("simulator", {}) or {}),
            "algorithm": {
                "name": "RuleBasedPolicy",
                "hyperparameters": rbc_hyper,
            },
        }

    def _refresh_episode_state(self, episode_index: int) -> None:
        """Refresh per-episode RNG and noisy/clean flag."""
        self._episode = episode_index
        self._timestep = 0
        self._current_episode_is_noisy = episode_index in self._noisy_episode_indices
        # Per-episode RNG so a re-run with the same base seed reproduces noise.
        self._episode_rng = np.random.default_rng(self._seed + episode_index)
        logger.info(
            "Episode {} starting (mode={}, noise_sigma={})",
            episode_index,
            "noisy" if self._current_episode_is_noisy else "clean",
            self._noise_sigma if self._current_episode_is_noisy else 0.0,
        )

    # ------------------------------------------------------------------
    # BaseAgent hooks
    # ------------------------------------------------------------------
    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Forward to the inner RBC so it builds its own EV/charger mapping.
        self._rbc.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )

        self._observation_names = [list(names) for names in observation_names]
        self._action_names = [list(names) for names in action_names]

        meta = metadata or {}
        building_names = meta.get("building_names")
        if isinstance(building_names, list) and building_names:
            self._building_names = [str(name) for name in building_names]
        else:
            self._building_names = [f"agent_{i}" for i in range(len(observation_names))]

        new_version = meta.get("topology_version")
        if isinstance(new_version, int):
            self._topology_version = new_version
            self._topology_versions_seen.add(new_version)
        else:
            # Wrapper might pass None on first attach if not initialised yet.
            self._topology_versions_seen.add(self._topology_version)

        # Initialise per-building buffers if not yet done. Under dynamic
        # topology, the building set can in principle change between attaches;
        # we lazily add new buckets but never drop old ones.
        for idx in range(len(observation_names)):
            self._transitions.setdefault(idx, [])

        # One-shot reward function probe (fail-fast on misconfiguration).
        if not self._reward_probed:
            self._probe_reward_function()
            self._reward_probed = True

        # Initialise episode 0 state on first attach. Subsequent topology-
        # change attaches must NOT reset the episode counter, otherwise we'd
        # corrupt per-episode noise reproducibility.
        if self._episode_rng is None:
            self._refresh_episode_state(episode_index=0)

        logger.info(
            "DistrictDataCollectionRBC attached: {} buildings, topology_version={}, "
            "reward_function={}",
            len(self._building_names),
            self._topology_version,
            self._reward_function_name,
        )

    def _probe_reward_function(self) -> None:
        """Fail-fast if the configured reward function is not the requested one.

        We only verify the *configured* name; instantiation/runtime probing of
        V2GPenaltyReward against this dataset is out of scope here. If the
        reward function fails at env.step() time, the runner will surface the
        original exception and the user can re-run with
        `simulator.reward_function: RewardFunction` per the agreed
        fail-fast-then-rerun contract.
        """
        configured = self._reward_function_name
        if configured not in {"RewardFunction", "V2GPenaltyReward",
                              "CostMinimizationReward", "CostHardConstraintReward"}:
            raise ValueError(
                f"Unknown reward_function '{configured}'. "
                "Re-run with simulator.reward_function set to one of: "
                "RewardFunction, V2GPenaltyReward, CostMinimizationReward, "
                "CostHardConstraintReward."
            )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        """Return executed actions; stash the clean reference for update()."""
        clean_actions = self._rbc.predict(observations, deterministic=deterministic)
        # Defensive deep-copy as plain Python floats so post-clip mutations
        # downstream don't mutate our stash.
        self._last_clean_actions = [
            [float(v) for v in agent_actions] for agent_actions in clean_actions
        ]

        if not self._current_episode_is_noisy or self._noise_sigma <= 0.0:
            return clean_actions

        if self._episode_rng is None:  # safety
            self._episode_rng = np.random.default_rng(self._seed + self._episode)

        noisy_actions: List[List[float]] = []
        for agent_actions in clean_actions:
            arr = np.asarray(agent_actions, dtype=np.float64)
            if arr.size == 0:
                noisy_actions.append([])
                continue
            noise = self._episode_rng.normal(0.0, self._noise_sigma, size=arr.shape)
            perturbed = np.clip(arr + noise, -1.0, 1.0)
            noisy_actions.append(perturbed.tolist())
        return noisy_actions

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
        """Record one row per building for this transition."""
        policy_mode = "noisy" if self._current_episode_is_noisy else "clean"
        sigma_applied = self._noise_sigma if self._current_episode_is_noisy else 0.0

        n_agents = min(len(observations), len(self._observation_names))
        for agent_idx in range(n_agents):
            obs_vec = np.asarray(observations[agent_idx], dtype=float).reshape(-1)
            act_vec = np.asarray(actions[agent_idx], dtype=float).reshape(-1)
            next_obs_vec = np.asarray(next_observations[agent_idx], dtype=float).reshape(-1)
            obs_names = self._observation_names[agent_idx]
            action_names = self._action_names[agent_idx]
            clean_vec = (
                self._last_clean_actions[agent_idx]
                if agent_idx < len(self._last_clean_actions)
                else [float(v) for v in act_vec]
            )

            row: Dict[str, Any] = {
                "episode": self._episode,
                "timestep": self._timestep,
                "topology_version": self._topology_version,
                "policy_mode": policy_mode,
                "noise_sigma_applied": sigma_applied,
            }

            for i, val in enumerate(obs_vec):
                col = obs_names[i] if i < len(obs_names) else f"col_{i}"
                row[f"obs_{col}"] = float(val)

            for i, val in enumerate(act_vec):
                col = action_names[i] if i < len(action_names) else f"col_{i}"
                row[f"action_{col}"] = float(val)

            for i, val in enumerate(clean_vec):
                col = action_names[i] if i < len(action_names) else f"col_{i}"
                row[f"action_clean_{col}"] = float(val)

            reward_value = float(rewards[agent_idx]) if agent_idx < len(rewards) else 0.0
            row["reward"] = reward_value

            for i, val in enumerate(next_obs_vec):
                col = obs_names[i] if i < len(obs_names) else f"col_{i}"
                row[f"next_obs_{col}"] = float(val)

            row["terminated"] = int(bool(terminated))
            row["truncated"] = int(bool(truncated))

            self._transitions.setdefault(agent_idx, []).append(row)

        self._timestep += 1

        if terminated or truncated:
            total_so_far = sum(len(v) for v in self._transitions.values())
            logger.info(
                "Episode {} finished after {} steps (total rows across buildings: {})",
                self._episode,
                self._timestep,
                total_so_far,
            )
            # Prepare for the next episode (the wrapper's learn() loop will
            # call predict()/update() again starting next iteration).
            self._refresh_episode_state(episode_index=self._episode + 1)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        export_root = Path(output_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        artifacts: List[Dict[str, Any]] = []
        per_building_counts: Dict[str, int] = {}

        for agent_idx, rows in sorted(self._transitions.items()):
            building_label = (
                self._building_names[agent_idx]
                if agent_idx < len(self._building_names)
                else f"agent_{agent_idx}"
            )
            safe_label = self._safe_filename(building_label)
            csv_name = f"offline_dataset_{safe_label}.csv"
            csv_path = export_root / csv_name

            fieldnames = (
                list(rows[0].keys())
                if rows
                else self._build_empty_fieldnames(agent_idx)
            )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            per_building_counts[building_label] = len(rows)
            artifacts.append(
                {
                    "agent_index": agent_idx,
                    "path": csv_name,
                    "format": "offline_dataset",
                    "config": {
                        "building": building_label,
                        "rows": len(rows),
                    },
                }
            )
            logger.info(
                "Wrote {} transitions for building '{}' → {}",
                len(rows),
                building_label,
                csv_path,
            )

        # Companion metadata file (single source of truth for downstream loaders).
        metadata_path = export_root / "dataset_metadata.json"
        metadata_payload: Dict[str, Any] = {
            "seed": self._seed,
            "noise_sigma": self._noise_sigma,
            "noisy_episode_indices": sorted(self._noisy_episode_indices),
            "episodes_completed": self._episode,
            "reward_function": self._reward_function_name,
            "dataset_schema": self._dataset_path,
            "topology_versions_seen": sorted(self._topology_versions_seen),
            "buildings": [
                {
                    "agent_index": idx,
                    "name": (
                        self._building_names[idx]
                        if idx < len(self._building_names)
                        else f"agent_{idx}"
                    ),
                    "num_observations": len(self._observation_names[idx])
                    if idx < len(self._observation_names) else 0,
                    "num_actions": len(self._action_names[idx])
                    if idx < len(self._action_names) else 0,
                    "rows": len(self._transitions.get(idx, [])),
                }
                for idx in sorted(self._transitions.keys())
            ],
            "per_building_row_counts": per_building_counts,
        }
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        logger.info("Dataset metadata written → {}", metadata_path)

        return {
            "format": "offline_dataset",
            "parameters": {
                "noise_sigma": self._noise_sigma,
                "noisy_episode_indices": sorted(self._noisy_episode_indices),
                "seed": self._seed,
                "reward_function": self._reward_function_name,
                "episodes_completed": self._episode,
                "buildings": list(per_building_counts.keys()),
            },
            "artifacts": artifacts,
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitise a building name so it is safe to use in a file path."""
        cleaned = "".join(
            ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_"
            for ch in str(name)
        )
        return cleaned or "agent"

    def _build_empty_fieldnames(self, agent_idx: int) -> List[str]:
        obs_names = (
            self._observation_names[agent_idx]
            if agent_idx < len(self._observation_names)
            else []
        )
        action_names = (
            self._action_names[agent_idx]
            if agent_idx < len(self._action_names)
            else []
        )
        fields = [
            "episode",
            "timestep",
            "topology_version",
            "policy_mode",
            "noise_sigma_applied",
        ]
        fields += [f"obs_{n}" for n in obs_names]
        fields += [f"action_{n}" for n in action_names]
        fields += [f"action_clean_{n}" for n in action_names]
        fields.append("reward")
        fields += [f"next_obs_{n}" for n in obs_names]
        fields += ["terminated", "truncated"]
        return fields
