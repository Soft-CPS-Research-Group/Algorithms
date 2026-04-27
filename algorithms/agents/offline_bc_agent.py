"""Inference agent that wraps a trained Behavioral Cloning policy.

The policy is loaded from disk (``model.pth`` + ``normalization_stats.json``)
and applied to a single target building's raw observations. Other agents in
the multi-agent setting receive zero actions, mirroring the data-collection
setup (we train BC for one specific building only).

This agent does **not** train online: :meth:`update` is a no-op. It exists so
that the trained policy can be evaluated under exactly the same CityLearn
conditions as the RBC behaviour policy.

M3 changes
----------

* **Name-aligned observation reordering.** The training CSV records the
  ``feature_names`` it was trained on. At inference we look up each training
  feature in the env's ``observation_names[target_idx]`` and reorder the
  current observation vector accordingly. This makes the agent robust to
  dynamic-topology reorderings and to environments whose obs space happens
  to be a permutation of the training one.
* **Fail-fast on schema drift.** If any training feature is missing from the
  env, or if the env's action dimensionality for the target building does
  not match the trained head, we raise a ``RuntimeError`` with an explicit
  remediation message instead of silently producing garbage actions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.offline.bc_policy import BCPolicy


class OfflineBCAgent(BaseAgent):
    """Behavioral Cloning policy plugged into the platform's BaseAgent contract."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.use_raw_observations = True

        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        model_path = hyper.get("model_path")
        stats_path = hyper.get("stats_path")
        if not model_path or not stats_path:
            raise ValueError(
                "OfflineBCAgent requires algorithm.hyperparameters.model_path and stats_path"
            )

        self._model_path = Path(model_path).expanduser().resolve()
        self._stats_path = Path(stats_path).expanduser().resolve()
        self._target_idx: int = int(hyper.get("target_building_index", 4))
        device_pref = str(hyper.get("device", "auto")).lower()
        if device_pref == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_pref)

        # Load normalization stats -----------------------------------------
        with self._stats_path.open("r", encoding="utf-8") as fh:
            stats = json.load(fh)
        # Support both legacy ('obs_mean'/'obs_std'/'obs_names') and current
        # ('mean'/'std'/'feature_names') key names emitted by the trainer.
        mean_key = "mean" if "mean" in stats else "obs_mean"
        std_key = "std" if "std" in stats else "obs_std"
        names_key = (
            "feature_names" if "feature_names" in stats
            else ("obs_names" if "obs_names" in stats else None)
        )
        self._obs_mean = np.asarray(stats[mean_key], dtype=np.float32)
        self._obs_std = np.asarray(stats[std_key], dtype=np.float32)
        # Avoid divide-by-zero for constant features.
        self._obs_std = np.where(self._obs_std < 1e-8, 1.0, self._obs_std)
        self._obs_names_train: List[str] = (
            list(stats.get(names_key, [])) if names_key else []
        )
        self._action_names_train: List[str] = list(stats.get("action_names", []))

        # Load policy -------------------------------------------------------
        checkpoint = torch.load(
            self._model_path, map_location=self._device, weights_only=False
        )
        arch = checkpoint["architecture"]
        self._policy = BCPolicy(
            obs_dim=int(arch["obs_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_layers=list(arch.get("hidden_layers", [256, 256])),
            dropout=float(arch.get("dropout", 0.0)),
        ).to(self._device)
        self._policy.load_state_dict(
            checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        )
        self._policy.eval()

        self._obs_dim = int(arch["obs_dim"])
        self._action_dim = int(arch["action_dim"])

        # Populated by attach_environment ---------------------------------
        self._all_action_names: List[List[str]] = []
        self._target_obs_names: List[str] = []
        self._target_action_names: List[str] = []
        # Index into env obs[target_idx] for each training feature, in
        # training order. Built once in attach_environment.
        self._index_map: Optional[np.ndarray] = None

        logger.info(
            "OfflineBCAgent loaded: model={}, target_building_index={}, device={}, "
            "obs_dim={}, action_dim={}",
            self._model_path,
            self._target_idx,
            self._device,
            self._obs_dim,
            self._action_dim,
        )

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
        self._all_action_names = [list(n) for n in action_names]

        if self._target_idx >= len(observation_names):
            raise RuntimeError(
                f"BC target_building_index={self._target_idx} exceeds env agent "
                f"count {len(observation_names)}. Re-train BC on the current "
                "dataset/topology, or set a valid target_building_index."
            )

        self._target_obs_names = list(observation_names[self._target_idx])
        self._target_action_names = list(action_names[self._target_idx])

        # Action-dim mismatch is a hard fail (cannot pad/truncate predictions).
        if len(self._target_action_names) != self._action_dim:
            raise RuntimeError(
                "BC action schema mismatch: env reports "
                f"{len(self._target_action_names)} actions for target building "
                f"(idx={self._target_idx}) but BC was trained with "
                f"{self._action_dim}. "
                "Re-train BC on the current dataset/topology — Building's "
                "action schema changed since training."
            )

        # Build index map by name. Requires training stats to include
        # feature_names (M2+ trainer always does).
        if not self._obs_names_train:
            logger.warning(
                "Training normalization_stats has no feature_names; falling back "
                "to positional obs feed. This is brittle under dynamic topology."
            )
            if len(self._target_obs_names) != self._obs_dim:
                raise RuntimeError(
                    f"BC obs dim mismatch: env reports {len(self._target_obs_names)} "
                    f"but BC was trained with {self._obs_dim}. Re-train BC."
                )
            self._index_map = np.arange(self._obs_dim, dtype=np.int64)
            return

        env_index_by_name = {n: i for i, n in enumerate(self._target_obs_names)}
        missing = [n for n in self._obs_names_train if n not in env_index_by_name]
        if missing:
            raise RuntimeError(
                "BC obs schema mismatch: training feature(s) not present in env "
                f"observation_names[{self._target_idx}]: {missing[:10]}"
                + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
                + ". Re-train BC on the current dataset/topology — "
                "Building's observation schema changed since training."
            )
        self._index_map = np.asarray(
            [env_index_by_name[n] for n in self._obs_names_train], dtype=np.int64
        )
        logger.info(
            "BC obs name-alignment OK: {} features mapped (env has {} total at target idx)",
            len(self._index_map),
            len(self._target_obs_names),
        )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        if self._index_map is None:
            raise RuntimeError(
                "OfflineBCAgent.predict called before attach_environment."
            )

        all_actions: List[List[float]] = []
        for agent_idx, obs in enumerate(observations):
            if agent_idx == self._target_idx:
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                # Reorder env obs to match training feature order.
                aligned = obs_arr[self._index_map]
                normalized = (aligned - self._obs_mean) / self._obs_std
                with torch.no_grad():
                    tensor = (
                        torch.from_numpy(normalized).unsqueeze(0).to(self._device)
                    )
                    action = self._policy(tensor).squeeze(0).cpu().numpy()
                all_actions.append([float(v) for v in action.tolist()])
            else:
                # Idle action vector for non-target agents.
                action_dim = (
                    len(self._all_action_names[agent_idx])
                    if agent_idx < len(self._all_action_names)
                    else 0
                )
                all_actions.append([0.0] * action_dim)
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
        # No online learning — BC is trained offline.
        return None

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Copy the trained model + stats into the job dir and emit manifest."""
        export_root = Path(output_dir)
        export_root.mkdir(parents=True, exist_ok=True)

        model_dst = export_root / self._model_path.name
        stats_dst = export_root / self._stats_path.name
        try:
            shutil.copy2(self._model_path, model_dst)
            shutil.copy2(self._stats_path, stats_dst)
        except Exception as exc:  # pragma: no cover - filesystem edge cases
            logger.warning("Failed to copy BC artifacts to job dir: {}", exc)

        return {
            "format": "behavioral_cloning",
            "parameters": {
                "target_building_index": self._target_idx,
                "obs_dim": self._obs_dim,
                "action_dim": self._action_dim,
                "model_path_source": str(self._model_path),
                "stats_path_source": str(self._stats_path),
            },
            "artifacts": [
                {
                    "agent_index": self._target_idx,
                    "path": model_dst.name,
                    "format": "behavioral_cloning",
                    "config": {"stats_path": stats_dst.name},
                }
            ],
        }
