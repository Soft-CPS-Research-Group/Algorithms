"""Inference agent that wraps a trained Behavioral Cloning policy.

The policy is loaded from disk (``model.pth`` + ``normalization_stats.json``)
and applied to a single target building's raw observations. Other agents in
the multi-agent setting receive zero actions, mirroring the data-collection
setup (we only train BC for Building 5 / agent index 4).

This agent does **not** train online: :meth:`update` is a no-op. It exists so
that the trained policy can be evaluated under exactly the same CityLearn
conditions as the RBC behaviour policy.
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
        self._obs_names_train: List[str] = list(stats.get(names_key, [])) if names_key else []
        self._action_names_train: List[str] = list(stats.get("action_names", []))

        # Load policy -------------------------------------------------------
        checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)
        arch = checkpoint["architecture"]
        self._policy = BCPolicy(
            obs_dim=int(arch["obs_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_layers=list(arch.get("hidden_layers", [256, 256])),
        ).to(self._device)
        self._policy.load_state_dict(checkpoint.get("state_dict", checkpoint.get("model_state_dict")))
        self._policy.eval()

        self._obs_dim = int(arch["obs_dim"])
        self._action_dim = int(arch["action_dim"])

        # Populated by attach_environment ---------------------------------
        self._all_action_names: List[List[str]] = []
        self._target_obs_names: List[str] = []
        self._target_action_names: List[str] = []

        logger.info(
            "OfflineBCAgent loaded: model={}, target_building_index={}, device={}",
            self._model_path,
            self._target_idx,
            self._device,
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
            logger.warning(
                "target_building_index={} exceeds agent count {}; falling back to 0",
                self._target_idx,
                len(observation_names),
            )
            self._target_idx = 0

        self._target_obs_names = list(observation_names[self._target_idx])
        self._target_action_names = list(action_names[self._target_idx])

        # Sanity checks vs. training metadata.
        if len(self._target_obs_names) != self._obs_dim:
            logger.warning(
                "Observation dim mismatch: env reports {} but BC was trained on {}",
                len(self._target_obs_names),
                self._obs_dim,
            )
        if len(self._target_action_names) != self._action_dim:
            logger.warning(
                "Action dim mismatch: env reports {} but BC was trained on {}",
                len(self._target_action_names),
                self._action_dim,
            )
        if self._obs_names_train and self._obs_names_train != self._target_obs_names:
            logger.warning(
                "Observation name order differs between training data and current env; "
                "BC inputs are positional — verify schema compatibility."
            )

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        all_actions: List[List[float]] = []
        for agent_idx, obs in enumerate(observations):
            if agent_idx == self._target_idx:
                obs_arr = np.asarray(obs, dtype=np.float32)
                # Standardize then forward.
                normalized = (obs_arr - self._obs_mean) / self._obs_std
                with torch.no_grad():
                    tensor = torch.from_numpy(normalized).unsqueeze(0).to(self._device)
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
