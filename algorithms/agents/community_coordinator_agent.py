


from algorithms.agents.base_agent import BaseAgent
from typing import Any, Dict, List, Optional, Sequence
import torch
from torch import nn
from loguru import logger
from pathlib import Path
from algorithms.constants import DEFAULT_ONNX_OPSET

import numpy as np


class CommunityCoordinatorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        hyper = (config.get("algorithm", {}).get("hyperparameters") or {})
        self.community = hyper.get("community")
        self.use_raw_observations = True

        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )


    def update(
        self,
        observations: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_observations: List[np.ndarray],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        pass



    def _build_community_state(
            self, 
            observations: List[np.ndarray]
    ) -> List[Any]:
        total_net_electricity_consumption = 0
        total_solar = 0
        total_load = 0

        for idx, building_obs in enumerate(observations):
            obs_indexes = self._obs_index[idx]

            net_electricity_consumption_idx = obs_indexes["net_electricity_consumption"]
            solar_idx = obs_indexes["solar_generation"]
            load_idx = obs_indexes["non_shiftable_load"]

            total_net_electricity_consumption += building_obs[net_electricity_consumption_idx]
            total_solar += building_obs[solar_idx]
            total_load += building_obs[load_idx]

        return [total_net_electricity_consumption, total_solar, total_load]


    def predict(
        self,
        observations: List[np.ndarray],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        community_state = self._build_community_state(observations)

        state_tensor = torch.tensor(community_state, dtype=torch.float32)
        o1 = self.network(state_tensor)
        logger.info("O1: {}", o1)

        return [[0.0] * self._action_dims[i] for i in range(len(observations))]
    


    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        """Persist training state and return the checkpoint path."""
        raise NotImplementedError("Agent does not implement checkpointing.")

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._obs_index = [
            {name: idx for idx, name in enumerate(names)}
            for names in observation_names
        ]
        self._action_dims = [len(names) for names in action_names]

    def export_artifacts(self, output_dir, context=None):

        export_root = Path(output_dir)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        export_path = onnx_dir / "community_coordinator.onnx"
        dummy_input = torch.randn(1, 3)

        torch.onnx.export(
            self.network,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=DEFAULT_ONNX_OPSET,
            do_constant_folding=True,
            input_names=["community_state"],
            output_names=["o1"],
            dynamic_axes={
                "community_state": {0: "batch_size"},
                "o1": {0: "batch_size"},
            },
        )

        relative_path = export_path.relative_to(export_root)

        return {
            "format": "onnx",
            "artifacts": [
                {
                    "path": str(relative_path),
                    "format": "onnx",
                    "agent_index": i,
                }
                for i in range(len(self._action_dims))
            ]
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        raise NotImplementedError("Agent does not implement checkpoint loading.")

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        pass