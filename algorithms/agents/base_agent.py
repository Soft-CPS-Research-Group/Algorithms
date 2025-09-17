from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt
from torch.nn import Module


class BaseAgent(Module, ABC):
    """Common interface for all training and inference agents."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
    ) -> List[List[float]]:
        """Return actions for the current time step."""

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update replay buffer(s) and network parameters."""

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        """Persist training state and return the checkpoint path."""
        raise NotImplementedError("Agent does not implement checkpointing.")

    @abstractmethod
    def export_artifacts(self, output_dir: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export inference artefacts and return metadata about the exports."""

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Optional hook for loading checkpoints outside MLflow resume logic."""
        raise NotImplementedError("Agent does not implement checkpoint loading.")
