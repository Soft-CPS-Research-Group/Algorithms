from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
        """Update replay buffers and network parameters for the current step.

        Subclasses must honour the scheduling flags provided by
        :class:`utils.wrapper_citylearn.Wrapper_CityLearn`:

        ``update_step``
            Indicates whether the learning update should occur this step.
        ``update_target_step``
            Signals that target networks should be synchronised.
        ``initial_exploration_done``
            Becomes ``True`` after the warm-up phase and can gate updates.
        ``global_learning_step``
            Monotonic counter across the entire run, useful for logging.
        """

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        """Persist training state and return the checkpoint path."""
        raise NotImplementedError("Agent does not implement checkpointing.")

    @abstractmethod
    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export inference artefacts and return metadata about the exports."""

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Optional hook for loading checkpoints outside MLflow resume logic."""
        raise NotImplementedError("Agent does not implement checkpoint loading.")
