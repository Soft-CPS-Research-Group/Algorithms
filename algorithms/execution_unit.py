"""Abstract interface that the training wrapper interacts with.

The wrapper holds a single :class:`ExecutionUnit` and stays agnostic to
whether it's a single agent, a pipeline of stages (vertical hierarchy), or
an ensemble of agents at the same level (horizontal fan-out). This is the
abstraction that allows the wrapper code to remain unchanged when the
architecture is extended to N levels.

Concrete implementations:
    * :class:`algorithms.agents.base_agent.BaseAgent` — single agent.
    * :class:`algorithms.pipeline.Pipeline` — ordered chain of stages.
      Threads ``context`` from each stage to the next.
    * :class:`algorithms.pipeline.Ensemble` — N agents at the same level.
      Fan-out across observations; same parent context broadcast to all.

Method signatures mirror the long-standing :class:`BaseAgent` contract so
that existing single-agent code paths (and the wrapper that drives them)
keep working unchanged once :class:`BaseAgent` adopts this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt


class ExecutionUnit(ABC):
    """Abstract contract for anything the wrapper drives.

    The default implementations cover the optional hooks (lifecycle,
    persistence) so that simple agents can override only what they need.
    Composite units (:class:`Pipeline`, :class:`Ensemble`) override these
    defaults to delegate to their children.
    """

    # Whether the unit wants raw (unencoded) observations from the wrapper.
    # Subclasses override at the instance or class level.
    use_raw_observations: bool = False

    # ------------------------------------------------------------------
    # Core interaction loop
    # ------------------------------------------------------------------
    @abstractmethod
    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> Any:
        """Return actions (or a signal for the next stage) for this step.

        ``context`` is the output produced by the parent stage in a
        :class:`Pipeline`. Single-agent leaves and the root stage receive
        ``None`` and may ignore the argument.
        """

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
        """Update internal state / network parameters for this step.

        Implementations must honour the scheduling flags supplied by
        :class:`utils.wrapper_citylearn.Wrapper_CityLearn`.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        """Return whether warm-up is complete. Default: always ``True``."""
        _ = global_learning_step
        return True

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Optional hook providing environment metadata after construction."""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        """Persist training state and return the checkpoint path."""
        raise NotImplementedError(
            "Execution unit does not implement checkpoint saving."
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restore training state from ``checkpoint_path``."""
        raise NotImplementedError(
            "Execution unit does not implement checkpoint loading."
        )

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export inference artefacts and return metadata describing them.

        Default implementation returns an empty payload; concrete units
        (single agents, composites) override to emit real artefacts.
        """
        _ = output_dir, context
        return {"format": "none", "artifacts": []}
