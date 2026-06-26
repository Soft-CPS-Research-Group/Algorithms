from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional
import os
import sys
import time

_BASE_AGENT_TRACE_ENABLED = (
    os.environ.get("OPEVA_STARTUP_TRACE", "1").strip().lower()
    not in {"0", "false", "no", "off"}
    and os.path.basename(sys.argv[0]) == "run_experiment.py"
)
_BASE_AGENT_TRACE_T0 = time.monotonic()


def _base_agent_trace(message: str) -> None:
    if not _BASE_AGENT_TRACE_ENABLED:
        return
    elapsed = time.monotonic() - _BASE_AGENT_TRACE_T0
    print(f"[opeva-base-agent +{elapsed:.3f}s] {message}", file=sys.stderr, flush=True)


_base_agent_trace("module import started")
_base_agent_trace("before numpy import")
import numpy as np
_base_agent_trace("after numpy import")
_base_agent_trace("before numpy.typing import")
import numpy.typing as npt
_base_agent_trace("after numpy.typing import")
_base_agent_trace("before torch.nn Module import")
from torch.nn import Module
_base_agent_trace("after torch.nn Module import")

_base_agent_trace("before execution unit import")
from algorithms.execution_unit import ExecutionUnit
_base_agent_trace("after execution unit import")


class BaseAgent(Module, ExecutionUnit):
    """Common interface for all training and inference agents.

    Inherits the wrapper-facing contract from :class:`ExecutionUnit` so that
    single agents and composite execution units (pipelines, ensembles) are
    interchangeable from the wrapper's perspective.
    """

    supports_dynamic_topology: ClassVar[bool] = False
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        """Return actions for the current time step.

        ``context`` is provided by the parent stage when this agent runs
        inside a :class:`~algorithms.pipeline.Pipeline`. Single agents may
        ignore it.
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

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Optional hook that provides environment metadata after instantiation."""

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

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        """Return whether the agent considers warm-up/exploration complete."""
        _ = global_learning_step
        return True


_base_agent_trace("class definitions loaded")
