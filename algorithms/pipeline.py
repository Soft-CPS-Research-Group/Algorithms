"""Composite execution units for hierarchical or ensemble architectures.

Two infrastructure classes:

* :class:`Pipeline` — vertical chain of stages from top (manager) to bottom
  (leaf). The output produced by ``predict`` of one stage is passed as the
  ``context`` argument to the next stage. The leaf stage produces the
  environment actions.
* :class:`Ensemble` — horizontal fan-out of N agents acting at the same
  level. Each agent receives its own observation slice and the same
  parent context. The combined output is the list of per-agent actions.

Both classes are pure orchestrators: they hold no domain logic. They
satisfy :class:`ExecutionUnit` so that the wrapper interacts with them
through the same surface as a single agent.

Adding new hierarchy levels does not require any change to either of
these classes nor to the wrapper — composition happens entirely through
the configuration that drives the builder in :mod:`run_experiment`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from algorithms.execution_unit import ExecutionUnit


class Pipeline(ExecutionUnit):
    """Ordered chain of execution units (top → bottom).

    The output of ``stages[i].predict(...)`` is forwarded as the
    ``context`` argument of ``stages[i + 1].predict(...)``. The leaf
    stage's output is what the wrapper sees as ``predict``'s return
    value.
    """

    def __init__(self, stages: Sequence[ExecutionUnit]):
        if not stages:
            raise ValueError("Pipeline requires at least one stage.")
        self.stages: List[ExecutionUnit] = list(stages)

    @property
    def use_raw_observations(self) -> bool:  # type: ignore[override]
        return any(stage.use_raw_observations for stage in self.stages)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def predict(
        self,
        observations,
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> Any:
        ctx = context
        result: Any = None
        for stage in self.stages:
            result = stage.predict(observations, deterministic, context=ctx)
            ctx = result
        return result

    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        for stage in self.stages:
            stage.update(
                observations,
                actions,
                rewards,
                next_observations,
                terminated,
                truncated,
                update_target_step=update_target_step,
                global_learning_step=global_learning_step,
                update_step=update_step,
                initial_exploration_done=initial_exploration_done,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return all(
            stage.is_initial_exploration_done(global_learning_step)
            for stage in self.stages
        )

    def attach_environment(self, **kwargs) -> None:
        for stage in self.stages:
            stage.attach_environment(**kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for index, stage in enumerate(self.stages):
            stage_dir = root / f"stage_{index}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            try:
                stage.save_checkpoint(str(stage_dir), step)
            except NotImplementedError:
                continue
        return str(root)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Pipeline checkpoint root not found: {root}")
        for index, stage in enumerate(self.stages):
            stage_dir = root / f"stage_{index}"
            if not stage_dir.exists():
                continue
            try:
                stage.load_checkpoint(str(stage_dir))
            except NotImplementedError:
                continue

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        stages_metadata: List[Dict[str, Any]] = []
        for index, stage in enumerate(self.stages):
            stage_dir = root / f"stage_{index}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            metadata = stage.export_artifacts(str(stage_dir), context)
            stages_metadata.append({"stage_index": index, **(metadata or {})})
        return {"format": "pipeline", "stages": stages_metadata}


class Ensemble(ExecutionUnit):
    """Horizontal fan-out: N units acting at the same hierarchy level.

    Each unit receives its own observation slice, the same parent context
    (when used inside a :class:`Pipeline`), and the same scheduling flags
    during ``update``. Outputs are returned as a list, one entry per
    member, in the order the members were registered.
    """

    def __init__(self, agents: Sequence[ExecutionUnit]):
        if not agents:
            raise ValueError("Ensemble requires at least one agent.")
        self.agents: List[ExecutionUnit] = list(agents)

    @property
    def use_raw_observations(self) -> bool:  # type: ignore[override]
        return any(agent.use_raw_observations for agent in self.agents)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def predict(
        self,
        observations,
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> List[Any]:
        results: List[Any] = []
        for index, agent in enumerate(self.agents):
            obs_slice = [observations[index]] if index < len(observations) else []
            output = agent.predict(obs_slice, deterministic, context=context)
            # Agents return a list-of-actions; for a single-agent slice we
            # unwrap the outer container so the ensemble result reads as
            # one row per member.
            if isinstance(output, list) and len(output) == 1:
                results.append(output[0])
            else:
                results.append(output)
        return results

    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        for index, agent in enumerate(self.agents):
            agent.update(
                [observations[index]] if index < len(observations) else [],
                [actions[index]] if index < len(actions) else [],
                [rewards[index]] if index < len(rewards) else [],
                [next_observations[index]] if index < len(next_observations) else [],
                terminated,
                truncated,
                update_target_step=update_target_step,
                global_learning_step=global_learning_step,
                update_step=update_step,
                initial_exploration_done=initial_exploration_done,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return all(
            agent.is_initial_exploration_done(global_learning_step)
            for agent in self.agents
        )

    def attach_environment(
        self,
        *,
        observation_names,
        action_names,
        action_space,
        observation_space,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        for index, agent in enumerate(self.agents):
            agent.attach_environment(
                observation_names=(
                    [observation_names[index]] if index < len(observation_names) else []
                ),
                action_names=(
                    [action_names[index]] if index < len(action_names) else []
                ),
                action_space=(
                    [action_space[index]] if index < len(action_space) else []
                ),
                observation_space=(
                    [observation_space[index]] if index < len(observation_space) else []
                ),
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for index, agent in enumerate(self.agents):
            agent_dir = root / f"agent_{index}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                agent.save_checkpoint(str(agent_dir), step)
            except NotImplementedError:
                continue
        return str(root)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Ensemble checkpoint root not found: {root}")
        for index, agent in enumerate(self.agents):
            agent_dir = root / f"agent_{index}"
            if not agent_dir.exists():
                continue
            try:
                agent.load_checkpoint(str(agent_dir))
            except NotImplementedError:
                continue

    def export_artifacts(
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        members_metadata: List[Dict[str, Any]] = []
        for index, agent in enumerate(self.agents):
            agent_dir = root / f"agent_{index}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            metadata = agent.export_artifacts(str(agent_dir), context)
            members_metadata.append({"agent_index": index, **(metadata or {})})
        return {"format": "ensemble", "agents": members_metadata}
