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

import numpy as np
from loguru import logger

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
        self._raw_observations: Optional[Any] = None
        self._encoded_observations: Optional[Any] = None
        self._raw_next_observations: Optional[Any] = None
        self._encoded_next_observations: Optional[Any] = None
        self._profiled_encoded_observations: Dict[str, Any] = {}
        self._profiled_encoded_next_observations: Dict[str, Any] = {}

    @property
    def use_raw_observations(self) -> bool:
        return all(stage.use_raw_observations for stage in self.stages)

    @property
    def requires_raw_observation_context(self) -> bool:
        return any(
            stage.use_raw_observations
            or bool(getattr(stage, "requires_raw_observation_context", False))
            or getattr(stage, "_warm_start_policy", None) is not None
            or bool(getattr(stage, "warm_start_policy_name", None))
            for stage in self.stages
        )

    def _observations_for_stage(self, stage: ExecutionUnit, fallback: Any) -> Any:
        if stage.use_raw_observations and self._raw_observations is not None:
            return self._raw_observations
        profile = str(getattr(stage, "observation_encoding_profile", "") or "").strip().lower()
        if profile and profile in self._profiled_encoded_observations:
            return self._profiled_encoded_observations[profile]
        if not stage.use_raw_observations and self._encoded_observations is not None:
            return self._encoded_observations
        return fallback

    def _next_observations_for_stage(self, stage: ExecutionUnit, fallback: Any) -> Any:
        if stage.use_raw_observations and self._raw_next_observations is not None:
            return self._raw_next_observations
        profile = str(getattr(stage, "observation_encoding_profile", "") or "").strip().lower()
        if profile and profile in self._profiled_encoded_next_observations:
            return self._profiled_encoded_next_observations[profile]
        if not stage.use_raw_observations and self._encoded_next_observations is not None:
            return self._encoded_next_observations
        return fallback

    def required_observation_encoding_profiles(self) -> List[str]:
        """Return non-default entity encodings requested by pipeline stages."""
        profiles = {
            str(getattr(stage, "observation_encoding_profile", "") or "").strip().lower()
            for stage in self.stages
        }
        return sorted(profile for profile in profiles if profile)

    def set_profiled_observation_context(
        self,
        profiled_encoded_observations: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._profiled_encoded_observations = dict(profiled_encoded_observations or {})

    def set_profiled_transition_context(
        self,
        *,
        profiled_encoded_observations: Optional[Dict[str, Any]] = None,
        profiled_encoded_next_observations: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._profiled_encoded_observations = dict(profiled_encoded_observations or {})
        self._profiled_encoded_next_observations = dict(
            profiled_encoded_next_observations or {}
        )

    def set_observation_context(
        self,
        *,
        raw_observations: Any = None,
        encoded_observations: Any = None,
    ) -> None:
        self._raw_observations = raw_observations
        self._encoded_observations = encoded_observations

        for stage in self.stages:
            hook = getattr(stage, "set_observation_context", None)
            if callable(hook):
                hook(
                    raw_observations=raw_observations,
                    encoded_observations=encoded_observations,
                )

    def set_transition_context(
        self,
        *,
        raw_observations: Any = None,
        raw_next_observations: Any = None,
        encoded_observations: Any = None,
        encoded_next_observations: Any = None,
    ) -> None:
        self._raw_observations = raw_observations
        self._raw_next_observations = raw_next_observations
        self._encoded_observations = encoded_observations
        self._encoded_next_observations = encoded_next_observations

        for stage in self.stages:
            hook = getattr(stage, "set_transition_context", None)
            if callable(hook):
                hook(
                    raw_observations=raw_observations,
                    raw_next_observations=raw_next_observations,
                    encoded_observations=encoded_observations,
                    encoded_next_observations=encoded_next_observations,
                )

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        """Propagate episode-local clock state independently of hierarchy context."""
        for stage in self.stages:
            hook = getattr(stage, "set_episode_context", None)
            if callable(hook):
                hook(
                    episode_step=episode_step,
                    next_episode_step=next_episode_step,
                )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def predict(
        self,
        observations: Any,
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> Any:
        ctx = context
        result: Any = None
        for stage in self.stages:
            stage_observations = self._observations_for_stage(stage, observations)
            # A frozen stage is an immutable behavioural dependency of the
            # trainable stages around it.  Letting it keep sampling actions
            # would inject leaf-policy noise into the manager's transition and
            # reward even though its parameters cannot adapt.  In particular,
            # a trainable CC must explore against the exact deterministic PPO
            # policy that will be used at evaluation/deployment time.
            stage_deterministic = (
                True if getattr(stage, "frozen", False) else deterministic
            )
            result = stage.predict(
                stage_observations,
                stage_deterministic,
                context=ctx,
            )
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
            if getattr(stage, "frozen", False):
                continue
            stage_observations = self._observations_for_stage(stage, observations)
            stage_next_observations = self._next_observations_for_stage(
                stage,
                next_observations,
            )
            stage.update(
                stage_observations,
                actions,
                rewards,
                stage_next_observations,
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

    def _collect_stage_metric_hook(self, hook_name: str) -> Dict[str, float]:
        """Expose diagnostics from every hierarchy stage without collisions."""

        metrics: Dict[str, float] = {}
        for index, stage in enumerate(self.stages):
            hook = getattr(stage, hook_name, None)
            if not callable(hook):
                continue
            payload = hook()
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    continue
                metrics[f"Pipeline/stage_{index}/{key}"] = numeric
        return metrics

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        return self._collect_stage_metric_hook("get_diagnostic_metrics")

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        return self._collect_stage_metric_hook("consume_latest_training_metrics")

    def attach_environment(self, **kwargs) -> None:
        metadata = kwargs.get("metadata") or {}
        raw_observation_names = metadata.get("raw_observation_names")
        encoded_observation_names = metadata.get("encoded_observation_names")
        profiled_encoded_observation_names = (
            metadata.get("profiled_encoded_observation_names") or {}
        )

        for stage in self.stages:
            stage_kwargs = dict(kwargs)
            profile = str(
                getattr(stage, "observation_encoding_profile", "") or ""
            ).strip().lower()
            if profile and profile in profiled_encoded_observation_names:
                stage_kwargs["observation_names"] = profiled_encoded_observation_names[
                    profile
                ]
            elif (
                stage.use_raw_observations
                or bool(getattr(stage, "requires_raw_observation_context", False))
                or getattr(stage, "_warm_start_policy", None) is not None
                or bool(getattr(stage, "warm_start_policy_name", None))
            ) and raw_observation_names is not None:
                stage_kwargs["observation_names"] = raw_observation_names
            elif not stage.use_raw_observations and encoded_observation_names is not None:
                stage_kwargs["observation_names"] = encoded_observation_names
            stage.attach_environment(**stage_kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        for index, stage in enumerate(self.stages):
            if getattr(stage, "frozen", False):
                logger.debug(
                    "Pipeline stage {} ({}) is frozen; skipping checkpoint save.",
                    index,
                    type(stage).__name__,
                )
                continue
            stage_dir = root / f"stage_{index}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            try:
                stage.save_checkpoint(str(stage_dir), step)
            except NotImplementedError:
                logger.debug(
                    "Pipeline stage {} ({}) does not implement save_checkpoint; skipping.",
                    index,
                    type(stage).__name__,
                )
                continue
        return str(root)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Pipeline checkpoint root not found: {root}")
        loaded_count = 0
        for index, stage in enumerate(self.stages):
            stage_dir = root / f"stage_{index}"
            if not stage_dir.exists():
                # Flat fallback: standalone checkpoint (no stage subdirs).
                # A model trained solo saves directly to the root dir.
                # Allow stage 0 to load from root if root contains checkpoint
                # files — covers the CC-trained-alone → HIRO-loaded scenario.
                if index == 0 and any(root.iterdir()):
                    logger.debug(
                        "Pipeline stage 0: no stage_0/ subdir found; "
                        "falling back to root '{}' (flat checkpoint format).",
                        root,
                    )
                    stage_dir = root
                else:
                    logger.debug(
                        "Pipeline stage {} ({}): checkpoint dir '{}' not found; skipping.",
                        index,
                        type(stage).__name__,
                        root / f"stage_{index}",
                    )
                    continue
            try:
                stage.load_checkpoint(str(stage_dir))
                loaded_count += 1
            except NotImplementedError:
                logger.debug(
                    "Pipeline stage {} ({}) does not implement load_checkpoint; skipping.",
                    index,
                    type(stage).__name__,
                )
                continue
        if loaded_count == 0:
            logger.warning(
                "Pipeline.load_checkpoint: no stage loaded anything from '{}'. "
                "All stages either had no checkpoint directory or do not implement "
                "load_checkpoint. The model was NOT restored.",
                checkpoint_path,
            )

    def load_stage_checkpoint(self, stage_index: int, checkpoint_path: str) -> None:
        """Restore exactly one stage from a standalone checkpoint root.

        This supports hierarchical composition where a new manager starts from
        scratch while a previously trained leaf is loaded and frozen. The path
        is passed directly to the selected stage, so an ``Ensemble`` can route
        its ``agent_<index>`` children exactly as in a standalone run.
        """
        index = int(stage_index)
        if index < 0 or index >= len(self.stages):
            raise IndexError(
                f"Pipeline stage index {index} is outside range 0:{len(self.stages) - 1}."
            )

        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Pipeline stage checkpoint root not found: {root}")

        stage = self.stages[index]
        try:
            stage.load_checkpoint(str(root))
        except NotImplementedError as exc:
            raise RuntimeError(
                f"Pipeline stage {index} ({type(stage).__name__}) does not support checkpoint loading."
            ) from exc

        logger.info(
            "Pipeline stage {} ({}) loaded from standalone checkpoint root {}",
            index,
            type(stage).__name__,
            root,
        )

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
        for agent in self.agents:
            # Child metrics are aggregated by the Ensemble and then logged by
            # the wrapper.  Direct child logging would make N independent
            # learners overwrite the same metric keys at every step.
            setattr(agent, "managed_by_ensemble", True)

    @property
    def use_raw_observations(self) -> bool:
        return any(agent.use_raw_observations for agent in self.agents)

    @property
    def requires_raw_observation_context(self) -> bool:
        """Whether any member needs raw observations beside its model input.

        Independent neural learners normally consume encoded observations, but
        a warm-start/behaviour-cloning teacher such as ``RBCSmartPolicy`` uses
        the raw semantic observation stream.  Exposing this recursively keeps
        the wrapper from selecting the direct encoded-only fast path.
        """
        return any(
            agent.use_raw_observations
            or bool(getattr(agent, "requires_raw_observation_context", False))
            or getattr(agent, "_warm_start_policy", None) is not None
            or bool(getattr(agent, "warm_start_policy_name", None))
            for agent in self.agents
        )

    @staticmethod
    def _member_observation_slice(observations: Any, index: int) -> Any:
        if observations is None:
            return None
        try:
            if index >= len(observations):
                return []
        except TypeError:
            return observations
        return [observations[index]]

    def set_observation_context(
        self,
        *,
        raw_observations: Any = None,
        encoded_observations: Any = None,
    ) -> None:
        """Fan wrapper observation context out to the matching member.

        This is deliberately separate from :meth:`predict`: teachers must see
        the raw observation for their own building while the learned policy
        continues to receive the encoded vector.
        """
        for index, agent in enumerate(self.agents):
            hook = getattr(agent, "set_observation_context", None)
            if callable(hook):
                hook(
                    raw_observations=self._member_observation_slice(
                        raw_observations,
                        index,
                    ),
                    encoded_observations=self._member_observation_slice(
                        encoded_observations,
                        index,
                    ),
                )

    def set_transition_context(
        self,
        *,
        raw_observations: Any = None,
        raw_next_observations: Any = None,
        encoded_observations: Any = None,
        encoded_next_observations: Any = None,
    ) -> None:
        """Fan current/next context out for teacher-aware replay and BC."""
        for index, agent in enumerate(self.agents):
            hook = getattr(agent, "set_transition_context", None)
            if callable(hook):
                hook(
                    raw_observations=self._member_observation_slice(
                        raw_observations,
                        index,
                    ),
                    raw_next_observations=self._member_observation_slice(
                        raw_next_observations,
                        index,
                    ),
                    encoded_observations=self._member_observation_slice(
                        encoded_observations,
                        index,
                    ),
                    encoded_next_observations=self._member_observation_slice(
                        encoded_next_observations,
                        index,
                    ),
                )

    def set_episode_context(
        self,
        *,
        episode_step: Optional[int] = None,
        next_episode_step: Optional[int] = None,
    ) -> None:
        """Give every independent member the same episode-local clock."""
        for agent in self.agents:
            hook = getattr(agent, "set_episode_context", None)
            if callable(hook):
                hook(
                    episode_step=episode_step,
                    next_episode_step=next_episode_step,
                )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def predict(
        self,
        observations: Any,
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> List[Any]:
        # Per-member context distribution. A hierarchical manager (e.g. the CC)
        # emits ONE signal per member as an array of length == ensemble size.
        # In that case member i receives its own element context[i]. A scalar
        # (or any non-matching context) is broadcast unchanged to every member.
        try:
            context_len = len(context) if isinstance(context, (list, tuple, np.ndarray)) else None
        except TypeError:
            context_len = None
        ctx_is_per_member = context_len == len(self.agents)

        results: List[Any] = []
        for index, agent in enumerate(self.agents):
            if index >= len(observations):
                logger.warning(
                    "Ensemble member {} received no observations "
                    "(observations length={}, ensemble size={}). "
                    "Member will see an empty obs slice.",
                    index,
                    len(observations),
                    len(self.agents),
                )
                obs_slice: List[Any] = []
            else:
                obs_slice = [observations[index]]

            member_context = context[index] if ctx_is_per_member else context
            output = agent.predict(obs_slice, deterministic, context=member_context)

            # Contract: each member receives one obs slice and must return
            # exactly one row (the action vector for that member). Anything
            # else is an agent-side bug — fail loud rather than silently
            # mangle the ensemble output.
            if isinstance(output, list):
                if len(output) != 1:
                    raise RuntimeError(
                        f"Ensemble member {index} returned {len(output)} rows "
                        f"for a single observation slice; expected exactly 1."
                    )
                results.append(output[0])
            else:
                # Non-list outputs (e.g. context tensors from a non-leaf
                # agent) are forwarded unchanged.
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
        n_obs = len(observations)
        if n_obs != len(self.agents):
            raise RuntimeError(
                f"Ensemble.update: observations length ({n_obs}) does not match "
                f"ensemble size ({len(self.agents)}). "
                "Ensure the wrapper supplies one observation slice per ensemble member."
            )
        for index, agent in enumerate(self.agents):
            agent.update(
                [observations[index]],
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

    def _aggregate_metric_hook(self, hook_name: str) -> Dict[str, float]:
        values_by_key: Dict[str, List[float]] = {}
        for agent in self.agents:
            hook = getattr(agent, hook_name, None)
            if not callable(hook):
                continue
            payload = hook()
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    continue
                values_by_key.setdefault(str(key), []).append(numeric)

        metrics: Dict[str, float] = {
            "Ensemble/member_count": float(len(self.agents)),
        }
        for key, values in values_by_key.items():
            array = np.asarray(values, dtype=np.float64)
            metrics[f"Ensemble/{key}_mean"] = float(np.mean(array))
            metrics[f"Ensemble/{key}_min"] = float(np.min(array))
            metrics[f"Ensemble/{key}_max"] = float(np.max(array))
        return metrics

    def get_diagnostic_metrics(self) -> Dict[str, float]:
        """Aggregate member status without leaking one member over another."""
        return self._aggregate_metric_hook("get_diagnostic_metrics")

    def consume_latest_training_metrics(self) -> Dict[str, float]:
        """Consume and aggregate the latest independent learner metrics."""
        return self._aggregate_metric_hook("consume_latest_training_metrics")

    def attach_environment(
        self,
        *,
        observation_names,
        action_names,
        action_space,
        observation_space,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        env_agent_count = len(action_names) if action_names is not None else 0
        if env_agent_count != len(self.agents):
            raise ValueError(
                f"Ensemble size mismatch: ensemble has {len(self.agents)} member(s) "
                f"but the environment exposes {env_agent_count} agent slot(s). "
                f"Adjust 'count' in the pipeline config to match the environment."
            )
        for index, agent in enumerate(self.agents):
            member_metadata = dict(metadata or {})
            for key in (
                "building_names",
                "raw_observation_names",
                "encoded_observation_names",
                "raw_observation_bounds",
            ):
                value = member_metadata.get(key)
                if isinstance(value, (list, tuple)) and index < len(value):
                    member_metadata[key] = [value[index]]
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
                metadata=member_metadata,
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
                logger.debug(
                    "Ensemble member {} ({}) does not implement save_checkpoint; skipping.",
                    index,
                    type(agent).__name__,
                )
                continue
        return str(root)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        root = Path(checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(f"Ensemble checkpoint root not found: {root}")
        if root.is_file():
            root = root.parent
        for index, agent in enumerate(self.agents):
            agent_dir = root / f"agent_{index}"
            if not agent_dir.exists():
                continue
            try:
                artifact_name = str(
                    getattr(agent, "checkpoint_artifact", "latest_checkpoint.pth")
                    or "latest_checkpoint.pth"
                )
                artifact_path = agent_dir / artifact_name
                agent.load_checkpoint(
                    str(artifact_path if artifact_path.exists() else agent_dir)
                )
            except NotImplementedError:
                logger.debug(
                    "Ensemble member {} ({}) does not implement load_checkpoint; skipping.",
                    index,
                    type(agent).__name__,
                )
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
            member_context = dict(context or {})
            member_context["agent_index_offset"] = index
            metadata = agent.export_artifacts(str(agent_dir), member_context)
            members_metadata.append({"agent_index": index, **(metadata or {})})
        return {"format": "ensemble", "agents": members_metadata}
