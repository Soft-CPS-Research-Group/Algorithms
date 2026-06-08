"""Algorithm registry and execution-unit builder for the training entrypoint."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from loguru import logger

from algorithms.agents.baseline_policies import (
    NormalNoBatteryPolicy,
    NormalPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartPolicy,
    RandomPolicy,
    SignalAwareRBC,
)
from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.building_agent import BuildingAgent
from algorithms.agents.cc_level1_agent import CCLevel1Agent
from algorithms.agents.community_coordinator_agent import CommunityCoordinatorAgent
from algorithms.agents.maddpg_agent import MADDPG
from algorithms.agents.masac_agent import MASAC
from algorithms.agents.matd3_agent import MATD3
from algorithms.agents.ppo_agents import HAPPO, IPPO, MAPPO
from algorithms.agents.rbc_agent import RuleBasedPolicy
from algorithms.execution_unit import ExecutionUnit
from algorithms.pipeline import Ensemble, Pipeline

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "BuildingAgent": BuildingAgent,
    "CCLevel1": CCLevel1Agent,
    "CommunityCoordinator": CommunityCoordinatorAgent,
    "HAPPO": HAPPO,
    "IPPO": IPPO,
    "MADDPG": MADDPG,
    "MAPPO": MAPPO,
    "MASAC": MASAC,
    "MATD3": MATD3,
    "NormalNoBatteryPolicy": NormalNoBatteryPolicy,
    "NormalPolicy": NormalPolicy,
    "RBCBasicPolicy": RBCBasicPolicy,
    "RBCCommunityPolicy": RBCCommunityPolicy,
    "RBCSmartPolicy": RBCSmartPolicy,
    "RandomPolicy": RandomPolicy,
    "RuleBasedPolicy": RuleBasedPolicy,
    "SignalAwareRBC": SignalAwareRBC,
}

PLACEHOLDER_ALGORITHMS = {
    "SingleAgentRL",
}

# Derived from the registry: algorithms whose class sets _use_raw_observations=False
# (i.e. neural agents that need the wrapper's observation encoding).
ENCODED_OBSERVATION_ALGORITHMS: frozenset[str] = frozenset(
    name for name, cls in ALGORITHM_REGISTRY.items()
    if not cls._use_raw_observations
)


def supported_algorithms() -> List[str]:
    """Return registered algorithm names sorted for stable error messages."""
    return sorted(ALGORITHM_REGISTRY.keys())


def placeholder_algorithms() -> List[str]:
    """Return known schema placeholders that are not yet runtime implementations."""
    return sorted(PLACEHOLDER_ALGORITHMS)


def is_algorithm_supported(name: str | None) -> bool:
    """Check whether ``name`` is backed by a runtime implementation."""
    return bool(name) and name in ALGORITHM_REGISTRY


def build_unsupported_algorithm_message(name: str | None) -> str:
    """Build a clear fail-fast message for unsupported/placeholder algorithms."""
    supported = ", ".join(supported_algorithms()) or "none"
    placeholders = ", ".join(placeholder_algorithms()) or "none"

    if not name:
        return (
            "Algorithm name is required in configuration. "
            f"Supported algorithms: {supported}. "
            f"Known placeholders: {placeholders}."
        )

    if name in PLACEHOLDER_ALGORITHMS:
        return (
            f"Algorithm '{name}' is a schema placeholder and has no runtime implementation yet. "
            f"Supported algorithms: {supported}. "
            f"Known placeholders: {placeholders}."
        )

    return (
        f"Unsupported algorithm '{name}'. "
        f"Supported algorithms: {supported}. "
        f"Known placeholders: {placeholders}."
    )


def _stage_to_agent_view(global_config: Dict[str, Any], stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesise an agent-facing config view for a single pipeline stage.

    Existing agent constructors read ``self.config["algorithm"][...]`` as
    historical convention. Rather than rewriting every agent in this
    branch, the builder copies the global config and substitutes a stage
    slice under ``algorithm`` so each agent sees the same shape it has
    always seen. The migration to per-stage config objects can happen
    independently in a follow-up, without changing this builder.
    """
    agent_view = dict(global_config)
    algorithm_block: Dict[str, Any] = {
        "name": stage_cfg["algorithm"],
        "hyperparameters": stage_cfg.get("hyperparameters", {}) or {},
    }
    for optional_key in ("networks", "replay_buffer", "exploration", "policy"):
        if optional_key in stage_cfg and stage_cfg[optional_key] is not None:
            algorithm_block[optional_key] = stage_cfg[optional_key]
    agent_view["algorithm"] = algorithm_block
    return agent_view


def build_execution_unit(config: Dict[str, Any]) -> ExecutionUnit:
    """Instantiate the model the wrapper drives.

    Reads ``config['pipeline']`` (an ordered list of stage descriptions)
    and produces:

    * a single :class:`BaseAgent` when the pipeline has exactly one
      stage with ``count == 1`` (current default — backwards compatible
      with the historical single-agent flow),
    * an :class:`Ensemble` for a single stage with ``count > 1``,
    * a :class:`Pipeline` of stages otherwise (each entry being either
      a single agent or an :class:`Ensemble` when ``count > 1``).

    Adding a new hierarchy level is purely a configuration change — no
    code change here, in the wrapper, or in any agent class.
    """
    pipeline_cfg = config.get("pipeline") or []
    if not pipeline_cfg:
        message = build_unsupported_algorithm_message(None)
        logger.error(message)
        raise ValueError(message)

    stages: List[ExecutionUnit] = []
    for stage_cfg in pipeline_cfg:
        algorithm_name = stage_cfg.get("algorithm")
        if not is_algorithm_supported(algorithm_name):
            message = build_unsupported_algorithm_message(algorithm_name)
            logger.error(message)
            raise ValueError(message)

        agent_cls = ALGORITHM_REGISTRY[algorithm_name]
        agent_view = _stage_to_agent_view(config, stage_cfg)
        count = int(stage_cfg.get("count", 1) or 1)
        if count < 1:
            raise ValueError(
                f"Stage '{algorithm_name}' has count={count}; must be >= 1."
            )

        frozen = bool(stage_cfg.get("frozen", False))

        if count == 1:
            unit = agent_cls(config=agent_view)
            unit.frozen = frozen
            stages.append(unit)
        else:
            members = [agent_cls(config=agent_view) for _ in range(count)]
            ensemble = Ensemble(members)
            ensemble.frozen = frozen
            stages.append(ensemble)

    if len(stages) == 1:
        return stages[0]
    return Pipeline(stages)
