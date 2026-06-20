"""Algorithm registry and execution-unit builder for the training entrypoint."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Type


_REGISTRY_TRACE_ENABLED = (
    os.environ.get("OPEVA_STARTUP_TRACE", "1").strip().lower() not in {"0", "false", "no", "off"}
    and os.path.basename(sys.argv[0]) == "run_experiment.py"
)
_REGISTRY_TRACE_T0 = time.monotonic()


def _registry_trace(message: str) -> None:
    if not _REGISTRY_TRACE_ENABLED:
        return
    elapsed = time.monotonic() - _REGISTRY_TRACE_T0
    print(f"[opeva-registry +{elapsed:.3f}s] {message}", file=sys.stderr, flush=True)


_registry_trace("module import started")

_registry_trace("before loguru import")
from loguru import logger
_registry_trace("after loguru import")

_registry_trace("before baseline policies import")
from algorithms.agents.baseline_policies import (
    NormalNoBatteryPolicy,
    NormalPolicy,
    RBCBasicPolicy,
    RBCCommunityPolicy,
    RBCSmartPolicy,
    RandomPolicy,
    SignalAwareRBC,
)
_registry_trace("after baseline policies import")
_registry_trace("before base agent import")
from algorithms.agents.base_agent import BaseAgent
_registry_trace("after base agent import")
_registry_trace("before building agent import")
from algorithms.agents.building_agent import BuildingAgent
_registry_trace("after building agent import")
_registry_trace("before cc level1 import")
from algorithms.agents.cc_level1_agent import CCLevel1Agent
_registry_trace("after cc level1 import")
_registry_trace("before community coordinator import")
from algorithms.agents.community_coordinator_agent import CommunityCoordinatorAgent
_registry_trace("after community coordinator import")
_registry_trace("before district data collection import")
from algorithms.agents.district_data_collection_agent import DistrictDataCollectionRBC
_registry_trace("after district data collection import")
_registry_trace("before ev data collection import")
from algorithms.agents.ev_data_collection_agent import EVDataCollectionRBC
_registry_trace("after ev data collection import")
_registry_trace("before MADDPG import")
from algorithms.agents.maddpg_agent import MADDPG
_registry_trace("after MADDPG import")
_registry_trace("before MASAC import")
from algorithms.agents.masac_agent import MASAC
_registry_trace("after MASAC import")
_registry_trace("before MATD3 import")
from algorithms.agents.matd3_agent import MATD3
_registry_trace("after MATD3 import")
_registry_trace("before PPO agents import")
from algorithms.agents.ppo_agents import HAPPO, IPPO, MAPPO
_registry_trace("after PPO agents import")
_registry_trace("before RuleBasedPolicy import")
from algorithms.agents.rbc_agent import RuleBasedPolicy
_registry_trace("after RuleBasedPolicy import")
_registry_trace("before CQL entity agent import")
from algorithms.offline_rl.cql_entity_agent import CQLEntityAgent
_registry_trace("after CQL entity agent import")
_registry_trace("before IQL entity agent import")
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent
_registry_trace("after IQL entity agent import")
_registry_trace("before execution unit import")
from algorithms.execution_unit import ExecutionUnit
_registry_trace("after execution unit import")
_registry_trace("before pipeline import")
from algorithms.pipeline import Ensemble, Pipeline
_registry_trace("after pipeline import")

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "BuildingAgent": BuildingAgent,
    "CCLevel1": CCLevel1Agent,
    "CQLEntityAgent": CQLEntityAgent,
    "CommunityCoordinator": CommunityCoordinatorAgent,
    "DistrictDataCollectionRBC": DistrictDataCollectionRBC,
    "EVDataCollectionRBC": EVDataCollectionRBC,
    "HAPPO": HAPPO,
    "IPPO": IPPO,
    "IQLEntityAgent": IQLEntityAgent,
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
_registry_trace("registry built")


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
