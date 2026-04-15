"""Algorithm registry for training entrypoint."""

from __future__ import annotations

from typing import Dict, List, Type

from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.ev_data_collection_agent import EVDataCollectionRBC
from algorithms.agents.maddpg_agent import MADDPG
from algorithms.agents.rbc_agent import RuleBasedPolicy

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    "RuleBasedPolicy": RuleBasedPolicy,
    "EVDataCollectionRBC": EVDataCollectionRBC,
}

PLACEHOLDER_ALGORITHMS = {
    "SingleAgentRL",
}


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


def create_agent(config: dict) -> BaseAgent:
    """Instantiate an agent based on the configuration."""
    algorithm_cfg = config.get("algorithm", {})
    name = algorithm_cfg.get("name")
    if not is_algorithm_supported(name):
        message = build_unsupported_algorithm_message(name)
        logger.error(message)
        raise ValueError(message)

    try:
        agent_cls = ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        # Defensive guard in case registry is mutated between checks.
        message = build_unsupported_algorithm_message(name)
        logger.error(message)
        raise ValueError(message) from exc

    return agent_cls(config=config)
