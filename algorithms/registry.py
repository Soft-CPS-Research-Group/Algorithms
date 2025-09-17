"""Algorithm registry for training entrypoint."""

from __future__ import annotations

from typing import Dict, Type

from loguru import logger

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.maddpg_agent import MADDPG

ALGORITHM_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "MADDPG": MADDPG,
    # Additional algorithms can be registered here.
}


def create_agent(config: dict) -> BaseAgent:
    """Instantiate an agent based on the configuration."""
    algorithm_cfg = config.get("algorithm", {})
    name = algorithm_cfg.get("name")
    if not name:
        raise ValueError("Algorithm name is required in configuration.")

    try:
        agent_cls = ALGORITHM_REGISTRY[name]
    except KeyError as exc:
        logger.error("Algorithm '{}' is not registered.", name)
        raise ValueError(f"Unsupported algorithm '{name}'.") from exc

    return agent_cls(config=config)
