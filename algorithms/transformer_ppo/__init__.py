"""Transformer PPO agent and its entity-interface training components."""

from typing import Any

__all__ = ["AgentTransformerPPO"]


def __getattr__(name: str) -> Any:
    if name == "AgentTransformerPPO":
        from algorithms.transformer_ppo.agent import AgentTransformerPPO

        return AgentTransformerPPO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
