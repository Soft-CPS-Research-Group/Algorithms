"""Typed-Interface Multi-Agent Reinforcement Learning.

The public runtime entrypoint is :class:`TIMARL`. Internal packages keep the
typed contract/compiler independent from its PPO or MAPPO learning backbone.
"""

from algorithms.ti_marl.agent import TIMARL

__all__ = ["TIMARL"]
