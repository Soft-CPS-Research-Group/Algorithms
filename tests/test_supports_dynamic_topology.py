"""Pin the ``supports_dynamic_topology`` ClassVar on each registered agent
class so the wrapper guardrail and config validator agree on capability."""

from __future__ import annotations


def test_base_agent_default_is_false() -> None:
    from algorithms.agents.base_agent import BaseAgent

    assert BaseAgent.supports_dynamic_topology is False


def test_maddpg_does_not_support_dynamic_topology() -> None:
    from algorithms.agents.maddpg_agent import MADDPG

    assert MADDPG.supports_dynamic_topology is False


def test_rule_based_policy_supports_dynamic_topology() -> None:
    from algorithms.agents.rbc_agent import RuleBasedPolicy

    assert RuleBasedPolicy.supports_dynamic_topology is True
