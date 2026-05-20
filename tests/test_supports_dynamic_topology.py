"""Tests for the registry-driven ``supports_dynamic_topology`` flag.

Spec ``docs/specv2.md`` §12.4: dynamic-topology permission is decided by a
ClassVar on the agent class, not by hardcoded class-name comparisons in the
wrapper or the schema. The cross-config validator behaviour is already
covered by ``tests/test_config_validation.py``
(``test_validate_config_rejects_maddpg_with_entity_dynamic`` and
``test_validate_config_accepts_rule_based_with_entity_dynamic``); this
module pins the per-class ClassVar values so the dispatch above keeps the
right answer for each concrete agent.
"""

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
