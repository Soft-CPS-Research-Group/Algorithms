import pytest

from algorithms.registry import (
    build_unsupported_algorithm_message,
    create_agent,
    is_algorithm_supported,
)


def test_registry_marks_single_agent_placeholder_as_unsupported():
    assert is_algorithm_supported("SingleAgentRL") is False


def test_create_agent_error_for_placeholder_includes_supported_and_placeholders():
    config = {"algorithm": {"name": "SingleAgentRL"}}
    with pytest.raises(ValueError) as exc_info:
        create_agent(config)

    message = str(exc_info.value)
    assert "SingleAgentRL" in message
    assert "Supported algorithms" in message
    assert "MADDPG" in message
    assert "RuleBasedPolicy" in message
    assert "Known placeholders" in message


def test_build_unsupported_algorithm_message_for_missing_name():
    message = build_unsupported_algorithm_message(None)
    assert "Algorithm name is required" in message
