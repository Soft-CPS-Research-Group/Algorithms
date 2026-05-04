import pytest

from algorithms.registry import (
    build_execution_unit,
    build_unsupported_algorithm_message,
    is_algorithm_supported,
)


def test_registry_marks_single_agent_placeholder_as_unsupported():
    assert is_algorithm_supported("SingleAgentRL") is False


def test_build_execution_unit_error_for_placeholder_includes_supported_and_placeholders():
    config = {"pipeline": [{"algorithm": "SingleAgentRL"}]}
    with pytest.raises(ValueError) as exc_info:
        build_execution_unit(config)

    message = str(exc_info.value)
    assert "SingleAgentRL" in message
    assert "Supported algorithms" in message
    assert "MADDPG" in message
    assert "RuleBasedPolicy" in message
    assert "Known placeholders" in message


def test_build_execution_unit_error_for_empty_pipeline():
    with pytest.raises(ValueError) as exc_info:
        build_execution_unit({"pipeline": []})
    assert "Algorithm name is required" in str(exc_info.value)


def test_build_unsupported_algorithm_message_for_missing_name():
    message = build_unsupported_algorithm_message(None)
    assert "Algorithm name is required" in message
