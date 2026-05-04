import pytest

from algorithms.registry import (
    _stage_to_agent_view,
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


def test_build_execution_unit_error_for_null_algorithm_in_stage():
    config = {"pipeline": [{"algorithm": None}]}
    with pytest.raises(ValueError) as exc_info:
        build_execution_unit(config)
    assert "Algorithm name is required" in str(exc_info.value)


def test_build_unsupported_algorithm_message_for_missing_name():
    message = build_unsupported_algorithm_message(None)
    assert "Algorithm name is required" in message


# ----------------------------------------------------------------------
# _stage_to_agent_view
# ----------------------------------------------------------------------
class TestStageToAgentView:
    def test_synthesises_algorithm_block_with_name_and_hyperparameters(self) -> None:
        global_config = {
            "metadata": {"experiment_name": "exp"},
            "training": {"seed": 7},
            "pipeline": [{"algorithm": "MADDPG", "hyperparameters": {"gamma": 0.99}}],
        }
        stage = global_config["pipeline"][0]

        view = _stage_to_agent_view(global_config, stage)

        assert view["algorithm"]["name"] == "MADDPG"
        assert view["algorithm"]["hyperparameters"] == {"gamma": 0.99}

    def test_preserves_global_config_keys(self) -> None:
        global_config = {
            "metadata": {"experiment_name": "exp"},
            "training": {"seed": 7},
            "simulator": {"central_agent": False},
            "pipeline": [{"algorithm": "RuleBasedPolicy"}],
        }
        stage = global_config["pipeline"][0]

        view = _stage_to_agent_view(global_config, stage)

        assert view["metadata"] == {"experiment_name": "exp"}
        assert view["training"] == {"seed": 7}
        assert view["simulator"] == {"central_agent": False}

    def test_omits_optional_subblocks_when_absent_or_none(self) -> None:
        stage = {"algorithm": "RuleBasedPolicy", "networks": None, "exploration": None}
        view = _stage_to_agent_view({}, stage)

        algorithm_block = view["algorithm"]
        assert "networks" not in algorithm_block
        assert "exploration" not in algorithm_block
        assert "replay_buffer" not in algorithm_block

    def test_includes_optional_subblocks_when_present(self) -> None:
        stage = {
            "algorithm": "MADDPG",
            "hyperparameters": {"gamma": 0.99},
            "networks": {"actor": {"layers": [8]}},
            "replay_buffer": {"class": "MultiAgentReplayBuffer", "capacity": 10, "batch_size": 2},
            "exploration": {"strategy": "GaussianNoise", "params": {}},
        }
        view = _stage_to_agent_view({}, stage)

        algorithm_block = view["algorithm"]
        assert algorithm_block["networks"] == {"actor": {"layers": [8]}}
        assert algorithm_block["replay_buffer"]["class"] == "MultiAgentReplayBuffer"
        assert algorithm_block["exploration"]["strategy"] == "GaussianNoise"

    def test_does_not_mutate_input_global_config(self) -> None:
        global_config = {"metadata": {"experiment_name": "exp"}}
        stage = {"algorithm": "RuleBasedPolicy"}

        _stage_to_agent_view(global_config, stage)

        assert "algorithm" not in global_config
