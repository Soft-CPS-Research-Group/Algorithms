import inspect

import pytest

from algorithms.registry import (
    ALGORITHM_REGISTRY,
    _stage_to_agent_view,
    build_execution_unit,
    build_unsupported_algorithm_message,
    is_algorithm_supported,
)
from algorithms.pipeline import Ensemble, Pipeline


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


def test_registered_agents_accept_predict_context_keyword():
    for name, agent_cls in ALGORITHM_REGISTRY.items():
        signature = inspect.signature(agent_cls.predict)
        parameter = signature.parameters.get("context")

        assert parameter is not None, f"{name}.predict must accept context"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_fixed_price_signal_is_registered_and_emits_configured_multiplier():
    unit = ALGORITHM_REGISTRY["FixedPriceSignal"](
        config={"algorithm": {"hyperparameters": {"multiplier": 1.0}}}
    )

    assert unit.use_raw_observations is True
    assert unit.predict([], deterministic=True) == 1.0


def test_fixed_price_signal_can_emit_one_multiplier_per_pipeline_member():
    unit = ALGORITHM_REGISTRY["FixedPriceSignal"](
        config={"algorithm": {"hyperparameters": {"multipliers": [0.9, 1.05]}}}
    )

    assert unit.predict([], deterministic=True) == [0.9, 1.05]
    artifact = unit.export_artifacts("unused")
    assert artifact["multipliers"] == [0.9, 1.05]
    assert artifact["output_contract"] == "per_member_price_multiplier_vector"


def test_fixed_price_signal_can_emit_a_step_schedule():
    unit = ALGORITHM_REGISTRY["FixedPriceSignal"](
        config={
            "algorithm": {
                "hyperparameters": {
                    "schedule": [
                        {"start_step": 0, "multiplier": 1.05},
                        {"start_step": 96, "multiplier": 1.025},
                    ]
                }
            }
        }
    )

    unit.set_episode_context(episode_step=95)
    assert unit.predict([], deterministic=True) == 1.05
    unit.set_episode_context(episode_step=96)
    assert unit.predict([], deterministic=True) == 1.025
    artifact = unit.export_artifacts("unused")
    assert artifact["schedule"][1] == {
        "start_step": 96,
        "multiplier": 1.025,
    }
    assert artifact["output_contract"] == "scheduled_global_price_multiplier"


def test_hierarchical_raw_observation_agents_are_registered():
    for name in (
        "BuildingAgent",
        "CommunityCoordinator",
        "SignalAwareRBC",
        "SignalAwareRBCSmartLocal",
    ):
        assert is_algorithm_supported(name)
        assert ALGORITHM_REGISTRY[name]._use_raw_observations is True


def test_fixed_service_oracle_replay_policy_is_registered_as_raw_observation_policy():
    assert is_algorithm_supported("FixedServiceOracleReplayPolicy")
    assert ALGORITHM_REGISTRY["FixedServiceOracleReplayPolicy"]._use_raw_observations is True
    assert is_algorithm_supported("TotalHomeOracleReplayPolicy")
    assert ALGORITHM_REGISTRY["TotalHomeOracleReplayPolicy"]._use_raw_observations is True
    assert is_algorithm_supported("TotalOracleReplayPolicy")
    assert ALGORITHM_REGISTRY["TotalOracleReplayPolicy"]._use_raw_observations is True


def test_strict_local_rbc_is_registered_as_raw_observation_policy():
    assert is_algorithm_supported("RBCSmartLocalPolicy")
    assert ALGORITHM_REGISTRY["RBCSmartLocalPolicy"]._use_raw_observations is True
    assert is_algorithm_supported("SignalAwareRBCSmartLocal")
    assert ALGORITHM_REGISTRY["SignalAwareRBCSmartLocal"]._use_raw_observations is True


def test_hierarchical_manager_agents_use_encoded_observations():
    for name in ("CCLevel1", "CCLevel2"):
        assert is_algorithm_supported(name)
        assert ALGORITHM_REGISTRY[name]._use_raw_observations is False


def test_build_execution_unit_supports_cc_level1_signal_aware_rbc_pipeline():
    config = {
        "pipeline": [
            {
                "algorithm": "CCLevel1",
                "count": 1,
                "hyperparameters": {"output_mode": "signal"},
            },
            {"algorithm": "SignalAwareRBC", "count": 2},
        ],
    }

    unit = build_execution_unit(config)

    assert isinstance(unit, Pipeline)
    assert unit.stages[0].__class__.__name__ == "CCLevel1Agent"
    assert isinstance(unit.stages[1], Ensemble)
    assert len(unit.stages[1].agents) == 2


def test_build_execution_unit_supports_cc_level2_signal_aware_rbc_pipeline():
    config = {
        "pipeline": [
            {
                "algorithm": "CCLevel2",
                "count": 1,
                "hyperparameters": {"num_buildings": 2},
            },
            {"algorithm": "SignalAwareRBC", "count": 2},
        ],
    }

    unit = build_execution_unit(config)

    assert isinstance(unit, Pipeline)
    assert unit.stages[0].__class__.__name__ == "CCLevel2Agent"
    assert isinstance(unit.stages[1], Ensemble)
    assert len(unit.stages[1].agents) == 2


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
            "behavior_cloning": {"enabled": True},
        }
        view = _stage_to_agent_view({}, stage)

        algorithm_block = view["algorithm"]
        assert algorithm_block["networks"] == {"actor": {"layers": [8]}}
        assert algorithm_block["replay_buffer"]["class"] == "MultiAgentReplayBuffer"
        assert algorithm_block["exploration"]["strategy"] == "GaussianNoise"
        assert algorithm_block["behavior_cloning"] == {"enabled": True}

    def test_does_not_mutate_input_global_config(self) -> None:
        global_config = {"metadata": {"experiment_name": "exp"}}
        stage = {"algorithm": "RuleBasedPolicy"}

        _stage_to_agent_view(global_config, stage)

        assert "algorithm" not in global_config
