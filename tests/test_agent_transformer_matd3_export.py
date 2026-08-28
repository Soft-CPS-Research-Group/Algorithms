from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest

from tests.test_agent_transformer_matd3 import _config, _make_agent, _transition
from utils.artifact_manifest import build_manifest
from utils.bundle_validator import validate_bundle_contract


def test_export_writes_one_topology_versioned_opset17_model_per_building(
    tmp_path: Path,
) -> None:
    agent, obs_dim = _make_agent(buildings=2)
    agent._per_building[0].topology_version = 3
    agent._per_building[1].topology_version = 4

    metadata = agent.export_artifacts(str(tmp_path))

    assert metadata["format"] == "onnx"
    assert metadata["supports_dynamic_topology"] is True
    assert len(metadata["artifacts"]) == 2
    assert len(metadata["agent_models"]) == 2
    for index, artifact in enumerate(metadata["artifacts"]):
        version = index + 3
        assert artifact["path"] == (
            f"onnx_models/agent_{index}__topology_v{version}.onnx"
        )
        model_path = tmp_path / artifact["path"]
        model = onnx.load(model_path)
        assert model.opset_import[0].version == 17
        assert [value.name for value in model.graph.input] == ["encoded_obs"]
        assert [value.name for value in model.graph.output] == ["actions"]
        input_shape = model.graph.input[0].type.tensor_type.shape.dim
        output_shape = model.graph.output[0].type.tensor_type.shape.dim
        assert input_shape[0].dim_param == "batch"
        assert input_shape[1].dim_value == obs_dim
        assert output_shape[0].dim_param == "batch"
        assert output_shape[1].dim_value == 2
        assert artifact["config"]["deployable"] is True
        assert artifact["config"]["requires_runtime_residual"] is False
        assert artifact["config"]["requires_runtime_local_action_safety"] is False
        assert artifact["config"]["requires_runtime_local_price_conditioning"] is False


def test_exported_model_matches_deterministic_actor(tmp_path: Path) -> None:
    from onnx.reference import ReferenceEvaluator

    agent, obs_dim = _make_agent(buildings=1)
    observation = np.linspace(-0.5, 0.5, obs_dim, dtype=np.float32)
    expected = np.asarray(
        agent.predict([observation], deterministic=True), dtype=np.float32
    )

    metadata = agent.export_artifacts(str(tmp_path))
    evaluator = ReferenceEvaluator(str(tmp_path / metadata["artifacts"][0]["path"]))
    actual = evaluator.run(None, {"encoded_obs": observation[None, :]})[0]

    assert actual == pytest.approx(expected, abs=1.0e-5)
    batch_actual = evaluator.run(
        None,
        {"encoded_obs": np.stack((observation, observation), axis=0)},
    )[0]
    assert batch_actual.shape == (2, 2)


def test_export_supports_layout_without_controllable_assets(tmp_path: Path) -> None:
    from onnx.reference import ReferenceEvaluator

    agent, _ = _make_agent(buildings=1)
    state = agent._per_building[0]
    state.layout = replace(
        state.layout,
        segments=tuple(
            segment
            for segment in state.layout.segments
            if segment.family != "ca"
        ),
        n_ca=0,
        ca_action_names=(),
    )
    state.action_low = state.action_low[:0]
    state.action_high = state.action_high[:0]

    metadata = agent.export_artifacts(str(tmp_path))
    model_path = tmp_path / metadata["artifacts"][0]["path"]
    evaluator = ReferenceEvaluator(str(model_path))
    actual = evaluator.run(
        None,
        {
            "encoded_obs": np.zeros(
                (2, metadata["agent_models"][0]["obs_dim"]),
                dtype=np.float32,
            )
        },
    )[0]

    assert metadata["agent_models"][0]["n_ca"] == 0
    assert actual.shape == (2, 0)


def test_runtime_only_export_flags_are_loaded_from_hyperparameters() -> None:
    from algorithms.transformer_matd3.agent import AgentTransformerMATD3

    agent = AgentTransformerMATD3(
        _config(
            residual_policy_runtime_only_export=True,
            local_action_safety_runtime_only_export=True,
            local_price_conditioning_runtime_only_export=True,
        )
    )

    assert agent.residual_policy_runtime_only_export is True
    assert agent._local_action_safety_runtime_only_export is True
    assert agent._local_price_conditioning_runtime_only_export is True


@pytest.mark.parametrize(
    ("enabled_attribute", "runtime_attribute", "config_field"),
    [
        (
            "residual_policy_enabled",
            "residual_policy_runtime_only_export",
            "residual_policy_runtime_only_export",
        ),
        (
            "_local_action_safety_enabled",
            "_local_action_safety_runtime_only_export",
            "local_action_safety_runtime_only_export",
        ),
        (
            "_local_price_conditioning_enabled",
            "_local_price_conditioning_runtime_only_export",
            "local_price_conditioning_runtime_only_export",
        ),
    ],
)
def test_export_guards_fail_before_writing_files(
    tmp_path: Path,
    enabled_attribute: str,
    runtime_attribute: str,
    config_field: str,
) -> None:
    agent, _ = _make_agent(buildings=1)
    setattr(agent, enabled_attribute, True)
    setattr(agent, runtime_attribute, False)

    with pytest.raises(RuntimeError, match=config_field):
        agent.export_artifacts(str(tmp_path))

    assert not (tmp_path / "onnx_models").exists()


def test_runtime_only_export_marks_manifest_and_bundle_non_deployable(
    tmp_path: Path,
) -> None:
    agent, obs_dim = _make_agent(buildings=1)
    agent.residual_policy_enabled = True
    agent.residual_policy_runtime_only_export = True
    agent._local_action_safety_enabled = True
    agent._local_action_safety_runtime_only_export = True
    agent._local_price_conditioning_enabled = True
    agent._local_price_conditioning_runtime_only_export = True

    agent_metadata = agent.export_artifacts(str(tmp_path))
    artifact_config = agent_metadata["artifacts"][0]["config"]
    assert artifact_config["deployable"] is False
    assert artifact_config["requires_runtime_residual"] is True
    assert artifact_config["requires_runtime_local_action_safety"] is True
    assert artifact_config["requires_runtime_local_price_conditioning"] is True

    manifest = build_manifest(
        {
            "metadata": {"experiment_name": "test", "run_name": "run"},
            "simulator": {},
            "training": {},
            "topology": {"num_agents": 1},
            "pipeline": [
                {
                    "algorithm": "AgentTransformerMATD3",
                    "count": 1,
                    "hyperparameters": {},
                }
            ],
        },
        {
            "observation_names": [[f"feature_{index}" for index in range(obs_dim)]],
            "encoders": [[{"type": "NoNormalization", "params": {}}]],
            "action_bounds": [[{"low": [-2.0, -0.5], "high": [1.0, 0.75]}]],
            "action_names": ["electrical_storage", "electric_vehicle_storage"],
            "action_names_by_agent": {
                "0": ["electrical_storage", "electric_vehicle_storage"]
            },
            "reward_function": {"name": "Reward", "params": {}},
        },
        agent_metadata,
    )

    assert manifest["agent"]["artifacts"][0]["config"] == artifact_config
    validate_bundle_contract(manifest, tmp_path)


def test_underfull_update_emits_explicit_finite_skip_metrics() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=2)
    observation = np.zeros(obs_dim, dtype=np.float32)
    actions = agent.predict([observation], deterministic=True)

    agent.update(
        [observation],
        actions,
        [0.0],
        [observation],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=True,
        initial_exploration_done=True,
    )
    metrics = agent.consume_latest_training_metrics()

    assert metrics["TransformerMATD3/update_skipped"] == 1.0
    assert metrics["TransformerMATD3/update_skip_replay_underfull"] == 1.0
    assert metrics["TransformerMATD3/update_skip_initial_exploration"] == 0.0
    assert metrics["TransformerMATD3/update_skip_schedule"] == 0.0
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize(
    ("initial_exploration_done", "update_step", "expected_reason"),
    [
        (False, True, "initial_exploration"),
        (True, False, "schedule"),
    ],
)
def test_disabled_update_paths_emit_explicit_finite_skip_metrics(
    initial_exploration_done: bool,
    update_step: bool,
    expected_reason: str,
) -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    observation = np.zeros(obs_dim, dtype=np.float32)
    actions = agent.predict([observation], deterministic=True)

    agent.update(
        [observation],
        actions,
        [0.0],
        [observation],
        False,
        False,
        update_target_step=False,
        global_learning_step=0,
        update_step=update_step,
        initial_exploration_done=initial_exploration_done,
    )
    metrics = agent.consume_latest_training_metrics()

    assert metrics[f"TransformerMATD3/update_skip_{expected_reason}"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_successful_update_exposes_all_locked_core_metrics() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)
    metrics = agent.consume_latest_training_metrics()
    expected = {
        "critic_1_loss_mean",
        "critic_2_loss_mean",
        "critic_loss_mean",
        "critic_td_abs_mean",
        "critic_gap_abs_mean",
        "critic_grad_norm_mean",
        "q1_expected_mean",
        "q2_expected_mean",
        "q_min_expected_mean",
        "q_target_mean",
        "actor_update_performed",
        "actor_loss_mean",
        "actor_policy_loss_mean",
        "actor_policy_q_abs_mean",
        "actor_grad_norm_mean",
        "reward_raw_mean",
        "reward_train_mean",
        "reward_train_std",
        "replay_buffer_size",
        "replay_bucket_size_current",
        "replay_bucket_count",
        "n_step_returns",
        "n_step_queue_size",
        "target_policy_smoothing",
        "target_policy_noise",
        "target_policy_noise_clip",
        "actor_update_interval",
        "exploration_sigma",
        "exploration_step",
        "training_step_time",
    }

    assert {f"TransformerMATD3/{name}" for name in expected} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())


def test_diagnostic_metrics_include_latest_training_metrics() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    _transition(agent, obs_dim, 0)

    metrics = agent.get_diagnostic_metrics()

    assert "TransformerMATD3/critic_loss_mean" in metrics
    assert "TransformerMATD3/training_step_time" in metrics
    assert all(np.isfinite(value) for value in metrics.values())


def test_critic_diagnostics_persist_replay_q_and_attribute_buildings() -> None:
    agent, obs_dim = _make_agent(buildings=2, batch_size=1)
    agent._merge_latest_training_metrics(
        {
            "TransformerMATD3/replay_action_q_mean": 2.5,
            "TransformerMATD3/replay_action_q_abs_mean": 3.0,
            "TransformerMATD3/actor_update_performed": 1.0,
        }
    )
    agent._merge_latest_training_metrics(
        {
            "TransformerMATD3/critic_loss_mean": 1.0,
            "TransformerMATD3/actor_update_performed": 0.0,
        }
    )
    metrics = agent.get_diagnostic_metrics()
    assert metrics["TransformerMATD3/replay_action_q_mean"] == pytest.approx(2.5)
    assert metrics["TransformerMATD3/replay_action_q_abs_mean"] == pytest.approx(3.0)

    _transition(agent, obs_dim, 0)
    metrics = agent.get_diagnostic_metrics()
    for index in range(2):
        assert f"TransformerMATD3/building_{index}_td_abs_max" in metrics
        assert f"TransformerMATD3/building_{index}_critic_1_grad_norm" in metrics
        assert np.isfinite(metrics[f"TransformerMATD3/building_{index}_target_min"])


def test_critic_action_sensitivity_is_sampled_on_bounded_cadence() -> None:
    agent, obs_dim = _make_agent(buildings=1, batch_size=1)
    agent.critic_update_count = 15
    _transition(agent, obs_dim, 0)
    metrics = agent.get_diagnostic_metrics()
    key = "TransformerMATD3/building_0_storage_critic_dq_da_abs_mean"
    assert metrics["TransformerMATD3/building_0_storage_action_count"] > 0.0
    assert metrics["TransformerMATD3/building_0_storage_critic_dq_da_available"] == 1.0
    assert key in metrics
    assert np.isfinite(metrics[key])
    assert metrics[key] > 0.0
    assert np.isfinite(metrics["TransformerMATD3/building_0_storage_critic_dq_da_abs_p95"])
    assert np.isfinite(metrics["TransformerMATD3/building_0_storage_critic_dq_da_abs_max"])


def test_critic_action_sensitivity_handles_mixed_topology() -> None:
    agent, obs_dim = _make_agent(buildings=2, batch_size=1)
    agent.critic_update_count = 15
    _transition(agent, obs_dim, 0)
    metrics = agent.get_diagnostic_metrics()
    for index in range(2):
        assert metrics[f"TransformerMATD3/building_{index}_storage_action_count"] >= 0.0
        assert metrics[f"TransformerMATD3/building_{index}_storage_critic_dq_da_available"] in {0.0, 1.0}
