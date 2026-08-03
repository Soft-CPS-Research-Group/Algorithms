from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithms.agents.ppo_agents import PPO
from algorithms.agents.td3_agent import TD3
from algorithms.pipeline import Ensemble
from algorithms.registry import build_execution_unit, is_algorithm_supported
from reward_function.cost_hard_constraint_reward import (
    IndividualScorecardAlignedRewardV3,
    LocalCostServiceRewardV1,
    LocalEconomicSafetyRewardV3,
    LocalScorecardGuardRewardV2,
)
from utils.config_schema import validate_config


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class _RecordingScheduleTeacher:
    def __init__(self) -> None:
        self.schedule_steps: list[int] = []

    def predict_at_step(
        self,
        observations,
        *,
        schedule_step: int,
        deterministic: bool | None = None,
    ):
        del observations, deterministic
        self.schedule_steps.append(int(schedule_step))
        return [[0.25]]


def _distributed_config(name: str) -> dict:
    off_policy = name == "TD3"
    return {
        "training": {
            "seed": 7,
            "steps_between_training_updates": 1,
            "target_update_interval": 1,
        },
        "tracking": {
            "mlflow_step_sample_interval": 1,
            "training_diagnostics_enabled": True,
        },
        "checkpointing": {
            "checkpoint_artifact": "latest_checkpoint.pth",
            "reset_replay_buffer": False,
            "freeze_pretrained_layers": False,
            "fine_tune": False,
        },
        "topology": {
            "num_agents": 2,
            "observation_dimensions": [3, 2],
            "action_dimensions": [1, 2],
        },
        "pipeline": [
            {
                "algorithm": name,
                "count": 2,
                "hyperparameters": {"gamma": 0.95, "require_cuda": False},
                "networks": {
                    "actor": {"class": "Actor", "layers": [16], "lr": 1.0e-3},
                    "critic": {"class": "Critic", "layers": [16], "lr": 1.0e-3},
                },
                "replay_buffer": {
                    "class": "MultiAgentReplayBuffer" if off_policy else "OnPolicyRolloutBuffer",
                    "capacity": 8,
                    "batch_size": 2,
                },
                "exploration": {
                    "strategy": "GaussianNoise" if off_policy else "PPO",
                    "params": {
                        "gamma": 0.95,
                        "tau": 0.01,
                        "sigma": 0.1,
                        "decay": 1.0,
                        "min_sigma": 0.0,
                        "bias": 0.0,
                        "use_amp": False,
                        "end_initial_exploration_time_step": 0,
                        "random_exploration_steps": 0,
                        "initial_exploration_strategy": "uniform_full_range",
                        "train_during_initial_exploration": True,
                        "actor_update_interval": 2 if off_policy else 1,
                        "target_policy_smoothing": off_policy,
                        "target_policy_noise": 0.01,
                        "target_policy_noise_clip": 0.05,
                        "reward_normalization": False,
                        "rollout_length": 2,
                        "minibatch_size": 2,
                        "ppo_epochs": 1,
                        "gae_lambda": 0.95,
                        "clip_ratio": 0.2,
                        "entropy_coef": 0.01,
                        "value_loss_coef": 0.5,
                        "max_grad_norm": 0.5,
                        "initial_log_std": -0.5,
                    },
                },
            }
        ],
    }


def _attach(ensemble: Ensemble) -> None:
    ensemble.attach_environment(
        observation_names=[["a", "b", "c"], ["a", "b"]],
        action_names=[["storage"], ["charger", "appliance"]],
        action_space=[_Box([-2.0], [2.0]), _Box([0.0, 0.0], [7.4, 1.0])],
        observation_space=[None, None],
        metadata={"building_names": ["Building_1", "Building_2"]},
    )


@pytest.mark.parametrize(("name", "agent_cls"), [("PPO", PPO), ("TD3", TD3)])
def test_distributed_builder_creates_strict_local_learners(name, agent_cls):
    assert is_algorithm_supported(name)
    unit = build_execution_unit(_distributed_config(name))

    assert isinstance(unit, Ensemble)
    assert all(isinstance(agent, agent_cls) for agent in unit.agents)
    assert [agent.num_agents for agent in unit.agents] == [1, 1]
    assert [agent.observation_dimension for agent in unit.agents] == [[3], [2]]
    assert [agent.action_dimension for agent in unit.agents] == [[1], [2]]
    assert [agent.seed for agent in unit.agents] == [7, 8]


@pytest.mark.parametrize("name", ["PPO", "TD3"])
def test_strict_single_agent_class_rejects_unsliced_multi_agent_topology(name):
    config = _distributed_config(name)
    config["pipeline"][0]["count"] = 1
    with pytest.raises(ValueError, match="controls exactly one environment slot"):
        build_execution_unit(config)


@pytest.mark.parametrize("name", ["PPO", "TD3"])
def test_distributed_predict_update_checkpoint_and_metrics(name, tmp_path: Path):
    unit = build_execution_unit(_distributed_config(name))
    assert isinstance(unit, Ensemble)
    _attach(unit)

    for step in range(2):
        observations = [
            np.asarray([0.1 + step, 0.2, 0.3], dtype=np.float32),
            np.asarray([0.4, 0.5 + step], dtype=np.float32),
        ]
        next_observations = [
            np.asarray([0.2 + step, 0.2, 0.3], dtype=np.float32),
            np.asarray([0.4, 0.6 + step], dtype=np.float32),
        ]
        actions = unit.predict(observations, deterministic=False)
        assert len(actions) == 2
        assert -2.0 <= actions[0][0] <= 2.0
        assert 0.0 <= actions[1][0] <= 7.4
        assert 0.0 <= actions[1][1] <= 1.0

        unit.update(
            observations,
            actions,
            [-1.0, -0.5],
            next_observations,
            terminated=False,
            truncated=step == 1,
            update_target_step=True,
            global_learning_step=step + 1,
            update_step=True,
            initial_exploration_done=True,
        )

    metrics = unit.get_diagnostic_metrics()
    assert metrics["Ensemble/member_count"] == pytest.approx(2.0)
    assert any(key.startswith(f"Ensemble/{name}/") for key in metrics)

    checkpoint_root = Path(unit.save_checkpoint(str(tmp_path / "checkpoints"), step=2))
    assert (checkpoint_root / "agent_0" / "latest_checkpoint.pth").exists()
    assert (checkpoint_root / "agent_1" / "latest_checkpoint.pth").exists()
    unit.load_checkpoint(str(checkpoint_root))


@pytest.mark.parametrize("name", ["PPO", "TD3"])
def test_distributed_export_keeps_global_agent_indices(name, tmp_path: Path):
    unit = build_execution_unit(_distributed_config(name))
    assert isinstance(unit, Ensemble)
    _attach(unit)

    metadata = unit.export_artifacts(str(tmp_path / "bundle"), context={})

    assert metadata["format"] == "ensemble"
    assert [member["agent_index"] for member in metadata["agents"]] == [0, 1]
    for member_index, member in enumerate(metadata["agents"]):
        artifact = member["artifacts"][0]
        assert artifact["agent_index"] == member_index
        assert (
            tmp_path / "bundle" / f"agent_{member_index}" / artifact["path"]
        ).is_file()


@pytest.mark.parametrize(
    ("template", "reward_name"),
    [
        ("ppo_distributed_local.yaml", "LocalCostServiceRewardV1"),
        (
            "ppo_distributed_local_total_energy_bc_smoke.yaml",
            "LocalEconomicSafetyRewardV3",
        ),
        ("td3_distributed_local.yaml", "LocalEconomicSafetyRewardV3"),
    ],
)
def test_distributed_templates_validate(template, reward_name):
    path = Path("configs/templates/rl") / template
    text = path.read_text(encoding="utf-8")
    model = validate_config(yaml.safe_load(text))

    assert model.pipeline[0].count == 17
    assert model.simulator.reward_function == reward_name
    assert model.simulator.reward_function_kwargs["reward_scale"] == pytest.approx(0.01)
    assert model.simulator.central_agent is False
    assert model.simulator.entity_encoding.profile == "building_local_v1"
    assert "/home/" not in text
    assert "runs/" not in text
    if template.startswith("td3_"):
        actor = model.pipeline[0].networks.actor
        params = model.pipeline[0].exploration.params
        assert actor.class_name == "SemanticMultiHeadActor"
        assert actor.head_layers == [64]
        assert params["local_action_safety_enabled"] is True
        assert params["local_action_safety_ev_minimum_mode"] == "deadline_feasible"
        assert params["local_action_safety_service_teacher_enabled"] is False
        assert params["local_price_conditioning_enabled"] is True


def test_ppo_total_energy_bc_smoke_is_autonomous_rollout_with_verified_local_teacher():
    path = Path("configs/templates/rl/ppo_distributed_local_total_energy_bc_smoke.yaml")
    text = path.read_text(encoding="utf-8")
    model = validate_config(yaml.safe_load(text))
    simulator = model.simulator
    params = model.pipeline[0].exploration.params

    assert model.pipeline[0].algorithm == "PPO"
    assert model.pipeline[0].count == 17
    assert simulator.central_agent is False
    assert simulator.entity_encoding.profile == "building_local_v1"
    assert simulator.community_market.enabled is False
    assert simulator.simulation_end_time_step - simulator.simulation_start_time_step == 672

    # The MILP labels states visited by the PPO actor. It cannot control,
    # blend, or replace actions in either training or deterministic finish.
    assert params["warm_start_policy"] == "TotalOracleReplayPolicy"
    assert params["random_exploration_steps"] == 0
    assert params["end_initial_exploration_time_step"] == 0
    assert params["warm_start_policy_phaseout_steps"] == 0
    assert params["actor_behavior_cloning_weight"] > 0.0
    assert params["actor_behavior_cloning_replay_capacity"] >= 672
    assert params["local_action_safety_enabled"] is True
    assert params["local_action_safety_headroom_reserve_kw"] == pytest.approx(0.1)
    assert params["local_action_safety_ev_minimum_mode"] == "deadline_feasible"
    assert params["local_action_safety_service_teacher_enabled"] is False
    assert params["local_action_safety_service_teacher_eval_enabled"] is False

    schedule_path = Path(params["warm_start_policy_hyperparameters"]["schedule_path"])
    assert schedule_path.is_file()
    assert params["warm_start_policy_hyperparameters"]["allow_attached_action_subset"] is True
    assert params["warm_start_policy_hyperparameters"]["repeat_schedule_for_training"] is True

    manifest_path = schedule_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settlement"] == "individual"
    assert manifest["audit"]["all_buildings_pass_local_gates"] is True
    assert manifest["audit"]["building_count"] == 17
    assert manifest["source_window"]["horizon"] == 672
    assert manifest["diagnostic_only"] is True
    assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == manifest["artifacts"][
        "replay_schedule"
    ]["sha256"]


def test_exact_local_total_energy_training_demonstration_is_portable_and_verified():
    root = Path(
        "configs/demonstrations/local_total_energy_v1/train_0_1316_exact"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schedule_path = root / manifest["artifacts"]["replay_schedule"]["path"]

    assert manifest["portable"] is True
    assert manifest["diagnostic_only"] is False
    assert manifest["boundary_service_exact"] is True
    assert manifest["settlement"] == "individual"
    assert manifest["source_window"] == {
        "start_time_step": 0,
        "end_time_step_exclusive": 1316,
        "horizon": 1316,
    }
    assert manifest["audit"]["building_count"] == 17
    assert manifest["audit"]["local_gate_pass_count"] == 17
    assert manifest["audit"]["all_buildings_pass_local_gates"] is True
    assert schedule_path.is_file()
    assert hashlib.sha256(schedule_path.read_bytes()).hexdigest() == manifest["artifacts"][
        "replay_schedule"
    ]["sha256"]


def test_local_cost_service_reward_has_no_cross_building_terms():
    reward = LocalCostServiceRewardV1(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )
    first = {
        "net_electricity_consumption": 2.0,
        "electricity_pricing": 0.5,
    }
    quiet_neighbor = {
        "net_electricity_consumption": 0.0,
        "electricity_pricing": 0.5,
    }
    importing_neighbor = {
        "net_electricity_consumption": 100.0,
        "electricity_pricing": 0.5,
    }

    reward_with_quiet_neighbor = reward.calculate([first, quiet_neighbor])[0]
    reward_with_importing_neighbor = reward.calculate([first, importing_neighbor])[0]

    assert reward_with_quiet_neighbor == pytest.approx(-0.01)
    assert reward_with_importing_neighbor == pytest.approx(reward_with_quiet_neighbor)
    assert reward.last_components_by_agent[0]["reward_total_unscaled"] == pytest.approx(-1.0)
    assert reward.last_components_by_agent[0]["reward_scale"] == pytest.approx(0.01)
    assert reward.community_import_penalty == pytest.approx(0.0)
    assert reward.community_peak_import_penalty == pytest.approx(0.0)
    assert reward.community_settlement_cost_weight == pytest.approx(0.0)


def test_local_cost_service_reward_validates_scale():
    with pytest.raises(ValueError, match="reward_scale must be > 0"):
        LocalCostServiceRewardV1(
            env_metadata={"central_agent": False, "seconds_per_time_step": 900},
            reward_scale=0.0,
        )

    with pytest.raises(ValueError, match="requires central_agent=false"):
        LocalCostServiceRewardV1(
            env_metadata={"central_agent": True, "seconds_per_time_step": 900}
        )


def test_local_cost_service_reward_supports_simulator_two_phase_construction():
    reward = LocalScorecardGuardRewardV2(env_metadata=None)

    assert reward.env_metadata is None


def test_local_scorecard_guard_strengthens_all_hard_service_terms():
    base = LocalCostServiceRewardV1(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )
    guarded = LocalScorecardGuardRewardV2(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )

    assert guarded.grid_violation_penalty > base.grid_violation_penalty
    assert guarded.ev_departure_missed_penalty > base.ev_departure_missed_penalty
    assert guarded.battery_soc_violation_penalty > base.battery_soc_violation_penalty
    assert guarded.deferrable_deadline_missed_penalty > base.deferrable_deadline_missed_penalty
    assert guarded.community_import_penalty == pytest.approx(0.0)
    assert guarded.community_settlement_cost_weight == pytest.approx(0.0)
    assert guarded.reward_scale == pytest.approx(base.reward_scale)


def test_local_economic_safety_reward_removes_only_dense_urgency_terms():
    reward = LocalEconomicSafetyRewardV3(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )

    assert reward.local_cost_weight == pytest.approx(1.0)
    assert reward.ev_connected_deficit_penalty == pytest.approx(0.0)
    assert reward.ev_schedule_deficit_penalty == pytest.approx(0.0)
    assert reward.deferrable_urgency_penalty == pytest.approx(0.0)
    assert reward.ev_departure_missed_penalty == pytest.approx(2400.0)
    assert reward.grid_violation_penalty == pytest.approx(600.0)
    assert reward.deferrable_deadline_missed_penalty == pytest.approx(2000.0)
    assert reward.community_settlement_cost_weight == pytest.approx(0.0)


def test_individual_scorecard_reward_uses_per_building_market_settlement():
    reward = IndividualScorecardAlignedRewardV3(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )
    importer = {
        "net_electricity_consumption": 2.0,
        "electricity_pricing": 0.5,
    }
    exporter = {
        "net_electricity_consumption": -1.0,
        "electricity_pricing": 0.5,
    }

    rewards = reward.calculate([importer, exporter])
    components = reward.get_last_components()

    assert reward.local_cost_weight == pytest.approx(0.0)
    assert reward.community_settlement_cost_weight == pytest.approx(1.0)
    assert rewards == pytest.approx([-0.009, 0.004])
    assert sum(rewards) == pytest.approx(-0.005)
    assert components["community"]["community_settlement_cost_total"] == pytest.approx(0.5)
    assert components["per_agent"][0]["community_settlement_cost"] == pytest.approx(0.9)
    assert components["per_agent"][1]["community_settlement_cost"] == pytest.approx(-0.4)


def test_individual_scorecard_reward_keeps_independent_learner_contract():
    reward = IndividualScorecardAlignedRewardV3(
        env_metadata={"central_agent": False, "seconds_per_time_step": 900}
    )

    assert reward.central_agent is False
    assert reward.grid_violation_penalty == pytest.approx(600.0)
    assert reward.ev_departure_missed_penalty == pytest.approx(2400.0)
    assert reward.deferrable_deadline_missed_penalty == pytest.approx(2000.0)
    assert reward.reward_scale == pytest.approx(0.01)


def test_ppo_uses_consistent_tanh_squashed_policy():
    unit = build_execution_unit(_distributed_config("PPO"))
    assert isinstance(unit, Ensemble)
    agent = unit.agents[0]
    observation = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32, device=agent.device)

    with torch.no_grad():
        distribution = agent.actors[0].distribution(observation)
        deterministic_action = agent.actors[0](observation)
    assert torch.allclose(deterministic_action, torch.tanh(distribution.mean))

    boundary_actions = torch.tensor(
        [[-1.0,], [0.0,], [1.0,]], dtype=torch.float32, device=agent.device
    )
    boundary_distribution = torch.distributions.Normal(
        torch.zeros_like(boundary_actions), torch.ones_like(boundary_actions)
    )
    log_prob = PPO._squashed_log_prob(boundary_distribution, boundary_actions)
    assert torch.isfinite(log_prob).all()


def test_ppo_behavior_cloning_supports_action_type_priorities():
    unit = build_execution_unit(_distributed_config("PPO"))
    assert isinstance(unit, Ensemble)
    agent = unit.agents[1]
    agent.action_names = [[
        "electrical_storage",
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_1",
        "other",
    ]]
    agent.actor_storage_behavior_cloning_multiplier = 0.25
    agent.actor_ev_behavior_cloning_multiplier = 8.0
    agent.actor_deferrable_behavior_cloning_multiplier = 12.0
    agent.actor_other_behavior_cloning_multiplier = 0.5

    weights = agent._actor_behavior_cloning_action_weights(
        0,
        action_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert weights.tolist() == pytest.approx([0.25, 8.0, 12.0, 0.5])


def test_ppo_behavior_cloning_upweights_active_ev_and_deferrable_targets():
    unit = build_execution_unit(_distributed_config("PPO"))
    assert isinstance(unit, Ensemble)
    agent = unit.agents[0]
    agent.action_names = [[
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_1",
    ]]
    agent.actor_ev_behavior_cloning_positive_target_weight = 3.0
    agent.actor_ev_behavior_cloning_zero_target_weight = 5.0
    agent.actor_ev_behavior_cloning_zero_target_threshold = 0.05
    agent.actor_deferrable_behavior_cloning_positive_target_weight = 7.0
    target = torch.tensor(
        [[1.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )

    weights = agent._actor_behavior_cloning_sample_weights(
        0,
        base_weights=torch.ones(2),
        normalized_target=target,
    )

    assert torch.allclose(
        weights,
        torch.tensor([[4.0, 8.0], [6.0, 1.0]], dtype=torch.float32),
    )


def test_ppo_residual_behavior_cloning_upweights_neutral_targets_for_all_actions():
    unit = build_execution_unit(_distributed_config("PPO"))
    assert isinstance(unit, Ensemble)
    agent = unit.agents[0]
    agent.action_names = [[
        "electrical_storage",
        "electric_vehicle_storage_charger_1",
        "deferrable_appliance_1",
    ]]
    agent.residual_policy_enabled = True
    agent.actor_residual_behavior_cloning_neutral_target_weight = 7.0
    agent.actor_residual_behavior_cloning_neutral_target_threshold = 0.02
    target = torch.tensor(
        [[0.0, 0.01, -0.03], [0.5, -0.5, 0.0]],
        dtype=torch.float32,
    )

    weights = agent._actor_behavior_cloning_sample_weights(
        0,
        base_weights=torch.ones(3),
        normalized_target=target,
    )

    assert torch.allclose(
        weights,
        torch.tensor([[8.0, 8.0, 1.0], [1.0, 1.0, 8.0]], dtype=torch.float32),
    )


def test_ppo_non_residual_behavior_cloning_ignores_residual_neutral_weight():
    unit = build_execution_unit(_distributed_config("PPO"))
    assert isinstance(unit, Ensemble)
    agent = unit.agents[0]
    agent.action_names = [["electrical_storage"]]
    agent.residual_policy_enabled = False
    agent.actor_residual_behavior_cloning_neutral_target_weight = 7.0

    weights = agent._actor_behavior_cloning_sample_weights(
        0,
        base_weights=torch.ones(1),
        normalized_target=torch.zeros((2, 1), dtype=torch.float32),
    )

    assert torch.allclose(weights, torch.ones((2, 1), dtype=torch.float32))


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            {"actor_behavior_cloning_source": "warm_start_policy"},
            "requires warm_start_policy",
        ),
        (
            {"residual_policy_enabled": True},
            "residual_policy_enabled requires warm_start_policy",
        ),
        (
            {
                "warm_start_policy": "RBCSmartPolicy",
                "initial_exploration_strategy": "uniform_full_range",
            },
            "teacher is never initialized",
        ),
    ],
)
def test_td3_rejects_silent_teacher_configuration_noops(params, expected):
    config = _distributed_config("TD3")
    config["pipeline"][0]["exploration"]["params"].update(params)

    with pytest.raises(ValueError, match=expected):
        build_execution_unit(config)


def test_td3_ensemble_routes_rbcsmart_teacher_actions_into_local_replay():
    config = _distributed_config("TD3")
    params = config["pipeline"][0]["exploration"]["params"]
    params.update(
        {
            "initial_exploration_strategy": "policy",
            "warm_start_policy": "RBCSmartPolicy",
            "warm_start_policy_deterministic": True,
            "actor_behavior_cloning_source": "warm_start_policy",
            "end_initial_exploration_time_step": 2,
            "random_exploration_steps": 2,
            "train_during_initial_exploration": False,
        }
    )
    unit = build_execution_unit(config)
    assert isinstance(unit, Ensemble)
    _attach(unit)

    observations = [
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.5], dtype=np.float32),
    ]
    next_observations = [
        np.asarray([0.2, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.6], dtype=np.float32),
    ]
    unit.set_observation_context(
        raw_observations=observations,
        encoded_observations=observations,
    )
    actions = unit.predict(observations, deterministic=False)
    teacher_actions = [
        np.asarray(agent._last_warm_start_policy_actions[0], dtype=np.float32)
        for agent in unit.agents
    ]
    unit.set_transition_context(
        raw_observations=observations,
        raw_next_observations=next_observations,
        encoded_observations=observations,
        encoded_next_observations=next_observations,
    )
    unit.update(
        observations,
        actions,
        [-1.0, -0.5],
        next_observations,
        terminated=False,
        truncated=False,
        update_target_step=False,
        global_learning_step=1,
        update_step=False,
        initial_exploration_done=False,
    )

    for agent, expected_teacher in zip(unit.agents, teacher_actions):
        assert agent._warm_start_policy is not None
        assert agent.replay_buffer._behavior_actions is not None
        np.testing.assert_allclose(
            agent.replay_buffer._behavior_actions[0][0],
            expected_teacher,
            atol=1.0e-6,
        )


def test_td3_teacher_clock_uses_episode_step_after_resume_and_reset():
    unit = build_execution_unit(_distributed_config("TD3"))
    assert isinstance(unit, Ensemble)
    _attach(unit)
    agent = unit.agents[0]
    teacher = _RecordingScheduleTeacher()
    agent._warm_start_policy = teacher
    agent.exploration_step = 5096

    unit.set_episode_context(episode_step=0, next_episode_step=None)
    unit.set_observation_context(
        raw_observations=[np.asarray([0.1, 0.2, 0.3]), np.asarray([0.4, 0.5])],
        encoded_observations=[np.asarray([0.1, 0.2, 0.3]), np.asarray([0.4, 0.5])],
    )
    agent._predict_warm_start_policy(apply_noise=False, deterministic=True)
    assert teacher.schedule_steps == [0]

    unit.set_episode_context(episode_step=0, next_episode_step=1)
    unit.set_transition_context(
        raw_observations=[np.asarray([0.1, 0.2, 0.3]), np.asarray([0.4, 0.5])],
        raw_next_observations=[np.asarray([0.2, 0.2, 0.3]), np.asarray([0.4, 0.6])],
        encoded_observations=[np.asarray([0.1, 0.2, 0.3]), np.asarray([0.4, 0.5])],
        encoded_next_observations=[np.asarray([0.2, 0.2, 0.3]), np.asarray([0.4, 0.6])],
    )
    assert teacher.schedule_steps == [0, 1]

    unit.set_episode_context(episode_step=1, next_episode_step=None)
    unit.set_transition_context(
        raw_observations=[np.asarray([0.2, 0.2, 0.3]), np.asarray([0.4, 0.6])],
        raw_next_observations=[np.asarray([0.3, 0.2, 0.3]), np.asarray([0.4, 0.7])],
        encoded_observations=[np.asarray([0.2, 0.2, 0.3]), np.asarray([0.4, 0.6])],
        encoded_next_observations=[np.asarray([0.3, 0.2, 0.3]), np.asarray([0.4, 0.7])],
    )
    assert teacher.schedule_steps == [0, 1]

    unit.set_episode_context(episode_step=0, next_episode_step=None)
    agent._predict_warm_start_policy(apply_noise=False, deterministic=True)
    assert teacher.schedule_steps == [0, 1, 0]
