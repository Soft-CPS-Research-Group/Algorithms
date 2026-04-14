"""Tests for AgentTransformerPPO (Phase 4, updated for actual CityLearn naming).

Verifies instantiation, attach_environment, predict shape, stochastic vs
deterministic mode, update mechanics, checkpoint round-trip, and multi-building.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from algorithms.agents.transformer_ppo_agent import AgentTransformerPPO


# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

ENCODER_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "encoders" / "default.json"
)


def _load_encoder_config() -> dict:
    with ENCODER_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


TOKENIZER_CONFIG: Dict[str, Any] = {
    "ca_types": {
        "ev_charger": {
            "features": [
                "connected_state",
                "departure_time",
                "required_soc_departure",
                "_soc",
                "battery_capacity",
                "incoming_state",
                "arrival_time",
            ],
            "action_name": "electric_vehicle_storage",
        },
        "battery": {
            "features": ["electrical_storage_soc"],
            "action_name": "electrical_storage",
        },
        "washing_machine": {
            "features": ["start_time_step", "end_time_step"],
            "action_name": "washing_machine",
        },
    },
    "sro_types": {
        "temporal": {"features": ["month", "hour", "day_type", "daylight_savings_status"]},
        "weather": {
            "features": [
                "outdoor_dry_bulb_temperature",
                "outdoor_relative_humidity",
                "diffuse_solar_irradiance",
                "direct_solar_irradiance",
            ],
        },
        "pricing": {"features": ["electricity_pricing"]},
        "carbon": {"features": ["carbon_intensity"]},
    },
    "rl": {
        "demand_feature": "non_shiftable_load",
        "generation_features": ["solar_generation"],
        "extra_features": ["net_electricity_consumption"],
    },
}


def _make_config(
    num_agents: int = 1,
    min_steps: int = 0,
) -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerPPO",
            "hyperparameters": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "ppo_epochs": 2,
                "minibatch_size": 4,
                "entropy_coeff": 0.01,
                "value_coeff": 0.5,
                "max_grad_norm": 0.5,
                "min_steps_before_update": min_steps,
            },
            "transformer": {
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.0,
            },
            "tokenizer": TOKENIZER_CONFIG,
        },
        "topology": {"num_agents": num_agents},
        "tracking": {"mlflow_step_sample_interval": 10},
        "checkpointing": {"checkpoint_artifact": "latest_checkpoint.pth"},
        "training": {},
    }


# --- Actual CityLearn observation/action names ---

# Building_1: 1 EV charger (charger_1_1) + 1 battery + PV + 1 washing machine
BUILDING_1_OBS = [
    "month", "day_type", "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "outdoor_relative_humidity",
    "outdoor_relative_humidity_predicted_1",
    "outdoor_relative_humidity_predicted_2",
    "outdoor_relative_humidity_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_3",
    "daylight_savings_status",
    # Per-charger observations (charger_1_1)
    "electric_vehicle_charger_charger_1_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_1_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_1_1_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_1_1_soc",
    "connected_electric_vehicle_at_charger_charger_1_1_battery_capacity",
    "electric_vehicle_charger_charger_1_1_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_1_1_estimated_arrival_time",
    # Washing machine
    "washing_machine_1_start_time_step",
    "washing_machine_1_end_time_step",
]
BUILDING_1_ACTIONS = [
    "electrical_storage",
    "electric_vehicle_storage_charger_1_1",
    "washing_machine_1",
]

# Single-charger building (like Building_4, no battery, no washing machine)
SINGLE_EV_OBS = [
    "month", "day_type", "hour",
    "electricity_pricing",
    "non_shiftable_load",
    "solar_generation",
    "net_electricity_consumption",
    "daylight_savings_status",
    # Per-charger observations (charger_4_1)
    "electric_vehicle_charger_charger_4_1_connected_state",
    "connected_electric_vehicle_at_charger_charger_4_1_departure_time",
    "connected_electric_vehicle_at_charger_charger_4_1_required_soc_departure",
    "connected_electric_vehicle_at_charger_charger_4_1_soc",
    "connected_electric_vehicle_at_charger_charger_4_1_battery_capacity",
    "electric_vehicle_charger_charger_4_1_incoming_state",
    "incoming_electric_vehicle_at_charger_charger_4_1_estimated_arrival_time",
]
SINGLE_EV_ACTIONS = ["electric_vehicle_storage_charger_4_1"]


class DummySpace:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape)
        self.low = -np.ones(shape)


def _get_obs_dim(obs_names: List[str]) -> int:
    """Compute obs dim using the tokenizer's encoded dims map."""
    from algorithms.utils.observation_tokenizer import _build_encoded_dims_map
    enc_cfg = _load_encoder_config()
    idx_map = _build_encoded_dims_map(obs_names, enc_cfg)
    return sum(s.n_dims for s in idx_map.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_create_agent(self):
        agent = AgentTransformerPPO(config=_make_config())
        assert agent.building_models is None  # Not built until attach_environment
        assert agent.lr == 3e-4
        assert agent.d_model == 32

    def test_predict_without_attach_raises(self):
        agent = AgentTransformerPPO(config=_make_config())
        with pytest.raises(RuntimeError, match="attach_environment"):
            agent.predict([np.zeros(10)])


class TestAttachEnvironment:
    def test_single_building(self):
        agent = AgentTransformerPPO(config=_make_config())
        obs_dim = _get_obs_dim(BUILDING_1_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((len(BUILDING_1_ACTIONS),))],
            observation_space=[DummySpace((obs_dim,))],
        )

        assert agent.building_models is not None
        assert len(agent.building_models) == 1
        # battery + ev_charger + washing_machine = 3
        assert agent._n_ca_per_building[0] == 3

    def test_multi_building(self):
        agent = AgentTransformerPPO(config=_make_config(num_agents=2))
        obs_dim_1 = _get_obs_dim(BUILDING_1_OBS)
        obs_dim_2 = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
            action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
            action_space=[
                DummySpace((len(BUILDING_1_ACTIONS),)),
                DummySpace((len(SINGLE_EV_ACTIONS),)),
            ],
            observation_space=[
                DummySpace((obs_dim_1,)),
                DummySpace((obs_dim_2,)),
            ],
        )

        assert len(agent.building_models) == 2
        assert agent._n_ca_per_building[0] == 3  # battery + ev_charger + washing_machine
        assert agent._n_ca_per_building[1] == 1  # ev_charger only


class TestPredict:
    @pytest.fixture()
    def agent(self):
        a = AgentTransformerPPO(config=_make_config())
        obs_dim = _get_obs_dim(BUILDING_1_OBS)
        a.attach_environment(
            observation_names=[BUILDING_1_OBS],
            action_names=[BUILDING_1_ACTIONS],
            action_space=[DummySpace((len(BUILDING_1_ACTIONS),))],
            observation_space=[DummySpace((obs_dim,))],
        )
        return a

    def test_predict_shape(self, agent):
        obs_dim = _get_obs_dim(BUILDING_1_OBS)
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        actions = agent.predict(obs, deterministic=True)

        assert len(actions) == 1
        assert len(actions[0]) == 3  # 3 actions (battery + ev_charger + washing_machine)

    def test_predict_action_range(self, agent):
        obs_dim = _get_obs_dim(BUILDING_1_OBS)
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        actions = agent.predict(obs, deterministic=True)

        for a in actions[0]:
            assert -1.0 <= a <= 1.0

    def test_stochastic_stores_in_buffer(self, agent):
        obs_dim = _get_obs_dim(BUILDING_1_OBS)
        obs = [np.random.randn(obs_dim).astype(np.float64)]

        assert len(agent.rollout_buffers[0]) == 0
        agent.predict(obs, deterministic=False)
        assert len(agent.rollout_buffers[0]) == 1

    def test_deterministic_does_not_store(self, agent):
        obs_dim = _get_obs_dim(BUILDING_1_OBS)
        obs = [np.random.randn(obs_dim).astype(np.float64)]

        agent.predict(obs, deterministic=True)
        assert len(agent.rollout_buffers[0]) == 0


class TestUpdate:
    def _make_agent_with_data(self, n_steps: int = 10):
        """Create an agent and push n_steps of synthetic transitions."""
        agent = AgentTransformerPPO(config=_make_config(min_steps=0))
        obs_dim = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[SINGLE_EV_OBS],
            action_names=[SINGLE_EV_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        # Push transitions via predict + update
        for step in range(n_steps):
            obs = [np.random.randn(obs_dim).astype(np.float64)]
            agent.predict(obs, deterministic=False)

            next_obs = [np.random.randn(obs_dim).astype(np.float64)]
            is_last = (step == n_steps - 1)

            agent.update(
                observations=obs,
                actions=[np.array([0.0])],
                rewards=[1.0],
                next_observations=next_obs,
                terminated=is_last,
                truncated=False,
                update_target_step=False,
                global_learning_step=step,
                update_step=is_last,  # only update on last step
                initial_exploration_done=True,
            )

        return agent

    def test_update_clears_buffer(self):
        agent = self._make_agent_with_data(n_steps=10)
        # After update, buffer should be cleared
        assert len(agent.rollout_buffers[0]) == 0

    def test_update_increments_training_step(self):
        agent = self._make_agent_with_data(n_steps=10)
        assert agent._training_step >= 1

    def test_parameters_change_after_update(self):
        """PPO update should modify network parameters."""
        agent = AgentTransformerPPO(config=_make_config(min_steps=0))
        obs_dim = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[SINGLE_EV_OBS],
            action_names=[SINGLE_EV_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        # Snapshot initial weights
        initial_params = {
            name: p.clone()
            for name, p in agent.building_models[0].named_parameters()
        }

        # Run predict + update cycle
        for step in range(20):
            obs = [np.random.randn(obs_dim).astype(np.float64)]
            agent.predict(obs, deterministic=False)

            next_obs = [np.random.randn(obs_dim).astype(np.float64)]
            agent.update(
                observations=obs,
                actions=[np.array([0.0])],
                rewards=[float(step) * 0.1],
                next_observations=next_obs,
                terminated=(step == 19),
                truncated=False,
                update_target_step=False,
                global_learning_step=step,
                update_step=(step == 19),
                initial_exploration_done=True,
            )

        # Check that at least some parameters changed
        changed = False
        for name, p in agent.building_models[0].named_parameters():
            if not torch.equal(p, initial_params[name]):
                changed = True
                break
        assert changed, "No parameters changed after PPO update"


class TestCheckpoint:
    def test_round_trip(self, tmp_path):
        """Save and load should preserve weights."""
        config = _make_config()
        obs_dim = _get_obs_dim(SINGLE_EV_OBS)

        # Agent 1 — train briefly
        agent1 = AgentTransformerPPO(config=config)
        agent1.attach_environment(
            observation_names=[SINGLE_EV_OBS],
            action_names=[SINGLE_EV_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )

        # Save checkpoint
        ckpt_path = agent1.save_checkpoint(str(tmp_path), step=0)
        assert Path(ckpt_path).exists()

        # Agent 2 — load from checkpoint
        agent2 = AgentTransformerPPO(config=config)
        agent2.attach_environment(
            observation_names=[SINGLE_EV_OBS],
            action_names=[SINGLE_EV_ACTIONS],
            action_space=[DummySpace((1,))],
            observation_space=[DummySpace((obs_dim,))],
        )
        agent2.load_checkpoint(ckpt_path)

        # Verify weights match
        for p1, p2 in zip(
            agent1.building_models[0].parameters(),
            agent2.building_models[0].parameters(),
        ):
            torch.testing.assert_close(p1, p2)

    def test_load_nonexistent_raises(self):
        agent = AgentTransformerPPO(config=_make_config())
        with pytest.raises(FileNotFoundError):
            agent.load_checkpoint("/nonexistent/checkpoint.pth")


class TestMultiBuilding:
    def test_different_ca_counts(self):
        """2 buildings with different CA counts → independent outputs."""
        agent = AgentTransformerPPO(config=_make_config(num_agents=2))
        obs_dim_1 = _get_obs_dim(BUILDING_1_OBS)
        obs_dim_2 = _get_obs_dim(SINGLE_EV_OBS)

        agent.attach_environment(
            observation_names=[BUILDING_1_OBS, SINGLE_EV_OBS],
            action_names=[BUILDING_1_ACTIONS, SINGLE_EV_ACTIONS],
            action_space=[
                DummySpace((len(BUILDING_1_ACTIONS),)),
                DummySpace((len(SINGLE_EV_ACTIONS),)),
            ],
            observation_space=[
                DummySpace((obs_dim_1,)),
                DummySpace((obs_dim_2,)),
            ],
        )

        obs = [
            np.random.randn(obs_dim_1).astype(np.float64),
            np.random.randn(obs_dim_2).astype(np.float64),
        ]
        actions = agent.predict(obs, deterministic=True)

        assert len(actions) == 2
        assert len(actions[0]) == 3  # Building_1: battery + ev_charger + washing_machine
        assert len(actions[1]) == 1  # Single EV building


class TestIsInitialExplorationDone:
    def test_default_zero(self):
        agent = AgentTransformerPPO(config=_make_config(min_steps=0))
        assert agent.is_initial_exploration_done(0)

    def test_with_min_steps(self):
        agent = AgentTransformerPPO(config=_make_config(min_steps=100))
        assert not agent.is_initial_exploration_done(50)
        assert agent.is_initial_exploration_done(100)
        assert agent.is_initial_exploration_done(200)
