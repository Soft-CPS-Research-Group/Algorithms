from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
from gymnasium import spaces

from utils.wrapper_citylearn import Wrapper_CityLearn


@dataclass
class _DummyEpisodeTracker:
    episode_time_steps: int = 24


class _DummyEntityEnv:
    def __init__(self):
        self.interface = "entity"
        self.topology_mode = "dynamic"
        self.seconds_per_time_step = 3600
        self.random_seed = 22
        self.time_step_ratio = 1.0
        self.episode_tracker = _DummyEpisodeTracker(episode_time_steps=24)
        self.unwrapped = self
        self.time_steps = 24
        self._version = 0
        self._expose_building_names = True

    @property
    def building_names(self) -> List[str]:
        if not self._expose_building_names:
            raise AttributeError("building_names unavailable")
        return self._building_ids(self._version)

    @property
    def entity_specs(self) -> Dict[str, Any]:
        return self._specs(self._version)

    @property
    def observation_space(self):
        specs = self._specs(self._version)
        n_buildings = len(specs["tables"]["building"]["ids"])
        n_chargers = len(specs["tables"]["charger"]["ids"])

        return spaces.Dict(
            {
                "tables": spaces.Dict(
                    {
                        "district": spaces.Box(
                            low=np.array([[0.0, 0.0]], dtype=np.float32),
                            high=np.array([[23.0, 59.0]], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "building": spaces.Box(
                            low=np.zeros((n_buildings, 2), dtype=np.float32),
                            high=np.full((n_buildings, 2), 100.0, dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "charger": spaces.Box(
                            low=np.zeros((n_chargers, 4), dtype=np.float32),
                            high=np.array([[1.0, 100.0, 100.0, 24.0] for _ in range(n_chargers)], dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "storage": spaces.Box(
                            low=np.zeros((n_buildings, 1), dtype=np.float32),
                            high=np.ones((n_buildings, 1), dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "pv": spaces.Box(
                            low=np.zeros((n_buildings, 1), dtype=np.float32),
                            high=np.full((n_buildings, 1), 25.0, dtype=np.float32),
                            dtype=np.float32,
                        ),
                        "ev": spaces.Box(
                            low=np.zeros((n_chargers, 1), dtype=np.float32),
                            high=np.full((n_chargers, 1), 100.0, dtype=np.float32),
                            dtype=np.float32,
                        ),
                    }
                )
            }
        )

    @property
    def action_names(self) -> List[List[str]]:
        if self._version == 0:
            return [["electrical_storage", "electric_vehicle_storage_C1"]]
        return [
            ["electrical_storage", "electric_vehicle_storage_C1"],
            ["electrical_storage", "electric_vehicle_storage_C2"],
        ]

    @property
    def flat_action_space(self) -> List[spaces.Box]:
        if self._version == 0:
            return [spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)]
        return [
            spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32),
            spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32),
        ]

    def get_metadata(self):
        return {"buildings": [{} for _ in self._building_ids(self._version)]}

    def reset(self):
        self._version = 0
        return self._observation_payload(version=0), {}

    @staticmethod
    def _building_ids(version: int) -> List[str]:
        return ["B1"] if version == 0 else ["B1", "B2"]

    def _specs(self, version: int) -> Dict[str, Any]:
        building_ids = self._building_ids(version)
        charger_ids = [f"{building}/C{idx + 1}" for idx, building in enumerate(building_ids)]
        return {
            "tables": {
                "district": {"ids": ["district_0"], "features": ["hour", "minutes"]},
                "building": {"ids": building_ids, "features": ["load_power_kw", "pv_power_kw"]},
                "charger": {
                    "ids": charger_ids,
                    "features": [
                        "connected_state",
                        "connected_ev_soc",
                        "connected_ev_required_soc_departure",
                        "connected_ev_departure_time_step",
                    ],
                },
                "storage": {"ids": [f"{b}/electrical_storage" for b in building_ids], "features": ["soc"]},
                "pv": {"ids": [f"{b}/pv" for b in building_ids], "features": ["generation_power_kw"]},
                "ev": {"ids": [f"EV_{i + 1}" for i in range(len(charger_ids))], "features": ["soc"]},
            },
            "actions": {
                "building": {"ids": building_ids, "features": ["electrical_storage"]},
                "charger": {"ids": charger_ids, "features": ["electric_vehicle_storage"]},
            },
        }

    def _observation_payload(self, version: int) -> Dict[str, Any]:
        building_ids = self._building_ids(version)
        n_buildings = len(building_ids)
        n_chargers = len(building_ids)

        building_to_charger = np.array([[idx, idx] for idx in range(n_buildings)], dtype=np.int32)
        building_to_storage = np.array([[idx, idx] for idx in range(n_buildings)], dtype=np.int32)
        building_to_pv = np.array([[idx, idx] for idx in range(n_buildings)], dtype=np.int32)
        charger_to_ev = np.array([[idx, idx] for idx in range(n_chargers)], dtype=np.int32)

        return {
            "tables": {
                "district": np.array([[12.0, 0.0]], dtype=np.float32),
                "building": np.array([[50.0, 2.0] for _ in range(n_buildings)], dtype=np.float32),
                "charger": np.array([[1.0, 40.0, 80.0, 18.0] for _ in range(n_chargers)], dtype=np.float32),
                "storage": np.array([[0.5] for _ in range(n_buildings)], dtype=np.float32),
                "pv": np.array([[8.0] for _ in range(n_buildings)], dtype=np.float32),
                "ev": np.array([[40.0] for _ in range(n_chargers)], dtype=np.float32),
            },
            "edges": {
                "building_to_charger": building_to_charger,
                "building_to_storage": building_to_storage,
                "building_to_pv": building_to_pv,
                "charger_to_ev_connected": charger_to_ev,
                "charger_to_ev_connected_mask": np.ones((n_chargers,), dtype=np.float32),
                "charger_to_ev_incoming": charger_to_ev,
                "charger_to_ev_incoming_mask": np.zeros((n_chargers,), dtype=np.float32),
            },
            "meta": {"topology_version": int(version)},
        }


class _DummyModel:
    def __init__(self):
        self.use_raw_observations = True
        self.attach_calls = 0

    def attach_environment(self, **_kwargs):
        self.attach_calls += 1

    def predict(self, observations, deterministic=None):
        _ = deterministic
        return [[0.0 for _ in obs[:2]] for obs in observations]

    def update(self, **_kwargs):
        return None

    def is_initial_exploration_done(self, _global_step):
        return True


def _entity_config() -> Dict[str, Any]:
    return {
        "runtime": {"log_dir": None},
        "training": {"steps_between_training_updates": 1, "target_update_interval": 0},
        "checkpointing": {"checkpoint_interval": None, "require_update_step": True, "require_initial_exploration_done": True},
        "tracking": {"mlflow_enabled": False},
        "algorithm": {"name": "RuleBasedPolicy"},
        "simulator": {
            "interface": "entity",
            "topology_mode": "dynamic",
            "episodes": 1,
            "entity_encoding": {"enabled": True, "normalization": "minmax_space", "clip": True},
            "wrapper_reward": {
                "enabled": False,
                "profile": "cost_limits_v1",
                "clip_enabled": True,
                "clip_min": -10.0,
                "clip_max": 10.0,
                "squash": "none",
            },
        },
    }


def test_wrapper_entity_rebuilds_and_reattaches_on_topology_change():
    env = _DummyEntityEnv()
    wrapper = Wrapper_CityLearn(env=env, config=_entity_config(), job_id="entity-test")
    model = _DummyModel()
    wrapper.set_model(model)

    assert model.attach_calls == 1
    assert len(wrapper.action_space) == 1

    env._version = 1
    new_observations = env._observation_payload(version=1)
    adapted = wrapper._apply_entity_layout(new_observations, force_attach=False)

    assert len(adapted) == 2
    assert len(wrapper.action_space) == 2
    assert model.attach_calls == 2


def test_wrapper_entity_converts_flat_actions_into_entity_tables():
    env = _DummyEntityEnv()
    wrapper = Wrapper_CityLearn(env=env, config=_entity_config(), job_id="entity-actions")

    payload = wrapper._to_env_actions([[0.3, 0.8]])

    assert "tables" in payload
    assert payload["tables"]["building"].shape == (1, 1)
    assert payload["tables"]["charger"].shape == (1, 1)
    assert payload["tables"]["building"][0, 0] == pytest.approx(0.3, abs=1e-6)
    assert payload["tables"]["charger"][0, 0] == pytest.approx(0.8, abs=1e-6)


def test_wrapper_entity_building_names_fallbacks_to_entity_specs():
    env = _DummyEntityEnv()
    env._expose_building_names = False

    wrapper = Wrapper_CityLearn(env=env, config=_entity_config(), job_id="entity-building-names")
    info = wrapper.describe_environment()
    assert info["building_names"] == ["B1"]

    env._version = 1
    wrapper._apply_entity_layout(env._observation_payload(version=1), force_attach=False)
    updated_info = wrapper.describe_environment()
    assert updated_info["building_names"] == ["B1", "B2"]
