from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from algorithms.agents.rbc_agent import RuleBasedPolicy
from utils.wrapper_citylearn import Wrapper_CityLearn


def _supports_entity_interface() -> bool:
    try:
        from citylearn.citylearn import CityLearnEnv
    except Exception:
        return False

    try:
        signature = inspect.signature(CityLearnEnv.__init__)
    except Exception:
        return False

    return "interface" in signature.parameters and "topology_mode" in signature.parameters


@pytest.mark.slow
def test_rule_based_dynamic_entity_smoke_1000_steps():
    if not _supports_entity_interface():
        pytest.skip("Installed simulator does not expose entity/topology_mode constructor arguments.")

    from citylearn.citylearn import CityLearnEnv

    schema_path = Path("datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json")
    if not schema_path.exists():
        pytest.skip("Dynamic topology demo dataset not available in Algorithms datasets directory.")

    config = {
        "runtime": {"log_dir": None},
        "training": {"steps_between_training_updates": 1, "target_update_interval": 0},
        "checkpointing": {"checkpoint_interval": None, "require_update_step": True, "require_initial_exploration_done": True},
        "tracking": {
            "mlflow_enabled": False,
            "log_frequency": 1000,
            "mlflow_step_sample_interval": 1000,
            "progress_updates_enabled": False,
        },
        "algorithm": {
            "name": "RuleBasedPolicy",
            "hyperparameters": {
                "pv_charge_threshold": 0.0,
                "flexibility_hours": 3.0,
                "emergency_hours": 1.0,
                "pv_preferred_charge_rate": 0.6,
                "flex_trickle_charge": 0.0,
                "min_charge_rate": 0.0,
                "emergency_charge_rate": 1.0,
                "energy_epsilon": 1e-3,
                "default_capacity_kwh": 60.0,
                "non_flexible_chargers": [],
            },
        },
        "simulator": {
            "dataset_name": "citylearn_three_phase_dynamic_topology_demo_v1",
            "dataset_path": str(schema_path),
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

    env = CityLearnEnv(
        schema=str(schema_path),
        interface="entity",
        topology_mode="dynamic",
        central_agent=False,
        offline=True,
        simulation_start_time_step=0,
        episode_time_steps=1000,
    )

    wrapper = Wrapper_CityLearn(
        env=env,
        config=config,
        job_id="dynamic-smoke",
    )
    agent = RuleBasedPolicy(config)
    wrapper.set_model(agent)

    wrapper.learn(episodes=1)

    assert wrapper.episode_time_steps == 1000
    assert wrapper.global_step >= 999
