import pytest

from algorithms.agents.baseline_policies import RBCCommunityPolicy
from algorithms.utils.behavior_cloning import BehaviorCloningRegularizer


class DummySpace:
    def __init__(self, low, high):
        self.low = low
        self.high = high


def _agent_config_template():
    return {
        "algorithm": {"name": "AgentTransformerPPO"},
        "simulator": {"interface": "entity"},
    }


def _algorithm_config(**behavior_cloning_overrides):
    behavior_cloning = {
        "enabled": True,
        "weight": 0.4,
        "min_weight": 0.05,
        "decay_start_step": 100,
        "decay_steps": 500,
        "ev_multiplier": 2.0,
        "storage_multiplier": 0.5,
        "warm_start": {
            "policy": "RBCCommunityPolicy",
            "deterministic": True,
            "noise_scale": 0.03,
            "phaseout_steps": 25,
            "phaseout_mode": "linear",
            "hyperparameters": {"pv_preferred_charge_rate": 0.37},
        },
    }
    behavior_cloning.update(behavior_cloning_overrides)
    return {"name": "AgentTransformerPPO", "behavior_cloning": behavior_cloning}


def _attach_args(building_count=2):
    observation_names = [
        ["hour", "pv_power_kw", "electrical_storage_soc"],
        ["hour", "load_power_kw"],
    ][:building_count]
    action_names = [
        ["electrical_storage"],
        ["electric_vehicle_storage_charger_2_1"],
    ][:building_count]
    action_space = [
        DummySpace([-1.0], [1.0]),
        DummySpace([0.0], [1.0]),
    ][:building_count]
    observation_space = [object() for _ in range(building_count)]
    metadata = {
        "building_names": [f"Building_{idx + 1}" for idx in range(building_count)],
        "seconds_per_time_step": 900,
    }
    return {
        "observation_names": observation_names,
        "action_names": action_names,
        "action_space": action_space,
        "observation_space": observation_space,
        "metadata": metadata,
    }


def test_from_config_returns_none_when_absent_disabled_or_missing_warm_start():
    assert BehaviorCloningRegularizer.from_config({}, _agent_config_template()) is None
    assert (
        BehaviorCloningRegularizer.from_config(
            _algorithm_config(enabled=False), _agent_config_template()
        )
        is None
    )
    algorithm_cfg_without_warm_start = _algorithm_config()
    del algorithm_cfg_without_warm_start["behavior_cloning"]["warm_start"]
    assert (
        BehaviorCloningRegularizer.from_config(
            algorithm_cfg_without_warm_start, _agent_config_template()
        )
        is None
    )
    assert (
        BehaviorCloningRegularizer.from_config(
            _algorithm_config(warm_start=None), _agent_config_template()
        )
        is None
    )


def test_from_config_parses_enabled_config_values():
    agent_config_template = _agent_config_template()
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), agent_config_template
    )

    assert regularizer is not None
    assert regularizer.enabled is True
    assert regularizer.weight == pytest.approx(0.4)
    assert regularizer.min_weight == pytest.approx(0.05)
    assert regularizer.decay_start_step == 100
    assert regularizer.decay_steps == 500
    assert regularizer.ev_multiplier == pytest.approx(2.0)
    assert regularizer.storage_multiplier == pytest.approx(0.5)
    assert regularizer.policy == "RBCCommunityPolicy"
    assert regularizer.deterministic is True
    assert regularizer.noise_scale == pytest.approx(0.03)
    assert regularizer.phaseout_steps == 25
    assert regularizer.phaseout_mode == "linear"
    assert regularizer.hyperparameters == {"pv_preferred_charge_rate": 0.37}
    assert regularizer.teacher_policy is None
    assert regularizer.latest_teacher_actions is None

    regularizer.agent_config_template["algorithm"]["name"] = "mutated"
    assert agent_config_template["algorithm"]["name"] == "AgentTransformerPPO"


def test_attach_environment_builds_teacher_and_initializes_buffers():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )

    regularizer.attach_environment(**_attach_args())

    assert isinstance(regularizer.teacher_policy, RBCCommunityPolicy)
    assert regularizer.teacher_policy._action_labels == [
        ["electrical_storage"],
        ["electric_vehicle_storage_charger_2_1"],
    ]
    assert regularizer.teacher_action_buffers == [[], []]


def test_record_transition_aligns_teacher_actions_by_building():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())

    actions = [[0.1, 0.2], [0.3]]
    regularizer.set_latest_teacher_actions(actions)
    actions[0][0] = 99.0
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    building_0_action = regularizer.teacher_action_for(0, 0)
    building_1_action = regularizer.teacher_action_for(1, 0)
    assert building_0_action == [0.1, 0.2]
    assert building_1_action == [0.3]

    building_0_action.append(99.0)
    assert regularizer.teacher_action_for(0, 0) == [0.1, 0.2]


def test_record_transition_appends_none_when_teacher_missing():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())

    regularizer.record_transition(0)
    regularizer.set_latest_teacher_actions([[0.1]])
    regularizer.record_transition(1)

    assert regularizer.teacher_action_for(0, 0) is None
    assert regularizer.teacher_action_for(1, 0) is None


def test_on_buffer_flushed_clears_only_one_building():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())
    regularizer.set_latest_teacher_actions([[0.1], [0.2]])
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    regularizer.on_buffer_flushed(0)

    assert regularizer.teacher_action_for(0, 0) is None
    assert regularizer.teacher_action_for(1, 0) == [0.2]


def test_on_topology_change_clears_buffers_and_reattaches_teacher():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())
    old_teacher = regularizer.teacher_policy
    regularizer.set_latest_teacher_actions([[0.1], [0.2]])
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    regularizer.on_topology_change(**_attach_args(building_count=1))

    assert regularizer.teacher_policy is not old_teacher
    assert isinstance(regularizer.teacher_policy, RBCCommunityPolicy)
    assert regularizer.latest_teacher_actions is None
    assert regularizer.teacher_action_buffers == [[]]
