import random

import numpy as np
import pytest
import torch

from algorithms.agents.baseline_policies import RBCCommunityPolicy
from algorithms.utils.behavior_cloning import BehaviorCloningRegularizer
from algorithms.utils.entity_token_layout import BuildingTokenLayout, TokenSegment


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


def _regularizer(**behavior_cloning_overrides):
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(**behavior_cloning_overrides), _agent_config_template()
    )
    assert regularizer is not None
    return regularizer


def _layout(*ca_types):
    segments = tuple(
        TokenSegment(
            family="ca",
            type_name=type_name,
            instance_id=f"ca_{idx}",
            feature_indices=(idx,),
            feature_names=(f"feature_{idx}",),
        )
        for idx, type_name in enumerate(ca_types)
    )
    return BuildingTokenLayout(
        building_id="Building_1",
        segments=segments,
        n_sro=0,
        n_ca=len(segments),
        ca_action_names=tuple(f"action_{idx}" for idx in range(len(segments))),
        excluded_feature_names=(),
    )


class FakeTeacher:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []

    def predict(self, observations, deterministic=None):
        self.calls.append((observations, deterministic))
        return [[float(value) for value in action] for action in self.actions]


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


@pytest.mark.parametrize(
    ("overrides", "step", "expected"),
    [
        ({"weight": 0.0}, 0, 0.0),
        ({"weight": -0.1}, 0, 0.0),
        ({"weight": 0.4, "decay_start_step": 100, "decay_steps": 50}, 99, 0.4),
        ({"weight": 0.4, "min_weight": 0.05, "decay_start_step": 100, "decay_steps": 0}, 200, 0.4),
        ({"weight": 0.4, "min_weight": 0.05, "decay_start_step": 100, "decay_steps": 100}, 150, 0.225),
        ({"weight": 0.4, "min_weight": 0.05, "decay_start_step": 100, "decay_steps": 100}, 250, 0.05),
        ({"weight": 0.4, "min_weight": 0.8, "decay_start_step": 100, "decay_steps": 100}, 150, 0.4),
    ],
)
def test_effective_weight_schedule_cases(overrides, step, expected):
    regularizer = _regularizer(**overrides)

    assert regularizer.effective_weight(step) == pytest.approx(expected)


def test_ca_type_weights_aligns_with_ca_layout_order():
    regularizer = _regularizer(ev_multiplier=3.0, storage_multiplier=0.25)
    layout = _layout("storage", "charger", "district")

    weights = regularizer.ca_type_weights(
        layout,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert weights.dtype == torch.float64
    assert weights.tolist() == pytest.approx([0.25, 3.0, 1.0])


def test_bc_loss_zero_when_predicted_means_equal_teacher():
    regularizer = _regularizer()
    regularizer.teacher_action_buffers = [[[0.1, -0.2], [0.3, 0.4]]]
    predictions = torch.tensor([[[0.1], [-0.2]], [[0.3], [0.4]]])

    loss = regularizer.bc_loss_term(
        building_idx=0,
        layout=_layout("charger", "storage"),
        predicted_means=predictions,
        step_indices=torch.tensor([0, 1]),
        global_learning_step=100,
    )

    assert loss.item() == pytest.approx(0.0)
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_loss"] == pytest.approx(0.0)
    assert metrics["behavior_cloning_weighted_loss"] == pytest.approx(0.0)
    assert metrics["behavior_cloning_effective_weight"] == pytest.approx(0.4)
    assert metrics["behavior_cloning_valid_samples"] == pytest.approx(2.0)


def test_bc_loss_ignores_missing_teacher_and_mismatched_lengths():
    regularizer = _regularizer()
    regularizer.teacher_action_buffers = [[[0.0, 0.0], None, [0.5], [0.2, -0.2]]]
    predictions = torch.tensor(
        [[[0.0], [0.0]], [[0.9], [0.9]], [[0.9], [0.9]], [[0.2], [-0.2]]]
    )

    loss = regularizer.bc_loss_term(
        building_idx=0,
        layout=_layout("charger", "storage"),
        predicted_means=predictions,
        step_indices=torch.tensor([0, 1, 2, 3]),
        global_learning_step=100,
    )

    assert loss.item() == pytest.approx(0.0)
    assert regularizer.snapshot_metrics()["behavior_cloning_valid_samples"] == pytest.approx(2.0)


def test_bc_loss_normalizes_by_active_ca_type_weights():
    regularizer = _regularizer(
        weight=1.0,
        min_weight=1.0,
        decay_start_step=0,
        decay_steps=0,
        ev_multiplier=10.0,
        storage_multiplier=1.0,
    )
    regularizer.teacher_action_buffers = [[[0.0, 0.0]]]
    predictions = torch.tensor([[[1.0], [0.0]]])

    loss = regularizer.bc_loss_term(
        building_idx=0,
        layout=_layout("charger", "storage"),
        predicted_means=predictions,
        step_indices=torch.tensor([0]),
        global_learning_step=0,
    )

    raw_loss = 10.0 / 11.0
    assert loss.item() == pytest.approx(raw_loss)
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_loss"] == pytest.approx(raw_loss)
    assert metrics["behavior_cloning_weighted_loss"] == pytest.approx(raw_loss)
    assert metrics["behavior_cloning_effective_weight"] == pytest.approx(1.0)
    assert metrics["behavior_cloning_valid_samples"] == pytest.approx(1.0)


def test_compute_teacher_actions_populates_latest_teacher_actions_with_fake_teacher():
    regularizer = _regularizer(warm_start={
        "policy": "RBCCommunityPolicy",
        "deterministic": False,
        "noise_scale": 0.0,
        "phaseout_steps": 25,
        "phaseout_mode": "probability",
        "hyperparameters": {},
    })
    teacher = FakeTeacher([[0.1, -0.2], [0.3]])
    regularizer.teacher_policy = teacher
    observations = [np.array([1.0, 2.0]), np.array([3.0])]

    actions = regularizer.compute_teacher_actions(observations)

    assert actions == [[0.1, -0.2], [0.3]]
    assert regularizer.latest_teacher_actions == [[0.1, -0.2], [0.3]]
    assert teacher.calls == [(observations, False)]
    actions[0][0] = 99.0
    assert regularizer.latest_teacher_actions == [[0.1, -0.2], [0.3]]


def test_compute_teacher_actions_without_teacher_clears_latest_actions():
    regularizer = _regularizer()
    regularizer.set_latest_teacher_actions([[0.1]])

    assert regularizer.compute_teacher_actions([np.array([1.0])]) is None
    assert regularizer.latest_teacher_actions is None


def test_maybe_phaseout_blend_interpolates_and_eventually_reaches_zero_probability():
    regularizer = _regularizer(warm_start={
        "policy": "RBCCommunityPolicy",
        "deterministic": True,
        "noise_scale": 0.0,
        "phaseout_steps": 2,
        "phaseout_mode": "blend",
        "hyperparameters": {},
    })
    regularizer.set_latest_teacher_actions([[1.0, -1.0]])
    actor_actions = [[-1.0, 1.0]]

    blended = regularizer.maybe_phaseout(actor_actions, deterministic=False)
    final_actions = regularizer.maybe_phaseout(actor_actions, deterministic=False)

    assert blended[0] == pytest.approx([0.0, 0.0])
    assert final_actions == actor_actions
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_phaseout_probability"] == pytest.approx(0.0)
    assert metrics["behavior_cloning_phaseout_used"] == pytest.approx(0.0)


def test_maybe_phaseout_probability_mode_can_return_teacher_actions(monkeypatch):
    regularizer = _regularizer(warm_start={
        "policy": "RBCCommunityPolicy",
        "deterministic": True,
        "noise_scale": 0.0,
        "phaseout_steps": 4,
        "phaseout_mode": "probability",
        "hyperparameters": {},
    })
    regularizer.set_latest_teacher_actions([[0.7]])
    monkeypatch.setattr(random, "random", lambda: 0.1)

    actions = regularizer.maybe_phaseout([[-0.7]], deterministic=False)

    assert actions == [[0.7]]
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_phaseout_probability"] == pytest.approx(0.75)
    assert metrics["behavior_cloning_phaseout_used"] == pytest.approx(1.0)


def test_maybe_phaseout_probability_mode_keeps_actor_on_teacher_shape_mismatch(monkeypatch):
    regularizer = _regularizer(warm_start={
        "policy": "RBCCommunityPolicy",
        "deterministic": True,
        "noise_scale": 0.0,
        "phaseout_steps": 4,
        "phaseout_mode": "probability",
        "hyperparameters": {},
    })
    regularizer.set_latest_teacher_actions([[0.7], [0.9]])
    actor_actions = [[-0.7], [-0.1, 0.1]]
    monkeypatch.setattr(random, "random", lambda: 0.1)

    actions = regularizer.maybe_phaseout(actor_actions, deterministic=False)

    assert actions == [[0.7], [-0.1, 0.1]]
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_phaseout_probability"] == pytest.approx(0.75)
    assert metrics["behavior_cloning_phaseout_used"] == pytest.approx(1.0)


def test_maybe_phaseout_deterministic_returns_actor_actions_unchanged():
    regularizer = _regularizer(warm_start={
        "policy": "RBCCommunityPolicy",
        "deterministic": True,
        "noise_scale": 0.0,
        "phaseout_steps": 4,
        "phaseout_mode": "blend",
        "hyperparameters": {},
    })
    regularizer.set_latest_teacher_actions([[0.7]])
    actor_actions = [[-0.7]]

    assert regularizer.maybe_phaseout(actor_actions, deterministic=True) is actor_actions
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_phaseout_probability"] == pytest.approx(0.0)
    assert metrics["behavior_cloning_phaseout_used"] == pytest.approx(0.0)


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


def test_set_latest_teacher_actions_none_clears_and_records_none():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())
    regularizer.set_latest_teacher_actions([[0.1], [0.2]])

    regularizer.set_latest_teacher_actions(None)
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    assert regularizer.latest_teacher_actions is None
    assert regularizer.teacher_action_buffers == [[None], [None]]
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


def test_on_topology_change_preserves_unchanged_building_buffers():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )
    regularizer.attach_environment(**_attach_args())
    old_teacher = regularizer.teacher_policy
    regularizer.set_latest_teacher_actions([[0.1], [0.2]])
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    regularizer.on_topology_change(**_attach_args(), changed_buildings=[0])

    assert regularizer.teacher_policy is not old_teacher
    assert isinstance(regularizer.teacher_policy, RBCCommunityPolicy)
    assert regularizer.latest_teacher_actions is None
    assert regularizer.teacher_action_buffers == [[], [[0.2]]]


def test_snapshot_metrics_reports_lifecycle_diagnostics():
    regularizer = BehaviorCloningRegularizer.from_config(
        _algorithm_config(), _agent_config_template()
    )

    assert regularizer.snapshot_metrics() == {
        "behavior_cloning_teacher_enabled": 0.0,
        "behavior_cloning_latest_teacher_available": 0.0,
        "behavior_cloning_teacher_buffer_size": 0.0,
        "behavior_cloning_effective_weight": 0.0,
        "behavior_cloning_loss": 0.0,
        "behavior_cloning_weighted_loss": 0.0,
        "behavior_cloning_valid_samples": 0.0,
        "behavior_cloning_phaseout_probability": 0.0,
        "behavior_cloning_phaseout_used": 0.0,
    }

    regularizer.attach_environment(**_attach_args())
    regularizer.set_latest_teacher_actions([[0.1], [0.2]])
    regularizer.record_transition(0)
    regularizer.record_transition(1)

    assert regularizer.snapshot_metrics() == {
        "behavior_cloning_teacher_enabled": 1.0,
        "behavior_cloning_latest_teacher_available": 1.0,
        "behavior_cloning_teacher_buffer_size": 2.0,
        "behavior_cloning_effective_weight": 0.0,
        "behavior_cloning_loss": 0.0,
        "behavior_cloning_weighted_loss": 0.0,
        "behavior_cloning_valid_samples": 0.0,
        "behavior_cloning_phaseout_probability": 0.0,
        "behavior_cloning_phaseout_used": 0.0,
    }

    regularizer.set_latest_teacher_actions(None)

    assert regularizer.snapshot_metrics() == {
        "behavior_cloning_teacher_enabled": 1.0,
        "behavior_cloning_latest_teacher_available": 0.0,
        "behavior_cloning_teacher_buffer_size": 2.0,
        "behavior_cloning_effective_weight": 0.0,
        "behavior_cloning_loss": 0.0,
        "behavior_cloning_weighted_loss": 0.0,
        "behavior_cloning_valid_samples": 0.0,
        "behavior_cloning_phaseout_probability": 0.0,
        "behavior_cloning_phaseout_used": 0.0,
    }
