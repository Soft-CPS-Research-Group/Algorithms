from dataclasses import replace

import numpy as np
import pytest
import torch
from loguru import logger
from pydantic import ValidationError

from algorithms.utils.behavior_cloning import (
    BehaviorCloningRegularizer,
    Demonstration,
)
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    NfcExpression,
    TokenSegment,
)
from utils.config_schema import TransformerPPOBehaviorCloningConfig


def _layout(instance_id: str = "charger_1") -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id="Building_1",
        segments=(
            TokenSegment(
                "nfc", "building_nfc", None, (0, 1), ("hour", "month"),
                NfcExpression("subtract", 0, 1),
            ),
            TokenSegment("ca", "charger", instance_id, (2,), ("soc",)),
        ),
        n_sro=0,
        n_ca=1,
        ca_action_names=("electric_vehicle_storage",),
        excluded_feature_names=(),
    )


def _six_feature_layout() -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id="Building_1",
        segments=(
            TokenSegment("nfc", "building_nfc", None, (0, 1, 2), ("hour", "month", "day")),
            TokenSegment("ca", "charger", "charger_1", (3, 4, 5), ("soc", "demand", "power")),
        ),
        n_sro=0,
        n_ca=1,
        ca_action_names=("electric_vehicle_storage",),
        excluded_feature_names=(),
    )


def _layout_with_trailing_excluded_features() -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id="Building_1",
        segments=(
            TokenSegment("nfc", "building_nfc", None, (0, 2), ("hour", "month")),
            TokenSegment("ca", "charger", "charger_1", (3,), ("soc",)),
        ),
        n_sro=0,
        n_ca=1,
        ca_action_names=("electric_vehicle_storage",),
        excluded_feature_names=("excluded_after_selected", "another_excluded"),
    )


def _regularizer(**overrides) -> BehaviorCloningRegularizer:
    behavior_cloning = {
        "enabled": True,
        "demonstration_episodes": 1,
        "max_samples_per_building": 2,
        "pretraining_epochs": 2,
        "batch_size": 1,
        "weight": 0.4,
        "min_weight": 0.1,
        "decay_start_step": 5,
        "decay_steps": 10,
        "ev_multiplier": 2.0,
        "storage_multiplier": 1.0,
        "teacher": {
            "policy": "RBCSmartPolicy",
            "deterministic": True,
            "hyperparameters": {},
        },
    }
    behavior_cloning.update(overrides)
    regularizer = BehaviorCloningRegularizer.from_config(
        {"behavior_cloning": behavior_cloning}, {"algorithm": {"name": "AgentTransformerPPO"}}
    )
    assert regularizer is not None
    return regularizer


def test_schema_requires_smart_teacher_and_rejects_legacy_phaseout_fields() -> None:
    parsed = TransformerPPOBehaviorCloningConfig.model_validate(
        _regularizer().config_dict
    )
    assert parsed.teacher.policy == "RBCSmartPolicy"

    legacy = _regularizer().config_dict
    legacy["teacher"]["phaseout_steps"] = 1
    with pytest.raises(ValidationError, match="phaseout_steps"):
        TransformerPPOBehaviorCloningConfig.model_validate(legacy)


def test_demonstration_is_frozen_and_groups_by_layout_signature() -> None:
    regularizer = _regularizer(max_samples_per_building=8)
    layout = _layout()
    observation = np.array([1.0, 2.0, 3.0])
    target = [0.25]

    regularizer.record_demonstration(0, observation, layout, target)
    observation[0] = 99.0
    target[0] = 99.0

    signature = regularizer.layout_signature(layout)
    demo = regularizer.demonstrations_by_signature[signature][0]
    assert isinstance(demo, Demonstration)
    assert demo.observation.tolist() == [1.0, 2.0, 3.0]
    assert demo.target.tolist() == [0.25]
    assert demo.layout is not layout
    assert demo.layout == layout
    with pytest.raises((AttributeError, TypeError, ValueError)):
        demo.target[0] = 0.0

    regularizer.record_demonstration(0, np.zeros(3), _layout("charger_2"), [0.5])
    assert len(regularizer.demonstrations_by_signature) == 2
    assert len(regularizer.demonstrations_for_building_by_signature(0)) == 2


def test_demonstration_stores_encoded_length() -> None:
    regularizer = _regularizer()
    layout = _six_feature_layout()

    regularizer.record_demonstration(
        0, np.arange(6, dtype=np.float32), layout, [0.25]
    )

    signature = regularizer.layout_signature(layout)
    demo = regularizer.demonstrations_for_building_by_signature(0)[signature][0]
    assert demo.encoded_length == 6


def test_record_demonstration_rejects_shape_mismatch() -> None:
    regularizer = _regularizer()
    layout = _six_feature_layout()

    regularizer.record_demonstration(0, np.zeros(6), layout, [0.25])
    regularizer.record_demonstration(0, np.zeros(7), layout, [0.25])

    assert regularizer.demonstration_count(0) == 1
    assert regularizer.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 1.0


@pytest.mark.parametrize(
    ("observation", "target", "reason", "expected_shape", "actual_shape", "rejected"),
    [
        (np.zeros(7), [0.25], "observation_shape_mismatch", "(6,)", "(7,)", 1.0),
        (np.zeros(6), [0.25, 0.5], "target_shape_mismatch", "(1,)", "(2,)", 0.0),
        (np.zeros(6), [float("nan")], "target_nonfinite", "(1,)", "(1,)", 0.0),
    ],
)
def test_record_demonstration_logs_rejection(
    observation, target, reason, expected_shape, actual_shape, rejected
) -> None:
    regularizer = _regularizer()
    layout = _six_feature_layout()
    messages = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        regularizer.record_demonstration(3, observation, layout, target)
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert f"reason={reason}" in messages[0]
    assert f"expected_shape={expected_shape}" in messages[0]
    assert f"actual_shape={actual_shape}" in messages[0]
    assert regularizer.snapshot_metrics()["behavior_cloning_rejected_at_record"] == rejected


def test_valid_record_and_reservoir_replacement_do_not_log_warning() -> None:
    regularizer = _regularizer(max_samples_per_building=1)
    layout = _six_feature_layout()
    regularizer._rng.randrange = lambda _seen: 0
    messages = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        regularizer.record_demonstration(0, np.zeros(6), layout, [0.25])
        regularizer.record_demonstration(0, np.ones(6), layout, [0.5])
    finally:
        logger.remove(sink_id)

    assert messages == []
    assert regularizer.demonstration_count(0) == 1
    stored_demo = next(iter(regularizer.demonstrations_for_building_by_signature(0).values()))[0]
    assert stored_demo.observation.tolist() == [1.0] * 6
    assert stored_demo.target.tolist() == [0.5]


def test_rejected_record_metric_survives_state_round_trip() -> None:
    regularizer = _regularizer()
    layout = _six_feature_layout()
    regularizer.record_demonstration(0, np.zeros(7), layout, [0.25])

    restored = _regularizer()
    restored.load_state_dict(regularizer.state_dict())

    assert restored.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 1.0


def test_load_state_dict_defaults_missing_rejected_record_metric_to_zero() -> None:
    regularizer = _regularizer()
    state = regularizer.state_dict()
    state.pop("rejected_at_record", None)

    restored = _regularizer()
    restored.load_state_dict(state)

    assert restored.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 0.0


def test_load_state_dict_rejects_missing_required_state_key() -> None:
    regularizer = _regularizer()
    state = regularizer.state_dict()
    state.pop("seen_per_building")

    with pytest.raises(RuntimeError, match="missing required key.*seen_per_building"):
        regularizer.load_state_dict(state)


def test_load_state_dict_rejects_mismatched_reservoir_building_keys() -> None:
    regularizer = _regularizer()
    state = regularizer.state_dict()
    state["seen_per_building"][0] = 0

    with pytest.raises(RuntimeError, match="reservoir building keys"):
        _regularizer().load_state_dict(state)


def test_load_state_dict_rejects_seen_count_below_stored_demonstrations() -> None:
    regularizer = _regularizer()
    layout = _layout()
    regularizer.record_demonstration(0, np.zeros(3), layout, [0.25])
    state = regularizer.state_dict()
    state["seen_per_building"][0] = 0

    with pytest.raises(RuntimeError, match="reservoir seen count"):
        _regularizer().load_state_dict(state)


def test_record_demonstration_accepts_full_width_with_trailing_excluded_features() -> None:
    regularizer = _regularizer()
    layout = _layout_with_trailing_excluded_features()

    regularizer.record_demonstration(0, np.zeros(5), layout, [0.25])

    assert regularizer.demonstration_count(0) == 1
    assert regularizer.snapshot_metrics()["behavior_cloning_rejected_at_record"] == 0.0


def test_load_state_dict_rejects_legacy_demonstrations_without_encoded_length() -> None:
    regularizer = _regularizer()
    layout = _layout()
    legacy_demo = Demonstration.__new__(Demonstration)
    object.__setattr__(legacy_demo, "observation", np.zeros(3, dtype=np.float32))
    object.__setattr__(legacy_demo, "layout", layout)
    object.__setattr__(
        legacy_demo,
        "layout_signature",
        regularizer.layout_signature(layout),
    )
    object.__setattr__(legacy_demo, "target", np.array([0.25], dtype=np.float32))
    state = regularizer.state_dict()
    state["demonstrations"] = {0: [legacy_demo]}
    state["seen_per_building"] = {0: 1}

    with pytest.raises(RuntimeError, match="predates BC data contract"):
        regularizer.load_state_dict(state)


def test_load_state_dict_rejects_out_of_bounds_demonstration_layout() -> None:
    regularizer = _regularizer()
    layout = _layout()
    regularizer.record_demonstration(0, np.zeros(3), layout, [0.25])
    state = regularizer.state_dict()
    demo = state["demonstrations"][0][0]
    bad_segment = replace(demo.layout.segments[0], feature_indices=(3, 1))
    bad_layout = replace(demo.layout, segments=(bad_segment,) + demo.layout.segments[1:])
    state["demonstrations"][0][0] = Demonstration(
        observation=demo.observation,
        encoded_length=demo.encoded_length,
        layout=bad_layout,
        layout_signature=regularizer.layout_signature(bad_layout),
        target=demo.target,
    )

    with pytest.raises(RuntimeError, match="invalid BC layout"):
        _regularizer().load_state_dict(state)


def test_demonstration_accessors_return_immutable_group_snapshots() -> None:
    regularizer = _regularizer()
    layout = _layout()
    regularizer.record_demonstration(0, np.zeros(3), layout, [0.25])

    signature = regularizer.layout_signature(layout)
    grouped = regularizer.demonstrations_by_signature
    building_grouped = regularizer.demonstrations_for_building_by_signature(0)

    assert isinstance(grouped[signature], tuple)
    assert isinstance(building_grouped[signature], tuple)
    with pytest.raises(AttributeError):
        grouped[signature].append(grouped[signature][0])
    with pytest.raises(AttributeError):
        building_grouped[signature].append(building_grouped[signature][0])
    assert regularizer.demonstration_count(0) == 1


def test_reservoir_sampling_is_bounded_per_building_and_auxiliary_loss_uses_demos() -> None:
    regularizer = _regularizer(max_samples_per_building=2, weight=1.0)
    layout = _layout()
    for value in range(10):
        regularizer.record_demonstration(0, np.full(3, value), layout, [0.0])

    assert regularizer.demonstration_count(0) == 2
    predictions = torch.tensor([[[1.0]], [[0.0]]])
    loss = regularizer.demonstration_loss(
        layout=layout,
        demonstrations=regularizer.sample_demonstrations(0, layout, batch_size=2),
        predicted_means=predictions,
        global_learning_step=0,
    )
    assert loss.item() == pytest.approx(0.5)
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_demonstration_samples"] == 2.0
    assert metrics["behavior_cloning_valid_samples"] == 2.0
