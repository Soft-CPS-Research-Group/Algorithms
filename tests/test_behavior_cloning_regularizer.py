import numpy as np
import pytest
import torch
from pydantic import ValidationError

from algorithms.utils.behavior_cloning import (
    BehaviorCloningRegularizer,
    Demonstration,
)
from algorithms.utils.entity_token_layout import BuildingTokenLayout, TokenSegment
from utils.config_schema import TransformerPPOBehaviorCloningConfig


def _layout(instance_id: str = "charger_1") -> BuildingTokenLayout:
    return BuildingTokenLayout(
        building_id="Building_1",
        segments=(
            TokenSegment("nfc", "building_nfc", None, (0, 1), ("hour", "month")),
            TokenSegment("ca", "charger", instance_id, (2,), ("soc",)),
        ),
        n_sro=0,
        n_ca=1,
        ca_action_names=("electric_vehicle_storage",),
        excluded_feature_names=(),
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


def test_reservoir_sampling_is_bounded_per_building_and_auxiliary_loss_uses_demos() -> None:
    regularizer = _regularizer(max_samples_per_building=2, weight=1.0)
    layout = _layout()
    for value in range(10):
        regularizer.record_demonstration(0, np.full(3, value), layout, [0.0])

    assert regularizer.demonstration_count(0) == 2
    predictions = torch.tensor([[[1.0]], [[0.0]]])
    loss = regularizer.demonstration_loss(
        layout=layout,
        demonstrations=regularizer.sample_demonstrations(layout, batch_size=2),
        predicted_means=predictions,
        global_learning_step=0,
    )
    assert loss.item() == pytest.approx(0.5)
    metrics = regularizer.snapshot_metrics()
    assert metrics["behavior_cloning_demonstration_samples"] == 2.0
    assert metrics["behavior_cloning_valid_samples"] == 2.0
