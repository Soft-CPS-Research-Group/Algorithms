import pytest

from algorithms.agents.baseline_policies import RBCCommunityPolicy, RBCSmartPolicy
from algorithms.utils.warm_start_policy import build_warm_start_policy


class DummySpace:
    def __init__(self, low, high):
        self.low = low
        self.high = high


def test_build_warm_start_policy_instantiates_and_attaches_rbc_community():
    observation_names = [
        ["hour", "pv_power_kw", "electrical_storage_soc"],
        ["hour", "load_power_kw"],
    ]
    action_names = [["electrical_storage"], ["electric_vehicle_storage_charger_2_1"]]
    action_space = [DummySpace([-1.0], [1.0]), DummySpace([0.0], [1.0])]
    observation_space = [object(), object()]
    metadata = {"building_names": ["Building_1", "Building_2"], "seconds_per_time_step": 900}

    policy = build_warm_start_policy(
        owner_name="TransformerPPO",
        policy_name="RBCCommunityPolicy",
        policy_hyperparameters=None,
        config_template={"algorithm": {"name": "TransformerPPO"}, "simulator": {}},
        observation_names=observation_names,
        action_names=action_names,
        action_space=action_space,
        observation_space=observation_space,
        metadata=metadata,
    )

    assert isinstance(policy, RBCCommunityPolicy)
    assert policy._action_labels == action_names
    assert policy._obs_index == [
        {"hour": 0, "pv_power_kw": 1, "electrical_storage_soc": 2},
        {"hour": 0, "load_power_kw": 1},
    ]
    assert policy._agent_buildings == {0: "Building_1", 1: "Building_2"}
    assert policy.step_hours == pytest.approx(0.25)


def test_build_warm_start_policy_passes_hyperparameters():
    policy = build_warm_start_policy(
        owner_name="TransformerPPO",
        policy_name="RBCSmartPolicy",
        policy_hyperparameters={"pv_preferred_charge_rate": 0.37},
        config_template={"algorithm": {"name": "TransformerPPO"}},
        observation_names=[["hour"]],
        action_names=[["electric_vehicle_storage_charger_1_1"]],
        action_space=[DummySpace([0.0], [1.0])],
        observation_space=[None],
        metadata=None,
    )

    assert isinstance(policy, RBCSmartPolicy)
    assert policy.pv_preferred_charge_rate == pytest.approx(0.37)


def test_build_warm_start_policy_rejects_unsupported_policy():
    with pytest.raises(ValueError) as exc_info:
        build_warm_start_policy(
            owner_name="TransformerPPO",
            policy_name="UnsupportedPolicy",
            policy_hyperparameters=None,
            config_template={},
            observation_names=[[]],
            action_names=[[]],
            action_space=[],
            observation_space=[],
            metadata=None,
        )

    message = str(exc_info.value)
    assert "TransformerPPO" in message
    assert "UnsupportedPolicy" in message
    assert "RBCCommunityPolicy" in message


def test_build_warm_start_policy_rejects_non_mapping_hyperparameters():
    with pytest.raises(ValueError) as exc_info:
        build_warm_start_policy(
            owner_name="TransformerPPO",
            policy_name="RBCCommunityPolicy",
            policy_hyperparameters=[("pv_preferred_charge_rate", 0.37)],
            config_template={},
            observation_names=[[]],
            action_names=[[]],
            action_space=[],
            observation_space=[],
            metadata=None,
        )

    message = str(exc_info.value)
    assert "TransformerPPO" in message
    assert "RBCCommunityPolicy" in message
    assert "hyperparameters" in message
