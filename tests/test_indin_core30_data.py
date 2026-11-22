"""Scientific guards for transition timing, resets and chronological separation."""

import numpy as np
import pytest

from scripts.indin_core30.data import RampState, RewardCapture, relabel_rewards, validate_windows
from algorithms.offline_rl.entity_layout import canonicalize_entity_names


def test_ramp_uses_signed_aggregate_power_and_excludes_reset_interval():
    state = RampState()
    assert state.observe(2.0, 900)["ramp_abs_kw"] == 0
    crossing = state.observe(-1.0, 900)
    assert crossing["community_net_power_kw"] == -4
    assert crossing["ramp_abs_kw"] == 12
    assert crossing["ramp_up_kw"] == 0
    assert state.observe(1.0, 900)["ramp_up_kw"] == 8
    assert RampState().observe(99.0, 900)["ramp_abs_kw"] == 0


def test_reward_tap_preserves_base_reward_and_captures_reward_time_data():
    class Reward:
        def calculate(self, observations):
            self.components = {"per_agent": [{"cost": 3}, {"cost": 4}], "community": {}}
            return [-3, -4]

        def get_last_components(self):
            return self.components

    reward = Reward()
    tap = RewardCapture(reward, 900)
    obs = [{"net_electricity_consumption": 3}, {"net_electricity_consumption": -2}]
    assert tap.calculate(obs) == [-3, -4]
    obs[0]["net_electricity_consumption"] = 1000
    reward.components["per_agent"][0]["cost"] = 999
    assert tap.last["community_net_energy_kwh"] == 1
    assert tap.last["components"][0]["cost"] == 3


def test_relabel_uses_captured_community_ramp_without_differencing_agent_rows():
    base = np.array([-1, -2, -3, -4])
    # Two agents at each instant: there is no spurious within-instant ramp.
    ramps = np.array([0, 0, 8, 8])
    np.testing.assert_allclose(relabel_rewards(base, ramps, weight=2, ramp_scale_kw=4),
                               [-1, -2, -7, -8])
    np.testing.assert_array_equal(relabel_rewards(base, ramps, weight=0, ramp_scale_kw=4), base)


def test_chronological_train_validation_boundaries():
    validate_windows([
        {"id": "jan", "split": "train", "start_step": 0, "transitions": 96},
        {"id": "feb", "split": "validation", "start_step": 31 * 96, "transitions": 14 * 96},
    ])
    with pytest.raises(ValueError, match="overlaps"):
        validate_windows([{"id": "leak", "split": "train", "start_step": 31 * 96 - 1, "transitions": 2}])
    with pytest.raises(ValueError, match="2023"):
        validate_windows([{"id": "test", "split": "train", "start_step": 365 * 96, "transitions": 1}])


def test_canonical_slots_preserve_action_binding_and_feature_semantics():
    left = ["charger::Member_003/CHG-M003-02::soc", "building.price",
            "charger::Member_003/CHG-M003-01::soc"]
    right = ["charger::Member_004/CHG-M004-02::soc", "building.price",
             "charger::Member_004/CHG-M004-01::soc"]
    left_actions = ["electric_vehicle_storage_CHG-M003-02", "electric_vehicle_storage_CHG-M003-01"]
    right_actions = ["electric_vehicle_storage_CHG-M004-02", "electric_vehicle_storage_CHG-M004-01"]
    canonical_obs, canonical_act = canonicalize_entity_names(left, left_actions)
    assert (canonical_obs, canonical_act) == canonicalize_entity_names(right, right_actions)
    assert canonical_obs == ["charger::slot_001::soc", "building.price", "charger::slot_000::soc"]
    assert canonical_act == ["electric_vehicle_storage_slot_001", "electric_vehicle_storage_slot_000"]
    changed = [name.replace("::soc", "::capacity") for name in right]
    assert canonicalize_entity_names(changed, right_actions)[0] != canonical_obs


def test_canonical_binding_rejects_ambiguous_or_missing_device():
    with pytest.raises(ValueError, match="unique"):
        canonicalize_entity_names(["charger::Member_1/CHG-01::soc"], ["electric_vehicle_storage_CHG-02"])
    with pytest.raises(ValueError, match="Duplicate"):
        canonicalize_entity_names(["building.price", "building.price"], [])
    assert canonicalize_entity_names(
        ["charger::Member_1/CHG-01::connected_ev::soc"], ["electric_vehicle_storage_CHG-01"]
    )[0] == ["charger::slot_000::connected_ev::soc"]


def test_declared_reward_scale_fallbacks_never_use_arbitrary_constants():
    from scripts.indin_core30.derive import reward_scale

    assert reward_scale([-1, -2, -3]) == (2, "median_absolute")
    assert reward_scale([0, 0, 0, 4])[1] == "p75_absolute_after_zero_median"
    assert reward_scale([0] * 9 + [4]) == (4, "mean_positive_absolute_after_zero_median_and_p75")
    with pytest.raises(ValueError, match="no non-zero"):
        reward_scale([0, 0])


def test_raw_collection_ignores_minmax_bounds_and_preserves_finite_features():
    from gymnasium.spaces import Box
    from scripts.indin_core30.data import collection_entity_adapter

    values = np.array([-3.0, 125.0, 0.5])
    bounds = Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 100.0, 1.0]), dtype=np.float64)
    raw = collection_entity_adapter(object(), {"entity_encoding": {
        "enabled": False, "clip": False, "normalization": "minmax_space"}})
    np.testing.assert_array_equal(raw.normalize_observation(0, values, ["a", "b", "c"], bounds), values)
    alternate = Box(low=np.array([-100.0, -100.0, -100.0]), high=np.array([1000.0, 1000.0, 1000.0]), dtype=np.float64)
    np.testing.assert_array_equal(raw.normalize_observation(0, values, ["a", "b", "c"], alternate), values)
    encoded = collection_entity_adapter(object(), {"entity_encoding": {
        "enabled": True, "clip": True, "normalization": "minmax_space"}})
    np.testing.assert_array_equal(encoded.normalize_observation(0, values, ["a", "b", "c"], bounds), [0, 1, 0.5])
