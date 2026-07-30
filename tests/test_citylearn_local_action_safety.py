from __future__ import annotations

import pytest

from algorithms.utils.citylearn_local_action_safety import (
    CityLearnLocalSafetyAdapter,
    CityLearnSafetyConfig,
    preserve_teacher_service_with_storage_fallback,
    replace_service_actions_with_teacher,
)


def _adapter_and_observation(*, headroom: float = 12.0):
    building = "Building_15"
    names = [
        "charging_building_headroom_kw",
        "charging_building_export_headroom_kw",
        "charging_phase_L1_headroom_kw",
        "charging_phase_L2_headroom_kw",
        f"storage::{building}/electrical_storage::soc",
        f"storage::{building}/electrical_storage::soc_min_ratio",
        f"storage::{building}/electrical_storage::nominal_power_kw",
        f"storage::{building}/electrical_storage::available_charge_action_normalized",
        f"storage::{building}/electrical_storage::available_discharge_action_normalized",
        f"storage::{building}/electrical_storage::phase_connection_L1",
        f"storage::{building}/electrical_storage::phase_connection_L2",
        f"storage::{building}/electrical_storage::phase_connection_L3",
    ]
    values = [headroom, 12.0, 7.0, 5.0, 0.5, 0.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for charger, max_power, phase, minimum in (
        ("charger_15_1", 7.4, "L1", 0.5),
        ("charger_15_2", 11.0, "L2", 0.4),
    ):
        prefix = f"charger::{building}/{charger}::"
        names.extend(
            [
                f"{prefix}connected_state",
                f"{prefix}max_charging_power_kw",
                f"{prefix}max_discharging_power_kw",
                f"{prefix}min_charging_power_kw",
                f"{prefix}available_charge_action_normalized",
                f"{prefix}available_discharge_action_normalized",
                f"{prefix}min_required_action_normalized",
                f"{prefix}phase_connection_{phase}",
            ]
        )
        values.extend([1.0, max_power, max_power, 0.0, 1.0, 1.0, minimum, 1.0])
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=[
            "electrical_storage",
            "electric_vehicle_storage_charger_15_1",
            "electric_vehicle_storage_charger_15_2",
        ],
        action_low=[-1.0, -1.0, -1.0],
        action_high=[1.0, 1.0, 1.0],
        metadata={"building_names": [building]},
    )
    return adapter, values


def test_citylearn_adapter_reserves_both_building_15_ev_minima() -> None:
    adapter, observation = _adapter_and_observation()

    result = adapter.project(observation, [1.0, -1.0, -1.0])

    assert result.feasible
    assert result.executed_actions[1:] == pytest.approx((0.5, 0.4))
    # L2 has only 0.6 kW after its mandatory 4.4 kW allocation. The balanced
    # storage may therefore use at most 1.8 kW total, i.e. action 0.36.
    assert result.executed_actions[0] == pytest.approx(0.36)


def test_citylearn_adapter_keeps_configured_margin_below_phase_limit() -> None:
    base, observation = _adapter_and_observation()
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=base.observation_names,
        action_names=base.action_names,
        action_low=base.action_low,
        action_high=base.action_high,
        metadata={"building_names": ["Building_15"]},
        config=CityLearnSafetyConfig(headroom_reserve_kw=0.1),
    )

    result = adapter.project(observation, [1.0, -1.0, -1.0])

    # L2 keeps 0.1 kW below its limit: (5.0 - 0.1 - 4.4) * 3 / 5.
    assert result.executed_actions[0] == pytest.approx(0.30)


def test_citylearn_adapter_rejects_negative_headroom_margin() -> None:
    with pytest.raises(ValueError, match="headroom_reserve_kw"):
        CityLearnSafetyConfig(headroom_reserve_kw=-0.1)

    with pytest.raises(ValueError, match="ev_minimum_mode"):
        CityLearnSafetyConfig(ev_minimum_mode="optimistic")


def test_citylearn_adapter_raises_on_structurally_infeasible_service_when_requested() -> None:
    adapter, observation = _adapter_and_observation(headroom=1.0)
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=adapter.observation_names,
        action_names=adapter.action_names,
        action_low=adapter.action_low,
        action_high=adapter.action_high,
        metadata={"building_names": ["Building_15"]},
        config=CityLearnSafetyConfig(fail_on_infeasible=True),
    )

    with pytest.raises(RuntimeError, match="Local safety projection is infeasible"):
        adapter.project(observation, [0.0, 0.0, 0.0])


def test_citylearn_adapter_blocks_non_mandatory_deferrable_start() -> None:
    building = "Building_1"
    prefix = f"deferrable_appliance::{building}/deferrable_appliance_1::"
    names = [
        f"{prefix}pending",
        f"{prefix}running",
        f"{prefix}can_start",
        f"{prefix}must_start_now",
        f"{prefix}start_power_kw",
    ]
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=["deferrable_appliance_deferrable_appliance_1"],
        action_low=[0.0],
        action_high=[1.0],
        metadata={"building_names": [building]},
    )

    ordinary = adapter.project([1.0, 0.0, 1.0, 0.0, 3.0], [1.0])
    mandatory = adapter.project([1.0, 0.0, 1.0, 1.0, 3.0], [0.0])

    assert ordinary.executed_actions == pytest.approx((0.0,))
    assert mandatory.executed_actions == pytest.approx((1.0,))


def test_citylearn_adapter_lifts_urgent_ev_action_above_charger_deadband() -> None:
    building = "Building_4"
    prefix = f"charger::{building}/charger_4_1::"
    names = [
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}min_charging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
    ]
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=["electric_vehicle_storage_charger_4_1"],
        action_low=[-1.0],
        action_high=[1.0],
        metadata={"building_names": [building]},
    )

    result = adapter.project([1.0, 22.0, 22.0, 3.7, 1.0, 1.0, 0.05], [0.0])

    assert result.executed_actions[0] == pytest.approx(3.7 / 22.0)


def test_deadline_feasible_ev_floor_allows_deferral_until_energy_is_mandatory() -> None:
    building = "Building_4"
    prefix = f"charger::{building}/charger_4_1::"
    names = [
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}min_charging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
        f"{prefix}departure_energy_margin_kwh",
        f"{prefix}available_charge_power_kw",
        f"{prefix}charger_efficiency_ratio",
    ]
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=["electric_vehicle_storage_charger_4_1"],
        action_low=[-1.0],
        action_high=[1.0],
        metadata={"building_names": [building], "seconds_per_time_step": 900},
        config=CityLearnSafetyConfig(ev_minimum_mode="deadline_feasible"),
    )

    def project(margin_kwh: float) -> float:
        values = [1.0, 22.0, 22.0, 3.7, 1.0, 1.0, 0.4, margin_kwh, 22.0, 0.9]
        return adapter.project(values, [0.0]).executed_actions[0]

    assert project(6.0) == pytest.approx(0.0)
    assert project(2.0) == pytest.approx(
        (22.0 * 0.25 * 0.9 - 2.0) / (0.9 * 0.25 * 22.0)
    )


def test_ev_service_target_cap_prevents_policy_overcharge() -> None:
    building = "Building_4"
    prefix = f"charger::{building}/charger_4_1::"
    names = [
        f"{prefix}connected_state",
        f"{prefix}max_charging_power_kw",
        f"{prefix}max_discharging_power_kw",
        f"{prefix}min_charging_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
        f"{prefix}min_required_action_normalized",
        f"{prefix}energy_to_required_soc_kwh",
        f"{prefix}charger_efficiency_ratio",
    ]
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=["electric_vehicle_storage_charger_4_1"],
        action_low=[-1.0],
        action_high=[1.0],
        metadata={"building_names": [building], "seconds_per_time_step": 900},
        config=CityLearnSafetyConfig(protect_ev_service_target=True),
    )

    result = adapter.project(
        [1.0, 22.0, 22.0, 3.7, 1.0, 1.0, 0.0, 1.2375, 0.9],
        [1.0],
    )

    assert result.executed_actions[0] == pytest.approx(0.25)


def test_zero_filled_headroom_is_unconfigured_until_positive_envelope_seen() -> None:
    building = "Building_1"
    prefix = f"storage::{building}/electrical_storage::"
    names = [
        "charging_building_headroom_kw",
        f"{prefix}soc",
        f"{prefix}soc_min_ratio",
        f"{prefix}nominal_power_kw",
        f"{prefix}available_charge_action_normalized",
        f"{prefix}available_discharge_action_normalized",
    ]
    adapter = CityLearnLocalSafetyAdapter(
        observation_names=names,
        action_names=["electrical_storage"],
        action_low=[-1.0],
        action_high=[1.0],
        metadata={"building_names": [building]},
    )

    no_service_limit = adapter.project([0.0, 0.5, 0.0, 5.0, 1.0, 1.0], [1.0])
    configured = adapter.project([2.0, 0.5, 0.0, 5.0, 1.0, 1.0], [1.0])
    saturated_after_detection = adapter.project([0.0, 0.5, 0.0, 5.0, 1.0, 1.0], [1.0])

    assert no_service_limit.executed_actions == pytest.approx((1.0,))
    assert configured.executed_actions == pytest.approx((0.4,))
    assert saturated_after_detection.executed_actions == pytest.approx((0.0,))


def test_service_teacher_replacement_leaves_learned_storage_untouched() -> None:
    merged = replace_service_actions_with_teacher(
        action_names=[
            [
                "electrical_storage",
                "electric_vehicle_storage_charger_1_1",
                "deferrable_appliance_deferrable_appliance_1",
            ]
        ],
        proposed_actions=[[-0.4, -0.8, 0.0]],
        teacher_actions=[[0.9, 0.35, 1.0]],
    )

    assert merged == [[-0.4, 0.35, 1.0]]


def test_teacher_service_change_restores_service_and_idles_storage() -> None:
    executed = preserve_teacher_service_with_storage_fallback(
        action_names=[
            "electrical_storage",
            "electric_vehicle_storage_charger_15_1",
            "electric_vehicle_storage_charger_15_2",
        ],
        teacher_merged_actions=[0.8, 0.5, 0.4],
        projected_actions=[0.2, 0.3, 0.4],
    )

    assert executed == [0.0, 0.5, 0.4]
