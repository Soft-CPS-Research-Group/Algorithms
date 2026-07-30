"""Translate CityLearn entity observations into local safety constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from algorithms.utils.local_action_safety import (
    ActionConstraint,
    ActionKind,
    ActionSpec,
    DeferrableConstraint,
    ElectricalHeadroom,
    EVConstraint,
    LocalActionProjectionRequest,
    LocalActionSafetyProjector,
    ProjectionResult,
    StorageConstraint,
)


_PHASES = ("L1", "L2", "L3")


def replace_service_actions_with_teacher(
    *,
    action_names: Sequence[Sequence[str]],
    proposed_actions: Sequence[Sequence[float]],
    teacher_actions: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Keep learned storage control while delegating hard-gated services."""

    if not (
        len(action_names) == len(proposed_actions) == len(teacher_actions)
    ):
        raise ValueError("Teacher service replacement requires aligned agent groups.")
    merged: list[list[float]] = []
    for names, proposed, teacher in zip(action_names, proposed_actions, teacher_actions):
        if not (len(names) == len(proposed) == len(teacher)):
            raise ValueError("Teacher service replacement requires aligned action vectors.")
        values: list[float] = []
        for name, proposed_value, teacher_value in zip(names, proposed, teacher):
            is_service = str(name).startswith(("electric_vehicle", "deferrable_appliance"))
            values.append(float(teacher_value if is_service else proposed_value))
        merged.append(values)
    return merged


def preserve_teacher_service_with_storage_fallback(
    *,
    action_names: Sequence[str],
    teacher_merged_actions: Sequence[float],
    projected_actions: Sequence[float],
) -> list[float]:
    """Never trade a teacher service command for learned storage dispatch.

    If the approximate local projector changes any EV/deferrable command, all
    stationary-storage control for that building is dropped for the step and
    the service command is restored.  This conservative fallback prevents a
    modelled headroom mismatch from causing a hard service-gate failure.
    """

    if not (
        len(action_names) == len(teacher_merged_actions) == len(projected_actions)
    ):
        raise ValueError("Service preservation requires aligned action vectors.")
    service_changed = any(
        str(name).startswith(("electric_vehicle", "deferrable_appliance"))
        and not np.isclose(float(proposed), float(projected), atol=1.0e-9, rtol=0.0)
        for name, proposed, projected in zip(
            action_names, teacher_merged_actions, projected_actions
        )
    )
    if not service_changed:
        return [float(value) for value in projected_actions]
    return [
        (
            0.0
            if str(name) == "electrical_storage"
            else float(proposed)
            if str(name).startswith(("electric_vehicle", "deferrable_appliance"))
            else float(projected)
        )
        for name, proposed, projected in zip(
            action_names, teacher_merged_actions, projected_actions
        )
    ]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _optional_non_negative(values: Mapping[str, float], key: str) -> float | None:
    if key not in values:
        return None
    return max(_finite(values[key]), 0.0)


@dataclass(frozen=True)
class CityLearnSafetyConfig:
    """Runtime policy for the CityLearn adapter."""

    fail_on_infeasible: bool = False
    protect_ev_minimum: bool = True
    ev_minimum_mode: str = "average"
    protect_ev_service_target: bool = False
    protect_deferrable_must_start: bool = True
    allow_discretionary_deferrable_start: bool = False
    headroom_reserve_kw: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.headroom_reserve_kw) or self.headroom_reserve_kw < 0.0:
            raise ValueError("headroom_reserve_kw must be finite and >= 0.")
        if self.ev_minimum_mode not in {"average", "deadline_feasible"}:
            raise ValueError(
                "ev_minimum_mode must be 'average' or 'deadline_feasible'."
            )


class CityLearnLocalSafetyAdapter:
    """Build and apply a one-building projection from raw entity features.

    The adapter intentionally consumes the *raw* entity vector. Encoded or
    normalized actor inputs are unsuitable for physical kW/SOC constraints.
    """

    def __init__(
        self,
        *,
        observation_names: Sequence[str],
        action_names: Sequence[str],
        action_low: Sequence[float],
        action_high: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        config: CityLearnSafetyConfig | None = None,
    ) -> None:
        self.observation_names = tuple(str(name) for name in observation_names)
        self.action_names = tuple(str(name) for name in action_names)
        self.action_low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        self.metadata = dict(metadata or {})
        self.config = config or CityLearnSafetyConfig()
        seconds_per_time_step = _finite(
            self.metadata.get("seconds_per_time_step"), 3600.0
        )
        self.step_hours = max(seconds_per_time_step / 3600.0, 1.0e-9)
        self.projector = LocalActionSafetyProjector()
        # Entity mode emits zero-filled electrical-service columns even for
        # buildings that have no configured service limit.  A positive local
        # total/phase headroom observation proves that the constraint exists;
        # after that, a later zero is a real saturated board and must be kept.
        self._electrical_headroom_observed = False

        expected = len(self.action_names)
        if self.action_low.size != expected or self.action_high.size != expected:
            raise ValueError("CityLearn safety action names and bounds must have equal length.")
        if np.any(~np.isfinite(self.action_low)) or np.any(~np.isfinite(self.action_high)):
            raise ValueError("CityLearn safety action bounds must be finite.")
        if np.any(self.action_high < self.action_low):
            raise ValueError("CityLearn safety action high bounds must be >= low bounds.")

        building_names = self.metadata.get("building_names")
        self.building_id = (
            str(building_names[0])
            if isinstance(building_names, (list, tuple)) and building_names
            else self._infer_building_id()
        )

    def _infer_building_id(self) -> str:
        for name in self.observation_names:
            if "::Building_" not in name:
                continue
            return name.split("::", 1)[1].split("/", 1)[0]
        return "unknown_building"

    def project(
        self,
        raw_observation: Sequence[float],
        proposed_actions: Sequence[float],
    ) -> ProjectionResult:
        values_array = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
        if values_array.size != len(self.observation_names):
            raise ValueError(
                "Raw observation length does not match the safety observation-name contract: "
                f"{values_array.size} != {len(self.observation_names)}."
            )
        proposed = np.asarray(proposed_actions, dtype=np.float64).reshape(-1)
        if proposed.size != len(self.action_names):
            raise ValueError(
                "Proposed action length does not match the safety action-name contract: "
                f"{proposed.size} != {len(self.action_names)}."
            )

        values = {
            name: _finite(value)
            for name, value in zip(self.observation_names, values_array)
        }
        specs: list[ActionSpec] = []
        constraints: list[ActionConstraint] = []
        for index, action_name in enumerate(self.action_names):
            spec, constraint = self._action_contract(
                action_name,
                low=float(self.action_low[index]),
                high=float(self.action_high[index]),
                values=values,
            )
            specs.append(spec)
            constraints.append(constraint)

        result = self.projector.project(
            LocalActionProjectionRequest(
                building_id=self.building_id,
                proposed_actions=tuple(float(value) for value in proposed),
                action_specs=tuple(specs),
                action_constraints=tuple(constraints),
                headroom=self._headroom(values),
            )
        )
        if self.config.fail_on_infeasible and not result.feasible:
            codes = ", ".join(reason.code.value for reason in result.infeasible_reasons)
            raise RuntimeError(
                f"Local safety projection is infeasible for {self.building_id}: {codes}."
            )
        return result

    def _action_contract(
        self,
        action_name: str,
        *,
        low: float,
        high: float,
        values: Mapping[str, float],
    ) -> tuple[ActionSpec, ActionConstraint]:
        if action_name == "electrical_storage":
            prefix = f"storage::{self.building_id}/electrical_storage::"
            nominal_power = max(self._value(values, prefix, "nominal_power_kw"), 0.0)
            phases = self._phases(values, prefix)
            spec = ActionSpec(
                action_name,
                ActionKind.STORAGE,
                low=low,
                high=high,
                import_kw_per_action=nominal_power,
                export_kw_per_action=nominal_power,
                phases=phases,
                priority=0,
            )
            constraint = ActionConstraint(
                storage=StorageConstraint(
                    soc=self._value(values, prefix, "soc", fallback="electrical_storage_soc"),
                    min_soc=self._value(values, prefix, "soc_min_ratio", default=0.0),
                    max_soc=1.0,
                    available_charge_action=self._optional_value(
                        values, prefix, "available_charge_action_normalized"
                    ),
                    available_discharge_action=self._optional_value(
                        values, prefix, "available_discharge_action_normalized"
                    ),
                )
            )
            return spec, constraint

        if action_name.startswith("electric_vehicle_storage_"):
            charger_id = action_name.removeprefix("electric_vehicle_storage_")
            prefix = f"charger::{self.building_id}/{charger_id}::"
            max_charge = max(self._value(values, prefix, "max_charging_power_kw"), 0.0)
            max_discharge = max(self._value(values, prefix, "max_discharging_power_kw"), 0.0)
            min_charge = max(self._value(values, prefix, "min_charging_power_kw"), 0.0)
            minimum = self._ev_minimum_action(
                values=values,
                prefix=prefix,
                max_charge_kw=max_charge,
            )
            if minimum > 0.0 and max_charge > 0.0:
                # CityLearn's charger has a true zero-output deadband below
                # min_charging_power; an urgent request must cross it.
                minimum = max(minimum, min_charge / max_charge)
            spec = ActionSpec(
                action_name,
                ActionKind.EV,
                low=low,
                high=high,
                import_kw_per_action=max_charge,
                export_kw_per_action=max_discharge,
                phases=self._phases(values, prefix),
                priority=200,
            )
            constraint = ActionConstraint(
                ev=EVConstraint(
                    connected=self._value(values, prefix, "connected_state") > 0.5,
                    min_required_action=minimum,
                    available_charge_action=self._ev_available_charge_action(
                        values=values,
                        prefix=prefix,
                        max_charge_kw=max_charge,
                    ),
                    available_discharge_action=self._optional_value(
                        values, prefix, "available_discharge_action_normalized"
                    ),
                )
            )
            return spec, constraint

        if action_name.startswith("deferrable_appliance_"):
            appliance_id = action_name.removeprefix("deferrable_appliance_")
            prefix = f"deferrable_appliance::{self.building_id}/{appliance_id}::"
            must_start = (
                self._value(values, prefix, "must_start_now") > 0.5
                if self.config.protect_deferrable_must_start
                else False
            )
            pending = self._value(values, prefix, "pending") > 0.5
            running = self._value(values, prefix, "running") > 0.5
            can_start = self._value(values, prefix, "can_start") > 0.5
            if not self.config.allow_discretionary_deferrable_start and not must_start:
                can_start = False
            spec = ActionSpec(
                action_name,
                ActionKind.DEFERRABLE,
                low=low,
                high=high,
                import_kw_per_action=max(self._value(values, prefix, "start_power_kw"), 0.0),
                phases=self._phases(values, prefix),
                priority=300,
            )
            constraint = ActionConstraint(
                deferrable=DeferrableConstraint(
                    pending=pending,
                    running=running,
                    can_start=can_start,
                    must_start=must_start,
                    start_action=1.0,
                )
            )
            return spec, constraint

        return (
            ActionSpec(action_name, ActionKind.OTHER, low=low, high=high),
            ActionConstraint(),
        )

    def _ev_minimum_action(
        self,
        *,
        values: Mapping[str, float],
        prefix: str,
        max_charge_kw: float,
    ) -> float:
        if not self.config.protect_ev_minimum or max_charge_kw <= 0.0:
            return 0.0
        average = max(
            self._value(values, prefix, "min_required_action_normalized"),
            0.0,
        )
        if self.config.ev_minimum_mode == "average":
            return average

        margin_key = f"{prefix}departure_energy_margin_kwh"
        available_key = f"{prefix}available_charge_power_kw"
        if margin_key not in values or available_key not in values:
            return average
        margin_kwh = _finite(values[margin_key], 0.0)
        available_charge_kw = max(_finite(values[available_key], 0.0), 0.0)
        efficiency = max(
            self._value(values, prefix, "charger_efficiency_ratio", default=1.0),
            1.0e-6,
        )
        deliverable_this_step_kwh = (
            available_charge_kw * self.step_hours * efficiency
        )
        energy_that_cannot_be_deferred_kwh = max(
            deliverable_this_step_kwh - margin_kwh,
            0.0,
        )
        required_grid_power_kw = energy_that_cannot_be_deferred_kwh / (
            efficiency * self.step_hours
        )
        return float(np.clip(required_grid_power_kw / max_charge_kw, 0.0, 1.0))

    def _ev_available_charge_action(
        self,
        *,
        values: Mapping[str, float],
        prefix: str,
        max_charge_kw: float,
    ) -> float | None:
        physical = self._optional_value(
            values, prefix, "available_charge_action_normalized"
        )
        if not self.config.protect_ev_service_target or max_charge_kw <= 0.0:
            return physical
        target_key = f"{prefix}energy_to_required_soc_kwh"
        if target_key not in values:
            return physical
        efficiency = max(
            self._value(values, prefix, "charger_efficiency_ratio", default=1.0),
            1.0e-6,
        )
        energy_to_target_kwh = max(_finite(values[target_key], 0.0), 0.0)
        target_limited = float(
            np.clip(
                energy_to_target_kwh
                / (efficiency * self.step_hours * max_charge_kw),
                0.0,
                1.0,
            )
        )
        return target_limited if physical is None else min(physical, target_limited)

    def _headroom(self, values: Mapping[str, float]) -> ElectricalHeadroom:
        import_total = self._first_optional(
            values,
            "charging_building_headroom_kw",
            "building_import_headroom_kw",
        )
        export_total = self._first_optional(
            values,
            "charging_building_export_headroom_kw",
            "building_export_headroom_kw",
        )
        phase_import = {
            phase: value
            for phase in _PHASES
            if (
                value := self._first_optional(
                    values,
                    f"charging_phase_{phase}_headroom_kw",
                )
            ) is not None
        }
        phase_export = {
            phase: value
            for phase in _PHASES
            if (
                value := self._first_optional(
                    values,
                    f"charging_phase_{phase}_export_headroom_kw",
                )
            ) is not None
        }
        observed_values = [
            value
            for value in (import_total, export_total, *phase_import.values(), *phase_export.values())
            if value is not None
        ]
        if any(value > 0.0 for value in observed_values):
            self._electrical_headroom_observed = True
        if not self._electrical_headroom_observed:
            # Zero-filled columns mean "not configured" until this adapter has
            # observed a positive electrical-service envelope for the building.
            return ElectricalHeadroom()
        reserve = float(self.config.headroom_reserve_kw)

        def reserved(value: float | None) -> float | None:
            return None if value is None else max(0.0, value - reserve)

        return ElectricalHeadroom(
            total_import_kw=reserved(import_total),
            total_export_kw=reserved(export_total),
            phase_import_kw={phase: reserved(value) for phase, value in phase_import.items()},
            phase_export_kw={phase: reserved(value) for phase, value in phase_export.items()},
        )

    @staticmethod
    def _first_optional(values: Mapping[str, float], *keys: str) -> float | None:
        for key in keys:
            value = _optional_non_negative(values, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _value(
        values: Mapping[str, float],
        prefix: str,
        suffix: str,
        *,
        default: float = 0.0,
        fallback: str | None = None,
    ) -> float:
        key = f"{prefix}{suffix}"
        if key in values:
            return _finite(values[key], default)
        if fallback is not None and fallback in values:
            return _finite(values[fallback], default)
        return default

    @staticmethod
    def _optional_value(
        values: Mapping[str, float], prefix: str, suffix: str
    ) -> float | None:
        key = f"{prefix}{suffix}"
        return _optional_non_negative(values, key)

    @staticmethod
    def _phases(values: Mapping[str, float], prefix: str) -> tuple[str, ...]:
        return tuple(
            phase
            for phase in _PHASES
            if _finite(values.get(f"{prefix}phase_connection_{phase}"), 0.0) > 0.5
        )
