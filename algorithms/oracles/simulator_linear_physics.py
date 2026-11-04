"""Simulator-aligned linear physics for explicitly supported static datasets.

This adapter never substitutes a worst-case constant for a nonlinear efficiency
curve. Constant efficiency, zero degradation and concave power/SOC curves are
required. Charger deadbands constrain *commands*: the scheduling model retains
the command deadband with disjunctive capacity/power saturation branches.
A candidate must always pass an independent simulator replay, including the
simulator's building safety allocation and floating-point state transitions.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from algorithms.oracles.total_energy_milp import TotalEnergyProblem


def constant_efficiency(name: str, attributes: Mapping[str, Any], *, battery: bool) -> float:
    if battery:
        if float(attributes.get("capacity_loss_coefficient", -1)) != 0:
            raise ValueError(
                f"{name}: simulator_linear requires explicit zero capacity degradation."
            )
        curves = [attributes.get("power_efficiency_curve")]
    else:
        curves = [
            attributes.get(k) for k in ("charge_efficiency_curve", "discharge_efficiency_curve")
        ]
    values = []
    for curve in curves:
        if curve is None:
            if battery:
                raise ValueError(f"{name}: an explicit constant efficiency curve is required.")
            values.append(float(attributes.get("efficiency", 1.0)))
            continue
        array = np.asarray(curve, dtype=float)
        if (
            array.ndim != 2
            or array.shape[1] != 2
            or len(array) < 2
            or not np.isfinite(array).all()
            or np.any(np.diff(array[:, 0]) <= 0)
            or not np.all(array[:, 1] == array[0, 1])
        ):
            raise ValueError(f"{name}: simulator_linear requires constant efficiency curves.")
        values.append(float(array[0, 1]))
    if len(set(values)) != 1 or not 0 < values[0] <= 1:
        raise ValueError(f"{name}: efficiency must be constant, symmetric and in (0, 1].")
    return math.sqrt(values[0]) if battery else values[0]


def power_envelope(name: str, attributes: Mapping[str, Any], *, ac_factor: float = 1.0):
    """Convert a concave SOC/power curve to exact affine AC-power ceilings."""
    curve = np.asarray(attributes.get("capacity_power_curve"), dtype=float)
    if (
        curve.ndim != 2
        or curve.shape[1] != 2
        or len(curve) < 2
        or not np.isfinite(curve).all()
        or np.any(np.diff(curve[:, 0]) <= 0)
        or curve[0, 0] != 0
        or curve[-1, 0] != 1
        or np.any(curve[:, 1] < 0)
    ):
        raise ValueError(f"{name}: an explicit power curve spanning SOC [0, 1] is required.")
    slopes = np.diff(curve[:, 1]) / np.diff(curve[:, 0])
    if np.any(np.diff(slopes) > 1e-10):
        raise ValueError(f"{name}: simulator_linear requires a concave power/SOC curve.")
    nominal = float(attributes["nominal_power"]) * ac_factor
    capacity = float(attributes["capacity"])
    return tuple(
        (float(m * nominal / capacity), float((y - m * x) * nominal))
        for m, (x, y) in zip(slopes, curve[:-1])
    )


def align_simulator_linear_problem(problem: TotalEnergyProblem, schema: Mapping[str, Any]):
    """Remove legacy safety margins and model the supported physical equations."""
    if problem.settlement == "community":
        market = schema.get("community_market") or {}
        if not market.get("enabled") or float(market.get("grid_export_price", 0)) != 0:
            raise ValueError(
                "simulator_linear community reference requires net settlement and zero grid export price."
            )
    storage = []
    for asset in problem.stationary_storage:
        definition = schema["buildings"][asset.building_id]["electrical_storage"]
        if (
            definition.get("type", "citylearn.energy_model.Battery")
            != "citylearn.energy_model.Battery"
        ):
            raise ValueError("simulator_linear does not support custom battery transitions.")
        attributes = definition["attributes"]
        efficiency = constant_efficiency(asset.building_id, attributes, battery=True)
        envelope = power_envelope(asset.building_id, attributes)
        max_power = float(attributes["nominal_power"]) * max(
            float(y) for _, y in attributes["capacity_power_curve"]
        )
        storage.append(
            replace(
                asset,
                capacity_kwh=float(attributes["capacity"]),
                initial_energy_kwh=float(attributes["capacity"])
                * float(
                    attributes.get(
                        "initial_soc", 1 - float(attributes.get("depth_of_discharge", 1))
                    )
                ),
                charge_efficiency=efficiency,
                discharge_efficiency=efficiency,
                max_charge_kw=max_power,
                max_discharge_kw=max_power,
                charge_power_envelope=envelope,
                discharge_power_envelope=envelope,
                reserve_is_discharge_only=True,
                loss_coefficient=float(attributes["loss_coefficient"]) * problem.timestep_hours,
            )
        )
    identities = problem.metadata["ev_identity_by_session"]
    sessions = []
    for session in problem.ev_sessions:
        ev_id = identities[session.session_id]
        battery = schema["electric_vehicles_def"][ev_id]["battery"]["attributes"]
        if (
            schema["electric_vehicles_def"][ev_id]["battery"].get(
                "type", "citylearn.energy_model.Battery"
            )
            != "citylearn.energy_model.Battery"
        ):
            raise ValueError("simulator_linear does not support custom EV battery transitions.")
        charger_id = session.action_name.removeprefix("electric_vehicle_storage_")
        charger = schema["buildings"][session.building_id]["chargers"][charger_id]
        attributes = charger["attributes"]
        if (
            charger.get("type", "citylearn.electric_vehicle_charger.Charger")
            != "citylearn.electric_vehicle_charger.Charger"
        ):
            raise ValueError("simulator_linear does not support custom charger transitions.")
        if float(charger.get("noise_std", 0)) != 0:
            raise ValueError("simulator_linear requires deterministic EV arrivals.")
        if (
            float(battery.get("loss_coefficient", -1)) != 0
            or float(battery.get("depth_of_discharge", 0)) != 1
            or float(attributes.get("max_discharging_power", -1)) != 0
        ):
            raise ValueError(
                "simulator_linear currently requires loss-free, charge-only EVs with zero reserve."
            )
        battery_eff = constant_efficiency(ev_id, battery, battery=True)
        charger_eff = constant_efficiency(charger_id, attributes, battery=False)
        envelope = power_envelope(ev_id, battery, ac_factor=1 / charger_eff)
        sessions.append(
            replace(
                session,
                charge_efficiency=battery_eff * charger_eff,
                discharge_efficiency=battery_eff * charger_eff,
                max_charge_kw=float(attributes["max_charging_power"]),
                charge_power_envelope=envelope,
                allow_charge_clipping=True,
            )
        )
    return replace(
        problem,
        stationary_storage=tuple(storage),
        ev_sessions=tuple(sessions),
        metadata={
            **problem.metadata,
            "physics": "simulator_linear",
            "stationary_model": "constant directional efficiency; exact capacity and concave power envelope; discharge-only reserve",
            "ev_model": "constant charger/battery efficiency; concave SOC taper; command deadband with capacity/power clipping",
            "delivered_power_clipping": True,
            "global_optimum_claim": False,
        },
    )


def charger_deadband_relaxation(problem: TotalEnergyProblem) -> TotalEnergyProblem:
    """Outer model includes actual charge truncated below a charger's deadband.

    It also allows controllable small charges away from saturation, so its
    objective/dual bound is a lower bound, not necessarily an executable plan.
    All other physical constraints and service/terminal requirements are kept.
    """
    if problem.metadata.get("physics") != "simulator_linear":
        raise ValueError("A simulator-aligned problem is required for this relaxation.")
    return replace(
        problem,
        ev_sessions=tuple(replace(s, min_charge_kw=0) for s in problem.ev_sessions),
        metadata={**problem.metadata, "charger_deadbands_relaxed": True},
    )
