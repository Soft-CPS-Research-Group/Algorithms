"""Pre-command energy exceedance at the site meter, phases, and declared groups."""

from __future__ import annotations

import math
import re

from algorithms.ti_marl.learning.operating_reward import proposal_exceedance_kwh

from .reservation_inputs import physical_reading


def prelimit_exceedance_kwh(snapshot, bundle, *, step_seconds, declared):
    """Evaluate the raw proposal on the measured electrical state (§7.3.3).

    Replace the previously applied controllable powers at each meter with the
    new proposal. Running programmes remain in the baseline until they finish.
    Return None for missing required meter/rating/phase membership instead of
    assuming a feasible proposal.
    """
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Invalid physical duration for electrical exceedance")
    site = bundle.agent_id
    groups = {g.group_id: g for g in snapshot.groups_for(site)}
    decisions = {d.group_id: d for d in bundle.decisions}
    if decisions.keys() != groups.keys() or not declared:
        return None
    readings = {
        (p.sensor_id, p.observation_id): p.values[0]
        for p in snapshot.parts_for(site)
        if physical_reading(p)
    }
    limits = [c for c in snapshot.constraints if c.owner_agent_id == site and c.active]
    changes, proposed = {}, {}
    for key, group in groups.items():
        if getattr(group, "output_unconfirmed", False):
            # A reservation uses worst-case watts here; an exact physical
            # reward requires downstream measurement of what happened.
            return None
        decision = decisions[key]
        port = next((p for p in group.ports if p.mode == decision.mode), None)
        if port is None or not math.isfinite(decision.fraction):
            return None
        if group.group_type == "deferrable":
            # Only a new START changes the meter; an already-running cycle
            # appears in the measured uncontrolled background for this check.
            value = group.activation_power_kw if decision.mode == "START" else 0.0
            if decision.mode == "START" and value <= 0.0:
                return None
            previous = 0.0
        else:
            previous = readings.get((group.module_id, "last_applied_power_kw"))
            if previous is None:
                return None
            rating = (
                group.max_charge_power_kw
                if decision.mode.startswith("CHARGE_")
                else -group.max_discharge_power_kw
                if decision.mode.startswith("DISCHARGE_")
                else 0.0
            )
            value = rating * port.upper_bound * decision.fraction
        proposed[key] = value
        changes[key] = value - previous
    site_excess = 0.0
    for side, sign in (("import", 1.0), ("export", -1.0)):
        declaration = declared.get("grid_" + side)
        if declaration is None:
            continue
        limit = declaration.get("max")
        net = readings.get(("self", "net_power_kw"))
        if limit is None or net is None:
            return None
        site_excess = max(site_excess, sign * (net + sum(changes.values())) - limit)
    if not any(k in declared for k in ("grid_import", "grid_export")):
        return None
    phase_excess = {}
    if "phase_import" in declared or "phase_export" in declared:
        # A charger can disconnect at this instant while its previous power
        # remains on the meter. Route that delta on its physical circuit.
        routing = {key: {} for key in groups}
        for constraint in limits:
            match = re.search(r"_phase_(L[123])_headroom_kw$", constraint.constraint_id)
            if match:
                for key, coefficient in constraint.member_group_coefficients:
                    routing[key][match[1]] = coefficient
        for key, delta in changes.items():
            weights = routing[key]
            if delta and not math.isclose(sum(weights.values()), 1.0):
                declared_wiring = dict(getattr(groups[key], "phase_weights", ()))
                if not math.isclose(sum(declared_wiring.values()), 1.0) or any(
                    not math.isclose(coefficient, declared_wiring.get(phase, 0.0))
                    for phase, coefficient in weights.items()
                ):
                    return None
                routing[key] = declared_wiring
        for side, sign in (("import", 1.0), ("export", -1.0)):
            ratings = declared.get("phase_" + side, {}).get("max", {})
            for phase, limit in ratings.items():
                net = readings.get(("self", f"charging_phase_{phase}_power_kw"))
                if net is None:
                    return None
                delta = sum(changes[k] * w.get(phase, 0.0) for k, w in routing.items())
                phase_excess[phase] = max(
                    phase_excess.get(phase, 0.0), sign * (net + delta) - limit
                )
    group_excess = 0.0
    for c in limits:
        if c.constraint_type == "group_import_limit_kw":
            if c.upper_bound is None:
                return None
            weights = dict(c.member_group_coefficients) or dict.fromkeys(c.member_group_ids, 1.0)
            group_excess = max(
                group_excess, sum(proposed[k] * v for k, v in weights.items()) - c.upper_bound
            )
    return proposal_exceedance_kwh(
        site_excess, tuple(phase_excess.values()), (group_excess,), step_seconds
    )
