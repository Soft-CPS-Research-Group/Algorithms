"""Authorised post-action physical evidence for the SoftCPS 2.1 adapter.

No values collected here enter observations. Unknowns are omitted so a strict
reward cannot turn a skipped, corrupted or uninstrumented outcome into zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def merge_reward_evidence(reported, measured):
    """Keep published interval facts; overlapping physical measurements must agree."""
    if not isinstance(reported, Mapping) or not isinstance(measured, Mapping):
        raise ValueError("Expected mappings of physical reward terms")
    combined = dict(reported)
    for key, value in measured.items():
        if value is None:
            continue
        if combined.get(key) is None:
            combined[key] = value
        elif not math.isclose(float(combined[key]), float(value), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Native and replayed reward evidence disagree for {key}")
    return combined


def _settled_cost(env, step):
    history = getattr(env, "_community_market_settlement_history", None)
    if not history or step >= len(history):
        return None
    rows = history[step]
    if not rows or any(row.get("settled_cost_eur") is None for row in rows):
        return None
    return math.fsum(row["settled_cost_eur"] for row in rows)


def _demand_response(env, step):
    service = getattr(env, "_demand_response_service", None)
    if service is None:
        return None
    if not service.enabled:
        return 0.0
    # Settlement runs on the action interval before the simulator increments t.
    row = service._settlement_by_time_step.get(step)
    if row is not None:
        if (
            row.get("baseline_valid") is not True
            or row.get("penalty_eur") is None
            or row.get("revenue_eur") is None
        ):
            return None
        return float(row["penalty_eur"]) - float(row["revenue_eur"])
    absolute_step = env.episode_tracker.episode_start_time_step + step
    if any(r.start_time_step <= absolute_step <= r.end_time_step for r in service.requests):
        return None
    return 0.0


def _fixed_service(env, step):
    audit = getattr(env, "_kpi_service", None)
    if audit is None or not hasattr(audit, "_service_energy"):
        return None
    deficits = []
    for building in audit._get_kpi_buildings():
        # The native audit selects completed active rows and converts durations.
        # Columns are cooling, heating, DHW and base load; no EV or battery load.
        expected, served = audit._service_energy(building, step, step)
        if not np.isfinite(expected).all() or not np.isfinite(served).all():
            return None
        if (expected < 0).any() or (served < 0).any():
            return None
        deficits.append(float(np.maximum(expected - served, 0.0).sum()))
    return math.fsum(deficits)


class SoftCPSRewardFeedback:
    """Resolve one-step simulator ledger increments, resetting at each episode."""

    def __init__(self, *, measure_ev_excess=False, measure_terminal_stock=False):
        self.measure_terminal_stock = measure_terminal_stock
        self.measure_ev_excess = measure_ev_excess
        self._programmes = {}
        self._ev_recognised = {}
        self._previous_step = None
        self._previous_terms = {}

    def transition_info(self, env, step, info):
        reported = info.get("thesis_reward_terms", {})
        return {"thesis_reward_terms": merge_reward_evidence(reported, self(env, step))}

    def __call__(self, env, step):
        if int(env.time_step) != step + 1:
            raise ValueError("Operating reward needs the just-completed simulator step")
        if step != 0 and self._previous_step == step:
            return self._previous_terms.copy()
        if step == 0:
            self._programmes.clear()
            self._ev_recognised.clear()
        elif self._previous_step != step - 1:
            raise ValueError("Operating reward missed a physical transition")

        evidence = {
            "energy_cost_eur": _settled_cost(env, step),
            "dr_cost_eur": _demand_response(env, step),
            "fixed_deficit_kwh": _fixed_service(env, step),
            "programme_deficit_kwh": self._programme_deficit(env),
        }
        if self.measure_terminal_stock:
            evidence["stationary_terminal_deficit_kwh"] = self._terminal_stock_deficit(env, step)
        evidence.update(self._ev_deficits(env, step))
        terms = {name: value for name, value in evidence.items() if value is not None}
        self._previous_step, self._previous_terms = step, terms
        return terms.copy()

    @staticmethod
    def _terminal_stock_deficit(env, step):
        """Stored kWh needed to restore initial inventory in a static, finite episode."""
        if getattr(env, "topology_mode", None) != "static":
            return None
        if not env.terminated and not env.truncated:
            return 0.
        missing = []
        for building in env.buildings:
            battery = building.electrical_storage
            if battery is None or battery.nominal_power == 0:
                continue
            initial = float(battery.initial_soc) * float(battery.capacity)
            final = float(battery.soc[step]) * float(battery.capacity)
            if not (math.isfinite(initial) and math.isfinite(final)):
                return None
            missing.append(max(0., initial-final))
        return math.fsum(missing)

    def _ev_deficits(self, env, step):
        """Charge each protected kWh once, including at a recorded departure.

        A positive shortfall after granting all remaining intervals at the
        charger's absolute rating and perfect efficiency is already physically
        unavoidable. A missing forecast or tighter forecast headroom cannot
        create such a certificate. At departure the measured battery state
        resolves the remaining shortfall and the separate tolerance band.
        """
        audit = getattr(env, "_kpi_service", None)
        if getattr(env, "topology_mode", None) != "static" or audit is None:
            return {}
        due, tolerance_band, excess = 0.0, 0.0, 0.0
        recognised = self._ev_recognised.copy()
        for building in env.buildings:
            for charger in building.electric_vehicle_chargers:
                schedule = charger.charger_simulation
                states = schedule.electric_vehicle_charger_state
                if states[step] != 1:
                    continue
                ev = charger.past_connected_evs[step]
                sid = audit._normal_ev_id(schedule.electric_vehicle_session_id[step])
                target = audit._normalize_soc_target(
                    schedule.electric_vehicle_required_soc_departure[step]
                )
                if ev is None or sid is None or target is None:
                    return {}
                if ev.battery.soc[step] is None or ev.battery.capacity is None:
                    return {}
                soc = float(ev.battery.soc[step])
                capacity = float(ev.battery.capacity)
                if (
                    not math.isfinite(soc)
                    or not 0.0 <= soc <= 1.0
                    or not capacity > 0.0
                    or not math.isfinite(capacity)
                ):
                    return {}
                if self.measure_ev_excess:
                    # The AC charger reading includes losses. The battery terminal
                    # ledger measures energy actually accepted in this interval.
                    charge = ev.battery.gross_energy_charge
                    if charge is None:
                        return {}
                    accepted_kwh = float(charge[step]) * float(ev.battery.round_trip_efficiency)
                    if not math.isfinite(accepted_kwh) or accepted_kwh < 0:
                        return {}
                    excess += min(accepted_kwh, max(0., (soc - target) * capacity))
                requested_kwh = max(target - soc, 0.0) * capacity
                accepted = max(target - audit._ev_departure_service_tolerance(), 0.0)
                protected_kwh = max(accepted - soc, 0.0) * capacity
                # Current and next connection identities are both physically
                # known after env.step; a truncated episode is not a departure.
                remaining_steps = schedule.electric_vehicle_departure_time[step]
                countdown = float(remaining_steps) if remaining_steps is not None else math.nan
                if step + 1 == len(states):
                    # Same terminal schedule rule as the simulator departure audit.
                    departing = math.isfinite(countdown) and 0 <= countdown <= 1
                elif states[step + 1] != 1:
                    next_state = states[step + 1]
                    if next_state is None or not math.isfinite(next_state):
                        return {}
                    departing = True
                else:
                    next_sid = audit._normal_ev_id(schedule.electric_vehicle_session_id[step + 1])
                    current_ev = audit._normal_ev_id(schedule.electric_vehicle_id[step])
                    next_ev = audit._normal_ev_id(schedule.electric_vehicle_id[step + 1])
                    if next_sid is None or current_ev is None or next_ev is None:
                        return {}
                    departing = next_sid != sid or next_ev != current_ev
                key = (building.name, charger.charger_id, sid)
                previous = recognised.get(key, 0.0)
                if departing:
                    irrevocable = protected_kwh
                    tolerance_band += requested_kwh - protected_kwh
                    recognised.pop(key, None)
                else:
                    if charger.max_charging_power is None:
                        return {}
                    rating = float(charger.max_charging_power)
                    if (
                        not math.isfinite(rating)
                        or rating < 0
                        or not math.isfinite(countdown)
                        or countdown < 1
                    ):
                        return {}
                    optimistic_kwh = (
                        (countdown - 1.0) * float(env.seconds_per_time_step) / 3600.0 * rating
                    )
                    irrevocable = max(0.0, protected_kwh - optimistic_kwh)
                    recognised[key] = max(previous, irrevocable)
                due += max(0.0, irrevocable - previous)
        self._ev_recognised = recognised
        terms = {"ev_deficit_kwh": due, "ev_tolerance_kwh": tolerance_band}
        if self.measure_ev_excess:
            terms["ev_excess_kwh"] = excess
        return terms

    def _programme_deficit(self, env):
        # Dynamic exits/reinstallations need the cross-instance service ledger
        # to tell confirmed cancellation from a loss. Do not infer coverage.
        if getattr(env, "topology_mode", None) != "static":
            return None
        missing = 0.0
        for building in env.buildings:
            for appliance in building.deferrable_appliances:
                ledger = getattr(appliance, "_service_ledger", None)
                if ledger is None or ledger.last_step is None:
                    return None
                # Index only once per episode. In operation look at last-start,
                # deadline and just-finished intervals, never scan an annual
                # calendar in full for each high-frequency control decision.
                if appliance not in self._programmes:
                    calendar = appliance.deferrable_appliance_simulation.flexibility_schedule
                    by_step = defaultdict(set)
                    by_id = {}
                    for cycle in calendar:
                        cid = cycle["cycle_id"]
                        by_id[cid] = cycle
                        by_step[int(cycle["latest_start_time_step"])].add(cid)
                        by_step[int(cycle["deadline_time_step"])].add(cid)
                    self._programmes[appliance] = by_step, by_id, set(), set()
                by_step, by_id, known_finished, charged = self._programmes[appliance]
                finished = ledger.finished - known_finished
                known_finished.update(finished)
                due = by_step.get(ledger.last_step, set()).union(finished) - charged
                for row in ledger.records([by_id[cid] for cid in due]):
                    if row["included"] and row["state"] == "missed":
                        missing += float(row["final_unserved_energy_kwh"])
                        charged.add(row["cycle_id"])
        return missing
