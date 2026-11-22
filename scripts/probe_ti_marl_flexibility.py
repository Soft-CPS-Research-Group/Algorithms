"""Paired evaluation of a saved TI-MARL actor on one controlled EV service.

Read the actual encoder's observation declarations from a trace of the same
job. Simulate four 15-minute intervals, 07:00--08:00, with a 20 kWh battery
initially at 50%, a 4 kW unidirectional charger, unit efficiency, and no
uncontrollable demand. Move only one published window between the first and
second half hour. This diagnostic tests learned timing separately from
the service reservation; it is not a Core-30 community KPI.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from algorithms.configuration.agents import TIMARLHierarchyConfig
from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.contracts.enums import HealthState
from algorithms.ti_marl.contracts.models import (
    ActionGroupInstance,
    ActionPortInstance,
    LocalConstraint,
    ObservationPart,
)
from algorithms.ti_marl.learning.community_signal import CommunitySignal
from algorithms.ti_marl.runtime.forecasts import HORIZON, OperatingSnapshot, SiteOutlook

CAPACITY = 20.0
POWER = 4.0
DT = 0.25


class EVProbe:
    def __init__(self, trace_file):
        with gzip.open(trace_file, "rt") as stream:
            frame = next(
                line["payload"] for line in map(json.loads, stream) if line["kind"] == "snapshot"
            )
        group = next(
            group for group in frame["action_groups"] if group["group_type"] == "ev_session"
        )
        self.site = group["owner_agent_id"]
        self.sensor = group["module_id"]
        ports = tuple(
            ActionPortInstance(
                p["port_id"],
                p["mode"],
                p["target_entity_id"],
                p["action_name"],
                0.35 if p["mode"] == "CHARGE_EV" else 0.0,
                1.0 if p["mode"] == "CHARGE_EV" else 0.0,
                valid=p["mode"] in ("IDLE", "CHARGE_EV"),
            )
            for p in group["ports"]
        )
        self.group = ActionGroupInstance(
            group["group_id"],
            "ev_session",
            self.site,
            self.sensor,
            ports,
            max_charge_power_kw=POWER,
            adapter_target_entity_id=group["adapter_target_entity_id"],
            reservation_profile=(
                ("battery_efficiency", 1.0),
                ("hourly_loss", 0.0),
                ("ev_battery_rating_kw", POWER),
                ("ev_taper_start_soc", 0.99),
                ("ev_taper_at_full", 1.0),
            ),
        )
        self.limit = LocalConstraint(
            "probe_grid", self.site, "group_import_limit_kw", upper_bound=POWER
        )
        self.templates = [
            ObservationPart(**{**part, "health": HealthState.HEALTHY})
            for part in frame["observation_parts"]
            if part["owner_agent_id"] == self.site
            and part["sensor_id"] in ("self", self.sensor)
            and len(part["values"]) == 1
        ]

    @staticmethod
    def window(variable, first, t):
        start = 0 if first else 2
        active = start <= t < start + 2
        price = 0.08 if active and variable == "price" else 0.24
        solar = POWER if active and variable == "solar" else 0.0
        incentive = -0.05 if active and variable == "incentive" else 0.0
        return price, solar, incentive

    def snapshot(self, variable, first, t, stored, target, *, start_step=28, quarters=4):
        now = (start_step + t) * 900.0
        price, solar, _ = self.window(variable, first, t)
        readings = {
            "connected_state": 1.0,
            "connected_ev_soc": stored / CAPACITY,
            "connected_ev_battery_capacity_kwh": CAPACITY,
            "connected_ev_soc_min_ratio": 0.0,
            "connected_ev_battery_nominal_power_kw": POWER,
            "connected_ev_required_soc_departure": target,
            "hours_until_departure": (quarters - t) * DT,
            "charge_efficiency_at_max_ratio": 1.0,
            "charger_efficiency_ratio": 1.0,
            "min_charging_power_kw": 1.4,
            "max_charging_power_kw": POWER,
            "available_charge_power_kw": POWER,
            "available_charge_action_normalized": 1.0,
            "can_charge": 1.0,
            "can_discharge": 0.0,
            "available_discharge_action_normalized": 0.0,
            "electricity_pricing": price,
            "non_shiftable_load": 0.0,
            "pv_power_kw": solar,
        }
        parts = tuple(
            replace(
                part,
                values=(readings[part.observation_id],),
                valid=True,
                validity_reasons=(),
                age_seconds=0.0,
                estimated=False,
            )
            for part in self.templates
            if part.observation_id in readings
        )
        future = tuple(self.window(variable, first, t + i) for i in range(HORIZON))
        prices, pv, signals = map(tuple, zip(*future))
        curve = SiteOutlook(
            self.site,
            (prices, (0.0,) * HORIZON, pv),
            ((2,) * HORIZON, (1,) * HORIZON, (1,) * HORIZON),
        )
        return OperatingSnapshot(
            contract_version="ti_marl_v1",
            compiler_version="controlled_probe",
            topology_version=0,
            time_step=start_step + t,
            agent_ids=(self.site,),
            modules=(),
            entities=(),
            fault_evidence=(),
            health=(),
            observation_parts=parts,
            action_groups=(self.group,),
            dependencies=(),
            constraints=(self.limit,),
            shared_resources=(),
            timestamp_seconds=now,
            clock_seconds=now,
            outlooks=(curve,),
            incentive=signals,
        )

    @torch.no_grad()
    def episode(self, actor, variable, first, urgent, manager=None):
        if manager is not None:
            manager.reset()
        stored = 0.5 * CAPACITY
        # Keep the checkpoint's service tolerance. The protected minima are
        # 2 kWh over the next hour (flexible) or 4 kWh (urgent).
        needed = 4.0 if urgent else 2.0
        target = (stored + needed) / CAPACITY + actor.tolerance_soc
        delivered, billed, supervised = [], 0.0, 0
        signals, free_modes, charge_probabilities = [], [], []
        for t in range(4):
            snapshot = self.snapshot(variable, first, t, stored, target)
            if manager is not None:
                snapshot = replace(snapshot, measured_peak_kw=manager.peak.peak)
                snapshot = replace(snapshot, incentive=manager.curve_for(snapshot, training=False))
            signals.append(snapshot.incentive[0])
            evaluation = actor(snapshot, deterministic=True)
            supervised += int(evaluation.traces[self.site].supervised)
            decision = evaluation.executed[0].decisions[0]
            choices = evaluation.traces[self.site].choices
            mode = next((c for c in choices if c.stage == "mode"), None)
            free_modes.append(bool(mode and mode.sampled))
            # The probe is unidirectional: the only alternatives are IDLE and CHARGE.
            probability = math.exp(mode.old_log_prob) if mode else None
            if probability is not None and decision.mode == "IDLE":
                probability = 1.0 - probability
            charge_probabilities.append(probability)
            kwh = decision.fraction * POWER * DT if decision.mode == "CHARGE_EV" else 0.0
            stored += kwh
            price, solar, _ = self.window(variable, first, t)
            billed += price * max(0.0, kwh - solar * DT)
            if manager is not None:
                balance = kwh - solar * DT
                manager.peak.observe(
                    snapshot.clock_seconds, 900.0, max(0.0, balance), max(0.0, -balance)
                )
            delivered.append(kwh)
        return {
            "charge_by_quarter_kwh": delivered,
            "free_mode": free_modes,
            "p_charge": charge_probabilities,
            "applied_incentive": signals,
            "first_half_kwh": sum(delivered[:2]),
            "charge_kwh": sum(delivered),
            "real_bill_eur": billed,
            "departure_soc": stored / CAPACITY,
            "requested_shortfall_kwh": max(0.0, target * CAPACITY - stored),
            "protected_shortfall_kwh": max(0.0, needed - sum(delivered)),
            "supervised_decisions": supervised,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local checkpoint")
    parser.add_argument(
        "--trace", type=Path, required=True, help="Trace chunk containing a snapshot"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = TIMARL(saved["configuration"])
    probe = EVProbe(args.trace)
    agent.actor.eval()
    agent.actor.step_seconds = 900.0
    agent.actor.order_seed = 0
    manager = None
    extra = saved.get("extra_policy_state", {})
    if extra.get("training_phase") == "coordinator":
        hyper = saved["configuration"]["algorithm"]["hyperparameters"]
        settings = TIMARLHierarchyConfig.model_validate(hyper["hierarchy"]).model_dump()
        manager = CommunitySignal(
            settings,
            agent.actor.d_model,
            agent.device,
            agent.seed,
            hyper.get("gamma", 0.99),
            hyper.get("discount_timebase_seconds") or 3600.0,
        )
        manager.network.load_state_dict(extra["manager"])
    for label in ("initial", "trained"):
        if label == "trained":
            agent.actor.load_state_dict(saved["actor"])
        drives = [("injected", None)]
        if label == "trained" and manager is not None:
            drives.append(("coordinator", manager))
        for drive, coordinator in drives:
            for urgent in (False, True):
                variables = (
                    ("price", "solar", "incentive") if coordinator is None else ("price", "solar")
                )
                for variable in variables:
                    early = probe.episode(agent.actor, variable, True, urgent, coordinator)
                    late = probe.episode(agent.actor, variable, False, urgent, coordinator)
                    print(
                        json.dumps(
                            {
                                "weights": label,
                                "signal_source": drive,
                                "urgent": urgent,
                                "variable": variable,
                                "shift_first_half_kwh": early["first_half_kwh"]
                                - late["first_half_kwh"],
                                "early": early,
                                "late": late,
                            },
                            allow_nan=False,
                        )
                    )


if __name__ == "__main__":
    main()
