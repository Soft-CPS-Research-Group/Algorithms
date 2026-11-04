"""Community cost plans from measured Simulator state, per step or per week."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from algorithms.agents.base_agent import BaseAgent
from algorithms.agents.total_oracle_replay_policy import TotalOracleReplayPolicy
from algorithms.configuration.milp import MILPStageConfig, WeeklyOracleHyperparameters
from algorithms.oracles.citylearn_contract import complete_idle_series
from algorithms.oracles.perfect_foresight_milp import SemanticSchedule, SolveOptions
from algorithms.oracles.rolling_energy import (
    BoundedDatasetForecast,
    update_measured_problem,
)
from algorithms.oracles.total_energy_milp import solve_total_energy_schedule


class MILPPolicy(BaseAgent):
    _use_raw_observations = True
    requires_final_pipeline_stage = True

    def __init__(self, config):
        super().__init__()
        self.settings = MILPStageConfig(
            algorithm="MILP",
            hyperparameters=(config.get("algorithm") or {}).get("hyperparameters") or {},
        ).hyperparameters
        self.weekly = isinstance(self.settings, WeeklyOracleHyperparameters)
        simulator = config["simulator"]
        if (config.get("training") or {}).get("episodes", 0) != 0:
            raise ValueError("MILP requires zero training episodes and an evaluation job.")
        if simulator.get("simulation_end_time_step") is None:
            raise ValueError("MILP requires an explicit finite simulation window.")
        self.forecast = BoundedDatasetForecast(
            simulator["dataset_path"],
            simulator.get("building_ids"),
            physics=self.settings.physics,
            weekly=self.weekly,
        )
        self.start = int(simulator.get("simulation_start_time_step") or 0)
        self.end = int(simulator["simulation_end_time_step"])
        steps = self.settings.horizon_hours / self.forecast.timestep_hours
        if not math.isclose(steps, round(steps)):
            raise ValueError("MILP horizon must contain an integer number of time steps.")
        self.horizon_steps = int(round(steps))
        self.options = SolveOptions(
            time_limit_seconds=self.settings.time_limit_seconds,
            mip_relative_gap=self.settings.mip_relative_gap,
            threads=self.settings.solver_threads,
            # A per-solve service allowance can accumulate after each measured
            # reset when a terminal EV target is physically unreachable.
            lexicographic_shortfall_tolerance_kwh=(
                0.0 if self.settings.physics == "simulator_linear" else 1e-3
            ),
        )
        if self.weekly:
            self.options = replace(self.options, throughput_tiebreaker_eur_per_kwh=0.0)
        initial = self.forecast.problem(self.start, self.start + 1)
        idle = SemanticSchedule(
            problem_id="milp-mpc-attachment",
            horizon=1,
            timestep_hours=initial.timestep_hours,
            series=complete_idle_series([], initial.metadata["action_inventory"], 1),
            metadata={},
        )
        self.converter = TotalOracleReplayPolicy({}, schedule=idle)
        self._attachment = None
        self._step = 0
        self._episode = 0
        self._started = {}
        self._rows = []
        self._decision_log = None
        self._plan_step = None
        self._planned_starts = {}

    def prepare_training(self, output_dir, context=None):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._decision_log = destination / "milp_decisions.jsonl"

    def attach_environment(self, **kwargs):
        metadata = kwargs.get("metadata") or {}
        if metadata.get("interface") != "entity" or metadata.get("topology_mode") != "static":
            raise ValueError("MILP currently requires the static entity interface.")
        if tuple(metadata.get("building_names") or ()) != tuple(
            self.forecast.building_ids
            or [
                bid
                for bid, b in self.forecast.schema["buildings"].items()
                if b.get("include", True)
            ]
        ):
            raise ValueError("MILP forecast and runtime building identities differ.")
        market = metadata.get("community_market") or {}
        if not market.get("enabled") or float(market.get("grid_export_price") or 0) != 0:
            raise ValueError(
                "MILP community cost requires an enabled market and zero grid export price."
            )
        self.converter.attach_environment(**kwargs)
        self._attachment = kwargs
        self._indices = [
            {str(k): i for i, k in enumerate(names)} for names in kwargs["observation_names"]
        ]
        self._owners = {bid: i for i, bid in enumerate(metadata["building_names"])}

    def on_episode_start(self, *, episode, training):
        if training:
            raise ValueError("MILP is an evaluation controller; configure zero training episodes.")
        self._step = 0
        self._episode = episode
        self._started.clear()
        self._plan_step = None
        self._planned_starts.clear()

    def set_episode_context(self, *, episode_step=None, next_episode_step=None):
        if episode_step is not None:
            self._step = int(episode_step)

    def _value(self, observations, bid, namespace, asset, *features):
        group = self._owners[bid]
        for feature in features:
            name = f"{namespace}::{bid}/{asset}::{feature}"
            index = self._indices[group].get(name)
            if index is not None:
                value = float(observations[group][index])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite observed state: {name}")
                return value
        raise ValueError(f"MILP requires observed {namespace}/{bid}/{asset}/{features}.")

    @staticmethod
    def _soc(value):
        # The entity contract publishes storage and connected-EV SOC as ratios.
        if not -1e-6 <= value <= 1 + 1e-6:
            raise ValueError("Observed SOC is outside [0, 1].")
        return float(np.clip(value, 0, 1))

    def _measured(self, problem, observations):
        storage, ev = {}, {}
        for asset in problem.stationary_storage:
            bid, name = asset.building_id, asset.action_name
            capacity = self._value(observations, bid, "storage", name, "capacity_kwh")
            soc = self._soc(self._value(observations, bid, "storage", name, "soc"))
            # Entity SOC is the previous completed interval's endogenous state.
            # The simulator applies standby loss before this interval's action.
            retention = 1.0 - asset.loss_coefficient if self._step > 0 else 1.0
            storage[(bid, name)] = (soc * capacity * retention, capacity)
        for session in problem.ev_sessions:
            if session.start_time_step != 0:
                continue
            bid, name = session.building_id, session.action_name
            cid = name.removeprefix("electric_vehicle_storage_")
            connected = self._value(observations, bid, "charger", cid, "connected_state")
            if connected <= 0:
                raise ValueError("Forecast connected EV is absent from observed state.")
            capacity = self._value(
                observations, bid, "charger", cid, "connected_ev_battery_capacity_kwh"
            )
            soc = self._soc(self._value(observations, bid, "charger", cid, "connected_ev_soc"))
            ev[(bid, name)] = (soc * capacity, capacity)
        for started_at, cycle in self._started.values():
            elapsed = self.start + self._step - started_at
            if 0 < elapsed < len(cycle.load_profile_kwh):
                aid = cycle.action_name.removeprefix("deferrable_appliance_")
                if (
                    self._value(
                        observations, cycle.building_id, "deferrable_appliance", aid, "running"
                    )
                    <= 0
                ):
                    raise ValueError(
                        "A committed appliance start was not applied by the Simulator."
                    )
                if self.weekly:
                    tail = cycle.load_profile_kwh[elapsed:]
                    state = (observations, cycle.building_id, "deferrable_appliance", aid)
                    if self._value(*state, "remaining_duration_steps") != len(
                        tail
                    ) or not math.isclose(
                        self._value(*state, "remaining_energy_kwh"),
                        sum(tail),
                        rel_tol=1e-5,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(f"Simulator appliance phase differs for {cycle.cycle_id}.")
        return storage, ev

    def predict(self, observations, deterministic=None, *, context=None):
        if self._attachment is None:
            raise RuntimeError("MILP must be attached before prediction.")
        if self.weekly and self._plan_step is not None:
            offset = self._step - self._plan_step
            if 0 < offset < self.converter.schedule.horizon:
                return self._execute(observations, offset)
        begin = time.perf_counter()
        now = self.start + self._step
        end = min(self.end, now + self.horizon_steps)
        problem = self.forecast.problem(now, end, episode_end=self.end if self.weekly else None)
        storage, ev = self._measured(problem, observations)
        problem, running = update_measured_problem(
            problem,
            storage_energy=storage,
            ev_energy=ev,
            started_cycles=self._started,
            now=now,
            terminal_storage_soc=self.settings.terminal_storage_soc,
            numerical_margin=not self.weekly,
        )
        prepared = time.perf_counter()
        result = solve_total_energy_schedule(problem, options=self.options)
        solved = time.perf_counter()
        row = {
            "episode": self._episode,
            "step": self._step,
            "source_start": now,
            "source_end_exclusive": end,
            "horizon_steps": problem.horizon,
            "build_seconds": prepared - begin,
            "solve_seconds": solved - prepared,
            "solver": result.solver.to_dict(),
            "model_cost_eur": result.cost_eur,
            "model_ev_shortfall_kwh": result.minimum_total_ev_shortfall_kwh,
            "problem_sha256": hashlib.sha256(problem.to_json().encode()).hexdigest(),
            "observed_storage_energy_kwh": {f"{b}/{a}": v[0] for (b, a), v in storage.items()},
            "observed_ev_energy_kwh": {f"{b}/{a}": v[0] for (b, a), v in ev.items()},
            "running_cycle_ids": sorted(running),
            "service_phase_status": result.service_phase_status,
            "service_shortfall_tolerance_kwh": self.options.lexicographic_shortfall_tolerance_kwh,
            "numerical_flow_projection_max_kw": (
                result.schedule.metadata.get("numerical_flow_projection_max_kw", 0)
                if result.schedule is not None
                else None
            ),
            "service_phase_optimal": result.service_phase_optimal,
        }
        if result.schedule is None:
            self._record({**row, "decision_seconds": time.perf_counter() - begin, "failed": True})
            raise RuntimeError(f"MILP has no feasible plan at step {self._step}: {result.solver}")
        schedule = result.schedule
        running_keys = {
            (c.building_id, c.action_name)
            for c in problem.deferrable_cycles
            if c.cycle_id in running
        }
        schedule = replace(
            schedule,
            series=tuple(
                replace(s, values=(0.0, *s.values[1:]))
                if (s.building_id, s.action_name) in running_keys
                else s
                for s in schedule.series
            ),
        )
        # Bind the same standard unit conversion against the new plan; no RBC
        # teacher, safety projection or precomputed weekly action is used.
        self.converter.bind_schedule(schedule, **self._attachment)
        self._plan_step = self._step
        self._planned_starts = {}
        for cycle in problem.deferrable_cycles:
            offset = result.selected_deferrable_starts.get(cycle.cycle_id)
            if cycle.cycle_id not in running and offset is not None:
                self._planned_starts.setdefault(offset, []).append(cycle)
        if self.weekly and self._decision_log is not None:
            block = self._decision_log.parent / f"episode-{self._episode}" / f"week-{now}"
            block.mkdir(parents=True, exist_ok=True)
            (block / "problem.json").write_text(problem.to_json())
            (block / "result.json").write_text(
                json.dumps(replace(result, schedule=schedule).to_dict(), allow_nan=False)
            )
        actions = self._execute(observations, 0)
        row.update(
            {
                "decision_seconds": time.perf_counter() - begin,
                "service_phase_optimal": result.service_phase_optimal,
                "actions": actions,
                "failed": False,
            }
        )
        self._record(row)
        return actions

    def _execute(self, observations, offset):
        actions = self.converter.predict_at_step(observations, schedule_step=offset)
        for cycle in self._planned_starts.get(offset, ()):
            aid = cycle.action_name.removeprefix("deferrable_appliance_")
            if (
                self._value(
                    observations, cycle.building_id, "deferrable_appliance", aid, "can_start"
                )
                <= 0
            ):
                raise ValueError(f"Simulator cannot start the scheduled request {cycle.cycle_id}.")
            self._started[cycle.cycle_id] = (self.start + self._step, cycle)
        self._step += 1
        return actions

    def _record(self, row):
        self._rows.append(row)
        if self._decision_log is not None:
            with self._decision_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def update(self, *args, **kwargs):
        return None

    def export_artifacts(self, output_dir, context=None):
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shared = {
            "policy_type": "weekly_community_oracle" if self.weekly else "milp_mpc",
            "forecast": "perfect_dataset_values_bounded_to_168h"
            if self.weekly
            else "perfect_dataset_values_bounded_to_24h",
            "hyperparameters": self.settings.model_dump(),
            "schema_sha256": self.forecast.schema_sha256,
            "reoptimizes_every_step": not self.weekly,
            "global_optimum_claim": False,
            "runtime_only_reason": "requires_live_state_forecasts_and_milp_solver",
        }
        (destination / "milp_decisions.json").write_text(json.dumps(self._rows, indent=2) + "\n")
        artifacts = []
        for i, bid in enumerate(self.converter._building_names):
            name = f"policy_agent_{i}.json"
            (destination / name).write_text(
                json.dumps({**shared, "building_id": bid}, indent=2) + "\n"
            )
            artifacts.append(
                {
                    "agent_index": i,
                    "path": name,
                    "format": "rule_based",
                    "config": {
                        "use_preprocessor": False,
                        "auxiliary_paths": ["milp_decisions.json"],
                    },
                }
            )
        return {
            "format": "rule_based",
            "parameters": shared,
            "artifacts": artifacts,
            "deployable": False,
            "runtime_only_reason": shared["runtime_only_reason"],
        }
