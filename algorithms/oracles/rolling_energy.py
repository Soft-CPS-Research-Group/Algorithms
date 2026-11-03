"""Bounded perfect forecasts and measured-state updates for the community MILP.

The data provider may cache Parquet files, but its extraction boundary returns
only [now, now + horizon). No weekly solution or other controller is consulted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from algorithms.oracles.citylearn_total_energy import (
    build_citylearn_total_energy_problem,
)
from algorithms.oracles.total_energy_milp import DeferrableCycle, TotalEnergyProblem


class BoundedDatasetForecast:
    """Perfect foresight restricted at the data boundary, not after optimization."""

    def __init__(self, schema_path: str | Path, building_ids=None):
        self.schema_path = Path(schema_path).resolve()
        payload = self.schema_path.read_bytes()
        self.schema_sha256 = hashlib.sha256(payload).hexdigest()
        self.schema = json.loads(payload)
        self.building_ids = building_ids
        self.timestep_hours = float(self.schema["seconds_per_time_step"]) / 3600.0
        self._cache: dict[Path, pd.DataFrame] = {}
        self._kinds: dict[str, str] = {}
        self._charger_files: dict[tuple[str, str], str] = {}
        for bid, building in self.schema["buildings"].items():
            if not building.get("include", True):
                continue
            for field in ("energy_simulation", "pricing"):
                self._kinds[str(building[field])] = "series"
            for cid, charger in (building.get("chargers") or {}).items():
                filename = str(charger["charger_simulation"])
                self._kinds[filename] = "series"
                self._charger_files[(bid, f"electric_vehicle_storage_{cid}")] = filename
            for appliance in (building.get("deferrable_appliances") or {}).values():
                self._kinds[str(appliance["cycle_profiles_file"])] = "profiles"
                self._kinds[str(appliance["flexibility_schedule_file"])] = "requests"

    def problem(self, start: int, end: int) -> TotalEnergyProblem:
        if start < 0 or end <= start or (end - start) * self.timestep_hours > 24.0 + 1e-9:
            raise ValueError("A forecast must cover a positive window of at most 24 hours.")
        visible: dict[str, pd.DataFrame] = {}

        def read_window(root: Path, filename: Any) -> pd.DataFrame:
            filename = str(filename)
            if filename in visible:
                return visible[filename]
            path = (root / filename).resolve()
            if path not in self._cache:
                self._cache[path] = pd.read_parquet(path)
            source = self._cache[path]
            kind = self._kinds[filename]
            if kind == "series":
                if end > len(source):
                    raise ValueError(f"Forecast exceeds available data: {filename}")
                frame = source.iloc[start:end].reset_index(drop=True)
            elif kind == "requests":
                # Retain past/pending requests, never requests first available
                # beyond the forecast. Completed cycles are removed by state.
                frame = source.loc[
                    (source.earliest_start_time_step < end)
                    & (source.latest_start_time_step >= start)
                ].copy()
                frame["earliest_start_time_step"] -= start
                frame["latest_start_time_step"] -= start
            else:
                frame = source  # Static equipment profiles, not future events.
            visible[filename] = frame
            return frame

        built = build_citylearn_total_energy_problem(
            schema_path=self.schema_path,
            start_time_step=0,
            end_time_step=end - start,
            settlement="community",
            building_ids=self.building_ids,
            read_data=read_window,
        )
        problem = built.problem
        sessions = []
        for session in problem.ev_sessions:
            frame = visible[self._charger_files[(session.building_id, session.action_name)]]
            # A right-censored connection has no invented departure. Its
            # announced user target is nevertheless the horizon reserve; this
            # prevents endless deferral and borrowing charge beyond the job.
            row = frame.iloc[session.start_time_step]
            raw_target = float(row["electric_vehicle_required_soc_departure"])
            target = raw_target / 100.0 if raw_target > 1.0 else raw_target
            if not np.isfinite(target) or not 0 <= target <= 1:
                raise ValueError("Forecast EV target is missing or outside [0, 1].")
            sessions.append(
                replace(
                    session,
                    required_final_energy_kwh=max(
                        session.minimum_energy_kwh, session.capacity_kwh * target
                    ),
                )
            )
        return replace(
            problem,
            problem_id=f"milp-mpc::{start}_{end}",
            ev_sessions=tuple(sessions),
            metadata={
                **problem.metadata,
                "source_start_time_step": start,
                "source_end_time_step_exclusive": end,
                "forecast": "perfect_dataset_values_bounded_to_24h",
                "global_optimum_claim": False,
            },
        )


def update_measured_problem(
    problem: TotalEnergyProblem,
    *,
    storage_energy: Mapping[tuple[str, str], tuple[float, float]],
    ev_energy: Mapping[tuple[str, str], tuple[float, float]],
    started_cycles: Mapping[str, tuple[int, DeferrableCycle]],
    now: int,
    terminal_storage_soc: float,
) -> tuple[TotalEnergyProblem, set[str]]:
    """Carry observed energy and committed appliance tails into a new solve."""
    storage = []
    for asset in problem.stationary_storage:
        energy, physical_capacity = storage_energy[(asset.building_id, asset.action_name)]
        if (
            not np.isfinite(energy)
            or not asset.minimum_energy_kwh - 1e-5 <= energy <= physical_capacity + 1e-5
        ):
            raise ValueError("Observed stationary energy is outside physical bounds.")
        storage.append(
            replace(
                asset,
                initial_energy_kwh=max(energy, asset.minimum_energy_kwh),
                capacity_kwh=max(asset.capacity_kwh, energy),
                final_energy_min_kwh=max(
                    asset.minimum_energy_kwh, terminal_storage_soc * physical_capacity
                ),
            )
        )
    sessions = []
    for session in problem.ev_sessions:
        if session.start_time_step == 0:
            energy, physical_capacity = ev_energy[(session.building_id, session.action_name)]
            if abs(physical_capacity - session.capacity_kwh) > 1e-4:
                raise ValueError("Connected EV capacity differs from the forecast identity.")
            if (
                not np.isfinite(energy)
                or not session.minimum_energy_kwh - 1e-5 <= energy <= physical_capacity + 1e-5
            ):
                raise ValueError("Observed EV energy is outside physical bounds.")
            session = replace(session, initial_energy_kwh=max(energy, session.minimum_energy_kwh))
        sessions.append(session)
    cycles = [c for c in problem.deferrable_cycles if c.cycle_id not in started_cycles]
    running_actions: set[str] = set()
    for cycle_id, (started_at, cycle) in started_cycles.items():
        elapsed = now - started_at
        if elapsed < 0:
            raise ValueError("A committed appliance cycle cannot start in the future.")
        tail = cycle.load_profile_kwh[elapsed : elapsed + problem.horizon]
        if not tail:
            continue
        # A forced tail retains the device's phase and occupancy, including
        # zero-power steps. The policy suppresses its synthetic start trigger.
        cycles.append(
            replace(
                cycle,
                earliest_start_time_step=0,
                latest_start_time_step=0,
                load_profile_kwh=tail,
                must_run=True,
            )
        )
        running_actions.add(cycle_id)
    return replace(
        problem,
        stationary_storage=tuple(storage),
        ev_sessions=tuple(sessions),
        deferrable_cycles=tuple(cycles),
        metadata={
            **problem.metadata,
            "terminal_storage_soc": terminal_storage_soc,
            "terminal_ev_policy": "announced_user_target",
            "measured_state": True,
            "stationary_model": "measured initial energy, conservative linear physics, fixed terminal SOC reserve",
            "community_observations_used": True,
        },
    ), running_actions
