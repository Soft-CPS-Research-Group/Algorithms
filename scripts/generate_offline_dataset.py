#!/usr/bin/env python3
"""Generate an Offline RL dataset for EV charging from CityLearn CSV data.

Processes static CSV files using a standalone Rule-Based Controller (RBC) as
the behavior policy.  For each charger the script:

1. Extracts charging sessions (episodes) from the charger CSV.
2. Simulates SoC forward from the arrival SoC using charger specs.
3. Computes RBC actions at each timestep.
4. Evaluates a reward function that prioritises SoC-target achievement.
5. Writes (s, a, r, s', done) tuples to a single CSV file.

Usage
-----
    python scripts/generate_offline_dataset.py \
        --dataset-path datasets/citylearn_challenge_2022_phase_all_plus_evs \
        --chargers charger_1_1 charger_4_1 \
        --output datasets/offline_rl/ev_charging.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RBCConfig:
    """Hyperparameters for the Rule-Based Controller (mirrors rbc_agent.py)."""
    pv_charge_threshold: float = 0.0
    flexibility_hours: float = 3.0
    emergency_hours: float = 1.0
    pv_preferred_charge_rate: float = 0.6
    flex_trickle_charge: float = 0.0
    min_charge_rate: float = 0.0
    emergency_charge_rate: float = 1.0
    energy_epsilon: float = 1e-3
    step_hours: float = 1.0


@dataclass
class RewardConfig:
    """Weights for the offline RL reward function.

    Reward formula
    --------------
    r_t  =  w_progress · r_progress
           + w_solar   · r_solar
           - w_cost    · r_cost
           + r_terminal

    Components
    ----------
    r_progress : (soc_after − soc_before) / initial_soc_gap
        Dense per-step signal rewarding progress toward the departure SoC
        target.  Normalised by the initial gap so that each unit of progress
        contributes equally regardless of absolute gap size.

    r_solar : min(1, solar_generation / energy_charged) · action
        Bonus proportional to how much of the charged energy could be covered
        by local PV generation.  Encourages the agent to shift charging into
        high-solar periods.

    r_cost : electricity_price · energy_charged_kwh
        Penalty for the monetary cost of drawing energy from the grid.
        Encourages off-peak / low-price charging.

    r_terminal (only at the last step of an episode):
        +terminal_bonus        if final SoC ≥ required SoC − 1 %
        −terminal_penalty_scale · shortfall   otherwise
        Provides a large end-of-episode signal for meeting / missing the
        departure target.
    """
    w_progress: float = 0.5
    w_solar: float = 0.2
    w_cost: float = 0.1
    terminal_bonus: float = 1.0
    terminal_penalty_scale: float = 0.01


# ---------------------------------------------------------------------------
# Schema & data loading
# ---------------------------------------------------------------------------

@dataclass
class ChargerSpec:
    charger_id: str
    building: str
    max_power: float
    min_power: float
    efficiency: float


def load_schema(schema_path: Path) -> Tuple[Dict[str, ChargerSpec], Dict[str, float]]:
    """Parse *schema.json* and return charger specs + EV capacities."""
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    ev_capacities: Dict[str, float] = {}
    for ev_name, ev_def in schema.get("electric_vehicles_def", {}).items():
        attrs = (ev_def.get("battery") or {}).get("attributes") or {}
        ev_capacities[ev_name] = float(attrs.get("capacity", 60.0))

    charger_specs: Dict[str, ChargerSpec] = {}
    for bld_name, bld_data in schema.get("buildings", {}).items():
        for cid, cdata in (bld_data.get("chargers") or {}).items():
            attrs = cdata.get("attributes") or {}
            charger_specs[cid] = ChargerSpec(
                charger_id=cid,
                building=bld_name,
                max_power=float(attrs.get("max_charging_power", attrs.get("nominal_power", 7.4))),
                min_power=float(attrs.get("min_charging_power", 0.0)),
                efficiency=float(attrs.get("efficiency", 0.95)),
            )

    return charger_specs, ev_capacities


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_column(path: Path, column: str) -> List[float]:
    rows = _load_csv_rows(path)
    return [_safe_float(r[column]) for r in rows]


# ---------------------------------------------------------------------------
# RBC action computation (standalone – no CityLearn dependency)
# ---------------------------------------------------------------------------

def compute_rbc_action(
    current_soc: float,
    required_soc: float,
    time_to_departure: float,
    solar_generation: float,
    max_power: float,
    capacity: float,
    cfg: RBCConfig,
    is_flexible: bool = True,
) -> float:
    """Return a normalised charge rate in [0, 1].

    The logic mirrors ``RuleBasedPolicy._compute_ev_action`` but operates on
    explicit scalar inputs instead of an observation vector.
    """
    soc_gap_pct = max(0.0, required_soc - current_soc)
    energy_needed = (soc_gap_pct / 100.0) * capacity

    if energy_needed <= cfg.energy_epsilon:
        return 0.0

    ttd = max(time_to_departure, cfg.step_hours)
    mp = max_power if max_power > 0 else 7.4

    required_power = energy_needed / ttd
    norm = min(1.0, required_power / mp)

    if solar_generation >= cfg.pv_charge_threshold:
        norm = max(norm, cfg.pv_preferred_charge_rate)

    if ttd <= cfg.emergency_hours or not is_flexible:
        norm = max(norm, cfg.emergency_charge_rate)
    elif (
        is_flexible
        and solar_generation < cfg.pv_charge_threshold
        and ttd > cfg.flexibility_hours
    ):
        norm = max(cfg.flex_trickle_charge, min(norm, cfg.flex_trickle_charge))

    if norm > 0:
        norm = max(norm, cfg.min_charge_rate)

    return float(max(0.0, min(1.0, norm)))


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    soc_before: float,
    soc_after: float,
    required_soc: float,
    initial_soc_gap: float,
    action: float,
    solar_generation: float,
    max_power: float,
    electricity_price: float,
    is_terminal: bool,
    cfg: RewardConfig,
    step_hours: float = 1.0,
) -> float:
    """Compute the per-step reward.

    See :class:`RewardConfig` for the full formula and component descriptions.
    """
    reward = 0.0

    # -- dense progress signal --
    if initial_soc_gap > 0:
        progress = (soc_after - soc_before) / initial_soc_gap
        reward += cfg.w_progress * max(0.0, progress)

    # -- solar utilisation bonus --
    if action > 0 and solar_generation > 0:
        energy_charged = action * max_power * step_hours
        solar_frac = min(1.0, solar_generation / (energy_charged + 1e-6))
        reward += cfg.w_solar * solar_frac * action

    # -- cost penalty --
    if action > 0:
        energy_kwh = action * max_power * step_hours
        reward -= cfg.w_cost * electricity_price * energy_kwh

    # -- terminal bonus / penalty --
    if is_terminal:
        gap = max(0.0, required_soc - soc_after)
        if gap <= 1.0:
            reward += cfg.terminal_bonus
        else:
            reward -= cfg.terminal_penalty_scale * gap

    return reward


# ---------------------------------------------------------------------------
# SoC simulation
# ---------------------------------------------------------------------------

def step_soc(
    current_soc: float,
    action: float,
    max_power: float,
    efficiency: float,
    capacity: float,
    step_hours: float = 1.0,
) -> float:
    """Advance SoC by one timestep.  Returns new SoC in [0, 100]."""
    energy = action * max_power * efficiency * step_hours  # kWh
    delta_pct = (energy / capacity) * 100.0
    return float(max(0.0, min(100.0, current_soc + delta_pct)))


# ---------------------------------------------------------------------------
# Episode / session extraction
# ---------------------------------------------------------------------------

@dataclass
class Session:
    initial_soc: float
    ev_id: Optional[str]
    steps: List[Tuple[int, Dict[str, str]]] = field(default_factory=list)


def extract_sessions(
    charger_rows: List[Dict[str, str]],
    max_steps: int = 8760,
) -> List[Session]:
    """Identify charging sessions from charger CSV rows.

    A session begins when:
      * ``charger_state`` transitions from 3 → 2 → 1 (normal arrival), or
      * ``charger_state`` is 1 at the start of the file (ongoing session).

    A session ends when ``charger_state`` becomes 3 (EV departed).
    Only ``charger_state == 1`` rows are kept as episode steps.
    """
    sessions: List[Session] = []
    current: Optional[Session] = None
    pending_soc: Optional[float] = None
    pending_ev: Optional[str] = None

    for t in range(min(len(charger_rows), max_steps)):
        row = charger_rows[t]
        state = int(_safe_float(row.get("electric_vehicle_charger_state", "0")))

        if state == 2:
            # Arrival marker – close any open session and stash arrival info.
            if current is not None:
                sessions.append(current)
                current = None
            pending_soc = _safe_float(
                row.get("electric_vehicle_estimated_soc_arrival"), default=None
            )
            ev_raw = (row.get("electric_vehicle_id") or "").strip()
            pending_ev = ev_raw if ev_raw else None

        elif state == 1:
            ev_raw = (row.get("electric_vehicle_id") or "").strip()
            ev_id = ev_raw if ev_raw else pending_ev

            if current is None:
                init_soc = pending_soc if pending_soc is not None else 50.0
                current = Session(initial_soc=init_soc, ev_id=ev_id)
                pending_soc = None
                pending_ev = None

            current.steps.append((t, row))

        elif state == 3:
            if current is not None:
                sessions.append(current)
                current = None
            pending_soc = None
            pending_ev = None

    if current is not None:
        sessions.append(current)

    return sessions


# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------

COLUMNS = [
    "episode_id",
    "timestep",
    "charger_id",
    "ev_id",
    # ---- state s ----
    "hour",
    "day_type",
    "soc",
    "required_soc",
    "time_to_departure",
    "solar_generation",
    "electricity_pricing",
    "carbon_intensity",
    "max_charging_power",
    "battery_capacity",
    # ---- action & reward ----
    "action",
    "reward",
    # ---- next state s' ----
    "next_soc",
    "next_hour",
    "next_day_type",
    "next_time_to_departure",
    "next_solar_generation",
    "next_electricity_pricing",
    "next_carbon_intensity",
    # ---- terminal flag ----
    "done",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_dataset(
    dataset_path: Path,
    charger_ids: List[str],
    output_path: Path,
    rbc_cfg: Optional[RBCConfig] = None,
    reward_cfg: Optional[RewardConfig] = None,
) -> Dict[str, Any]:
    """Build the offline RL dataset CSV and return summary statistics."""

    rbc_cfg = rbc_cfg or RBCConfig()
    reward_cfg = reward_cfg or RewardConfig()

    charger_specs, ev_capacities = load_schema(dataset_path / "schema.json")

    pricing = _load_column(dataset_path / "pricing.csv", "electricity_pricing")
    carbon = _load_column(dataset_path / "carbon_intensity.csv", "carbon_intensity")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    episode_counter = 0
    stats: Dict[str, Any] = {
        "chargers_processed": [],
        "episodes": 0,
        "transitions": 0,
        "skipped_short_sessions": 0,
    }

    for charger_id in charger_ids:
        if charger_id not in charger_specs:
            print(f"  [WARN] charger '{charger_id}' not in schema – skipping")
            continue

        spec = charger_specs[charger_id]
        building_csv = dataset_path / f"{spec.building}.csv"
        charger_csv = dataset_path / f"{charger_id}.csv"

        if not building_csv.exists() or not charger_csv.exists():
            print(f"  [WARN] missing CSV for {charger_id} – skipping")
            continue

        building_rows = _load_csv_rows(building_csv)
        charger_rows = _load_csv_rows(charger_csv)

        max_steps = min(len(building_rows), len(charger_rows), len(pricing), len(carbon))
        sessions = extract_sessions(charger_rows, max_steps)

        for session in sessions:
            if len(session.steps) < 2:
                stats["skipped_short_sessions"] += 1
                continue

            ev_id = session.ev_id or "unknown"
            capacity = ev_capacities.get(ev_id, 60.0)

            first_row = session.steps[0][1]
            required_soc = _safe_float(
                first_row.get("electric_vehicle_required_soc_departure"), default=100.0
            )
            initial_gap = max(1.0, required_soc - session.initial_soc)

            current_soc = session.initial_soc
            ep_rows: List[Dict[str, Any]] = []

            for step_idx, (t, row) in enumerate(session.steps):
                is_last = step_idx == len(session.steps) - 1

                b = building_rows[t] if t < len(building_rows) else building_rows[-1]
                hour = int(_safe_float(b.get("hour", 0)))
                day_type = int(_safe_float(b.get("day_type", 1)))
                solar_gen = _safe_float(b.get("solar_generation", 0))

                price = pricing[t] if t < len(pricing) else pricing[-1]
                carb = carbon[t] if t < len(carbon) else carbon[-1]

                req_soc = _safe_float(
                    row.get("electric_vehicle_required_soc_departure"), default=required_soc
                )
                ttd = _safe_float(row.get("electric_vehicle_departure_time"), default=1.0)

                action = compute_rbc_action(
                    current_soc=current_soc,
                    required_soc=req_soc,
                    time_to_departure=ttd,
                    solar_generation=solar_gen,
                    max_power=spec.max_power,
                    capacity=capacity,
                    cfg=rbc_cfg,
                )

                next_soc = step_soc(
                    current_soc=current_soc,
                    action=action,
                    max_power=spec.max_power,
                    efficiency=spec.efficiency,
                    capacity=capacity,
                    step_hours=rbc_cfg.step_hours,
                )

                # Next-state observation fields
                if not is_last:
                    nt, nrow = session.steps[step_idx + 1]
                    nb = building_rows[nt] if nt < len(building_rows) else building_rows[-1]
                    n_hour = int(_safe_float(nb.get("hour", 0)))
                    n_day = int(_safe_float(nb.get("day_type", 1)))
                    n_solar = _safe_float(nb.get("solar_generation", 0))
                    n_price = pricing[nt] if nt < len(pricing) else pricing[-1]
                    n_carb = carbon[nt] if nt < len(carbon) else carbon[-1]
                    n_ttd = _safe_float(nrow.get("electric_vehicle_departure_time"), default=0.0)
                else:
                    n_hour, n_day, n_solar = hour, day_type, solar_gen
                    n_price, n_carb, n_ttd = price, carb, 0.0

                reward = compute_reward(
                    soc_before=current_soc,
                    soc_after=next_soc,
                    required_soc=req_soc,
                    initial_soc_gap=initial_gap,
                    action=action,
                    solar_generation=solar_gen,
                    max_power=spec.max_power,
                    electricity_price=price,
                    is_terminal=is_last,
                    cfg=reward_cfg,
                    step_hours=rbc_cfg.step_hours,
                )

                ep_rows.append(
                    {
                        "episode_id": episode_counter,
                        "timestep": step_idx,
                        "charger_id": charger_id,
                        "ev_id": ev_id,
                        "hour": hour,
                        "day_type": day_type,
                        "soc": round(current_soc, 4),
                        "required_soc": round(req_soc, 4),
                        "time_to_departure": round(ttd, 4),
                        "solar_generation": round(solar_gen, 4),
                        "electricity_pricing": round(price, 6),
                        "carbon_intensity": round(carb, 6),
                        "max_charging_power": spec.max_power,
                        "battery_capacity": capacity,
                        "action": round(action, 6),
                        "reward": round(reward, 6),
                        "next_soc": round(next_soc, 4),
                        "next_hour": n_hour,
                        "next_day_type": n_day,
                        "next_time_to_departure": round(n_ttd, 4),
                        "next_solar_generation": round(n_solar, 4),
                        "next_electricity_pricing": round(n_price, 6),
                        "next_carbon_intensity": round(n_carb, 6),
                        "done": 1 if is_last else 0,
                    }
                )

                current_soc = next_soc

            if ep_rows:
                all_rows.extend(ep_rows)
                episode_counter += 1

        stats["chargers_processed"].append(charger_id)

    # -- write CSV --
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    stats["episodes"] = episode_counter
    stats["transitions"] = len(all_rows)

    # -- write companion metadata JSON --
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "dataset_path": str(dataset_path),
        "chargers": charger_ids,
        "rbc_config": rbc_cfg.__dict__,
        "reward_config": reward_cfg.__dict__,
        "stats": stats,
        "columns": COLUMNS,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an Offline RL dataset for EV charging.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the CityLearn dataset directory (contains schema.json).",
    )
    parser.add_argument(
        "--chargers",
        nargs="+",
        required=True,
        help="Charger IDs to include (e.g. charger_1_1 charger_4_1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    print(f"Dataset path : {args.dataset_path}")
    print(f"Chargers     : {args.chargers}")
    print(f"Output       : {args.output}")
    print()

    stats = generate_dataset(
        dataset_path=Path(args.dataset_path),
        charger_ids=args.chargers,
        output_path=Path(args.output),
    )

    print()
    print("=== Dataset Summary ===")
    print(f"  Chargers processed : {stats['chargers_processed']}")
    print(f"  Episodes           : {stats['episodes']}")
    print(f"  Total transitions  : {stats['transitions']}")
    print(f"  Skipped (short)    : {stats['skipped_short_sessions']}")
    print(f"  Output             : {args.output}")


if __name__ == "__main__":
    main()
