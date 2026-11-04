from dataclasses import replace

import numpy as np
import pytest

from algorithms.oracles.perfect_foresight_milp import SolveOptions
from algorithms.oracles.simulator_linear_physics import (
    align_simulator_linear_problem,
    constant_efficiency,
    power_envelope,
)
from algorithms.oracles.total_energy_milp import (
    EVSession,
    StorageAsset,
    TotalEnergyProblem,
    solve_total_energy_schedule,
)


def _problem(**kwargs):
    return TotalEnergyProblem(
        problem_id="physical-contract",
        timestep_hours=0.25,
        building_ids=("home",),
        price_eur_per_kwh=np.array([1.0, 2.0]),
        base_net_load_kwh=np.zeros((1, 2)),
        **kwargs,
    )


def test_constant_physics_uses_simulator_efficiency_without_safety_discount():
    attributes = dict(
        capacity=10,
        nominal_power=5,
        capacity_loss_coefficient=0,
        power_efficiency_curve=[[0, 0.94], [1, 0.94]],
        capacity_power_curve=[[0, 1], [0.9, 1], [1, 0.25]],
    )
    assert constant_efficiency("battery", attributes, battery=True) == pytest.approx(np.sqrt(0.94))
    lines = power_envelope("battery", attributes)
    for soc, expected in [(0, 5), (0.9, 5), (0.95, 3.125), (1, 1.25)]:
        assert min(m * soc * 10 + b for m, b in lines) == pytest.approx(expected)
    for bad in (
        {"capacity_loss_coefficient": 1e-5},
        {"power_efficiency_curve": [[0, 0.9], [1, 0.95]]},
        {"power_efficiency_curve": None},
    ):
        with pytest.raises(ValueError):
            constant_efficiency("battery", {**attributes, **bad}, battery=True)
    with pytest.raises(ValueError, match="concave"):
        power_envelope(
            "battery", {**attributes, "capacity_power_curve": [[0, 0.2], [0.5, 0.3], [1, 1]]}
        )


def test_discharge_reserve_does_not_force_recharge_after_passive_loss():
    storage = StorageAsset(
        "home",
        "electrical_storage",
        10,
        1,
        1,
        5,
        5,
        minimum_energy_kwh=1,
        loss_coefficient=0.01,
        reserve_is_discharge_only=True,
    )
    # Terminal floor is 1; one early charge replaces the real passive loss.
    result = solve_total_energy_schedule(
        _problem(stationary_storage=(storage,)), SolveOptions(throughput_tiebreaker_eur_per_kwh=0)
    )
    assert result.cost_eur == pytest.approx(1 / 0.99 - 1)
    # Below-reserve starting energy remains physically admissible.
    below = replace(storage, initial_energy_kwh=0.9)
    result = solve_total_energy_schedule(_problem(stationary_storage=(below,)))
    assert result.solver.optimal
    assert result.cost_eur == pytest.approx(1 / 0.99 - 0.9)


@pytest.mark.parametrize("clipping", [False, True])
def test_capacity_clipped_ev_can_finish_below_command_deadband(clipping):
    ev = EVSession(
        "ev",
        "home",
        "charger",
        0,
        1,
        10,
        9.9,
        10,
        0,
        4,
        0,
        min_charge_kw=2,
        allow_charge_clipping=clipping,
    )
    result = solve_total_energy_schedule(_problem(ev_sessions=(ev,)))
    if clipping:
        assert result.solver.optimal
        assert result.cost_eur == pytest.approx(0.1)
        assert result.ev_final_energy_kwh["ev"] == pytest.approx(10)
    else:
        assert result.solver.status == "infeasible"


@pytest.mark.parametrize("capacity", [10, 40])
def test_soc_power_clipping_branch_matches_battery_transfer(capacity):
    from citylearn.base import EpisodeTracker
    from citylearn.energy_model import Battery

    tracker = EpisodeTracker(0, 2)
    tracker.next_episode(3, False, False, 0)
    battery = Battery(
        capacity=capacity,
        nominal_power=4,
        initial_soc=0.95,
        efficiency=1,
        loss_coefficient=0,
        depth_of_discharge=1,
        capacity_loss_coefficient=0,
        power_efficiency_curve=[[0, 1], [1, 1]],
        capacity_power_curve=[[0, 1], [0.9, 1], [1, 0.1]],
        episode_tracker=tracker,
        seconds_per_time_step=900,
        time_step_ratio=1,
    )
    battery.reset()
    battery.charge(1.0)  # 4 kW command; SOC taper clips to 2.2 kW.
    actual = float(battery.energy_balance[0])
    ev = EVSession(
        "ev",
        "home",
        "charger",
        0,
        0,
        capacity,
        capacity * 0.95,
        capacity,
        0,
        4,
        0,
        min_charge_kw=3,
        allow_departure_shortfall=True,
        allow_charge_clipping=True,
        charge_power_envelope=((0, 4), (-36 / capacity, 36.4)),
    )
    result = solve_total_energy_schedule(
        _problem(ev_sessions=(ev,)),
        SolveOptions(lexicographic_shortfall_tolerance_kwh=0),
    )
    # The 10 kWh battery fills at 0.5 kWh; the 40 kWh battery remains below
    # capacity and transfers 0.55 kWh at the SOC-dependent power ceiling.
    assert result.solver.optimal
    assert result.cost_eur == pytest.approx(actual, abs=1e-6)
    assert result.ev_final_energy_kwh["ev"] == pytest.approx(capacity * 0.95 + actual, abs=1e-6)
    roundtrip = TotalEnergyProblem.from_json(_problem(ev_sessions=(ev,)).to_json())
    assert roundtrip.ev_sessions == (ev,)


def test_hourly_standby_loss_is_scaled_to_the_control_interval():
    from citylearn.base import EpisodeTracker
    from citylearn.energy_model import Battery

    attributes = dict(
        capacity=10,
        nominal_power=5,
        efficiency=0.94,
        initial_soc=0.5,
        depth_of_discharge=0.9,
        loss_coefficient=0.01,
        capacity_loss_coefficient=0,
        power_efficiency_curve=[[0, 0.94], [1, 0.94]],
        capacity_power_curve=[[0, 1], [1, 1]],
    )
    p = _problem(
        stationary_storage=(StorageAsset("home", "electrical_storage", 10, 5, 5, 5, 5),),
        metadata={"ev_identity_by_session": {}},
    )
    aligned = align_simulator_linear_problem(
        p, {"buildings": {"home": {"electrical_storage": {"attributes": attributes}}}}
    )
    tracker = EpisodeTracker(0, 2)
    tracker.next_episode(3, False, False, 0)
    battery = Battery(
        **attributes, episode_tracker=tracker, seconds_per_time_step=900, time_step_ratio=1
    )
    battery.reset()
    battery.charge(0)
    battery.next_time_step()
    battery.charge(0)
    loss = aligned.stationary_storage[0].loss_coefficient
    assert loss == pytest.approx(0.0025)
    assert float(battery.soc[1]) * 10 == pytest.approx(5 * (1 - loss), abs=1e-6)


@pytest.mark.parametrize("residue_kw", [4e-7, 3.73e-6, 1e-3])
@pytest.mark.parametrize("charging", [False, True])
def test_solver_residue_cannot_request_out_of_bounds_ev_power(monkeypatch, residue_kw, charging):
    import algorithms.oracles.total_energy_milp as module

    ev = EVSession("ev", "home", "charger", 0, 1, 10, 5, 6 if charging else 5, 0, 4, 0)
    p = _problem(ev_sessions=(ev,))
    index = module._Layout.build(p).ev[0].charge[0]
    solve = module._milp_primal_only

    def with_residue(*args, **kwargs):
        raw = solve(*args, **kwargs)
        raw.x[index] = ((4 + residue_kw) if charging else -residue_kw) * p.timestep_hours
        return raw

    monkeypatch.setattr(module, "_milp_primal_only", with_residue)
    if residue_kw * p.timestep_hours > 1e-6:
        with pytest.raises(ValueError, match="physical bound"):
            solve_total_energy_schedule(p)
    else:
        result = solve_total_energy_schedule(p)
        assert result.schedule.series[0].values == (4.0 if charging else 0.0, 0.0)
        assert result.schedule.metadata["numerical_flow_projection_max_kw"] == pytest.approx(
            residue_kw
        )
