"""Numerical checks on the isolated EV study's reward and discrete action law."""

from dataclasses import replace

import pytest
import torch
from torch.distributions import Categorical

from algorithms.ti_marl.experiments.ev_levels import LevelDistribution, LevelEVActor
from algorithms.ti_marl.policy.joint import Choice
from scripts.train_ti_marl_ev_sessions import excess_delivered_kwh


def test_only_new_excess_is_charged():
    target = 10.0
    stored = 9.5
    charges = []
    for delivered in (0.25, 0.5, 0.0, 0.5):
        stored += delivered
        charges.append(excess_delivered_kwh(stored, target, delivered))
    assert charges == [0.0, 0.25, 0.0, 0.5]
    assert excess_delivered_kwh(12.0, target, 0.0) == 0.0
    assert excess_delivered_kwh(12.5, target, 0.5) == 0.5


def test_discrete_support_and_log_probability_use_physical_levels():
    choice = Choice("one", "ev_session", "power", 1, lower=0.35, upper=1.0, scale_kw=4)
    assert LevelEVActor.levels(choice) == (0.5, 1.0)
    assert LevelEVActor.levels(replace(choice, lower=0.6)) == (1.0,)
    with pytest.raises(ValueError):
        LevelEVActor.levels(replace(choice, upper=0.4))
    logits = torch.tensor([1.0, 2.0], requires_grad=True)
    dist = LevelDistribution((0.5, 1.0), Categorical(logits=logits))
    logp, entropy = LevelEVActor._score(dist, replace(choice, fraction=1.0), None)
    assert torch.allclose(logp, logits.log_softmax(0)[1])
    assert torch.isfinite(entropy)
    logp.backward()
    assert logits.grad[1] > 0 and logits.grad[0] < 0


@pytest.mark.parametrize(
    "quarters,stored,need",
    [
        (1, 10.0, 0.75),
        (1, 10.0, 1.45),
        (4, 10.0, 2.0),
        (4, 10.0, 2.75),
        (3, 19.5, 0.48),
        (2, 19.7, 0.25),
        (8, 12.0, -0.2),
    ],
)
def test_isolated_port_domains_match_native_reservation(quarters, stored, need):
    from algorithms.ti_marl.runtime.reservation import SiteReservation
    from algorithms.ti_marl.runtime.reservation_inputs import (
        Battery,
        Equipment,
        PowerLimit,
        PowerMode,
        SiteInputs,
    )
    from scripts.ti_marl_ev.dose_training import legal_interval

    modes = (PowerMode("IDLE", 0, 0.0, 0.0, 0.0), PowerMode("CHARGE_EV", 1, 1.4, 4.0, 4.0))
    ev = Equipment(
        "ev", modes, Battery(20.0, stored), quarters * 900.0, max(0.0, stored + need - 1.0)
    )
    limit = PowerLimit("site", {"ev": 1.0}, "import", 20.0, constant=True)
    site = SiteReservation(SiteInputs((ev,), (limit,)))
    idle, charge, lower, upper = legal_interval(
        torch.tensor(need, dtype=torch.float64), torch.tensor(stored, dtype=torch.float64), quarters
    )
    feasible = site.admissible_modes("ev")
    assert bool(idle) == ("IDLE" in feasible)
    assert bool(charge) == ("CHARGE_EV" in feasible)
    if charge:
        site.select_mode("ev", "CHARGE_EV")
        assert site.fraction_bounds("ev") == pytest.approx((float(lower), float(upper)), abs=1e-7)


@pytest.mark.parametrize("requested_kwh", [1.23, 1.6, 0.0, -0.25])
def test_continuous_training_ledger_matches_operating_reward(requested_kwh):
    from types import SimpleNamespace

    import numpy as np

    from algorithms.ti_marl.learning.operating_reward import TERM_WEIGHTS, operating_reward
    from scripts.ti_marl_ev.dose_training import GAMMA, rollout
    from scripts.ti_marl_ev.intermediate import PRICE

    gate = torch.nn.Linear(15, 1)
    watts = torch.nn.Linear(17, 1)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.fill_(4.0)
        watts.weight.zero_()
        watts.bias.zero_()
    price = np.array([[0.08, 0.24]])
    net = np.array([[-0.2, 0.5]])
    loss, q = rollout(
        SimpleNamespace(ev_gate=gate, ev_watts=watts),
        price,
        net,
        np.array([requested_kwh]),
        np.array([10.0]),
        explore=False,
    )
    first = 0.8 if requested_kwh == 1.6 else 0.675
    assert q.detach().numpy()[0] == pytest.approx([first, 0.675])
    need = requested_kwh
    expected = 0.0
    for t, delta in enumerate(q.detach().numpy()[0]):
        need -= delta
        ledger = dict.fromkeys(TERM_WEIGHTS, 0.0)
        ledger.update(
            energy_cost_eur=float(
                price[0, t] * (max(0.0, delta - net[0, t]) - max(0.0, -net[0, t]))
            ),
            ev_tolerance_kwh=max(0.0, need) if t == 1 else 0.0,
        )
        cost = -30.0 * operating_reward({"thesis_reward_terms": ledger}, reference_sites=30).value
        expected += GAMMA**t * (cost + PRICE * min(delta, max(0.0, -need)))
    assert loss.item() == pytest.approx(expected, abs=1e-9)
    loss.mean().backward()
    assert torch.isfinite(watts.bias.grad).all()


@pytest.mark.parametrize(
    "energy,expected",
    [
        (0.0, [0.0, 0.0, 0.0, 0.0]),
        (0.41, [0.41, 0.0, 0.0, 0.0]),
        (3.23, [1.0, 1.0, 0.88, 0.35]),
        (3.95, [1.0, 1.0, 1.0, 0.95]),
        (4.0, [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_same_energy_immediate_reference_obeys_port_minimum(energy, expected):
    from scripts.ti_marl_ev.native_curves import immediate_schedule

    assert immediate_schedule(energy, 4) == pytest.approx(expected)
    with pytest.raises(ValueError):
        immediate_schedule(0.2, 4)


@pytest.mark.parametrize("target,solar,sign", [(0.9, 0.0, -1), (0.4, 1.0, 1)])
def test_finite_power_gradient_respects_actual_service_and_overfill(target, solar, sign):
    from types import SimpleNamespace

    import numpy as np

    from scripts.ti_marl_ev.dose_context import power_gradient

    actor = SimpleNamespace(ev_gate=torch.nn.Linear(15, 1), ev_watts=torch.nn.Linear(17, 1))
    with torch.no_grad():
        actor.ev_gate.weight.zero_()
        actor.ev_gate.bias.fill_(4.0)
        actor.ev_watts.weight.zero_()
        actor.ev_watts.bias.zero_()
    loss = power_gradient(
        actor, np.array([[0.12]]), np.array([[solar]]), np.array([target]), np.array([10.0]), 0.1
    )
    loss.backward()
    assert sign * actor.ev_watts.bias.grad.item() > 0


def test_session_router_keeps_the_selected_readout_on_retry_and_until_departure(monkeypatch):
    from types import SimpleNamespace

    from algorithms.ti_marl.experiments import ev_pretraining as ev

    actor = ev.SessionEVActor.__new__(ev.SessionEVActor)
    torch.nn.Module.__init__(actor)
    actor.entry_router = torch.nn.Linear(14, 1)
    with torch.no_grad():
        actor.entry_router.weight.zero_()
        actor.entry_router.bias.fill_(1.0)
    actor.start_session()

    def context(self, obs):
        self.ev_input = (0.75, obs.quarters / 4)
        return None

    monkeypatch.setattr(ev.LongEVActor, "_contexts", context)
    monkeypatch.setattr(
        ev,
        "samples_for",
        lambda obs, site: {
            "connected_ev_required_soc_departure": 0.8,
            "max_charging_power_kw": 4.0,
        },
    )

    def frame(step):
        channel = [0.1] * (4 - step)
        return SimpleNamespace(
            quarters=4 - step,
            clock_seconds=step * 900.0,
            time_step=step,
            agent_ids=("house",),
            outlooks=(SimpleNamespace(values=(channel, channel, channel)),),
        )

    actor._contexts(frame(0))
    assert actor.other
    with torch.no_grad():
        actor.entry_router.bias.fill_(-1.0)
    actor._contexts(frame(1))
    actor._contexts(frame(1))
    assert actor.other
    actor.start_session()
    actor._contexts(frame(0))
    assert not actor.other


def test_response_interpolates_continuous_watts_and_can_exceed_request():
    from algorithms.ti_marl.experiments.ev_pretraining import ContinuousDose

    head = ContinuousDose(torch.nn.Linear(17, 1))
    with torch.no_grad():
        head.curve[8] = 0.5
        head.curve[9] = 0.8
    x = torch.zeros((3, 17))
    x[:, 0] = torch.tensor([0.41, 0.425, 0.44])
    x[:, 1] = 1.0
    x[:, -2] = 0.35
    x[:, -1] = 1.0
    applied = 0.35 + 0.65 * head(x).squeeze(1).sigmoid()
    assert applied.detach().numpy() == pytest.approx([0.56, 0.65, 0.74], abs=1e-6)
    assert (applied > x[:, 0]).all()
    applied.sum().backward()
    assert head.curve.grad.nonzero().flatten().tolist() == [8, 9]
