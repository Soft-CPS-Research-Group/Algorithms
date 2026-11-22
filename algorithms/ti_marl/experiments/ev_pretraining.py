"""Research adapters for V2 EV pretraining on an isolated, unidirectional port.

Inputs come from the observed EV and the issued quarter-hour outlook. One
actor instance serves one snapshot at a time, as in the V2 PPO replay loop.
"""

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical

from algorithms.ti_marl.policy.ev_dose import FEATURES, ContinuousDose, scalar_head
from algorithms.ti_marl.policy.joint import JointActor
from algorithms.ti_marl.runtime.forecasts import samples_for


def fields(snapshot):
    """Local service requirement and announced power and tariff summaries until departure."""
    if len(snapshot.agent_ids) != 1 or len(snapshot.outlooks) != 1:
        raise ValueError("Expected one EV site with its published outlook")
    (site,) = snapshot.agent_ids
    groups = snapshot.groups_for(site)
    if len(groups) != 1 or groups[0].group_type != "ev_session":
        raise ValueError("Expected one connected EV and no other controllable equipment")
    z = samples_for(snapshot, site)
    if z.get("connected_state") != 1 or z.get("can_discharge", 0) != 0:
        raise ValueError("This study requires a connected, unidirectional EV")
    remaining_quarters = z["hours_until_departure"] * 4
    n = round(remaining_quarters)
    if abs(remaining_quarters - n) > 1e-5:
        raise ValueError("This study requires a departure on a quarter-hour boundary")
    (curve,) = snapshot.outlooks
    if curve.site != site or n < 1 or any(len(v) < n for v in (*curve.values, *curve.quality)):
        raise ValueError("An outlook covering every remaining quarter is required")
    price, demand, pv = (np.asarray(v[:n], dtype=float) for v in curve.values)
    if any(not np.isfinite(x).all() for x in (price, demand, pv)) or any(
        min(q[:n]) <= 0 for q in curve.quality
    ):
        raise ValueError("Incomplete EV outlook")
    power = z["max_charging_power_kw"]
    capacity = z["connected_ev_battery_capacity_kwh"]
    if power <= 0 or capacity <= 0:
        raise ValueError("EV charging power and capacity must be positive")
    need = (
        (z["connected_ev_required_soc_departure"] - z["connected_ev_soc"])
        * capacity
        / (power * 0.25)
    )
    surplus = np.clip((pv - demand) / power, 0.0, 1.0)
    return (
        need,
        n / 4.0,
        (n - need) / 4.0,
        price[0] / 0.1,
        (price[0] - np.mean(price)) / 0.1,
        surplus[0],
        surplus[0] - np.mean(surplus),
        float(need <= 1e-7),
        (price[0] - price.min()) / 0.1,
        (price.max() - price[0]) / 0.1,
        surplus.max() - surplus[0],
    )


class EVActor(JointActor):
    """Learn continuous power in the original allowed interval and mode in the original mask.

    The residual heads condition the existing V2 distributions on local EV observations
    and the published curves. Zero initial output reproduces the starting policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ev_gate = scalar_head(FEATURES, self.d_model)
        self.ev_watts = scalar_head(FEATURES + 2, self.d_model)

    def _contexts(self, snapshot):
        if self.step_seconds != 900.0:
            raise ValueError("The EV research adapter requires quarter-hour decisions")
        self.ev_input = fields(snapshot)
        return super()._contexts(snapshot)

    def _readout(self, name, x):
        return getattr(self, name)(x)

    def _distribution(self, context, state, c):
        base = super()._distribution(context, state, c)
        x = context.new_tensor(self.ev_input)
        if c.stage == "mode":
            active = self.group_modes["ev_session"].index("CHARGE_EV")
            mask = x.new_tensor([i == active for i in c.allowed])
            return Categorical(logits=base.logits + self._readout("ev_gate", x) * mask)
        inp = torch.cat((x, x.new_tensor([c.lower, c.upper])))
        residual = self._readout("ev_watts", inp).squeeze().double().clamp(-14.0, 14.0)
        # Log-odds shift of the beta mean; both shape parameters stay above one.
        return Beta(
            base.concentration1.double() * torch.exp(residual.clamp_min(0)),
            base.concentration0.double() * torch.exp((-residual).clamp_min(0)),
        )


class BoundaryActor(EVActor):
    """Learn whether to use either legal endpoint or the continuous interior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_band = nn.Sequential(
            nn.Linear(FEATURES + 2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 3),
        )

    def _distribution(self, context, state, c):
        dist = super()._distribution(context, state, c)
        if c.stage != "power":
            return dist
        x = context.new_tensor((*self.ev_input, c.lower, c.upper))
        return (Categorical(logits=self.ev_band(x)), dist)

    @staticmethod
    def _score(dist, c, ref):
        if not isinstance(dist, tuple):
            return JointActor._score(dist, c, ref)
        which, inside = dist
        component = 0 if c.unit_sample == 0.0 else 2 if c.unit_sample == 1.0 else 1
        logw = np.log((c.upper - c.lower) * c.scale_kw)
        logp = which.logits[component]
        if component == 1:
            logp = logp + inside.log_prob(inside.concentration1.new_tensor(c.unit_sample)) - logw
        entropy = which.entropy() + which.probs[1] * (inside.entropy() + logw)
        return logp, entropy

    def _choose(self, c, group_context, state, deterministic):
        if c.stage != "power" or not c.sampled:
            return super()._choose(c, group_context, state, deterministic)
        which, inside = self._distribution(group_context, state, c)
        component = int(which.probs.argmax() if deterministic else which.sample())
        if component == 1:
            u = float((inside.mean if deterministic else inside.sample()).clamp(1e-9, 1.0 - 1e-9))
        else:
            u = 0.0 if component == 0 else 1.0
        c = replace(c, unit_sample=u, fraction=c.lower + (c.upper - c.lower) * u)
        logp, _ = self._score((which, inside), c, group_context)
        return replace(c, old_log_prob=float(logp.detach()))


class LongEVActor(BoundaryActor):
    """Include remaining cheap quarters and forecast solar surplus before departure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("ev_gate", "ev_watts", "ev_band"):
            seq = getattr(self, name)
            seq[0] = nn.Linear(seq[0].in_features + 4, self.d_model)

    def _contexts(self, snapshot):
        context = super()._contexts(snapshot)
        (outlook,) = snapshot.outlooks
        n = round(4 * self.ev_input[1])
        prices = np.asarray(outlook.values[0][:n], dtype=float)
        later = prices[1:]
        need = self.ev_input[0]
        future_surplus = max(0.0, n * (self.ev_input[5] - self.ev_input[6]) - self.ev_input[5])
        self.ev_input = (
            *self.ev_input,
            np.clip(need - np.count_nonzero(later < prices[0]), -6, 6),
            np.clip(need - np.count_nonzero(later <= prices[0]), -6, 6),
            min(future_surplus, 6.0),
            np.clip(need - future_surplus, -6, 6),
        )
        return context


def entry_features(price, net, need):
    """Plug-in facts only. Price and net surplus cover the announced stay."""
    p = np.asarray(price, dtype=float)
    s = np.clip(np.asarray(net, dtype=float), 0.0, 1.0)
    need = np.asarray(need, dtype=float)
    n = p.shape[1]
    cheap = p <= p.min(1, keepdims=True) + 0.005
    transitions = (abs(np.diff(p, axis=1)) > 0.005) | (abs(np.diff(s, axis=1)) > 0.05)
    return np.column_stack(
        (
            need,
            np.full(len(need), n / 8),
            need / n,
            np.mod(np.round(need, 9), 1.0),
            p[:, 0] / 0.1,
            p.mean(1) / 0.1,
            np.ptp(p, axis=1) / 0.1,
            cheap.mean(1),
            cheap.sum(1) / 8,
            s.max(1),
            s.mean(1),
            (s > 0.05).mean(1),
            s.sum(1) / 8,
            transitions.sum(1) / 8,
        )
    ).astype(np.float32)


class SessionEVActor(LongEVActor):
    """Two learned EV readouts share the V2 encoder. Select one on plug-in."""

    def __init__(self, *args, **kwargs):
        from copy import deepcopy

        super().__init__(*args, **kwargs)
        self.other_gate = deepcopy(self.ev_gate)
        self.other_watts = deepcopy(self.ev_watts)
        self.entry_router = nn.Sequential(nn.Linear(14, 24), nn.SiLU(), nn.Linear(24, 1))
        self.start_session()

    def start_session(self):
        self.session_key = None
        self.last_time = None
        self.other = False

    def _contexts(self, snapshot):
        ctx = super()._contexts(snapshot)
        n = round(self.ev_input[1] * 4)
        site = snapshot.agent_ids[0]
        key = (
            site,
            round(snapshot.clock_seconds + n * 900),
            samples_for(snapshot, site)["connected_ev_required_soc_departure"],
        )
        if self.session_key != key or self.last_time is None or snapshot.time_step < self.last_time:
            (curve,) = snapshot.outlooks
            p, d, pv = (np.array(channel[:n])[None, :] for channel in curve.values)
            power = samples_for(snapshot, site)["max_charging_power_kw"]
            data = entry_features(p, (pv - d) / power, [self.ev_input[0]])
            tensor = torch.as_tensor(data, device=next(self.parameters()).device)
            self.other = bool(self.entry_router(tensor).detach().item() > 0.0)
            self.session_key = key
        self.last_time = snapshot.time_step
        return ctx

    def _readout(self, name, x):
        if self.other:
            return (self.other_gate if name == "ev_gate" else self.other_watts)(x)
        return super()._readout(name, x)


class CalibratedEVActor(LongEVActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_watts = ContinuousDose(self.ev_watts)


ACTORS = {
    "calibrated": CalibratedEVActor,
    "continuous": EVActor,
    "boundary": BoundaryActor,
    "wide": LongEVActor,
    "session": SessionEVActor,
}
FORMAT = "ti_marl.v2.isolated_ev.1"


def load_ev_policy(path):
    """Load a research actor on CPU from its standalone weights and type registry."""
    model = torch.load(path, map_location="cpu", weights_only=True)
    if model["format"] != FORMAT:
        raise ValueError("Unsupported EV research policy format")
    actor = ACTORS[model["adapter"]](model["type_registry"], **model["actor_parameters"])
    actor.load_state_dict(model["weights"])
    for attr in ("step_seconds", "tolerance_soc", "departure_margin_seconds", "order_seed"):
        setattr(actor, attr, model["operating_parameters"][attr])
    return actor.eval()
