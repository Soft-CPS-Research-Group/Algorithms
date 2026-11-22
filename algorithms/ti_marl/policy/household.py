"""V2 household readouts shared across devices of the same type.

The physical decoder still chooses admissible modes and power intervals jointly.
Forecast conditioning and transferred EV dosage operate on each port's own state.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical

from algorithms.ti_marl.policy.ev_dose import FEATURES, ContinuousDose, scalar_head
from algorithms.ti_marl.policy.joint import JointActor
from algorithms.ti_marl.runtime.reservation_inputs import _parts_by_sensor


def car_features(group, obs, curves, dt, *, use_partial_outlook=False):
    """Keep the announced departure time and summarise only the issued forecast samples."""
    n = int(np.ceil(obs.get("hours_until_departure", 0.0) * 4 - 1e-6))
    if not (obs.get("connected_state") == 1 and n > 0 and curves is not None and dt == 900):
        return None
    if (
        curves.shape[1] == 0
        or (n > curves.shape[1] and not use_partial_outlook)
        or not {
            "connected_ev_required_soc_departure",
            "connected_ev_soc",
            "connected_ev_battery_capacity_kwh",
        }
        <= obs.keys()
    ):
        return None
    price, load, pv = curves[:, :n]
    charger_eta = obs.get("charge_efficiency_at_max_ratio", obs.get("charger_efficiency_ratio"))
    battery_eta = dict(group.reservation_profile).get("battery_efficiency")
    if charger_eta is None or battery_eta is None:
        return None
    eta = charger_eta * battery_eta**0.5
    power = group.max_charge_power_kw
    if power <= 0 or eta <= 0 or obs.get("can_discharge", 0):
        return None
    need = (
        (obs["connected_ev_required_soc_departure"] - obs["connected_ev_soc"])
        * obs["connected_ev_battery_capacity_kwh"]
        / (0.25 * power * eta)
    )
    sun = np.clip((pv - load) / power, 0.0, 1.0)
    later = sun[1:].sum()
    return (
        need,
        n / 4,
        (n - need) / 4,
        price[0] / 0.1,
        (price[0] - price.mean()) / 0.1,
        sun[0],
        sun[0] - sun.mean(),
        float(need <= 1e-7),
        (price[0] - price.min()) / 0.1,
        (price.max() - price[0]) / 0.1,
        sun.max() - sun[0],
        np.clip(need - (price[1:] < price[0]).sum(), -6, 6),
        np.clip(need - (price[1:] <= price[0]).sum(), -6, 6),
        min(later, 6),
        np.clip(need - later, -6, 6),
    )


def dose_bounds(choice, rated_kw):
    """Express the native feasible interval in the nominal EV head's units."""
    ratio = choice.scale_kw / rated_kw
    return choice.lower * ratio, choice.upper * ratio


class HouseholdActor(JointActor):
    """Per-port continuous EV head and learned adjustments for joint household decisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ev_gate = scalar_head(FEATURES + 4, self.d_model)
        self.ev_watts = ContinuousDose(scalar_head(FEATURES + 6, self.d_model))
        self.observe_programme_start_solar = False
        self.use_partial_ev_outlook = False
        self.coordination = nn.Parameter(torch.zeros(5, 8))

    def _contexts(self, snapshot):
        contexts = JointActor._contexts(self, snapshot)
        curves = {
            c.site: np.asarray(c.values)
            for c in snapshot.outlooks
            if all(min(q) > 0 for q in c.quality)
        }
        self.household_inputs = {}
        for site in snapshot.agent_ids:
            obs = _parts_by_sensor(snapshot, site)
            groups = snapshot.groups_for(site)
            curve = curves.get(site)
            outstanding = []
            for g in groups:
                reading = obs.get(g.module_id, {})
                if g.group_type == "ev_session":
                    x = car_features(
                        g, reading, curve, self.step_seconds,
                        use_partial_outlook=self.use_partial_ev_outlook,
                    )
                    if x is not None:
                        outstanding.append((g.group_id, x))
            by_port = dict(outstanding)
            for g in groups:
                z = obs.get(g.module_id, {})
                inputs = None
                if curve is not None:
                    duration = min(24, curve.shape[1])
                    p, load, pv = curve[:, :duration]
                    n = max(
                        1,
                        min(
                            duration,
                            int(
                                np.ceil(z.get("cycle_duration_steps", 1) * self.step_seconds / 900)
                            ),
                        ),
                    )
                    price = p[:n].mean()
                    sun = (pv[:n] - load[:n]).mean()
                    rivals = [x for key, x in outstanding if g.group_id != key and x[0] > 0]
                    urgency = max((x[0] / max(1, 4 * x[1]) for x in rivals), default=0.0)
                    device_state = z.get("soc", 0.0) - 0.5
                    if self.observe_programme_start_solar and g.group_type == "deferrable":
                        # Start demand is incurred immediately. The mean across the
                        # cycle cannot distinguish solar now from solar arriving later.
                        device_state = np.clip((pv[0] - load[0]) / 4, 0.0, 2.0)
                    inputs = (
                        1.0,
                        (price - p.mean()) / 0.1,
                        (price - p.min()) / 0.1,
                        (p.max() - price) / 0.1,
                        np.clip(sun / 4, -2, 2),
                        device_state,
                        urgency,
                        z.get("urgency_ratio", 0.0),
                    )
                self.household_inputs[g.group_id] = (
                    by_port.get(g.group_id),
                    inputs,
                    g.max_charge_power_kw,
                )
        return contexts

    def _distribution(self, context, state, c):
        base = JointActor._distribution(self, context, state, c)
        ev, inputs, rated_kw = self.household_inputs[c.group_id]
        if inputs is None:
            return base
        x = context.new_tensor(inputs)
        modes = self.group_modes[c.group_type]
        if c.group_type == "ev_session" and ev is not None:
            features = context.new_tensor(ev)
            if c.stage == "mode":
                gate = self.ev_gate(features).squeeze() + self.coordination[0] @ x
                active = modes.index("CHARGE_EV")
                return Categorical(
                    logits=base.logits + gate * x.new_tensor([i == active for i in c.allowed])
                )
            z = torch.cat((features, x.new_tensor(dose_bounds(c, rated_kw))))
            residual = self.ev_watts(z).squeeze().double().clamp(-14, 14)
            return Beta(
                base.concentration1.double() * residual.clamp_min(0).exp(),
                base.concentration0.double() * (-residual).clamp_min(0).exp(),
            )
        if c.group_type == "stationary_storage" and c.stage == "power":
            # A separate learned magnitude can reduce a discharge even when the
            # physical decoder has enough headroom for the full battery rating.
            shift = (self.coordination[4] @ x).clamp(-10, 10)
            return Beta(
                base.concentration1 * shift.clamp_min(0).exp(),
                base.concentration0 * (-shift).clamp_min(0).exp(),
            )
        if c.stage != "mode" or c.group_type not in {"stationary_storage", "deferrable"}:
            return base
        storage = c.group_type == "stationary_storage"
        score = self.coordination[1 if storage else 2] @ x
        direction = {"START": 1, "CHARGE_STATIONARY": 1, "DISCHARGE_STATIONARY": -1}
        adjustments = x.new_tensor([direction.get(modes[i], 0) for i in c.allowed]) * score
        if storage:
            active = x.new_tensor([modes[i] != "IDLE" for i in c.allowed])
            adjustments = adjustments + active * (self.coordination[3] @ x)
        return Categorical(logits=base.logits + adjustments)

    def initialise_from_ev(self, source):
        """Transfer identical V2 architecture, including the learned continuous power law."""
        if source.d_model != self.d_model or source.group_modes != self.group_modes:
            raise ValueError("Incompatible V2 EV checkpoint")
        # The isolated experiment also contains an endpoint classifier. This controller
        # uses its continuous interior power law and the native joint decoder.
        transferred = {k: v for k, v in source.state_dict().items() if not k.startswith("ev_band.")}
        missing, extra = self.load_state_dict(transferred, strict=False)
        if missing != ["coordination"] or extra:
            raise ValueError("Incompatible V2 EV parameter names")
