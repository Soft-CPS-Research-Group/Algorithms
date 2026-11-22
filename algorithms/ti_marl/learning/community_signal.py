"""Quarter-hour community incentives and their separate, realised return."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from algorithms.ti_marl.runtime.forecasts import BLOCK_SECONDS, HORIZON


class ImportPeak:
    """Daily maximum of completed quarter-hour means, calculated from meter energy."""

    def __init__(self):
        self.day = self.block = None
        self.peak = self.kwh = self.covered = 0.0

    def observe(self, start, duration, energy, exported_kwh=0.0, *, day=None):
        if not all(math.isfinite(v) and v >= 0 for v in (energy, exported_kwh)):
            raise ValueError("Community grid import must be non-negative kWh")
        day = math.floor(start / 86400) if day is None else day
        block = math.floor(start / BLOCK_SECONDS)
        if day != self.day:
            self.day, self.peak = day, 0.0
        if block != self.block:
            self.block, self.kwh, self.covered = block, 0.0, 0.0
        if start + duration > (block + 1) * BLOCK_SECONDS + 1e-6:
            raise ValueError("Meter interval crosses an incentive block")
        self.covered += duration
        self.kwh += energy - exported_kwh
        if not math.isclose(self.covered, BLOCK_SECONDS):
            return 0.0
        previous = self.peak
        self.peak = max(previous, self.kwh * 3600 / BLOCK_SECONDS)
        return self.peak - previous


def community_input(snapshot):
    """No household commitments or flexible-load decisions enter the manager."""
    if not snapshot.outlooks:
        raise ValueError("The coordinator requires community forecast inputs")
    values = np.asarray([row.values for row in snapshot.outlooks], dtype=np.float32)[:, :, :HORIZON]
    quality = np.asarray([row.quality for row in snapshot.outlooks])[:, :, :HORIZON]
    known = quality > 0
    # Prices are intensive. Physical loads and solar are extensive.
    price = (values[:, 0] * known[:, 0]).sum(0) / known[:, 0].sum(0).clip(1)
    net_load = np.log1p(np.maximum((values[:, 1] * known[:, 1]).sum(0), 0))
    solar = np.log1p(np.maximum((values[:, 2] * known[:, 2]).sum(0), 0))
    angle = 2 * math.pi * (snapshot.clock_seconds % 86400) / 86400
    coverage = known.mean(0).reshape(-1)
    return np.concatenate(
        (
            price,
            net_load,
            solar,
            coverage,
            [math.log1p(snapshot.measured_peak_kw or 0), math.sin(angle), math.cos(angle)],
        )
    ).astype(np.float32)


class CommunityPolicy(nn.Module):
    def __init__(self, knots, width):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(6 * HORIZON + 3, width), nn.Tanh())
        self.location = nn.Linear(width, knots)
        self.log_scale = nn.Parameter(torch.full((knots,), -0.7))
        self.value = nn.Linear(width, 1)

    def forward(self, x):
        latent = self.encode(x)
        return Normal(self.location(latent), self.log_scale.clamp(-4, 1).exp()), self.value(
            latent
        ).squeeze(-1)


@dataclass
class BlockDecision:
    inputs: np.ndarray
    raw: np.ndarray
    reward: float = 0.0
    duration: float = 0.0


class CommunitySignal(nn.Module):
    def __init__(self, config, width, device, seed, gamma, timebase_seconds=3600.0):
        super().__init__()
        self.kind = config["signal"]
        self.phase = config["training_phase"]
        self.bound = config["max_adjustment_eur_kwh"]
        self.peak_weight = config["peak_weight_per_kw"]
        self.knots = config["curve_knots"]
        self.gamma = gamma
        self.timebase_seconds = timebase_seconds
        self.rng = np.random.default_rng(seed)
        self.network = CommunityPolicy(self.knots, width).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config["learning_rate"])
        self.reset()

    def reset(self):
        self.peak = ImportPeak()
        self.block = None
        self.curve = ()
        self.decisions = []

    def curve_for(self, snapshot, training):
        block = math.floor(snapshot.clock_seconds / BLOCK_SECONDS)
        if block == self.block:
            return self.curve
        x = community_input(snapshot)
        if self.kind == "neutral":
            curve = np.zeros(HORIZON)
        elif self.kind == "rule":
            net = np.expm1(x[HORIZON : 2 * HORIZON]) - np.expm1(x[2 * HORIZON : 3 * HORIZON])
            centered = net - np.median(net)
            curve = self.bound * centered / max(1.0, float(np.max(np.abs(centered))))
        else:
            if self.phase == "local" and training:
                # Vary the signal independently of the tariff so the leaf can
                # learn its separate effect. Only the real bill is settled.
                raw = self.rng.normal(size=self.knots)
            else:
                with torch.no_grad():
                    device = self.network.location.weight.device
                    dist, _ = self.network(torch.as_tensor(x, device=device))
                    training_manager = self.phase == "coordinator" and training
                    raw = (dist.sample() if training_manager else dist.mean).cpu().numpy()
                if training_manager:
                    self.decisions.append(BlockDecision(x, raw.copy()))
            curve = self.bound * np.interp(
                np.arange(HORIZON), np.linspace(0, HORIZON - 1, self.knots), np.tanh(raw)
            )
        self.block = block
        self.curve = tuple(float(v) for v in curve)
        return self.curve

    def observe_reward(self, reward, seconds):
        if self.decisions:
            record = self.decisions[-1]
            record.reward += self.gamma ** (record.duration / self.timebase_seconds) * reward
            record.duration += seconds

    def finish(self, bootstrap=0.0):
        """One on-policy actor/critic update per episode, using elapsed time."""
        if not self.decisions:
            return {}
        decisions, self.decisions = self.decisions, []
        returns = []
        value = float(bootstrap)
        for step in reversed(decisions):
            value = step.reward + self.gamma ** (step.duration / self.timebase_seconds) * value
            returns.append(value)
        device = self.network.location.weight.device
        observations = torch.as_tensor(np.stack([d.inputs for d in decisions]), device=device)
        actions = torch.as_tensor(np.stack([d.raw for d in decisions]), device=device)
        dist, values = self.network(observations)
        target = values.new_tensor(returns[::-1])
        advantage = target - values
        actor_loss = -(dist.log_prob(actions).sum(-1) * advantage.detach()).mean()
        critic_loss = torch.nn.functional.smooth_l1_loss(values, target)
        self.optimizer.zero_grad()
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        return {
            "TI_MARL/manager_blocks": len(decisions),
            "TI_MARL/manager_value_loss": float(critic_loss.detach()),
        }
