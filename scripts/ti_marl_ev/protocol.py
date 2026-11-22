"""Shared input paths and date partitions for the controlled 2023 V2 EV study.

Set TI_MARL_EV_BASE to the directory containing the two starting checkpoints
(models/v2-{32,64}.pth) and inputs/probe-trace.jsonl.gz. TI_MARL_EV_SCHEMA
can select the Core-30 Normal 2023 schema directly. TI_MARL_EV_RESULTS is the
working directory for labels, fitted weights and teacher-free PPO results.
"""

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from algorithms.ti_marl.agent import TIMARL
from algorithms.ti_marl.experiments.ev_pretraining import ACTORS
from scripts.train_ti_marl_ev_sessions import Session, make_cases

HERE = Path(os.environ.get("TI_MARL_EV_RESULTS", "runs/ti-marl-2023-ev-pretrain"))
PRE = Path(os.environ.get("TI_MARL_EV_BASE", "runs/ti-marl-2023-08h"))
TRACE = PRE / "inputs/probe-trace.jsonl.gz"


def schema_path():
    override = os.environ.get("TI_MARL_EV_SCHEMA")
    if override:
        return Path(override)
    return Path(json.loads((PRE / "comparison.json").read_text())["dataset_schema"])


RESERVED_DATES = (
    "2023-12-10",
    "2023-10-27",
    "2023-09-30",
    "2023-12-24",
    "2023-10-20",
    "2023-09-06",
    "2023-09-01",
    "2023-12-14",
    "2023-12-17",
    "2023-11-19",
    "2023-09-09",
    "2023-09-18",
    "2023-11-21",
    "2023-09-23",
    "2023-10-24",
    "2023-10-31",
    "2023-09-21",
    "2023-10-10",
    "2023-09-27",
    "2023-09-02",
)


def split():
    train, _ = make_cases(schema_path(), 1536, 48)
    dates = sorted(
        {s.day for s in train}, key=lambda d: hashlib.sha256(("ev-dev:" + d).encode()).hexdigest()
    )
    dev_days = set(dates[:22])
    fit = [s for s in train if s.day not in dev_days]
    # Preserve complete four-session batches, without moving a dev date back to training.
    fit = fit[: (len(fit) // 4) * 4]
    dev = [s for s in train if s.day in dev_days and s.variable in ("price", "solar")][:64]
    dev = [replace(s, early=True) for s in dev]
    _, pool = make_cases(schema_path(), 0, 100)
    new = [s for s in pool if s.day in RESERVED_DATES]
    assert len(new) == 40 and len({s.day for s in new}) == 20
    assert not ({s.day for s in fit} & ({s.day for s in dev} | {s.day for s in new}))
    assert not ({s.day for s in dev} & {s.day for s in new})
    stress = [
        replace(s, request_kwh=s.request_kwh + (0.23 if i % 4 < 2 else -0.27))
        for i, s in enumerate(new)
    ]
    canonical = Session("fixed", "08:00", 28, 4, 10.0, 3.0, 0.08, 0.24, 4.0, "price", True)
    return (
        fit,
        dev,
        {"new": new, "off_grid": stress, "08h": [canonical, replace(canonical, variable="solar")]},
    )


def load(width, seed, variant="continuous"):
    checkpoint = torch.load(
        PRE / "models" / f"v2-{width}.pth", map_location="cpu", weights_only=False
    )
    with patch("algorithms.ti_marl.policy.joint.JointActor", ACTORS[variant]):
        agent = TIMARL(checkpoint["configuration"])
    miss, extra = agent.actor.load_state_dict(checkpoint["actor"], strict=False)
    assert not extra and all(k.startswith(("ev_gate.", "ev_watts.", "ev_band.")) for k in miss)
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor.step_seconds = 900.0
    assert agent.manager is None and agent.reward_profile == "thesis_common_reward_v1"
    torch.manual_seed(seed)
    # Independent seeded initial readouts; the last layer remains exactly neutral.
    for head in (agent.actor.ev_gate, agent.actor.ev_watts):
        for layer in head:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    if variant in ("boundary", "wide"):
        for layer in agent.actor.ev_band:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        nn.init.zeros_(agent.actor.ev_band[-1].weight)
        with torch.no_grad():
            agent.actor.ev_band[-1].bias.copy_(torch.tensor([-9.0, 0.0, -9.0]))
    return agent


@dataclass(frozen=True)
class OvernightPrice(Session):
    start_cheap: int = 2
    cheap_length: int = 4

    def window(self, t):
        begin = (
            self.start_cheap if self.early else self.quarters - self.start_cheap - self.cheap_length
        )
        return (self.low if begin <= t < begin + self.cheap_length else self.high), 0.0, 0.0


def next_price_features(x):
    # Two-level announced tariffs in this controlled study. Count potential quarter-hour
    # opportunities before departure; these are outlook summaries, not reserved capacity.
    x = np.asarray(x)
    n = x[..., 1] * 4
    spread = x[..., 8] + x[..., 9]
    ratio = x[..., 4] / np.maximum(spread, 1e-9)
    cheaper = np.where(x[..., 8] > 1e-7, np.maximum(0.0, n * ratio), 0.0)
    no_dearer = np.where(x[..., 9] > 1e-7, np.maximum(0.0, n * (1.0 + ratio) - 1.0), n - 1.0)
    pv_later = np.maximum(0.0, n * (x[..., 5] - x[..., 6]) - x[..., 5])
    return np.stack(
        (
            np.clip(x[..., 0] - cheaper, -6, 6),
            np.clip(x[..., 0] - no_dearer, -6, 6),
            np.clip(pv_later, 0, 6),
            np.clip(x[..., 0] - pv_later, -6, 6),
        ),
        axis=-1,
    )


def wide_sessions(pool, seed, count):
    """Variable overnight plug-in duration, cheap-window location and requested energy."""
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        s = pool[rng.integers(len(pool))]
        n = int(rng.choice([12, 16, 20, 24, 32, 36, 40]))
        length = int(rng.integers(3, min(7, n // 2) + 1))
        pad = int(rng.integers(0, max(1, n // 2 - length + 1)))
        # Anchor to the physical 2023 date and vary the morning departure.
        from datetime import datetime
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/Lisbon")
        hr = int(rng.choice([6, 7, 8, 9]))
        dt = datetime.fromisoformat(s.day + f"T{hr:02d}:00:00").replace(tzinfo=tz)
        origin = datetime(2023, 1, 1, tzinfo=tz)
        start = round((dt.timestamp() - origin.timestamp()) / 900) - n
        # Skip overnight starts crossing into another (potentially reserved) date.
        # The caller verifies both physical calendar dates where training is concerned.
        request = float(rng.choice([2.0, 2.5, 3.0, 3.23, 3.5, 3.73, 4.0, 4.5, 5.0]))
        result.append(
            OvernightPrice(
                s.day,
                dt.strftime("%H:%M"),
                start,
                n,
                10.0,
                request,
                s.low,
                s.high,
                0.0,
                "price",
                bool(rng.integers(2)),
                pad,
                length,
            )
        )
    return result


def fit_nights(pool, seed, count):
    from datetime import datetime, timedelta

    # Every physical interval including the previous evening belongs to training dates.
    approved = {s.day for s in pool}
    dates = [
        s
        for s in pool
        if (datetime.fromisoformat(s.day) - timedelta(days=1)).date().isoformat() in approved
    ]
    return wide_sessions(dates, seed, count)
