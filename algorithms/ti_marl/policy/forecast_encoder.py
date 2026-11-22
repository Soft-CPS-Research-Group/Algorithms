"""Preserve the order of the 24-hour operating and incentive curves."""

import torch
from torch import nn

from algorithms.ti_marl.runtime.forecasts import HORIZON


class ForecastEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(7 * HORIZON + 7, d_model), nn.Tanh())

    def forward(self, snapshot, agent_ids):
        reference = self.projection[0].weight
        by_site = {row.site: row for row in getattr(snapshot, "outlooks", ())}
        incentive = getattr(snapshot, "incentive", ()) or (0.0,) * HORIZON
        if not agent_ids:
            return reference.new_zeros((0, reference.shape[0]))
        rows = []
        tails = []
        available = []
        for site in agent_ids:
            curve = by_site.get(site)
            available.append(curve is not None)
            if curve is None:
                rows.append(((0.0,) * HORIZON,) * 7)
                tails.append((0.0,) * 7)
            else:
                channels = (*curve.values, *curve.quality)
                rows.append((*[v[:HORIZON] for v in channels], incentive))
                offset = (snapshot.clock_seconds % 900) / 900
                tails.append((*[v[HORIZON] if len(v) > HORIZON else 0.0 for v in channels], offset))
        batch = reference.new_tensor(rows)
        # Only compress physical kW channels; preserve prices and offsets.
        batch[:, 1:3] = torch.log1p(batch[:, 1:3].clamp_min(0))
        tail = reference.new_tensor(tails)
        tail[:, 1:3] = torch.log1p(tail[:, 1:3].clamp_min(0))
        return (
            self.projection(torch.cat((batch.flatten(1), tail), dim=1))
            * reference.new_tensor(available)[:, None]
        )
