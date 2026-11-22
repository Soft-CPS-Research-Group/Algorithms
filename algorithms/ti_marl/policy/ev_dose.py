"""Continuous EV power readout used at inference and during training."""

import torch
from torch import nn

FEATURES = 11


def scalar_head(inputs, width):
    net = nn.Sequential(
        nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ContinuousDose(nn.Module):
    """Learned piecewise-linear power response for the low-energy end of an EV stay.

    Knots are in observed remaining service, not in output power. Every interior
    power in the physical interval remains expressible. No table value is set
    to the requested dose. The original head serves larger remaining requests.
    """

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.curve = nn.Parameter(torch.full((43,), 0.675))
        self.terminal = nn.Parameter(torch.full((43,), 0.675))

    def forward(self, x):
        # Interpolate in units of one full-power quarter of energy.
        need = x[..., 0]
        location = (20.0 * need).clamp(0.0, 41.9999)
        index = location.long()
        weight = location - index
        last = (x[..., 1] <= 0.250001) & (need > 1.0)
        low_value = torch.where(last, self.terminal[index], self.curve[index])
        high_value = torch.where(last, self.terminal[index + 1], self.curve[index + 1])
        value = low_value + (high_value - low_value) * weight
        lo, hi = x[..., -2], x[..., -1]
        unit = ((value - lo) / (hi - lo).clamp_min(1e-8)).clamp(1e-6, 1.0 - 1e-6)
        original = self.original(x).squeeze(-1).clamp(-14.0, 14.0).sigmoid()
        blend = ((2.1 - need) / 0.25).clamp(0.0, 1.0)
        combined = unit * blend + original * (1.0 - blend)
        return torch.logit(combined.clamp(1e-6, 1.0 - 1e-6)).unsqueeze(-1)
