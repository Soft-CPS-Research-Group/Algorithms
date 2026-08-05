"""Shared CityLearn electric-vehicle extraction helpers."""

from __future__ import annotations

import hashlib

import numpy as np


def deterministic_ev_initial_soc(
    *,
    schema_random_seed: int,
    electric_vehicle_id: str,
) -> float:
    """Mirror CityLearn's fallback when an EV has no configured initial SOC."""

    seed_source = f"{int(schema_random_seed)}:{electric_vehicle_id}:initial_soc"
    deterministic_seed = int(
        hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8],
        16,
    )
    return float(np.random.RandomState(deterministic_seed).uniform(0.0, 1.0))


__all__ = ["deterministic_ev_initial_soc"]
