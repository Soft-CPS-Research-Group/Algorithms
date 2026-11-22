"""Calibrate the passive export forecast against metered demand and solar together."""

import numpy as np


class PassiveExportFeedback:
    """Maximum observed export underprediction per lead for the next three hours.

    Both meters must span the same complete quarter. A lead becomes eligible
    after a full day of consecutive pairs. Missing data interrupt eligibility
    without erasing any of the historical extrema.
    """

    LEADS = np.arange(1, 13)
    MIN_PAIRS = 96

    def __init__(self):
        size = len(self.LEADS) + 2
        self.origins = np.full(size, -1, dtype=np.int64)
        self.forecasts = np.full((size, len(self.LEADS)), np.nan)
        self.upper = np.zeros(len(self.LEADS))
        self.consecutive = np.zeros(len(self.LEADS), dtype=int)
        self.last_block = None

    def observe(self, block, completed, forecasts):
        size = len(self.origins)
        if self.last_block is not None and block != self.last_block + 1:
            self.consecutive.fill(0)
        # At the start of block b, the latest complete paired meter is b-1.
        origins = block - 1 - self.LEADS
        slots = origins % size
        prior = self.forecasts[slots, self.LEADS - 1]
        valid = (origins >= 0) & (self.origins[slots] == origins) & np.isfinite(prior)
        if completed is None:
            valid.fill(False)
        else:
            self.upper[valid] = np.maximum(self.upper[valid], completed - prior[valid])
        self.consecutive = np.where(valid, self.consecutive + 1, 0)
        self.origins[block % size] = block
        self.forecasts[block % size] = forecasts
        self.last_block = block

    def margins(self, declared_floor):
        if declared_floor is None:
            return ()
        return tuple(
            max(float(margin), sum(declared_floor)) if count >= self.MIN_PAIRS else None
            for margin, count in zip(self.upper, self.consecutive)
        )
