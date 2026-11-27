"""Near-term passive meter envelope for sites with non-interruptible programmes."""

import math
from collections.abc import Mapping

import numpy as np


class ProgrammeMeterEnvelope:
    """Check prevalidated delivery-quarter bounds against subsequent paired meters."""

    def __init__(self):
        self.last_block = None
        self.first_slot = None
        self.misses = np.zeros(2)
        self.previous_all_hours = None
        self.all_hours_misses = np.zeros(2)

    def forecast(self, record, block, completed_export, floor_kw):
        now = block * 900
        info = record.get("programme_meter_envelope")
        if not isinstance(info, Mapping) or not all(
            record.get(k) == now
            for k in ("issued_at_seconds", "received_at_seconds", "valid_from_seconds")
        ):
            return ()
        times = [info.get(key) for key in ("history_start", "fit_end", "validation_end")]
        if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in times):
            return ()
        start, fit, end = times
        if start < 0 or fit - start < 28 * 86400 or end - fit < 28 * 86400 or end > now:
            return ()
        try:
            curves = np.asarray([info["import_upper_kw"], info["export_upper_kw"]], dtype=float)
            all_hours = np.asarray(info["all_quarters_upper_kw"], dtype=float)
        except (KeyError, ValueError, TypeError):
            return ()
        if (
            curves.shape != (2, 13)
            or not np.isfinite(curves).all()
            or np.any(curves.sum(axis=0) < 0)
            or all_hours.shape != (2,)
            or not np.isfinite(all_hours).all()
            or np.any(all_hours < curves.max(axis=1))
        ):
            return ()
        if block != self.last_block:
            if (
                self.last_block == block - 1
                and completed_export is not None
                and math.isfinite(completed_export)
            ):
                actual = np.array([-completed_export, completed_export])
                self.misses = np.maximum(self.misses, actual - self.first_slot)
                self.all_hours_misses = np.maximum(
                    self.all_hours_misses, actual - self.previous_all_hours
                )
            self.first_slot = curves[:, 0].copy()
            self.previous_all_hours = all_hours
            self.last_block = block
        margin = np.maximum(self.misses, 2.0 * floor_kw)
        # Slot zero remains under the actual meter constraints. Only complete
        # future quarters in the three-hour certificate can use these bounds.
        local = curves[:, 1:] + margin[:, None]
        # A miss of a particular delivery quarter need not have exceeded the
        # separately observed maximum across *all* hours. Monitor both envelopes.
        global_upper = all_hours + np.maximum(self.all_hours_misses, 2.0 * floor_kw)
        return tuple(map(tuple, np.minimum(local, global_upper[:, None])))
