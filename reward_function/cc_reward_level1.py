"""CC Level-1 community reward function.

Objective: teach the global community signal to reduce import cost and
peak import, while penalising unnecessary export (wasted PV surplus).

Only uses community-aggregated variables — exactly the information the
Level-1 CC observes. No per-building terms, no battery/EV specifics.

Formula (per step)
------------------
    import_kwh  = max(community_net, 0)
    export_kwh  = max(-community_net, 0)

    community_cost     = import_kwh × price
    peak_penalty       = max(import_kwh − peak_reference, 0)
    export_penalty     = export_kwh

    reward = −cost_weight   × community_cost
             −peak_weight   × peak_penalty
             −export_weight × export_penalty

The ``peak_reference`` is the rolling mean of community import over the
last ``peak_window`` steps (default 96 = 24 h at 15-min resolution).
This makes the penalty dynamic: it fires when the current step's import
exceeds the *recent average*, rewarding reduction of unusual peaks.

Return value
------------
Returns the same scalar split equally across buildings so that the CC's
``sum(rewards)`` accumulates the full community signal.
"""

from __future__ import annotations

from collections import deque
from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CCRewardLevel1(RewardFunction):
    """Community-aggregate reward for the Level-1 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        cost_weight:   float = 1.0,
        peak_weight:   float = 0.5,
        export_weight: float = 0.1,
        peak_window:   int   = 96,    # steps — default 24 h at 15 min
        **kwargs,
    ) -> None:
        super().__init__(env_metadata, **kwargs)
        self._cost_weight   = float(cost_weight)
        self._peak_weight   = float(peak_weight)
        self._export_weight = float(export_weight)
        self._import_buf    = deque(maxlen=int(peak_window))

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):
            return default
        return v

    # ── main interface ────────────────────────────────────────────────────────

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        if not observations:
            return []

        # ── Aggregate community metrics from per-building building_ops obs ──
        # Each building contributes its own net electricity consumption.
        # Summing across buildings gives the community total.
        community_net = sum(
            self._safe(obs.get("net_electricity_consumption")) for obs in observations
        )
        community_import = max(community_net, 0.0)   # kWh drawn from grid this step
        community_export = max(-community_net, 0.0)  # kWh pushed to grid this step

        price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)

        # ── Community cost: import × price ──────────────────────────────────
        community_cost = community_import * price

        # ── Peak penalty: excess above rolling mean import ───────────────────
        # Fires only when this step's import exceeds recent average.
        # Trains the agent to flatten the import curve, not just reduce it.
        self._import_buf.append(community_import)
        peak_ref   = float(sum(self._import_buf) / len(self._import_buf))
        peak_extra = max(community_import - peak_ref, 0.0)

        # ── Export penalty: wasted generation ────────────────────────────────
        # Small weight — exporting is not as bad as peak import, but the
        # agent should prefer self-consumption over sending surplus to grid.
        export_penalty = community_export

        # ── Combined scalar reward ───────────────────────────────────────────
        scalar = (
            -self._cost_weight   * community_cost
            - self._peak_weight  * peak_extra
            - self._export_weight * export_penalty
        )

        # Split equally so CC.sum(rewards) = scalar
        per_building = scalar / len(observations)
        return [per_building] * len(observations)
