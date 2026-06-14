"""CC Level-1 community reward function.

Implements the supervisor's reward design (Phase 1):

    reward_t = - w_cost   * cost_norm
               - w_peak   * peak_import_norm
               - w_export * export_norm

Where:
    import_t           = max(community_net, 0)
    community_cost_t   = import_t * grid_price_t
    peak_penalty_t     = max(0, import_t - target_import) ** 2
    export_t           = max(-community_net, 0)

    cost_norm          = community_cost_t  / reference_cost
    peak_import_norm   = peak_penalty_t    / reference_peak_penalty
    export_norm        = export_t          / reference_export

Reference values derived from the 15-min dataset (17 buildings):
    target_import      = 4.14  kWh  (p75 community import)
    reference_cost     = 1.045      (p90 community cost)
    reference_peak     = 2.72       (p90 excess squared)
    reference_export   = 7.52  kWh  (p90 community export)

Factor penalty (factor_t - 1.0)^2 and smoothness penalty are applied
inside the agent's update() because the reward function has no access
to the CC's action.

Return value
------------
Same scalar split equally across buildings so CC.sum(rewards) = scalar.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CCRewardLevel1(RewardFunction):
    """Community-aggregate reward for the Level-1 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        w_cost:   float = 1.0,
        w_peak:   float = 0.3,
        w_export: float = 0.1,
        # Reference values from dataset (15-min, 17 buildings)
        target_import:       float = 4.14,   # kWh — p75 community import
        reference_cost:      float = 1.045,  # p90 community cost
        reference_peak:      float = 2.72,   # p90 peak excess squared
        reference_export:    float = 7.52,   # kWh — p90 community export
        **kwargs,
    ) -> None:
        super().__init__(env_metadata, **kwargs)
        self._w_cost   = float(w_cost)
        self._w_peak   = float(w_peak)
        self._w_export = float(w_export)

        self._target_import    = float(target_import)
        self._ref_cost         = max(float(reference_cost),   1e-8)
        self._ref_peak         = max(float(reference_peak),   1e-8)
        self._ref_export       = max(float(reference_export), 1e-8)

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

        # ── Community aggregates ─────────────────────────────────────────────
        community_net = sum(
            self._safe(obs.get("net_electricity_consumption")) for obs in observations
        )
        import_t = max(community_net, 0.0)
        export_t = max(-community_net, 0.0)

        price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)

        # ── Cost term ────────────────────────────────────────────────────────
        community_cost = import_t * price
        cost_norm = community_cost / self._ref_cost

        # ── Peak penalty (squared excess above target) ───────────────────────
        peak_excess   = max(import_t - self._target_import, 0.0)
        peak_penalty  = peak_excess ** 2
        peak_norm     = peak_penalty / self._ref_peak

        # ── Export penalty ───────────────────────────────────────────────────
        export_norm = export_t / self._ref_export

        # ── Combined scalar ──────────────────────────────────────────────────
        scalar = (
            - self._w_cost   * cost_norm
            - self._w_peak   * peak_norm
            - self._w_export * export_norm
        )

        per_building = scalar / len(observations)
        return [per_building] * len(observations)
