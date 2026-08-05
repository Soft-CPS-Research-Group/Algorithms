"""CC Level-1 community reward function.

Implements the supervisor's reward design (Phase 1):

    reward_t = - w_cost              * cost_norm
               - w_member_retail_cost * member_retail_cost_norm
               - w_peak              * peak_import_norm
               - w_ramp              * ramp_norm
               - w_export            * export_norm
               - w_violation         * violation_norm

Where:
    import_t           = max(community_net, 0)
    community_cost_t   = import_t * grid_price_t
    peak_penalty_t     = max(0, import_t - target_import) ** 2
    ramp_t             = abs(import_t - import_{t-1})
    export_t           = max(-community_net, 0)
    violation_t        = sum of charging_constraint_violation_kwh across buildings

    cost_norm          = community_cost_t      / reference_cost
    member_retail_cost_norm = member_retail_cost_t / reference_member_retail_cost
    peak_import_norm   = peak_penalty_t    / reference_peak
    ramp_norm          = ramp_t            / reference_ramping
    export_norm        = export_t          / reference_export
    violation_norm     = violation_t       / reference_violation

Reference values derived from the 15-min dataset (17 buildings):
    target_import      = 4.14  kWh  (p75 community import)
    reference_cost     = 1.045      (p90 community cost)
    reference_peak     = 2.72       (p90 excess squared)
    reference_ramping  = 1.878      (p90 step-to-step import change, from dataset)
    reference_export   = 7.52  kWh  (p90 community export)
    reference_violation = 1.0  kWh  (any violation is penalised at full w_violation per kWh)

These constructor defaults preserve the historical V1 recipe.  Current
experiments should pass references calibrated from their matching neutral
baseline explicitly; the V2 cost-focus configs do so.

Factor penalty (factor_t - 1.0)^2 and smoothness penalty are applied
inside the agent's update() because the reward function has no access
to the CC's action.

Return value
------------
``cost_aggregation="community_net"`` preserves the historical formula above.
When the community market is disabled, ``cost_aggregation="member_retail"``
instead uses the sum of each member's positive grid import at its retail
price.  This matches ``district_cost_total_control_eur``, which is the cost
used by the frozen-leaf CC scorecard.

``w_member_retail_cost`` is an optional second economic term.  It lets a
settlement-focused run keep ``community_net`` as its primary objective while
also discouraging regressions in the settlement-free/member-retail
counterfactual.  Its default is zero, so historical recipes are unchanged.

Same scalar split equally across buildings so CC.sum(rewards) = scalar.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


_VIOLATION_KEYS = (
    "charging_constraint_violation_kwh",   # confirmed present in current dataset
    "electrical_service_violation_kwh",    # alias used in other simulator versions
    "electrical_service_violation",
    "service_violation_kwh",
    "service_violation",
)


class CCRewardLevel1(RewardFunction):
    """Community-aggregate reward for the Level-1 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        w_cost:      float = 1.0,
        w_member_retail_cost: float = 0.0,
        w_peak:      float = 0.6,
        w_ramp:      float = 0.4,
        w_export:    float = 0.05,
        w_violation: float = 2.0,
        # Reference values from dataset (15-min, 17 buildings)
        target_import:       float = 4.14,   # kWh — p75 community import
        reference_cost:      float = 1.045,  # p90 community cost
        reference_member_retail_cost: float = 1.045,
        reference_peak:      float = 2.72,   # p90 peak excess squared
        reference_ramping:   float = 1.878,  # p90 step-to-step import change
        reference_export:    float = 7.52,   # kWh — p90 community export
        reference_violation: float = 1.0,    # kWh — 1 kWh of violation = full w_violation
        cost_aggregation: str = "community_net",
        **kwargs,
    ) -> None:
        super().__init__(env_metadata, **kwargs)
        self._w_cost      = float(w_cost)
        self._w_member_retail_cost = float(w_member_retail_cost)
        self._w_peak      = float(w_peak)
        self._w_ramp      = float(w_ramp)
        self._w_export    = float(w_export)
        self._w_violation = float(w_violation)

        self._target_import     = float(target_import)
        self._ref_cost          = max(float(reference_cost),      1e-8)
        self._ref_member_retail_cost = max(
            float(reference_member_retail_cost), 1e-8
        )
        self._ref_peak          = max(float(reference_peak),      1e-8)
        self._ref_ramping       = max(float(reference_ramping),   1e-8)
        self._ref_export        = max(float(reference_export),    1e-8)
        self._ref_violation     = max(float(reference_violation), 1e-8)
        self._cost_aggregation = str(cost_aggregation).strip().lower()
        if self._cost_aggregation not in {"community_net", "member_retail"}:
            raise ValueError(
                "CCRewardLevel1 cost_aggregation must be 'community_net' or "
                "'member_retail'"
            )
        self._prev_import       = 0.0

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

    @classmethod
    def _violation_kwh(cls, obs: Mapping[str, Any]) -> float:
        for key in _VIOLATION_KEYS:
            if key in obs:
                return max(cls._safe(obs[key]), 0.0)
        return 0.0

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

        # ── Cost term ────────────────────────────────────────────────────────
        member_retail_cost = sum(
            max(self._safe(obs.get("net_electricity_consumption")), 0.0)
            * max(self._safe(obs.get("electricity_pricing")), 0.0)
            for obs in observations
        )
        if self._cost_aggregation == "member_retail":
            community_cost = member_retail_cost
        else:
            price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)
            community_cost = import_t * price
        cost_norm = community_cost / self._ref_cost
        member_retail_cost_norm = (
            member_retail_cost / self._ref_member_retail_cost
        )

        # ── Peak penalty (squared excess above target) ───────────────────────
        peak_excess   = max(import_t - self._target_import, 0.0)
        peak_penalty  = peak_excess ** 2
        peak_norm     = peak_penalty / self._ref_peak

        # ── Ramping penalty (step-to-step import change) ─────────────────────
        ramp_norm = abs(import_t - self._prev_import) / self._ref_ramping
        self._prev_import = import_t

        # ── Export penalty ───────────────────────────────────────────────────
        export_norm = export_t / self._ref_export

        # ── Electrical violation penalty (hard constraint) ───────────────────
        total_violation = sum(self._violation_kwh(obs) for obs in observations)
        violation_norm  = total_violation / self._ref_violation

        # ── Combined scalar ──────────────────────────────────────────────────
        scalar = (
            - self._w_cost      * cost_norm
            - self._w_member_retail_cost * member_retail_cost_norm
            - self._w_peak      * peak_norm
            - self._w_ramp      * ramp_norm
            - self._w_export    * export_norm
            - self._w_violation * violation_norm
        )

        per_building = scalar / len(observations)
        return [per_building] * len(observations)
