"""Community Coordinator reward function.

Objective: teach the CC to coordinate the community toward lower-cost grid
usage. The CC emits one signal per building; in Phase 1 that signal drives an
internal RBC that charges/discharges each building's battery (and gates
flexible EV charging). This reward must therefore reward the *thing the CC
controls* — battery arbitrage — not total community cost.

Why not total cost?
-------------------
The previous reward was ``-(max(net, 0) * price)`` summed over buildings.
That is dominated by the *uncontrollable baseline load* (non-shiftable load is
~3-10x the battery throughput). The CC's battery action moves net by ~5 kW
against a 50-85 kW baseline, so the arbitrage gradient drowns in baseline
noise. Worse, per-step cost is myopic: charging ALWAYS looks bad (raises
import now) and discharging ALWAYS looks good, so the policy cannot learn
"store cheap, spend dear". Empirically the CC learned the *wrong* direction
(corr(price, o1) = +0.37 — it charged at peak price).

The fix: isolate the controllable signal (the battery's own grid draw) and
reward arbitrage against a local price reference.

    ref_price = mean(price, price_pred_1, price_pred_2, price_pred_3)
    arbitrage = -battery_consumption * (price - ref_price)

    battery_consumption = electrical_storage_electricity_consumption
        > 0 when charging (drawing from grid)
        < 0 when discharging (returning to grid)

Behaviour
---------
- charge when price < ref  → +reward  (store while cheap)        ✓
- charge when price > ref  → -reward  (don't buy dear)           ✓
- discharge when price > ref → +reward (spend stored energy dear) ✓
- discharge when price < ref → -reward (don't waste cheap energy) ✓
- do nothing               →  0       (acting-cheap beats idle → no collapse)

This mirrors how the battle-tested production rewards (V2G_Reward,
CostHardConstraint V46) work: they do NOT rely on raw per-step cost to teach
storage — they add explicit *shaping* terms (SoC targets, transfer bonuses)
that carry the learning signal. Here the shaping term is price-arbitrage on
the battery.

A small residual community-cost term (off by default) can be re-enabled to
keep the CC grounded in absolute cost once the direction is learned.

NOTE: reads CityLearn *building_ops* observations. Keys confirmed present in
the building_ops format (see reward_obs reference):
``electrical_storage_electricity_consumption``, ``electricity_pricing``,
``electricity_pricing_predicted_{1,2,3}``, ``net_electricity_consumption``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class CCReward(RewardFunction):
    """Battery-arbitrage shaping reward for the Community Coordinator."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        arbitrage_weight: float = 1.0,
        cost_weight: float = 0.0,
        export_credit_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        # arbitrage_weight: weight on the controllable battery-arbitrage term
        #                   (the primary learning signal).
        # cost_weight:      weight on the absolute community-cost term
        #                   (off by default — grounds the policy once direction
        #                   is learned; turning it on early re-introduces the
        #                   baseline-load noise that broke the old reward).
        self._arbitrage_weight = float(arbitrage_weight)
        self._cost_weight = float(cost_weight)
        self._export_credit_ratio = float(export_credit_ratio)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return default
        return parsed

    def _ref_price(self, obs: Mapping[str, Union[int, float]], price: float) -> float:
        """Local price reference = mean of current + predicted prices.

        Uses the price forecast already present in the observation, so the
        reference is stateless and needs no rolling history. Falls back to the
        current price if forecasts are missing.
        """
        prices = [price]
        for k in (
            "electricity_pricing_predicted_1",
            "electricity_pricing_predicted_2",
            "electricity_pricing_predicted_3",
        ):
            if k in obs:
                prices.append(self._safe_float(obs.get(k), default=price))
        return sum(prices) / float(len(prices))

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        """Return per-building shaped rewards.

        The CC's update loop sums these across buildings into the
        community-level scalar its centralized critic learns.
        """
        if not observations:
            return []

        rewards = []
        for obs in observations:
            price = max(self._safe_float(obs.get("electricity_pricing")), 0.0)
            ref = self._ref_price(obs, price)

            # ── Controllable arbitrage term (primary signal) ──────────────
            # Battery's own grid draw: + charging, - discharging.
            batt = self._safe_float(
                obs.get("electrical_storage_electricity_consumption")
            )
            # -draw * (price - ref):
            #   charge (draw>0) below ref (price<ref) → +  (store cheap)
            #   charge        above ref             → -  (buy dear)
            #   discharge (draw<0) above ref        → +  (spend dear)
            arbitrage = -batt * (price - ref)

            # ── Optional absolute-cost grounding term (off by default) ────
            net = self._safe_float(obs.get("net_electricity_consumption"))
            import_cost = max(net, 0.0) * price
            export_credit = max(-net, 0.0) * price * self._export_credit_ratio
            cost = -(import_cost - export_credit)

            rewards.append(
                self._arbitrage_weight * arbitrage + self._cost_weight * cost
            )

        return rewards
