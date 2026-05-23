"""Building Agent reward function — Phase 1 (constraint learning).

Goal: teach the agent to satisfy the EV departure constraint before optimising
for cost arbitrage or carbon. Three components only:

  1. Building cost  — net consumption × price signal.
  2. EV streaming deficit  — per-step gradient while soc < required.
  3. EV departure penalty  — large one-time hit when EV leaves with soc < req.
  4. V2G block  — immediate penalty if agent discharges EV while soc < req.

Dropped for Phase 1 (add back in Phase 2):
  - Carbon intensity
  - Community import benefit
  - Building battery SoC safety
  - Schedule-feasibility floor (replaced by simpler V2G block)

Departure is detected when connected=True AND hours_until_departure==0.0
(the final step of a session before the EV leaves).

NOTE: The reward function receives observations from CityLearn's building_ops layer.
EV data lives in obs["electric_vehicles_chargers_dict"][charger_id], NOT in flat
prefixed keys like "electric_vehicle_charger_{id}_*".
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

from citylearn.reward_function import RewardFunction


class BAReward(RewardFunction):
    """Phase 1 reward for Building Agents — EV constraint satisfaction first."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        building_cost_weight: float = 1.0,
        ev_deficit_weight: float = 10.0,
        ev_departure_weight: float = 50.0,
        ev_v2g_block_weight: float = 20.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self._alpha              = building_cost_weight
        self._ev_deficit_weight  = float(ev_deficit_weight)
        self._ev_departure_weight = float(ev_departure_weight)
        self._ev_v2g_block_weight = float(ev_v2g_block_weight)

    def calculate(
        self,
        observations: List[Mapping[str, Union[int, float]]],
    ) -> List[float]:
        rewards = []
        for obs in observations:
            net   = self._safe_float(obs.get("net_electricity_consumption"))
            price = self._safe_float(obs.get("electricity_pricing"))
            building_cost = -net * price

            ev_penalty = self._ev_service_penalty(obs)

            rewards.append(self._alpha * building_cost + ev_penalty)
        return rewards

    # ------------------------------------------------------------------
    # EV service penalty
    # ------------------------------------------------------------------

    def _ev_service_penalty(self, obs: Mapping[str, Union[int, float]]) -> float:
        """Three EV penalties per connected charger.

        A. V2G block: immediate penalty if agent discharges while soc < req.
           Couples penalty directly to action — no temporal delay.
        B. Streaming deficit: per-step gradient while soc < req.
           Urgency ramps from 1/6 (floor) to 1/hours as departure nears (knee at 6h).
        C. Departure penalty: large one-time hit on final session step (hours==0).
           Provides clean terminal signal for credit assignment.
        """
        ev_chargers = obs.get("electric_vehicles_chargers_dict")
        if not ev_chargers:
            return 0.0

        penalty = 0.0

        for _charger_id, data in ev_chargers.items():
            if not (isinstance(data, dict) and data.get("connected")):
                continue

            soc      = self._safe_float(data.get("battery_soc"),     default=1.0)
            req      = self._safe_float(data.get("required_soc"),     default=0.8)
            last_kwh = self._safe_float(data.get("last_charged_kwh"), default=0.0)

            raw_hours = data.get("hours_until_departure")
            if raw_hours is None or self._safe_float(raw_hours, default=-1.0) < 0:
                hours_until_departure = float("inf")
            else:
                hours_until_departure = max(self._safe_float(raw_hours, default=0.0), 0.0)

            deficit = max(req - soc, 0.0)

            # A. V2G block — penalise discharge when below required SoC.
            if last_kwh < 0 and soc < req:
                penalty += last_kwh * self._ev_v2g_block_weight

            # B. Streaming deficit gradient.
            if deficit > 0 and hours_until_departure != float("inf"):
                urgency = max(1.0 / max(hours_until_departure, 0.5), 1.0 / 6.0)
                penalty += -deficit * self._ev_deficit_weight * urgency

            # C. Departure terminal penalty — fires on last step of session.
            if hours_until_departure == 0.0 and deficit > 0:
                penalty += -deficit * self._ev_departure_weight

        return penalty

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed != parsed or parsed in (float("inf"), float("-inf")):
            return default
        return parsed
