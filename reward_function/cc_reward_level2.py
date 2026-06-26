"""CC Level-2 community reward function.

Extends CCRewardLevel1 with a per-building EV service term so the CC is
penalised whenever its high price signals cause a building's EV to miss its
required SoC by departure time.

    reward_t = community_term − w_ev · ev_penalty_t

Community term (identical to CCRewardLevel1):
    community_term = − w_cost   * cost_norm
                     − w_peak   * peak_import_norm
                     − w_export * export_norm

EV penalty (average urgency-weighted SoC deficit across all buildings):
    For each building i with a connected EV:
        gap_i      = max(required_soc_i − battery_soc_i, 0)      ∈ [0, 1]
        urgency_i  = clip(1 − hours_until_departure_i / H, 0, 1)  ∈ [0, 1]
        harm_i     = urgency_i · gap_i                             ∈ [0, 1]
    Buildings without a connected EV contribute harm_i = 0.

    ev_penalty_t = (Σ_i harm_i) / N_buildings

Design notes
------------
* urgency ramps linearly from 0 (H+ hours before departure) to 1 (at departure).
  Default H = 4 h — gives the RBC 4 hours of warning before the signal matters.
* Dividing by N_buildings keeps ev_penalty on the same scale as the per-building
  community terms regardless of how many EVs are present.
* w_ev = 0.5 default — EV safety carries half the weight of cost, comparable
  to the community signal at a mildly bad timestep but clearly secondary.

Return value
------------
Same scalar split equally across buildings (same pattern as CCRewardLevel1).
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

import mlflow
from citylearn.reward_function import RewardFunction


class CCRewardLevel2(RewardFunction):
    """Community + EV-service reward for the Level-2 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        # Community weights (mirror CCRewardLevel1)
        w_cost:   float = 1.0,
        w_peak:   float = 0.3,
        w_export: float = 0.1,
        # EV service weight
        w_ev:     float = 0.5,
        # Community reference values (15-min dataset, 17 buildings)
        target_import:    float = 4.14,   # kWh — p75 community import
        reference_cost:   float = 1.045,  # p90 community cost
        reference_peak:   float = 2.72,   # p90 peak excess squared
        reference_export: float = 7.52,   # kWh — p90 community export
        # EV urgency horizon in hours
        urgency_horizon:  float = 4.0,    # harm starts H hours before departure
        **kwargs,
    ) -> None:
        super().__init__(env_metadata, **kwargs)
        self._w_cost   = float(w_cost)
        self._w_peak   = float(w_peak)
        self._w_export = float(w_export)
        self._w_ev     = float(w_ev)

        self._target_import    = float(target_import)
        self._ref_cost         = max(float(reference_cost),   1e-8)
        self._ref_peak         = max(float(reference_peak),   1e-8)
        self._ref_export       = max(float(reference_export), 1e-8)

        self._urgency_horizon  = max(float(urgency_horizon), 1e-6)

        # MLflow logging — sample every N calls to avoid flooding
        self._log_interval: int = 50
        self._step: int = 0

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

    def _ev_harm(self, obs: Mapping[str, Any]) -> float:
        """Urgency-weighted SoC deficit for a single building's EVs.

        Returns a value in [0, 1].  Zero if no EV is connected.
        If a building has multiple chargers, harm is averaged over them.
        """
        ev_dict = obs.get("electric_vehicles_chargers_dict")
        if not ev_dict:
            return 0.0

        total = 0.0
        count = 0
        for ev in ev_dict.values():
            if not ev.get("connected", False):
                continue
            soc   = self._safe(ev.get("battery_soc"),           default=1.0)
            req   = self._safe(ev.get("required_soc"),          default=0.0)
            hours = self._safe(ev.get("hours_until_departure"), default=self._urgency_horizon)

            gap      = max(req - soc, 0.0)                                   # [0, 1]
            urgency  = max(1.0 - hours / self._urgency_horizon, 0.0)         # [0, 1]
            total   += urgency * gap
            count   += 1

        return total / count if count > 0 else 0.0

    # ── main interface ────────────────────────────────────────────────────────

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        if not observations:
            return []

        n = len(observations)

        # ── Community aggregates ─────────────────────────────────────────────
        community_net = sum(
            self._safe(obs.get("net_electricity_consumption")) for obs in observations
        )
        import_t = max(community_net, 0.0)
        export_t = max(-community_net, 0.0)

        price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)

        # ── Community term (identical to CCRewardLevel1) ─────────────────────
        cost_norm   = (import_t * price) / self._ref_cost
        peak_norm   = (max(import_t - self._target_import, 0.0) ** 2) / self._ref_peak
        export_norm = export_t / self._ref_export

        community_term = (
            - self._w_cost   * cost_norm
            - self._w_peak   * peak_norm
            - self._w_export * export_norm
        )

        # ── EV service term ──────────────────────────────────────────────────
        ev_harms = [self._ev_harm(obs) for obs in observations]
        ev_harm_sum = sum(ev_harms)
        ev_penalty  = ev_harm_sum / n

        n_ev_connected = sum(
            1 for obs in observations
            if any(ev.get("connected", False)
                   for ev in (obs.get("electric_vehicles_chargers_dict") or {}).values())
        )
        n_ev_urgent = sum(
            1 for h in ev_harms if h > 0.0
        )

        # ── Combined scalar ──────────────────────────────────────────────────
        scalar = community_term - self._w_ev * ev_penalty

        # ── MLflow logging (sampled) ─────────────────────────────────────────
        self._step += 1
        if mlflow.active_run() and self._step % self._log_interval == 0:
            mlflow.log_metrics(
                {
                    "CC2_rf/community_term":  community_term,
                    "CC2_rf/ev_penalty":      self._w_ev * ev_penalty,
                    "CC2_rf/ev_harm_sum":     ev_harm_sum,
                    "CC2_rf/cost_norm":       cost_norm,
                    "CC2_rf/peak_norm":       peak_norm,
                    "CC2_rf/export_norm":     export_norm,
                    "CC2_rf/n_ev_connected":  float(n_ev_connected),
                    "CC2_rf/n_ev_urgent":     float(n_ev_urgent),
                    "CC2_rf/total_reward":    scalar,
                },
                step=self._step,
            )

        per_building = scalar / n
        return [per_building] * n
