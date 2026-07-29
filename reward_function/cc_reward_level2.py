"""CC Level-2 community reward function.

Extends CCRewardLevel1 with a per-building EV service term so the CC is
penalised whenever its high price signals cause a building's EV to miss its
required SoC by departure time.

    reward_t = community_scalar − w_ev · ev_penalty_t

Community scalar (inherited verbatim from CCRewardLevel1, all 5 terms):
    − w_cost      * cost_norm
    − w_peak      * peak_import_norm
    − w_ramp      * ramp_norm
    − w_export    * export_norm
    − w_violation * violation_norm

By subclassing CCRewardLevel1 and calling its ``_community_scalar`` we guarantee
the community reward can never drift from the Level-1 definition: any change to
the Level-1 terms (e.g. the ramp/violation penalties) is inherited automatically.

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
import numpy as np

from reward_function.cc_reward_level1 import CCRewardLevel1


class CCRewardLevel2(CCRewardLevel1):
    """Community (CCRewardLevel1) + EV-service reward for the Level-2 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        # EV service weight
        w_ev: float = 0.5,
        # EV urgency horizon in hours
        urgency_horizon: float = 4.0,    # harm starts H hours before departure
        # Credit-assignment mode:
        #   "aggregate"    → every building receives the identical community
        #                    scalar (Phase-2-as-shipped; collapses to Level 1).
        #   "per_building" → each building receives its own reward so PPO gets a
        #                    gradient to differentiate multipliers per building.
        reward_mode: str = "aggregate",
        # Weight on the zero-mean per-building spatial cost-credit term
        # (only used when reward_mode == "per_building").
        w_spatial: float = 1.0,
        **kwargs,
    ) -> None:
        # All community weights / reference values (w_cost, w_peak, w_ramp,
        # w_export, w_violation, target_import, reference_*) are forwarded to
        # CCRewardLevel1 via **kwargs, so Level 2 inherits the exact Level-1
        # community scalar — including the ramp and violation terms.
        super().__init__(env_metadata, **kwargs)

        self._w_ev            = float(w_ev)
        self._urgency_horizon = max(float(urgency_horizon), 1e-6)

        mode = str(reward_mode).strip().lower()
        if mode not in ("aggregate", "per_building"):
            raise ValueError(
                f"CCRewardLevel2: unknown reward_mode '{reward_mode}' "
                "(expected 'aggregate' or 'per_building')."
            )
        self._reward_mode = mode
        self._w_spatial   = float(w_spatial)

        # MLflow logging — sample every N calls to avoid flooding
        self._log_interval: int = 50
        self._step: int = 0

    # ── helpers ──────────────────────────────────────────────────────────────

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

    # ── per-building credit assignment ─────────────────────────────────────────

    def _per_building_rewards(
        self,
        observations: List[Mapping[str, Union[int, float]]],
        community_term: float,
        ev_harms: List[float],
    ) -> List[float]:
        """Distribute the reward per building so PPO can credit each multiplier.

        r_i = community_term / N                       # shared community objective
              + w_spatial · (spatial_i − mean_spatial) # zero-mean spatial cost credit
              − w_ev · harm_i                          # per-building EV attribution

        The spatial term is each building's own import-cost responsibility,
        normalised onto the community cost scale (community cost ≈ N × per-building
        cost) and mean-centred so it sums to zero — it redistributes credit
        *without* changing the total community objective, telling the CC *which*
        buildings to target rather than moving the overall reward level. The EV
        term is attributed to the building whose EV is actually at risk.
        """
        n = len(observations)
        shared = community_term / n
        price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)
        ref_cost_pb = self._ref_cost / max(n, 1)

        raw_spatial = [
            -(max(self._safe(obs.get("net_electricity_consumption")), 0.0) * price)
            / ref_cost_pb
            for obs in observations
        ]
        mean_spatial = sum(raw_spatial) / n

        return [
            shared
            + self._w_spatial * (raw_spatial[i] - mean_spatial)
            - self._w_ev * ev_harms[i]
            for i in range(n)
        ]

    # ── main interface ────────────────────────────────────────────────────────

    def calculate(
        self, observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        if not observations:
            return []

        n = len(observations)

        # ── Community scalar (inherited 5-term Level-1 reward) ───────────────
        # Called exactly once per timestep — advances _prev_import as a side effect.
        community_term = self._community_scalar(observations)

        # ── EV service harm (per building; used by both credit modes) ────────
        ev_harms    = [self._ev_harm(obs) for obs in observations]
        ev_harm_sum = sum(ev_harms)

        # ── Distribute the reward across buildings ───────────────────────────
        if self._reward_mode == "per_building":
            rewards = self._per_building_rewards(observations, community_term, ev_harms)
        else:
            # Aggregate mode (default): identical community scalar per building.
            scalar  = community_term - self._w_ev * (ev_harm_sum / n)
            rewards = [scalar / n] * n

        # ── MLflow logging (sampled) ─────────────────────────────────────────
        self._step += 1
        if mlflow.active_run() and self._step % self._log_interval == 0:
            n_ev_connected = sum(
                1 for obs in observations
                if any(ev.get("connected", False)
                       for ev in (obs.get("electric_vehicles_chargers_dict") or {}).values())
            )
            n_ev_urgent = sum(1 for h in ev_harms if h > 0.0)
            reward_arr  = np.asarray(rewards, dtype=float)
            mlflow.log_metrics(
                {
                    "CC2_rf/community_term":  community_term,
                    "CC2_rf/ev_penalty":      self._w_ev * (ev_harm_sum / n),
                    "CC2_rf/ev_harm_sum":     ev_harm_sum,
                    "CC2_rf/n_ev_connected":  float(n_ev_connected),
                    "CC2_rf/n_ev_urgent":     float(n_ev_urgent),
                    "CC2_rf/total_reward":    float(reward_arr.sum()),
                    # Per-building spread — 0 in aggregate mode, > 0 once the
                    # reward differentiates buildings (per_building mode).
                    "CC2_rf/reward_spread":   float(reward_arr.std()),
                },
                step=self._step,
            )

        return rewards
