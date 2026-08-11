"""CC Level-2 community reward function.

Extends CCRewardLevel1 with a per-building EV service term so the CC is
penalised whenever its high price signals cause a building's EV to miss its
required SoC by departure time.

    reward_t = community_term − w_ev · ev_penalty_t

Community term (identical to CCRewardLevel1):
    community_term = − w_cost   * cost_norm
                     − w_peak   * peak_import_norm
                     − w_ramp   * import_ramp_norm
                     − w_export * export_norm
                     − w_violation * electrical_violation_norm

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
* ``w_ramp`` and ``w_violation`` default to zero for compatibility with the
  historical Level-2 runs. Current scorecard-aware campaigns set both
  explicitly.
* w_ev = 0.5 default — EV safety carries half the weight of cost, comparable
  to the community signal at a mildly bad timestep but clearly secondary.

Return value
------------
``credit_assignment="global"`` preserves the historical scalar split equally
across buildings. ``credit_assignment="member_decomposed"`` returns one
reward per building: exact member settlement cost, local EV/violation terms,
and community peak/ramp/export terms allocated to the members that contributed
to them. The member rewards always sum to the same community objective.
Set ``cost_aggregation="member_retail"`` while the community market is
disabled to align the cost term with ``district_cost_total_control_eur``.
Set ``cost_aggregation="community_settled"`` with the community market enabled
to optimize the exact member-level settlement reported by
``community_settled_cost_total_eur``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Union

import mlflow
from citylearn.reward_function import RewardFunction

from reward_function.community_settlement import community_settlement_components


class CCRewardLevel2(RewardFunction):
    """Community + EV-service reward for the Level-2 CC."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        *,
        # Community weights (mirror CCRewardLevel1)
        w_cost:   float = 1.0,
        w_peak:   float = 0.3,
        w_ramp:   float = 0.0,
        w_export: float = 0.1,
        w_violation: float = 0.0,
        # EV service weight
        w_ev:     float = 0.5,
        # Community reference values (15-min dataset, 17 buildings)
        target_import:    float = 4.14,   # kWh — p75 community import
        reference_cost:   float = 1.045,  # p90 community cost
        reference_peak:   float = 2.72,   # p90 peak excess squared
        reference_ramping: float = 1.878,  # p90 import ramp
        reference_export: float = 7.52,   # kWh — p90 community export
        reference_violation: float = 1.0,
        # EV urgency horizon in hours
        urgency_horizon:  float = 4.0,    # harm starts H hours before departure
        cost_aggregation: str = "community_net",
        community_local_price_ratio: float | None = None,
        community_grid_export_price: float | None = None,
        community_import_member_weights: Mapping[str, float] | None = None,
        credit_assignment: str = "global",
        ramp_credit_allocation: str = "gross_import_change",
        **kwargs,
    ) -> None:
        super().__init__(env_metadata, **kwargs)
        self._w_cost   = float(w_cost)
        self._w_peak   = float(w_peak)
        self._w_ramp   = float(w_ramp)
        self._w_export = float(w_export)
        self._w_violation = float(w_violation)
        self._w_ev     = float(w_ev)

        self._target_import    = float(target_import)
        self._ref_cost         = max(float(reference_cost),   1e-8)
        self._ref_peak         = max(float(reference_peak),   1e-8)
        self._ref_ramping      = max(float(reference_ramping), 1e-8)
        self._ref_export       = max(float(reference_export), 1e-8)
        self._ref_violation    = max(float(reference_violation), 1e-8)
        self._cost_aggregation = str(cost_aggregation).strip().lower()
        if self._cost_aggregation not in {
            "community_net",
            "community_settled",
            "member_retail",
        }:
            raise ValueError(
                "CCRewardLevel2 cost_aggregation must be 'community_net', "
                "'community_settled', or 'member_retail'"
            )
        self._credit_assignment = str(credit_assignment).strip().lower()
        if self._credit_assignment not in {"global", "member_decomposed"}:
            raise ValueError(
                "CCRewardLevel2 credit_assignment must be 'global' or "
                "'member_decomposed'"
            )
        self._ramp_credit_allocation = str(
            ramp_credit_allocation
        ).strip().lower()
        if self._ramp_credit_allocation not in {
            "gross_import_change",
            "causal_net",
        }:
            raise ValueError(
                "CCRewardLevel2 ramp_credit_allocation must be "
                "'gross_import_change' or 'causal_net'"
            )

        initial_metadata = env_metadata if isinstance(env_metadata, Mapping) else {}
        self._community_local_price_ratio_explicit = (
            community_local_price_ratio is not None
        )
        self._community_grid_export_price_explicit = (
            community_grid_export_price is not None
        )
        self._community_import_member_weights_explicit = (
            community_import_member_weights is not None
        )
        self._explicit_community_import_member_weights = dict(
            community_import_member_weights or {}
        )
        market_metadata = initial_metadata.get("community_market") or {}
        if not isinstance(market_metadata, Mapping):
            market_metadata = {}
        configured_ratio = (
            market_metadata.get(
                "local_price_ratio_to_grid_import",
                market_metadata.get("intra_community_sell_ratio", 0.8),
            )
            if community_local_price_ratio is None
            else community_local_price_ratio
        )
        self._community_local_price_ratio = min(
            max(self._safe(configured_ratio, 0.8), 0.0),
            1.0,
        )
        configured_export_price = (
            market_metadata.get("grid_export_price", 0.0)
            if community_grid_export_price is None
            else community_grid_export_price
        )
        self._community_grid_export_price = max(
            self._safe(configured_export_price, 0.0),
            0.0,
        )
        configured_weights = (
            market_metadata.get("import_member_weights") or {}
            if community_import_member_weights is None
            else community_import_member_weights
        )
        buildings = initial_metadata.get("buildings") or []
        building_names = [
            str(building.get("name") or "")
            for building in buildings
            if isinstance(building, Mapping)
        ]
        self._community_import_member_weights = (
            [
                max(self._safe(configured_weights.get(name, 1.0), 1.0), 0.0)
                for name in building_names
            ]
            if isinstance(configured_weights, Mapping) and building_names
            else None
        )
        self.last_community_settlement: Mapping[str, float] = {}
        self.last_member_components: List[Mapping[str, float]] = []

        self._urgency_horizon  = max(float(urgency_horizon), 1e-6)
        self._prev_import = 0.0
        self._prev_member_imports: List[float] = []
        self._prev_member_net_consumptions: List[float] = []

        # MLflow logging — sample every N calls to avoid flooding
        self._log_interval: int = 50
        self._step: int = 0

    def reset(self) -> None:
        """Reset temporal reward state at the episode boundary."""

        super().reset()
        self._prev_import = 0.0
        self._prev_member_imports = []
        self._prev_member_net_consumptions = []

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

    @classmethod
    def _violation_kwh(cls, obs: Mapping[str, Any]) -> float:
        for key in (
            "charging_constraint_violation_kwh",
            "electrical_service_violation_kwh",
            "electrical_service_violation",
            "service_violation_kwh",
            "service_violation",
        ):
            if key in obs:
                return max(cls._safe(obs[key]), 0.0)
        return 0.0

    def _refresh_community_settlement_contract(self) -> None:
        """Resolve market metadata after CityLearn completes environment loading."""

        metadata = self.env_metadata if isinstance(self.env_metadata, Mapping) else {}
        market = metadata.get("community_market") or {}
        if not isinstance(market, Mapping):
            market = {}
        if not self._community_local_price_ratio_explicit:
            ratio = market.get(
                "local_price_ratio_to_grid_import",
                market.get("intra_community_sell_ratio", 0.8),
            )
            self._community_local_price_ratio = min(
                max(self._safe(ratio, 0.8), 0.0),
                1.0,
            )
        if not self._community_grid_export_price_explicit:
            self._community_grid_export_price = max(
                self._safe(market.get("grid_export_price", 0.0), 0.0),
                0.0,
            )

        configured_weights = (
            self._explicit_community_import_member_weights
            if self._community_import_member_weights_explicit
            else market.get("import_member_weights") or {}
        )
        buildings = metadata.get("buildings") or []
        building_names = [
            str(building.get("name") or "")
            for building in buildings
            if isinstance(building, Mapping)
        ]
        if isinstance(configured_weights, Mapping) and building_names:
            self._community_import_member_weights = [
                max(self._safe(configured_weights.get(name, 1.0), 1.0), 0.0)
                for name in building_names
            ]

    @staticmethod
    def _allocate_total(total: float, weights: List[float]) -> List[float]:
        """Allocate a signed total while preserving it exactly."""

        if not weights:
            return []
        normalized = [max(float(value), 0.0) for value in weights]
        denominator = sum(normalized)
        if denominator <= 1.0e-12:
            return [float(total) / len(normalized) for _ in normalized]
        return [float(total) * value / denominator for value in normalized]

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

        # ── Community term (identical to CCRewardLevel1) ─────────────────────
        settlement_rows: List[Mapping[str, float]] = []
        if self._cost_aggregation == "member_retail":
            member_costs = [
                imported
                * max(self._safe(obs.get("electricity_pricing")), 0.0)
                for imported, obs in zip(
                    [max(value, 0.0) for value in (
                        self._safe(item.get("net_electricity_consumption"))
                        for item in observations
                    )],
                    observations,
                )
            ]
            community_cost = sum(member_costs)
        elif self._cost_aggregation == "community_settled":
            self._refresh_community_settlement_contract()
            weights = self._community_import_member_weights
            if weights is not None and len(weights) != len(observations):
                weights = None
            settlement_rows, settlement = community_settlement_components(
                observations,
                local_price_ratio=self._community_local_price_ratio,
                grid_export_price=self._community_grid_export_price,
                import_member_weights=weights,
            )
            self.last_community_settlement = settlement
            community_cost = settlement["community_settlement_cost_total"]
            member_costs = [
                float(row["community_settlement_cost"])
                for row in settlement_rows
            ]
        else:
            price = max(self._safe(observations[0].get("electricity_pricing")), 0.0)
            community_cost = import_t * price
            member_costs = self._allocate_total(community_cost, [
                max(self._safe(obs.get("net_electricity_consumption")), 0.0)
                for obs in observations
            ])
        cost_norm   = community_cost / self._ref_cost
        peak_norm   = (max(import_t - self._target_import, 0.0) ** 2) / self._ref_peak
        ramp_norm   = abs(import_t - self._prev_import) / self._ref_ramping
        member_net_consumptions = [
            self._safe(obs.get("net_electricity_consumption"))
            for obs in observations
        ]
        member_imports = [max(value, 0.0) for value in member_net_consumptions]
        member_exports = [max(-value, 0.0) for value in member_net_consumptions]
        member_import_changes = (
            [abs(current) for current in member_imports]
            if len(self._prev_member_imports) != n
            else [
                abs(current - previous)
                for current, previous in zip(
                    member_imports,
                    self._prev_member_imports,
                )
            ]
        )
        previous_member_nets = (
            self._prev_member_net_consumptions
            if len(self._prev_member_net_consumptions) == n
            else [0.0] * n
        )
        member_net_changes = [
            current - previous
            for current, previous in zip(
                member_net_consumptions,
                previous_member_nets,
            )
        ]
        community_import_change = import_t - self._prev_import
        if self._ramp_credit_allocation == "causal_net":
            # Give ramp credit to members whose signed net-load change agrees
            # with the aggregate import ramp. This includes export changes and
            # avoids blaming an offsetting member merely because its gross
            # import moved.
            ramp_allocation_weights = [
                max(change, 0.0)
                if community_import_change >= 0.0
                else max(-change, 0.0)
                for change in member_net_changes
            ]
            if sum(ramp_allocation_weights) <= 1.0e-12:
                ramp_allocation_weights = [
                    abs(change) for change in member_net_changes
                ]
        else:
            ramp_allocation_weights = member_import_changes
        self._prev_import = import_t
        self._prev_member_imports = member_imports
        self._prev_member_net_consumptions = member_net_consumptions
        export_norm = export_t / self._ref_export
        member_violations = [self._violation_kwh(obs) for obs in observations]
        violation_norm = sum(member_violations) / self._ref_violation

        community_term = (
            - self._w_cost   * cost_norm
            - self._w_peak   * peak_norm
            - self._w_ramp   * ramp_norm
            - self._w_export * export_norm
            - self._w_violation * violation_norm
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

        peak_components = self._allocate_total(
            -self._w_peak * peak_norm,
            member_imports,
        )
        ramp_components = self._allocate_total(
            -self._w_ramp * ramp_norm,
            ramp_allocation_weights,
        )
        export_components = self._allocate_total(
            -self._w_export * export_norm,
            member_exports,
        )
        member_rewards = []
        self.last_member_components = []
        for index in range(n):
            cost_component = -self._w_cost * member_costs[index] / self._ref_cost
            violation_component = (
                -self._w_violation
                * member_violations[index]
                / self._ref_violation
            )
            ev_component = -self._w_ev * ev_harms[index] / n
            reward = (
                cost_component
                + peak_components[index]
                + ramp_components[index]
                + export_components[index]
                + violation_component
                + ev_component
            )
            member_rewards.append(reward)
            self.last_member_components.append(
                {
                    "cost": cost_component,
                    "peak": peak_components[index],
                    "ramp": ramp_components[index],
                    "export": export_components[index],
                    "violation": violation_component,
                    "ev": ev_component,
                    "total": reward,
                }
            )

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
                    "CC2_rf/ramp_norm":       ramp_norm,
                    "CC2_rf/export_norm":     export_norm,
                    "CC2_rf/violation_norm":  violation_norm,
                    "CC2_rf/n_ev_connected":  float(n_ev_connected),
                    "CC2_rf/n_ev_urgent":     float(n_ev_urgent),
                    "CC2_rf/total_reward":    scalar,
                },
                step=self._step,
            )

        if self._credit_assignment == "member_decomposed":
            # Guard against numerical drift in future component changes.
            correction = scalar - sum(member_rewards)
            member_rewards[-1] += correction
            return member_rewards
        per_building = scalar / n
        return [per_building] * n
