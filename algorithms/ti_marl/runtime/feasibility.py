"""Analytic joint projection over one building's action groups."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

from algorithms.ti_marl.contracts.models import (
    ActionDecision,
    ActionGroupInstance,
    InterfaceSnapshot,
    LocalActionBundle,
)


class AnalyticLocalProjector:
    """Project typed bundles without performing community optimization."""

    def project(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Iterable[LocalActionBundle],
    ) -> Tuple[LocalActionBundle, ...]:
        groups = {group.group_id: group for group in snapshot.action_groups}
        return tuple(self._project_agent(snapshot, bundle, groups) for bundle in bundles)

    def _project_agent(
        self,
        snapshot: InterfaceSnapshot,
        bundle: LocalActionBundle,
        groups: Mapping[str, ActionGroupInstance],
    ) -> LocalActionBundle:
        interventions = list(bundle.interventions)
        decisions: Dict[str, ActionDecision] = {}
        for decision in bundle.decisions:
            group = groups.get(decision.group_id)
            if group is None:
                interventions.append(
                    self._intervention(decision.group_id, "removed_group", decision.mode, "IDLE", decision.fraction, 0.0)
                )
                continue
            port = next((item for item in group.ports if item.mode == decision.mode), None)
            if not group.enabled or port is None or not port.valid:
                decisions[group.group_id] = replace(decision, mode="IDLE", fraction=0.0, mode_index=0)
                interventions.append(
                    self._intervention(group.group_id, "invalid_port_fallback", decision.mode, "IDLE", decision.fraction, 0.0)
                )
                continue
            # ``fraction`` is relative to the *currently compiled* port.  The
            # runtime bound itself is applied once by the CityLearn codec.
            # Clipping the fraction to ``port.upper_bound`` here would apply
            # that contraction a second time.
            bounded = float(np.clip(decision.fraction, 0.0, 1.0))
            if decision.mode == "IDLE":
                bounded = 0.0
            if abs(bounded - decision.fraction) > 1.0e-9:
                interventions.append(
                    self._intervention(group.group_id, "fraction_domain", decision.mode, decision.mode, decision.fraction, bounded)
                )
            decisions[group.group_id] = replace(decision, fraction=bounded)

        self._enforce_deferrable_must_start(snapshot, bundle.agent_id, groups, decisions, interventions)
        self._scale_direction(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
            direction="charge",
            constraint_type="charging_headroom_kw",
        )
        self._scale_direction(
            snapshot,
            bundle.agent_id,
            groups,
            decisions,
            interventions,
            direction="discharge",
            constraint_type="export_headroom_kw",
        )
        ordered = tuple(decisions[key] for key in sorted(decisions))
        return LocalActionBundle(
            agent_id=bundle.agent_id,
            decisions=ordered,
            interventions=tuple(interventions),
        )

    def _enforce_deferrable_must_start(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
    ) -> None:
        parts = {part.source_entity_id: part for part in snapshot.parts_for(agent_id) if part.valid}
        for group_id, decision in tuple(decisions.items()):
            group = groups[group_id]
            if group.group_type != "deferrable":
                continue
            part = parts.get(group.module_id)
            if part is None:
                continue
            values = dict(zip(part.feature_names, part.values))
            must_start = (
                values.get("pending", 0.0) > 0.5
                and values.get("can_start", 0.0) > 0.5
                and values.get("slack_steps", 1.0) <= 0.0
            )
            start_port = next((port for port in group.ports if port.mode == "START" and port.valid), None)
            if must_start and start_port is not None and decision.mode != "START":
                decisions[group_id] = replace(decision, mode="START", fraction=1.0, mode_index=1)
                interventions.append(
                    self._intervention(group_id, "deferrable_must_start", decision.mode, "START", decision.fraction, 1.0)
                )

    def _scale_direction(
        self,
        snapshot: InterfaceSnapshot,
        agent_id: str,
        groups: Mapping[str, ActionGroupInstance],
        decisions: Dict[str, ActionDecision],
        interventions: list[Mapping[str, object]],
        *,
        direction: str,
        constraint_type: str,
    ) -> None:
        constraint = next(
            (
                item
                for item in snapshot.constraints
                if item.owner_agent_id == agent_id and item.constraint_type == constraint_type and item.active
            ),
            None,
        )
        if constraint is None or constraint.upper_bound is None or not np.isfinite(constraint.upper_bound):
            return
        selected = []
        total_power = 0.0
        for group_id, decision in decisions.items():
            expected = "CHARGE" in decision.mode if direction == "charge" else "DISCHARGE" in decision.mode
            if not expected:
                continue
            group = groups[group_id]
            rated = group.max_charge_power_kw if direction == "charge" else group.max_discharge_power_kw
            port = next((item for item in group.ports if item.mode == decision.mode), None)
            available_fraction = 0.0 if port is None else max(float(port.upper_bound), 0.0)
            power = (
                max(rated, 0.0)
                * max(decision.fraction, 0.0)
                * available_fraction
            )
            total_power += power
            selected.append((group_id, decision, rated))
        limit = max(float(constraint.upper_bound), 0.0)
        if total_power <= limit + 1.0e-9 or total_power <= 0.0:
            return
        scale = limit / total_power
        for group_id, decision, _rated in selected:
            updated = decision.fraction * scale
            decisions[group_id] = replace(decision, fraction=updated)
            interventions.append(
                self._intervention(group_id, constraint.constraint_id, decision.mode, decision.mode, decision.fraction, updated)
            )

    def assert_feasible(self, snapshot: InterfaceSnapshot, bundles: Iterable[LocalActionBundle]) -> None:
        groups = {group.group_id: group for group in snapshot.action_groups}
        for bundle in bundles:
            for direction, constraint_type in (
                ("charge", "charging_headroom_kw"),
                ("discharge", "export_headroom_kw"),
            ):
                limit = next(
                    (
                        item.upper_bound
                        for item in snapshot.constraints
                        if item.owner_agent_id == bundle.agent_id and item.constraint_type == constraint_type
                    ),
                    None,
                )
                if limit is None or not np.isfinite(limit):
                    continue
                total = 0.0
                for decision in bundle.decisions:
                    if (direction == "charge" and "CHARGE" not in decision.mode) or (
                        direction == "discharge" and "DISCHARGE" not in decision.mode
                    ):
                        continue
                    group = groups[decision.group_id]
                    rated = group.max_charge_power_kw if direction == "charge" else group.max_discharge_power_kw
                    port = next(
                        (item for item in group.ports if item.mode == decision.mode),
                        None,
                    )
                    available_fraction = (
                        0.0 if port is None else max(float(port.upper_bound), 0.0)
                    )
                    total += rated * decision.fraction * available_fraction
                if total > float(limit) + 1.0e-6:
                    raise AssertionError(
                        f"Projected {direction} power {total} exceeds {bundle.agent_id} limit {limit}"
                    )

    @staticmethod
    def _intervention(
        group_id: str,
        reason: str,
        raw_mode: str,
        final_mode: str,
        raw_fraction: float,
        final_fraction: float,
    ) -> Mapping[str, object]:
        return {
            "group_id": group_id,
            "reason": reason,
            "raw_mode": raw_mode,
            "final_mode": final_mode,
            "raw_fraction": float(raw_fraction),
            "final_fraction": float(final_fraction),
            "magnitude": abs(float(final_fraction) - float(raw_fraction)),
        }
