"""Build technology-neutral commands after local feasibility projection."""

from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

from algorithms.ti_marl.contracts.models import InterfaceSnapshot, LocalActionBundle
from algorithms.ti_marl.runtime.contracts import TypedActionCommand


class TypedCommandBuilder:
    def build(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Iterable[LocalActionBundle],
    ) -> Tuple[TypedActionCommand, ...]:
        groups = {group.group_id: group for group in snapshot.action_groups}
        commands = []
        for bundle in bundles:
            for decision in bundle.decisions:
                group = groups.get(decision.group_id)
                if group is None:
                    continue
                port = next(
                    (item for item in group.ports if item.mode == decision.mode),
                    None,
                )
                available = 0.0 if port is None else max(float(port.upper_bound), 0.0)
                if decision.mode.startswith("CHARGE_"):
                    rated = group.max_charge_power_kw
                    value = rated * decision.fraction * available
                    action_id = "charge"
                    unit = "kW"
                elif decision.mode.startswith("DISCHARGE_"):
                    rated = group.max_discharge_power_kw
                    value = rated * decision.fraction * available
                    action_id = "discharge"
                    unit = "kW"
                elif decision.mode == "START":
                    value = 1.0
                    action_id = "start"
                    unit = "boolean"
                else:
                    value = 0.0
                    action_id = "idle"
                    unit = "fraction"
                actuator_id = group.group_id.split(":", 1)[-1]
                material = (
                    snapshot.snapshot_hash,
                    bundle.agent_id,
                    actuator_id,
                    action_id,
                )
                command_id = hashlib.sha256(repr(material).encode("utf-8")).hexdigest()
                commands.append(
                    TypedActionCommand(
                        agent_id=bundle.agent_id,
                        actuator_id=actuator_id,
                        action_id=action_id,
                        mode=decision.mode,
                        value=float(value),
                        unit=unit,
                        target_entity_id=group.module_id,
                        timestamp_seconds=snapshot.timestamp_seconds,
                        command_id=command_id,
                        constraints_applied=tuple(
                            str(item.get("reason", ""))
                            for item in bundle.interventions
                            if item.get("group_id") == group.group_id
                        ),
                        fallback_reason=group.fallback_reason,
                    )
                )
        return tuple(commands)
