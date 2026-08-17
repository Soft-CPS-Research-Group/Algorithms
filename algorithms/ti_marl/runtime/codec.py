"""Convert typed fractions to the current CityLearn flat action vector."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from algorithms.ti_marl.contracts.models import InterfaceSnapshot, LocalActionBundle
from algorithms.ti_marl.runtime.contracts import TypedActionCommand


class CityLearnTypedActionCodec:
    def __init__(self) -> None:
        self.building_names: Tuple[str, ...] = ()
        self.action_names: Tuple[Tuple[str, ...], ...] = ()
        self.action_bounds: Tuple[Tuple[np.ndarray, np.ndarray], ...] = ()

    def attach(
        self,
        *,
        building_names: Sequence[str],
        action_names: Sequence[Sequence[str]],
        action_space: Sequence[Any],
    ) -> None:
        self.building_names = tuple(str(item) for item in building_names)
        self.action_names = tuple(tuple(str(name) for name in names) for names in action_names)
        bounds = []
        for space in action_space:
            low = np.asarray(getattr(space, "low", []), dtype=np.float64).reshape(-1)
            high = np.asarray(getattr(space, "high", []), dtype=np.float64).reshape(-1)
            bounds.append((low, high))
        self.action_bounds = tuple(bounds)

    def encode(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Sequence[LocalActionBundle],
    ) -> list[list[float]]:
        if not self.building_names:
            raise RuntimeError("CityLearnTypedActionCodec.attach() must be called first")
        bundle_by_agent = {bundle.agent_id: bundle for bundle in bundles}
        groups = {group.group_id: group for group in snapshot.action_groups}
        commands = []
        for index, agent_id in enumerate(self.building_names):
            names = self.action_names[index] if index < len(self.action_names) else ()
            low, high = self.action_bounds[index] if index < len(self.action_bounds) else (
                np.zeros(len(names)),
                np.zeros(len(names)),
            )
            vector = np.zeros(len(names), dtype=np.float64)
            bundle = bundle_by_agent.get(agent_id)
            if bundle is not None:
                for decision in bundle.decisions:
                    group = groups.get(decision.group_id)
                    if group is None:
                        continue
                    action_name = self._flat_action_name(
                        group.group_type,
                        group.adapter_target_entity_id or group.module_id,
                    )
                    if action_name not in names:
                        if decision.mode != "IDLE":
                            raise ValueError(
                                f"No CityLearn action slot {action_name!r} for {agent_id}/{group.group_id}"
                            )
                        continue
                    position = names.index(action_name)
                    port = next((item for item in group.ports if item.mode == decision.mode), None)
                    bound = 0.0 if port is None else float(port.upper_bound)
                    vector[position] = self._scalar(
                        decision.mode,
                        decision.fraction,
                        bound,
                        low[position],
                        high[position],
                    )
            vector = np.clip(vector, low, high) if len(vector) else vector
            commands.append(vector.tolist())
        return commands

    def encode_typed(
        self,
        snapshot: InterfaceSnapshot,
        typed_commands: Sequence[TypedActionCommand],
    ) -> list[list[float]]:
        """Simulator-adapter translation from neutral commands to flat slots."""

        if not self.building_names:
            raise RuntimeError("CityLearnTypedActionCodec.attach() must be called first")
        groups = {
            (group.owner_agent_id, group.module_id): group
            for group in snapshot.action_groups
        }
        by_agent: Dict[str, list[TypedActionCommand]] = {}
        for command in typed_commands:
            by_agent.setdefault(command.agent_id, []).append(command)
        result = []
        for index, agent_id in enumerate(self.building_names):
            names = self.action_names[index] if index < len(self.action_names) else ()
            low, high = self.action_bounds[index] if index < len(self.action_bounds) else (
                np.zeros(len(names)),
                np.zeros(len(names)),
            )
            vector = np.zeros(len(names), dtype=np.float64)
            for command in by_agent.get(agent_id, []):
                group = groups.get((agent_id, str(command.target_entity_id)))
                if group is None:
                    continue
                slot = self._flat_action_name(
                    group.group_type,
                    group.adapter_target_entity_id or group.module_id,
                )
                if slot not in names:
                    if command.mode != "IDLE":
                        raise ValueError(f"No CityLearn action slot {slot!r} for {agent_id}")
                    continue
                position = names.index(slot)
                if command.mode.startswith("CHARGE_"):
                    rated = group.max_charge_power_kw
                elif command.mode.startswith("DISCHARGE_"):
                    rated = group.max_discharge_power_kw
                else:
                    rated = 1.0
                fraction = 0.0 if rated <= 0.0 else float(command.value) / float(rated)
                vector[position] = self._scalar(
                    command.mode,
                    fraction,
                    1.0,
                    low[position],
                    high[position],
                )
            result.append(np.clip(vector, low, high).tolist() if len(vector) else [])
        return result

    @staticmethod
    def _flat_action_name(group_type: str, module_id: str) -> str:
        if group_type == "stationary_storage":
            return "electrical_storage"
        suffix = str(module_id).split("/", 1)[-1]
        if group_type == "ev_session":
            return f"electric_vehicle_storage_{suffix}"
        if group_type == "deferrable":
            return f"deferrable_appliance_{suffix}"
        raise ValueError(f"Unsupported TI-MARL action group type: {group_type}")

    @staticmethod
    def _scalar(
        mode: str,
        fraction: float,
        compiled_bound: float,
        low: float,
        high: float,
    ) -> float:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        bound = float(np.clip(compiled_bound, 0.0, 1.0))
        if mode in {"CHARGE_STATIONARY", "CHARGE_EV"}:
            return fraction * bound * max(float(high), 0.0)
        if mode in {"DISCHARGE_STATIONARY", "DISCHARGE_EV"}:
            return -fraction * bound * abs(min(float(low), 0.0))
        if mode == "START":
            return max(float(high), 0.0)
        return 0.0
