"""Adapter boundary between technology payloads and typed runtime frames."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    QualityState,
)
from algorithms.ti_marl.contracts.interface_definition import (
    InterfaceRegistry,
    SensorDefinition,
)
from algorithms.ti_marl.contracts.profile_registry import SENSOR_ENTITY_TYPES
from algorithms.ti_marl.runtime.contracts import (
    RUNTIME_CONTRACT_VERSION,
    TypedExecutionFeedback,
    TypedHealthEvidence,
    TypedObservationSample,
    TypedRuntimeEntity,
    TypedRuntimeFrame,
)
from algorithms.ti_marl.runtime.bindings import SimulatorBindingMap


class SimulatorAdapter:
    """Translate Simulator 1.7 contracts without leaking them into public YAMLs."""

    provenance = "softcpsrecsimulator_1.7_entity_adapter"

    def __init__(
        self,
        registry: InterfaceRegistry,
        *,
        bindings: SimulatorBindingMap | None = None,
    ) -> None:
        self.registry = registry
        self.bindings = bindings or SimulatorBindingMap()
        self.entity_specs: Dict[str, Any] = {}
        self.seconds_per_time_step = 1.0

    def attach_entity_specs(
        self,
        entity_specs: Mapping[str, Any],
        *,
        seconds_per_time_step: float = 1.0,
    ) -> None:
        specs = deepcopy(dict(entity_specs or {}))
        if str(specs.get("version")) != "entity_v1":
            raise ValueError("SimulatorAdapter requires entity_v1")
        if str(dict(specs.get("runtime_status_contract", {})).get("version")) != "runtime_status_v1":
            raise ValueError("SimulatorAdapter requires runtime_status_v1")
        if str(dict(specs.get("action_execution_contract", {})).get("version")) != "entity_action_execution_v1":
            raise ValueError("SimulatorAdapter requires entity_action_execution_v1")
        self.entity_specs = specs
        self.seconds_per_time_step = max(float(seconds_per_time_step), 1e-9)
        self._validate_bindings()

    def _validate_bindings(self) -> None:
        tables = self.entity_specs.get("tables", {})
        available_ids = {
            entity_type: set(str(item) for item in table.get("ids", []))
            for entity_type, table in tables.items()
        }
        for interface in self.registry.interfaces.values():
            # A registered member may be absent from the current technological
            # catalog until a later join event. Registration is not runtime
            # activity and must not be inferred from Simulator row presence.
            if interface.agent_id not in available_ids.get("building", set()):
                continue
            for sensor in interface.sensors:
                entity_type = SENSOR_ENTITY_TYPES[sensor.sensor_type]
                available_features = set(
                    str(item) for item in tables.get(entity_type, {}).get("features", [])
                )
                for observation in sensor.observations:
                    if observation.use == "excluded":
                        continue
                    bound_feature = self.bindings.observation_feature(
                        interface.agent_id,
                        sensor.sensor_id,
                        observation.observation_id,
                    )
                    if bound_feature not in available_features:
                        raise ValueError(
                            f"Simulator catalog cannot bind observation {interface.agent_id}."
                            f"{observation.path}: missing {entity_type}."
                            f"{bound_feature}"
                        )

    def to_frame(self, payload: Mapping[str, Any]) -> TypedRuntimeFrame:
        if not self.entity_specs:
            raise RuntimeError("SimulatorAdapter.attach_entity_specs() must be called first")
        meta = dict(payload.get("meta", {}) or {})
        if str(meta.get("spec_version")) != "entity_v1":
            raise ValueError("SimulatorAdapter received a payload that is not entity_v1")
        status = dict(meta.get("runtime_status", {}) or {})
        if str(status.get("version")) != "runtime_status_v1":
            raise ValueError("SimulatorAdapter requires runtime_status_v1 in every frame")
        if status.get("emits_health_state") is not False:
            raise ValueError("Simulator runtime status must remain facts-only")

        sequence = int(meta.get("time_step", 0))
        timestamp_seconds = sequence * self.seconds_per_time_step
        owners = self._owners(payload.get("edges", {}))
        entities, row_lookup = self._entities(payload, owners)
        # Preserve the technological adapter's canonical action/reward order;
        # registry file ordering is deliberately not a runtime ordering API.
        technological_agents = tuple(
            str(agent_id)
            for agent_id in self.entity_specs.get("tables", {}).get("building", {}).get("ids", [])
            if ("building", str(agent_id)) in row_lookup
        )
        unregistered = sorted(set(technological_agents) - set(self.registry.interfaces))
        if unregistered:
            raise ValueError(
                "SimulatorAdapter found active members without typed interfaces: "
                f"{unregistered}"
            )
        active_agents = technological_agents
        evidence = self._health_evidence(status, timestamp_seconds)
        evidence_by_target: Dict[str, list[TypedHealthEvidence]] = {}
        communication_evidence = []
        for item in evidence:
            evidence_by_target.setdefault(item.target_id, []).append(item)
            if item.event_domain == EventDomain.COMMUNICATION_LINK:
                communication_evidence.append(item)
        samples = self._samples(
            payload,
            row_lookup,
            owners,
            active_agents,
            timestamp_seconds,
            evidence_by_target,
            tuple(communication_evidence),
        )
        execution = self._execution_feedback(
            meta,
            timestamp_seconds,
            owners,
        )
        topology_events = tuple(
            deepcopy(dict(item))
            for item in (meta.get("topology_events", []) or [])
            if isinstance(item, Mapping)
        )
        frame_material = (
            sequence,
            int(meta.get("topology_version", 0)),
            tuple(item.sample_id for item in samples),
        )
        frame_id = hashlib.sha256(repr(frame_material).encode("utf-8")).hexdigest()
        return TypedRuntimeFrame(
            version=RUNTIME_CONTRACT_VERSION,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            sequence=sequence,
            topology_version=int(meta.get("topology_version", 0)),
            registered_agent_ids=self.registry.agent_ids,
            active_agent_ids=active_agents,
            samples=samples,
            health_evidence=evidence,
            entities=entities,
            execution_feedback=execution,
            topology_events=topology_events,
            provenance=self.provenance,
        )

    def _entities(
        self,
        payload: Mapping[str, Any],
        owners: Mapping[Tuple[str, int], str],
    ) -> Tuple[Tuple[TypedRuntimeEntity, ...], Mapping[Tuple[str, str], int]]:
        table_specs = self.entity_specs.get("tables", {})
        tables = payload.get("tables", {})
        result = []
        lookup: Dict[Tuple[str, str], int] = {}
        for entity_type, spec in table_specs.items():
            ids = [str(item) for item in spec.get("ids", [])]
            features = [str(item) for item in spec.get("features", [])]
            matrix = np.asarray(tables.get(entity_type, []), dtype=np.float64)
            if matrix.ndim == 1 and matrix.size:
                matrix = matrix.reshape(1, -1)
            for row, entity_id in enumerate(ids):
                if matrix.ndim != 2 or row >= matrix.shape[0]:
                    continue
                lookup[(str(entity_type), entity_id)] = row
                owner = owners.get((str(entity_type), row))
                if entity_type == "building":
                    owner = entity_id
                values = {
                    feature: self._finite(matrix[row, index])
                    for index, feature in enumerate(features)
                    if index < matrix.shape[1]
                }
                result.append(
                    TypedRuntimeEntity(
                        entity_id=entity_id,
                        entity_type=str(entity_type),
                        owner_agent_id=owner,
                        active=True,
                        values=values,
                    )
                )
        return tuple(sorted(result, key=lambda item: (item.entity_type, item.entity_id))), lookup

    def _samples(
        self,
        payload: Mapping[str, Any],
        row_lookup: Mapping[Tuple[str, str], int],
        owners: Mapping[Tuple[str, int], str],
        active_agents: Sequence[str],
        timestamp_seconds: float,
        evidence_by_target: Mapping[str, Sequence[TypedHealthEvidence]],
        communication_evidence: Sequence[TypedHealthEvidence],
    ) -> Tuple[TypedObservationSample, ...]:
        tables = payload.get("tables", {})
        table_specs = self.entity_specs.get("tables", {})
        matrices = {
            str(entity_type): np.asarray(value, dtype=np.float64)
            for entity_type, value in tables.items()
        }
        feature_indices = {
            str(entity_type): {
                str(feature): index
                for index, feature in enumerate(spec.get("features", []))
            }
            for entity_type, spec in table_specs.items()
        }
        result = []
        for agent_id in active_agents:
            interface = self.registry.for_agent(agent_id)
            for sensor in interface.sensors:
                entity_type = SENSOR_ENTITY_TYPES[sensor.sensor_type]
                if sensor.scope == "community":
                    source_id = self._community_source_id(entity_type, row_lookup)
                else:
                    source_id = self._local_source_id(
                        sensor,
                        agent_id,
                        entity_type,
                        owners,
                    )
                if source_id is None:
                    continue
                row = row_lookup.get((entity_type, source_id))
                if row is None:
                    continue
                feature_index = feature_indices.get(entity_type, {})
                matrix = matrices.get(entity_type, np.asarray([], dtype=np.float64))
                if matrix.ndim == 1 and matrix.size:
                    matrix = matrix.reshape(1, -1)
                for observation in sensor.observations:
                    bound_feature = self.bindings.observation_feature(
                        agent_id,
                        sensor.sensor_id,
                        observation.observation_id,
                    )
                    index = feature_index.get(bound_feature)
                    if observation.use == "excluded" or index is None or matrix.ndim != 2:
                        continue
                    # Preserve non-finite telemetry for TIC validity checks;
                    # never silently turn a bad sample into a nominal zero.
                    value = float(matrix[row, index])
                    runtime_facts = self._sample_runtime_facts(
                        source_id,
                        bound_feature,
                        observation.channel_id,
                        sensor.scope,
                        evidence_by_target,
                        communication_evidence,
                    )
                    result.append(
                        TypedObservationSample(
                            agent_id=agent_id,
                            sensor_id=sensor.sensor_id,
                            channel_id=observation.channel_id,
                            observation_id=observation.observation_id,
                            value=(value,),
                            shape=(1,),
                            unit=observation.unit,
                            timestamp_seconds=timestamp_seconds,
                            age_seconds=runtime_facts[3],
                            availability=runtime_facts[0],
                            quality=runtime_facts[1],
                            fault_mode=runtime_facts[2],
                            provenance=self.provenance,
                            source_entity_type=entity_type,
                            source_entity_id=source_id,
                            source_feature=bound_feature,
                        )
                    )
        return tuple(sorted(result, key=lambda item: item.sample_id))

    @staticmethod
    def _sample_runtime_facts(
        source_id: str,
        source_feature: str,
        channel_id: str,
        scope: str,
        evidence_by_target: Mapping[str, Sequence[TypedHealthEvidence]],
        communication_evidence: Sequence[TypedHealthEvidence],
    ) -> tuple[AvailabilityState, QualityState, Optional[str], float]:
        candidates = list(evidence_by_target.get(source_id, ()))
        candidates.extend(evidence_by_target.get("*", ()))
        if scope == "community":
            candidates.extend(communication_evidence)
        relevant = []
        for item in candidates:
            if item.event_domain == EventDomain.ACTUATOR_CHANNEL:
                relevant_match = channel_id == "execution_feedback"
            elif item.event_domain == EventDomain.ASSET_CONNECTION:
                relevant_match = channel_id == "connection"
            elif item.event_domain == EventDomain.COMMUNICATION_LINK:
                relevant_match = scope == "community"
            else:
                relevant_match = item.target_feature in {
                    "*",
                    "",
                    "both",
                    source_feature,
                }
            if relevant_match:
                relevant.append(item)
        if not relevant:
            return (
                AvailabilityState.AVAILABLE,
                QualityState.NOMINAL,
                None,
                0.0,
            )
        availability = (
            AvailabilityState.UNAVAILABLE
            if any(item.availability == AvailabilityState.UNAVAILABLE for item in relevant)
            else AvailabilityState.AVAILABLE
            if any(item.availability == AvailabilityState.AVAILABLE for item in relevant)
            else AvailabilityState.UNKNOWN
        )
        quality_order = {
            QualityState.NOMINAL: 0,
            QualityState.UNKNOWN: 1,
            QualityState.IMPAIRED: 2,
            QualityState.INVALID: 3,
        }
        quality = max(
            (item.quality for item in relevant),
            key=lambda item: quality_order[item],
        )
        fault_modes = sorted(
            {str(item.fault_mode) for item in relevant if item.fault_mode is not None}
        )
        age = max(
            (float(item.age_seconds) for item in relevant if item.age_seconds is not None),
            default=0.0,
        )
        return availability, quality, ("+".join(fault_modes) or None), age

    def _local_source_id(
        self,
        sensor: SensorDefinition,
        agent_id: str,
        entity_type: str,
        owners: Mapping[Tuple[str, int], str],
    ) -> Optional[str]:
        if sensor.source_entity_id is not None:
            return sensor.source_entity_id
        bound = self.bindings.sensor_entity_id(agent_id, sensor.sensor_id)
        if bound is not None:
            return bound
        if entity_type == "building":
            return agent_id
        ids = [
            str(item)
            for item in self.entity_specs.get("tables", {}).get(entity_type, {}).get("ids", [])
        ]
        candidates = [
            ids[row]
            for (kind, row), owner in owners.items()
            if kind == entity_type and owner == agent_id and 0 <= row < len(ids)
        ]
        suffix = sensor.sensor_id.rsplit("_", 1)[-1]
        try:
            index = max(int(suffix) - 1, 0)
        except ValueError:
            index = 0
        ordered = sorted(candidates)
        return ordered[index] if index < len(ordered) else None

    def actuator_entity_id(self, agent_id: str, actuator_id: str) -> Optional[str]:
        return self.bindings.actuator_entity_id(agent_id, actuator_id)

    @staticmethod
    def _community_source_id(
        entity_type: str,
        row_lookup: Mapping[Tuple[str, str], int],
    ) -> Optional[str]:
        candidates = sorted(
            entity_id
            for (kind, entity_id) in row_lookup
            if kind == entity_type
        )
        return candidates[0] if candidates else None

    def _health_evidence(
        self,
        status: Mapping[str, Any],
        timestamp_seconds: float,
    ) -> Tuple[TypedHealthEvidence, ...]:
        domains = {
            "asset_connections": EventDomain.ASSET_CONNECTION,
            "asset_availability": EventDomain.ASSET_AVAILABILITY,
            "sensor_channels": EventDomain.SENSOR_CHANNEL,
            "actuator_channels": EventDomain.ACTUATOR_CHANNEL,
            "communication_links": EventDomain.COMMUNICATION_LINK,
            "value_quality": EventDomain.VALUE_QUALITY,
        }
        events = {
            str(row.get("event_id")): dict(row)
            for row in status.get("active_events", []) or []
            if isinstance(row, Mapping) and row.get("event_id") is not None
        }
        result = []
        consumed_event_ids = set()
        for collection, domain in domains.items():
            for index, raw in enumerate(status.get(collection, []) or []):
                row = dict(raw)
                target_id = str(row.get("target_id") or row.get("source_id") or "*")
                target_type = str(row.get("target_type") or row.get("source_type") or "*")
                target_feature = str(row.get("target_feature") or row.get("relation") or "*")
                event_ids = tuple(sorted(str(item) for item in row.get("event_ids", []) or []))
                raw_id = str(row.get("event_id") or (event_ids[0] if event_ids else f"{collection}:{index}"))
                active = events.get(raw_id, {})
                consumed_event_ids.update(event_ids)
                if raw_id in events:
                    consumed_event_ids.add(raw_id)
                duration_steps = float(
                    row.get("active_duration_steps", active.get("active_duration_steps", 0)) or 0
                )
                age_steps = row.get("age_steps")
                fault_mode = row.get("fault_mode", active.get("fault_mode"))
                result.append(
                    TypedHealthEvidence(
                        evidence_id=f"{domain.value}:{target_type}:{target_id}:{target_feature}",
                        agent_id=self._owner_for_target(target_type, target_id),
                        sensor_id=None,
                        channel_id=None,
                        observation_id=None,
                        event_domain=domain,
                        fault_mode=None if fault_mode is None else str(fault_mode),
                        target_type=target_type,
                        target_id=target_id,
                        target_feature=target_feature,
                        availability=self._enum(AvailabilityState, row.get("availability"), AvailabilityState.UNKNOWN),
                        connection=self._enum(ConnectionState, row.get("connection"), ConnectionState.UNKNOWN),
                        quality=self._enum(QualityState, row.get("quality"), QualityState.UNKNOWN),
                        started_at_seconds=self._step_to_seconds(
                            row.get("start_time_step", active.get("start_time_step"))
                        ),
                        active_duration_seconds=max(duration_steps, 0.0) * self.seconds_per_time_step,
                        last_update_seconds=self._step_to_seconds(row.get("last_update_time_step")),
                        last_fresh_seconds=self._step_to_seconds(row.get("last_fresh_time_step")),
                        age_seconds=(
                            None
                            if age_steps is None
                            else max(float(age_steps), 0.0) * self.seconds_per_time_step
                        ),
                        event_ids=event_ids or (raw_id,),
                        provenance=self.provenance,
                    )
                )
        for event_id, row in events.items():
            if event_id in consumed_event_ids:
                continue
            try:
                domain = EventDomain(str(row.get("event_domain")))
            except ValueError as exc:
                raise ValueError(
                    f"Simulator runtime event {event_id!r} has unknown domain"
                ) from exc
            target_type = str(row.get("target_type") or "*")
            target_id = str(row.get("target_id") or "*")
            target_feature = str(row.get("target_feature") or "*")
            duration_steps = float(row.get("active_duration_steps", 0) or 0)
            fault_mode = row.get("fault_mode")
            result.append(
                TypedHealthEvidence(
                    evidence_id=(
                        f"{domain.value}:{target_type}:{target_id}:"
                        f"{target_feature}:{event_id}"
                    ),
                    agent_id=self._owner_for_target(target_type, target_id),
                    sensor_id=None,
                    channel_id=None,
                    observation_id=None,
                    event_domain=domain,
                    fault_mode=None if fault_mode is None else str(fault_mode),
                    target_type=target_type,
                    target_id=target_id,
                    target_feature=target_feature,
                    started_at_seconds=self._step_to_seconds(
                        row.get("start_time_step")
                    ),
                    active_duration_seconds=max(duration_steps, 0.0)
                    * self.seconds_per_time_step,
                    event_ids=(event_id,),
                    provenance=self.provenance,
                )
            )
        return tuple(sorted(result, key=lambda item: item.evidence_id))

    def _execution_feedback(
        self,
        meta: Mapping[str, Any],
        timestamp_seconds: float,
        owners: Mapping[Tuple[str, int], str],
    ) -> Tuple[TypedExecutionFeedback, ...]:
        raw = dict(meta.get("entity_action_execution", {}) or {})
        if not raw:
            return ()
        if str(raw.get("version")) != "entity_action_execution_v1":
            raise ValueError("SimulatorAdapter received unsupported action execution contract")
        result = []
        for row in raw.get("entries", []) or []:
            agent_id = str(row.get("agent_id") or row.get("owner_module_id") or "")
            target = str(row.get("target_entity_id") or "")
            actuator_id = self._logical_actuator_id(
                agent_id,
                target,
                str(row.get("action_name") or ""),
                owners,
            )
            result.append(
                TypedExecutionFeedback(
                    agent_id=agent_id,
                    actuator_id=actuator_id,
                    target_entity_id=actuator_id,
                    action_name=str(row.get("action_name") or ""),
                    requested_value=self._optional_float(row.get("requested_value")),
                    post_channel_value=self._optional_float(row.get("post_channel_value")),
                    limited_value=self._optional_float(row.get("limited_value")),
                    applied_value=self._optional_float(row.get("applied_value")),
                    applied_power_kw=self._optional_float(row.get("applied_power_kw")),
                    limitation_reasons=tuple(str(item) for item in row.get("limitation_reasons", []) or []),
                    timestamp_seconds=timestamp_seconds,
                    provenance=self.provenance,
                )
            )
        return tuple(result)

    def _logical_actuator_id(
        self,
        agent_id: str,
        physical_target_id: str,
        action_name: str,
        owners: Mapping[Tuple[str, int], str],
    ) -> str:
        interface = self.registry.for_agent(agent_id)
        for actuator in interface.actuators:
            bound = self.bindings.actuator_entity_id(
                agent_id,
                actuator.actuator_id,
            )
            if bound == physical_target_id:
                return actuator.actuator_id
        if action_name == "electrical_storage":
            entity_type = "storage"
        elif action_name.startswith("electric_vehicle_storage"):
            entity_type = "charger"
        elif action_name == "start" or action_name.startswith("deferrable_appliance"):
            entity_type = "deferrable_appliance"
        else:
            entity_type = None
        if entity_type is not None:
            physical_ids = [
                str(item)
                for item in self.entity_specs.get("tables", {})
                .get(entity_type, {})
                .get("ids", [])
            ]
            candidates = sorted(
                physical_ids[row]
                for (kind, row), owner in owners.items()
                if kind == entity_type
                and owner == agent_id
                and 0 <= row < len(physical_ids)
            )
            logical = sorted(
                actuator.actuator_id
                for actuator in interface.actuators
                if actuator.source_entity_type == entity_type
            )
            if physical_target_id in candidates:
                index = candidates.index(physical_target_id)
                if index < len(logical):
                    return logical[index]
        raise ValueError(
            "Simulator execution feedback cannot be bound to a public "
            f"actuator: {agent_id}/{physical_target_id} ({action_name})"
        )

    def _owners(self, edges: Mapping[str, Any]) -> Mapping[Tuple[str, int], str]:
        building_ids = [
            str(item)
            for item in self.entity_specs.get("tables", {}).get("building", {}).get("ids", [])
        ]
        owners: Dict[Tuple[str, int], str] = {
            ("building", row): agent_id for row, agent_id in enumerate(building_ids)
        }
        relations = {
            "building_to_storage": "storage",
            "building_to_charger": "charger",
            "building_to_deferrable_appliance": "deferrable_appliance",
            "building_to_pv": "pv",
        }
        for relation, entity_type in relations.items():
            for source, target in self._edge_pairs(edges.get(relation)):
                if 0 <= source < len(building_ids):
                    key = (entity_type, target)
                    owner = building_ids[source]
                    previous = owners.get(key)
                    if previous is not None and previous != owner:
                        raise ValueError(
                            f"SimulatorAdapter ambiguous binding for {entity_type} row "
                            f"{target}: {previous!r} and {owner!r}"
                        )
                    owners[key] = owner
        charger_owner = {
            row: owner
            for (kind, row), owner in owners.items()
            if kind == "charger"
        }
        for relation in ("charger_to_ev_connected", "charger_to_ev_incoming"):
            pairs = tuple(self._edge_pairs(edges.get(relation)))
            mask = np.asarray(edges.get(f"{relation}_mask", []), dtype=np.float64).reshape(-1)
            for index, (charger_row, ev_row) in enumerate(pairs):
                if index < len(mask) and mask[index] <= 0.5:
                    continue
                owner = charger_owner.get(charger_row)
                if owner is not None and ev_row >= 0:
                    owners[("ev", ev_row)] = owner
        return owners

    def _owner_for_target(self, target_type: str, target_id: str) -> Optional[str]:
        if target_type == "building" and target_id in self.registry.interfaces:
            return target_id
        for interface in self.registry.interfaces.values():
            for sensor in interface.sensors:
                source = (
                    self.bindings.sensor_entity_id(interface.agent_id, sensor.sensor_id)
                    or sensor.source_entity_id
                    or sensor.sensor_id
                )
                if source == target_id:
                    return interface.agent_id
            for actuator in interface.actuators:
                source = (
                    self.bindings.actuator_entity_id(
                        interface.agent_id,
                        actuator.actuator_id,
                    )
                    or actuator.target_entity_id
                    or actuator.actuator_id
                )
                if source == target_id:
                    return interface.agent_id
        return None

    def _step_to_seconds(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value) * self.seconds_per_time_step

    @staticmethod
    def _edge_pairs(value: Any) -> Iterable[Tuple[int, int]]:
        matrix = np.asarray([] if value is None else value, dtype=np.float64)
        if matrix.size == 0:
            return ()
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return tuple((int(row[0]), int(row[1])) for row in matrix if len(row) >= 2)

    @staticmethod
    def _enum(enum_type: Any, value: Any, default: Any) -> Any:
        try:
            return enum_type(str(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _finite(value: Any) -> float:
        result = float(value)
        return result if np.isfinite(result) else 0.0

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) else None


class MappingTelemetryAdapter:
    """Reference adapter for MQTT/Modbus/API gateways after field mapping.

    A gateway supplies already mapped logical IDs; this class proves that the
    TIC does not depend on Simulator tables once a ``TypedRuntimeFrame`` exists.
    """

    def __init__(self, registry: InterfaceRegistry, *, provenance: str) -> None:
        self.registry = registry
        self.provenance = str(provenance)

    def to_frame(self, payload: Mapping[str, Any]) -> TypedRuntimeFrame:
        samples = tuple(
            TypedObservationSample(
                agent_id=str(row["agent_id"]),
                sensor_id=str(row["sensor_id"]),
                channel_id=str(row["channel_id"]),
                observation_id=str(row["observation_id"]),
                value=tuple(float(item) for item in row.get("value", ())),
                shape=tuple(int(item) for item in row.get("shape", (1,))),
                unit=str(row["unit"]),
                timestamp_seconds=float(row.get("timestamp_seconds", payload["timestamp_seconds"])),
                age_seconds=float(row.get("age_seconds", 0.0)),
                availability=SimulatorAdapter._enum(
                    AvailabilityState,
                    row.get("availability", "AVAILABLE"),
                    AvailabilityState.UNKNOWN,
                ),
                quality=SimulatorAdapter._enum(
                    QualityState,
                    row.get("quality", "NOMINAL"),
                    QualityState.UNKNOWN,
                ),
                fault_mode=None if row.get("fault_mode") is None else str(row["fault_mode"]),
                provenance=self.provenance,
                source_entity_type=row.get("source_entity_type"),
                source_entity_id=row.get("source_entity_id"),
                source_feature=row.get("source_feature"),
                estimated=bool(row.get("estimated", False)),
            )
            for row in payload.get("samples", [])
        )
        entities = tuple(
            TypedRuntimeEntity(
                entity_id=str(row["entity_id"]),
                entity_type=str(row["entity_type"]),
                owner_agent_id=row.get("owner_agent_id"),
                active=bool(row.get("active", True)),
                values={str(key): float(value) for key, value in row.get("values", {}).items()},
            )
            for row in payload.get("entities", [])
        )
        health_evidence = tuple(
            TypedHealthEvidence(
                evidence_id=str(row["evidence_id"]),
                agent_id=row.get("agent_id"),
                sensor_id=row.get("sensor_id"),
                channel_id=row.get("channel_id"),
                observation_id=row.get("observation_id"),
                event_domain=EventDomain(str(row["event_domain"])),
                fault_mode=row.get("fault_mode"),
                target_type=str(row.get("target_type", "*")),
                target_id=str(row.get("target_id", "*")),
                target_feature=str(row.get("target_feature", "*")),
                availability=SimulatorAdapter._enum(
                    AvailabilityState,
                    row.get("availability", "UNKNOWN"),
                    AvailabilityState.UNKNOWN,
                ),
                connection=SimulatorAdapter._enum(
                    ConnectionState,
                    row.get("connection", "UNKNOWN"),
                    ConnectionState.UNKNOWN,
                ),
                quality=SimulatorAdapter._enum(
                    QualityState,
                    row.get("quality", "UNKNOWN"),
                    QualityState.UNKNOWN,
                ),
                started_at_seconds=row.get("started_at_seconds"),
                active_duration_seconds=float(row.get("active_duration_seconds", 0.0)),
                last_update_seconds=row.get("last_update_seconds"),
                last_fresh_seconds=row.get("last_fresh_seconds"),
                age_seconds=row.get("age_seconds"),
                event_ids=tuple(str(item) for item in row.get("event_ids", ())),
                provenance=self.provenance,
            )
            for row in payload.get("health_evidence", [])
        )
        execution_feedback = tuple(
            TypedExecutionFeedback(
                agent_id=str(row["agent_id"]),
                actuator_id=str(row["actuator_id"]),
                target_entity_id=str(row.get("target_entity_id", row["actuator_id"])),
                action_name=str(row.get("action_name", row["actuator_id"])),
                requested_value=SimulatorAdapter._optional_float(row.get("requested_value")),
                post_channel_value=SimulatorAdapter._optional_float(row.get("post_channel_value")),
                limited_value=SimulatorAdapter._optional_float(row.get("limited_value")),
                applied_value=SimulatorAdapter._optional_float(row.get("applied_value")),
                applied_power_kw=SimulatorAdapter._optional_float(row.get("applied_power_kw")),
                limitation_reasons=tuple(str(item) for item in row.get("limitation_reasons", ())),
                timestamp_seconds=float(row.get("timestamp_seconds", payload["timestamp_seconds"])),
                provenance=self.provenance,
            )
            for row in payload.get("execution_feedback", [])
        )
        active = tuple(str(item) for item in payload.get("active_agent_ids", []))
        return TypedRuntimeFrame(
            version=RUNTIME_CONTRACT_VERSION,
            frame_id=str(payload.get("frame_id", "mapping-frame")),
            timestamp_seconds=float(payload["timestamp_seconds"]),
            sequence=int(payload.get("sequence", 0)),
            topology_version=int(payload.get("topology_version", 0)),
            registered_agent_ids=self.registry.agent_ids,
            active_agent_ids=active,
            samples=samples,
            health_evidence=health_evidence,
            entities=entities,
            execution_feedback=execution_feedback,
            topology_events=tuple(payload.get("topology_events", ())),
            provenance=self.provenance,
        )
