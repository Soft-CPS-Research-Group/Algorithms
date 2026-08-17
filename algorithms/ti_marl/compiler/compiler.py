"""Typed Interface Compiler over deployment-neutral runtime frames."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from algorithms.ti_marl.compiler.health import HealthDeriver
from algorithms.ti_marl.contracts.compatibility import CompatibilitySignature
from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    HEALTH_SEVERITY,
    HealthState,
    QualityState,
)
from algorithms.ti_marl.contracts.interface_definition import (
    ActionDefinition,
    ActuatorDefinition,
    InterfaceRegistry,
    ObservationDefinition,
    RegistryDelta,
    TypedAgentInterface,
)
from algorithms.ti_marl.contracts.models import (
    ActionGroupInstance,
    ActionPortInstance,
    AgentSchema,
    Dependency,
    EntityInstance,
    FaultEvidence,
    HealthAssessment,
    InterfaceSnapshot,
    LocalConstraint,
    ModuleInstance,
    ObservationPart,
    SharedResource,
    canonical_value,
)
from algorithms.ti_marl.contracts.profile_registry import (
    ACTION_PROFILES,
    SENSOR_ENTITY_TYPES,
    CapabilityProfileRegistry,
)
from algorithms.ti_marl.runtime.adapters import SimulatorAdapter
from algorithms.ti_marl.runtime.bindings import SimulatorBindingMap
from algorithms.ti_marl.runtime.contracts import (
    RUNTIME_CONTRACT_VERSION,
    TypedHealthEvidence,
    TypedObservationSample,
    TypedRuntimeEntity,
    TypedRuntimeFrame,
)


COMPILER_VERSION = "tic_v2"


class TypedInterfaceCompiler:
    """Compile stable typed frames into policy snapshots and safe action ports."""

    def __init__(
        self,
        *,
        contract_version: str,
        typed_interfaces_dir: str | Path,
        interface_polling: bool = False,
        simulator_bindings_path: str | Path | None = None,
    ) -> None:
        self.contract_version = str(contract_version)
        if self.contract_version != "ti_marl_v1":
            raise ValueError(f"Unsupported TI-MARL contract version: {self.contract_version!r}")
        self.profiles = CapabilityProfileRegistry()
        self.interface_registry = InterfaceRegistry(
            typed_interfaces_dir,
            polling_enabled=interface_polling,
            profiles=self.profiles,
        )
        self.simulator_bindings = SimulatorBindingMap.load(simulator_bindings_path)
        self.adapter = SimulatorAdapter(
            self.interface_registry,
            bindings=self.simulator_bindings,
        )
        self.health_rules = deepcopy(dict(self.profiles.health_rules()))
        self.health_deriver = HealthDeriver(self.health_rules)
        self.agent_schema_config = self._generic_agent_schema()
        self.type_registry = self._generic_type_registry()
        self.agent_schema = self._agent_schema()
        self.entity_specs: Dict[str, Any] = {}
        self.seconds_per_time_step = 1.0
        self._mismatch_counts: Dict[Tuple[str, str], int] = {}
        self._structure_key: Optional[Tuple[Any, ...]] = None
        self._observation_plan: Tuple[
            Tuple[str, str, str, ObservationDefinition], ...
        ] = ()
        self._action_plan: Tuple[Tuple[str, ActuatorDefinition, str], ...] = ()
        self.structure_recompilations = 0
        self.compatibility_signature = CompatibilitySignature.build(
            contract_version=self.contract_version,
            agent_schema=self.agent_schema_config,
            type_registry=self.type_registry,
            health_rules=self.health_rules,
            compiler_version=COMPILER_VERSION,
        )

    def _generic_agent_schema(self) -> Mapping[str, Any]:
        return {
            "version": "ti_marl_agent_schema_v2",
            "agent_entity_type": "building",
            "module_types": sorted(set(SENSOR_ENTITY_TYPES.values())),
            "observation_semantic_types": [
                "local_energy",
                "local_constraint",
                "storage_state",
                "ev_service",
                "deferrable_state",
                "community_signal",
            ],
            "action_group_types": sorted(
                set(str(profile["group_type"]) for profile in ACTION_PROFILES.values())
            ),
            "interface_shape": "per_agent_instance_free",
        }

    def _generic_type_registry(self) -> Mapping[str, Any]:
        entity_types = {
            entity_type: {
                "semantic_type": (
                    "community_signal" if entity_type == "district" else "local_energy"
                )
            }
            for entity_type in sorted(set(SENSOR_ENTITY_TYPES.values()))
        }
        action_groups: Dict[str, Any] = {}
        for profile in ACTION_PROFILES.values():
            group_type = str(profile["group_type"])
            modes = tuple(str(item) for item in profile["modes"])
            previous = action_groups.get(group_type)
            if previous is None or len(modes) > len(previous["modes"]):
                action_groups[group_type] = {
                    "entity_type": str(profile["entity_type"]),
                    "modes": list(modes),
                }
        return {
            "version": "ti_marl_type_registry_v2",
            "entity_types": entity_types,
            "sensor_types": sorted(SENSOR_ENTITY_TYPES),
            "semantic_types": self._generic_agent_schema()["observation_semantic_types"],
            "action_group_types": action_groups,
            "hierarchy": ["observation", "channel", "sensor", "agent"],
        }

    def _agent_schema(self) -> AgentSchema:
        cfg = self.agent_schema_config
        return AgentSchema(
            version=str(cfg["version"]),
            agent_entity_type=str(cfg["agent_entity_type"]),
            module_types=tuple(str(item) for item in cfg["module_types"]),
            action_group_types=tuple(str(item) for item in cfg["action_group_types"]),
            observation_semantic_types=tuple(
                str(item) for item in cfg["observation_semantic_types"]
            ),
        )

    def attach_entity_specs(
        self,
        entity_specs: Mapping[str, Any],
        *,
        seconds_per_time_step: float = 1.0,
    ) -> None:
        next_specs = deepcopy(dict(entity_specs or {}))
        next_seconds = max(float(seconds_per_time_step), 1e-9)
        if next_specs == self.entity_specs and next_seconds == self.seconds_per_time_step:
            return
        self.entity_specs = next_specs
        self.seconds_per_time_step = next_seconds
        # A topology change can keep the same public version while changing
        # the technological entity catalog.  Never retain bindings compiled
        # against the previous catalog.
        self._structure_key = None
        self.adapter.attach_entity_specs(
            self.entity_specs,
            seconds_per_time_step=self.seconds_per_time_step,
        )

    def reload_interfaces(self) -> RegistryDelta:
        """Reload all files atomically; the previous registry survives any error."""

        delta = self.interface_registry.reload_interfaces()
        self._structure_key = None
        if self.entity_specs:
            self.adapter.attach_entity_specs(
                self.entity_specs,
                seconds_per_time_step=self.seconds_per_time_step,
            )
        return delta

    def resolved_typed_interface(self) -> Mapping[str, Any]:
        return self.interface_registry.resolved_bundle()

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "entity_specs": deepcopy(self.entity_specs),
            "seconds_per_time_step": self.seconds_per_time_step,
            "health": self.health_deriver.snapshot_state(),
            "registry_hash": self.interface_registry.registry_hash,
            "mismatch_counts": deepcopy(self._mismatch_counts),
        }

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        specs = deepcopy(dict(payload.get("entity_specs", {})))
        if specs:
            self.attach_entity_specs(
                specs,
                seconds_per_time_step=float(payload.get("seconds_per_time_step", 1.0)),
            )
        self.health_deriver.restore_state(payload.get("health", {}))
        self._mismatch_counts = {
            tuple(key) if not isinstance(key, tuple) else key: int(value)
            for key, value in payload.get("mismatch_counts", {}).items()
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        return {
            "health": self.health_deriver.snapshot_state(),
            "registry_hash": self.interface_registry.registry_hash,
            "mismatch_counts": deepcopy(self._mismatch_counts),
        }

    def reset_runtime_state(self) -> None:
        self.health_deriver.reset()
        self._mismatch_counts.clear()

    def load_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        self.health_deriver.restore_state(payload.get("health", {}))
        self._mismatch_counts = {
            tuple(key) if not isinstance(key, tuple) else key: int(value)
            for key, value in payload.get("mismatch_counts", {}).items()
        }

    def compile(self, payload: Mapping[str, Any]) -> InterfaceSnapshot:
        self.interface_registry.maybe_reload()
        return self.compile_frame(self.adapter.to_frame(payload))

    def compile_frame(self, frame: TypedRuntimeFrame) -> InterfaceSnapshot:
        if frame.version != RUNTIME_CONTRACT_VERSION:
            raise ValueError(f"Unsupported typed runtime frame: {frame.version!r}")
        unknown_active = sorted(set(frame.active_agent_ids) - set(frame.registered_agent_ids))
        if unknown_active:
            raise ValueError(f"Active runtime agents have no typed interface: {unknown_active}")
        entities = self._entities(frame.entities)
        self._ensure_structure(frame)
        evidence = tuple(self._fault_evidence(item) for item in frame.health_evidence)
        nominal = self._nominal_health_subjects(frame)
        health = self.health_deriver.derive(
            evidence,
            time_step=frame.sequence,
            timestamp_seconds=frame.timestamp_seconds,
            nominal_subjects=nominal,
        )
        health_by_subject = {item.subject_id: item for item in health}
        evidence_by_target: Dict[str, list[FaultEvidence]] = {}
        communication_evidence = []
        for item in evidence:
            evidence_by_target.setdefault(item.target_id, []).append(item)
            if item.event_domain == EventDomain.COMMUNICATION_LINK:
                communication_evidence.append(item)
        sample_ids = tuple(item.sample_id for item in frame.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Typed runtime frame contains duplicate observation samples")
        expected_sample_ids = {
            f"{agent_id}:{sensor_id}:{definition.channel_id}:{definition.observation_id}"
            for agent_id, sensor_id, _scope, definition in self._observation_plan
        }
        unknown_sample_ids = sorted(set(sample_ids) - expected_sample_ids)
        if unknown_sample_ids:
            raise ValueError(
                "Typed runtime frame contains observations outside the registered "
                f"interfaces: {unknown_sample_ids[:10]}"
            )
        samples = {item.sample_id: item for item in frame.samples}
        self._update_mismatch_counts(frame)
        parts, path_health = self._observation_parts(
            frame,
            samples,
            evidence_by_target,
            tuple(communication_evidence),
            health_by_subject,
        )
        groups = self._action_groups(frame, samples)
        groups, closure_log = self._apply_safety(
            frame,
            groups,
            parts,
            path_health,
            evidence,
            health_by_subject,
        )
        constraints = self._constraints(frame, samples, groups, parts)
        modules = self._modules(frame.entities)
        execution = tuple(canonical_value(asdict(item)) for item in frame.execution_feedback)
        return InterfaceSnapshot(
            contract_version=self.contract_version,
            compiler_version=COMPILER_VERSION,
            topology_version=frame.topology_version,
            time_step=frame.sequence,
            agent_ids=frame.active_agent_ids,
            modules=modules,
            entities=entities,
            fault_evidence=evidence,
            health=health,
            observation_parts=parts,
            action_groups=groups,
            dependencies=self._dependencies(),
            constraints=constraints,
            shared_resources=(
                SharedResource(
                    resource_id="community_sensor",
                    resource_type="community_observation",
                    member_agent_ids=frame.active_agent_ids,
                    observable_only=True,
                ),
            ),
            closure_log=closure_log,
            timestamp_seconds=frame.timestamp_seconds,
            registered_agent_ids=frame.registered_agent_ids,
            agent_metadata=tuple(
                (
                    agent_id,
                    self.interface_registry.for_agent(agent_id).role,
                    self.interface_registry.for_agent(agent_id).agent_type,
                )
                for agent_id in frame.active_agent_ids
            ),
            registry_hash=self.interface_registry.registry_hash,
            execution_feedback=execution,
            topology_events=frame.topology_events,
        )

    @staticmethod
    def _entities(items: Sequence[TypedRuntimeEntity]) -> Tuple[EntityInstance, ...]:
        result = []
        row_by_type: Dict[str, int] = {}
        for item in sorted(items, key=lambda value: (value.entity_type, value.entity_id)):
            features = tuple(sorted(str(key) for key in item.values))
            result.append(
                EntityInstance(
                    entity_id=item.entity_id,
                    entity_type=item.entity_type,
                    owner_agent_id=item.owner_agent_id,
                    row_index=row_by_type.get(item.entity_type, 0),
                    feature_names=features,
                    values=tuple(float(item.values[key]) for key in features),
                )
            )
            row_by_type[item.entity_type] = row_by_type.get(item.entity_type, 0) + 1
        return tuple(result)

    @staticmethod
    def _modules(items: Sequence[TypedRuntimeEntity]) -> Tuple[ModuleInstance, ...]:
        return tuple(
            ModuleInstance(
                module_id=item.entity_id,
                module_type=item.entity_type,
                owner_agent_id=item.owner_agent_id,
                entity_id=item.entity_id,
                available=(
                    AvailabilityState.AVAILABLE
                    if item.active
                    else AvailabilityState.UNAVAILABLE
                ),
            )
            for item in sorted(items, key=lambda value: (value.owner_agent_id or "", value.entity_id))
            if item.owner_agent_id is not None
        )

    @staticmethod
    def _fault_evidence(item: TypedHealthEvidence) -> FaultEvidence:
        return FaultEvidence(
            evidence_id=item.evidence_id,
            event_domain=item.event_domain,
            fault_mode=item.fault_mode,
            target_type=item.target_type,
            target_id=item.target_id,
            target_feature=item.target_feature,
            availability=item.availability,
            connection=item.connection,
            quality=item.quality,
            event_ids=item.event_ids,
            start_time_seconds=item.started_at_seconds,
            active_duration_seconds=item.active_duration_seconds,
            last_update_seconds=item.last_update_seconds,
            last_fresh_seconds=item.last_fresh_seconds,
            age_seconds=item.age_seconds,
            agent_id=item.agent_id,
            sensor_id=item.sensor_id,
            channel_name=item.channel_id,
            observation_id=item.observation_id,
        )

    def _nominal_health_subjects(
        self,
        frame: TypedRuntimeFrame,
    ) -> Mapping[str, Tuple[str, str]]:
        subjects = {}
        for item in frame.health_evidence:
            evidence = self._fault_evidence(item)
            subjects[HealthDeriver.subject_id(evidence)] = (
                HealthDeriver.semantic_type(evidence),
                HealthDeriver.criticality(evidence),
            )
        return subjects

    def _observation_parts(
        self,
        frame: TypedRuntimeFrame,
        samples: Mapping[str, TypedObservationSample],
        evidence_by_target: Mapping[str, Sequence[FaultEvidence]],
        communication_evidence: Sequence[FaultEvidence],
        health_by_subject: Mapping[str, HealthAssessment],
    ) -> Tuple[Tuple[ObservationPart, ...], Mapping[Tuple[str, str], HealthState]]:
        parts = []
        path_health: Dict[Tuple[str, str], HealthState] = {}
        for agent_id, sensor_id, sensor_scope, definition in self._observation_plan:
            sample_id = (
                f"{agent_id}:{sensor_id}:{definition.channel_id}:"
                f"{definition.observation_id}"
            )
            sample = samples.get(sample_id)
            validity_reasons = self._sample_validation(definition, sample)
            state = self._health_for_observation(
                (
                    definition.source_feature
                    if sample is None or sample.source_feature is None
                    else sample.source_feature
                ),
                definition.channel_id,
                sensor_scope,
                None if sample is None else sample.source_entity_id,
                evidence_by_target,
                communication_evidence,
                health_by_subject,
                sample,
            )
            if validity_reasons:
                state = max(
                    (state, HealthState.UNKNOWN),
                    key=lambda candidate: HEALTH_SEVERITY[candidate],
                )
            path_health[(agent_id, definition.path)] = state
            missing = sample is None or bool(validity_reasons)
            values = (0.0,) if missing else tuple(sample.value)
            parts.append(
                ObservationPart(
                    part_id=f"{agent_id}:{definition.path}",
                    owner_agent_id=agent_id,
                    source_entity_id=(
                        sensor_id
                        if sample is None or sample.source_entity_id is None
                        else sample.source_entity_id
                    ),
                    semantic_type=definition.semantic_type,
                    feature_names=(definition.observation_id,),
                    values=values,
                    health=state,
                    shape=(1,) if sample is None else sample.shape,
                    valid=not missing and state not in {
                        HealthState.MISSING,
                        HealthState.FAILED,
                        HealthState.UNKNOWN,
                    },
                    validity_reasons=validity_reasons,
                    estimated=False if sample is None else sample.estimated,
                    sensor_id=sensor_id,
                    channel_id=definition.channel_id,
                    observation_id=definition.observation_id,
                    unit=definition.unit,
                    scope=sensor_scope,
                    use=definition.use,
                    policy_input=definition.policy_input,
                    criticality=definition.criticality,
                    age_seconds=0.0 if sample is None else sample.age_seconds,
                    normalisation=definition.normalisation,
                )
            )
        return tuple(sorted(parts, key=lambda item: item.part_id)), path_health

    @staticmethod
    def _sample_validation(
        definition: ObservationDefinition,
        sample: Optional[TypedObservationSample],
    ) -> Tuple[str, ...]:
        if sample is None:
            return ()
        reasons = []
        if sample.unit != definition.unit:
            reasons.append("unit_mismatch")
        expected_size = 1
        if definition.dimensions:
            expected_size = int(
                np.prod(
                    [len(values) for values in definition.dimensions.values()],
                    dtype=np.int64,
                )
            )
        observed_size = int(np.prod(sample.shape, dtype=np.int64)) if sample.shape else 0
        if observed_size != len(sample.value) or observed_size != expected_size:
            reasons.append("shape_mismatch")
        if not all(np.isfinite(float(value)) for value in sample.value):
            reasons.append("non_finite_value")
        return tuple(sorted(set(reasons)))

    def _ensure_structure(self, frame: TypedRuntimeFrame) -> None:
        key = (
            self.interface_registry.registry_hash,
            frame.topology_version,
            frame.active_agent_ids,
            tuple(
                (item.owner_agent_id, item.entity_type, item.entity_id, item.active)
                for item in frame.entities
            ),
        )
        if key == self._structure_key:
            return
        observations = []
        actions = []
        active_entities = tuple(item for item in frame.entities if item.active)
        for agent_id in frame.active_agent_ids:
            interface = self.interface_registry.for_agent(agent_id)
            for sensor in interface.sensors:
                for definition in sensor.observations:
                    observations.append(
                        (agent_id, sensor.sensor_id, sensor.scope, definition)
                    )
            for actuator in interface.actuators:
                candidates = [
                    item
                    for item in active_entities
                    if item.owner_agent_id == agent_id
                    and item.entity_type == actuator.source_entity_type
                ]
                bound_entity_id = self.adapter.actuator_entity_id(
                    agent_id,
                    actuator.actuator_id,
                )
                entity = (
                    next(
                        (item for item in candidates if item.entity_id == bound_entity_id),
                        None,
                    )
                    if bound_entity_id is not None
                    else self._select_runtime_entity(actuator, candidates)
                )
                if entity is not None:
                    actions.append((agent_id, actuator, entity.entity_id))
        self._observation_plan = tuple(observations)
        self._action_plan = tuple(actions)
        self._structure_key = key
        self.structure_recompilations += 1

    def _health_for_observation(
        self,
        source_feature: str,
        channel_id: str,
        scope: str,
        source_entity_id: Optional[str],
        evidence_by_target: Mapping[str, Sequence[FaultEvidence]],
        communication_evidence: Sequence[FaultEvidence],
        assessments: Mapping[str, HealthAssessment],
        sample: Optional[TypedObservationSample],
    ) -> HealthState:
        states = []
        candidates = list(evidence_by_target.get(str(source_entity_id), ()))
        candidates.extend(evidence_by_target.get("*", ()))
        if scope == "community":
            candidates.extend(communication_evidence)
        for item in candidates:
            community_match = (
                scope == "community"
                and item.event_domain == EventDomain.COMMUNICATION_LINK
            )
            entity_match = item.target_id in {"*", source_entity_id}
            feature_match = item.target_feature in {"*", "", "both", source_feature}
            if item.event_domain == EventDomain.ASSET_CONNECTION:
                evidence_match = entity_match and channel_id == "connection"
            elif item.event_domain == EventDomain.ACTUATOR_CHANNEL:
                evidence_match = entity_match and channel_id == "execution_feedback"
            else:
                evidence_match = entity_match and feature_match
            if community_match or evidence_match:
                assessment = assessments.get(HealthDeriver.subject_id(item))
                if assessment is not None:
                    states.append(assessment.state)
        if sample is None:
            states.append(HealthState.MISSING)
        elif sample.availability == AvailabilityState.UNAVAILABLE:
            states.append(HealthState.MISSING)
        elif sample.quality == QualityState.INVALID:
            states.append(HealthState.MISSING)
        elif sample.quality == QualityState.IMPAIRED:
            states.append(HealthState.DEGRADED)
        else:
            states.append(HealthState.HEALTHY)
        return max(states, key=lambda state: HEALTH_SEVERITY[state])

    def _action_groups(
        self,
        frame: TypedRuntimeFrame,
        samples: Mapping[str, TypedObservationSample],
    ) -> Tuple[ActionGroupInstance, ...]:
        active_entities = {item.entity_id: item for item in frame.entities if item.active}
        groups = []
        for agent_id, actuator, entity_id in self._action_plan:
            entity = active_entities.get(entity_id)
            if entity is None:
                raise RuntimeError("TI-MARL structural action binding became stale")
            values = dict(entity.values)
            by_mode = {action.mode: action for action in actuator.actions}
            ports = [
                ActionPortInstance(
                    port_id=f"{agent_id}:{actuator.actuator_id}:IDLE",
                    mode="IDLE",
                    target_entity_id=entity.entity_id,
                    action_name="idle",
                    lower_bound=0.0,
                    upper_bound=0.0,
                )
            ]
            for mode in actuator.modes:
                if mode == "IDLE":
                    continue
                action = by_mode.get(mode)
                if action is None:
                    continue
                runtime_bound = self._runtime_bound(values, mode)
                valid = runtime_bound > 0.0
                ports.append(
                    ActionPortInstance(
                        port_id=f"{agent_id}:{actuator.actuator_id}:{mode}",
                        mode=mode,
                        target_entity_id=entity.entity_id,
                        action_name=action.action_id,
                        lower_bound=0.0,
                        upper_bound=runtime_bound,
                        valid=valid,
                        invalid_reasons=() if valid else ("unknown_or_zero_runtime_bound",),
                        contracted_by=("runtime_capability",) if runtime_bound < 1.0 else (),
                    )
                )
            max_charge = max(
                (
                    action.upper_bound
                    for action in actuator.actions
                    if action.mode.startswith("CHARGE_")
                ),
                default=0.0,
            )
            max_discharge = max(
                (
                    action.upper_bound
                    for action in actuator.actions
                    if action.mode.startswith("DISCHARGE_")
                ),
                default=0.0,
            )
            groups.append(
                ActionGroupInstance(
                    group_id=f"{agent_id}:{actuator.actuator_id}",
                    group_type=actuator.group_type,
                    owner_agent_id=agent_id,
                    module_id=actuator.actuator_id,
                    ports=tuple(ports),
                    max_charge_power_kw=max_charge,
                    max_discharge_power_kw=max_discharge,
                    adapter_target_entity_id=entity.entity_id,
                )
            )
        return tuple(sorted(groups, key=lambda item: item.group_id))

    @staticmethod
    def _select_runtime_entity(
        actuator: ActuatorDefinition,
        candidates: Sequence[TypedRuntimeEntity],
    ) -> Optional[TypedRuntimeEntity]:
        if actuator.target_entity_id is not None:
            return next(
                (item for item in candidates if item.entity_id == actuator.target_entity_id),
                None,
            )
        if not candidates:
            return None
        suffix = actuator.actuator_id.rsplit("_", 1)[-1]
        try:
            index = max(int(suffix) - 1, 0)
        except ValueError:
            index = 0
        ordered = sorted(candidates, key=lambda item: item.entity_id)
        return ordered[index] if index < len(ordered) else None

    @staticmethod
    def _runtime_bound(values: Mapping[str, float], mode: str) -> float:
        if mode.startswith("CHARGE_"):
            aliases = (
                "available_charge_action_normalized",
                "available_charging_action_normalized",
            )
        elif mode.startswith("DISCHARGE_"):
            aliases = (
                "available_discharge_action_normalized",
                "available_discharging_action_normalized",
            )
        elif mode == "START":
            aliases = ("available_start_action_normalized", "can_start")
        else:
            return 0.0
        for name in aliases:
            if name in values and np.isfinite(values[name]):
                return float(np.clip(values[name], 0.0, 1.0))
        return 0.0

    def _apply_safety(
        self,
        frame: TypedRuntimeFrame,
        groups: Sequence[ActionGroupInstance],
        parts: Sequence[ObservationPart],
        path_health: Mapping[Tuple[str, str], HealthState],
        evidence: Sequence[FaultEvidence],
        assessments: Mapping[str, HealthAssessment],
    ) -> Tuple[Tuple[ActionGroupInstance, ...], Tuple[Mapping[str, Any], ...]]:
        part_by_path = {
            (part.owner_agent_id, f"{part.sensor_id}.{part.channel_id}.{part.observation_id}"): part
            for part in parts
        }
        result = []
        log = []
        for group in groups:
            interface = self.interface_registry.for_agent(group.owner_agent_id)
            actuator_id = group.group_id.split(":", 1)[1]
            actuator = next(item for item in interface.actuators if item.actuator_id == actuator_id)
            updated = group
            # Physical asset/actuator failures are independent from sensor health.
            relevant_channel_failure = False
            disconnected = False
            runtime_target = group.adapter_target_entity_id or group.module_id
            mismatch_key = (group.owner_agent_id, group.module_id)
            if self._mismatch_counts.get(mismatch_key, 0) >= 3:
                updated = self._safe_idle(updated, "repeated_requested_applied_mismatch")
                log.append(
                    self._closure_row(updated, "repeated_requested_applied_mismatch")
                )
            outage = any(
                str(item.fault_mode or "").lower()
                in {"outage", "power_outage", "grid_outage"}
                and (
                    item.target_type in {"district", "community", "grid"}
                    or item.target_id
                    in {"*", group.owner_agent_id, runtime_target}
                )
                for item in evidence
            )
            if outage:
                prohibited = {
                    port.mode
                    for port in updated.ports
                    if port.mode.startswith("CHARGE_")
                    or port.mode in {"DISCHARGE_EV", "START"}
                }
                updated = self._invalidate_modes(
                    updated,
                    prohibited,
                    "power_outage",
                )
                log.append(self._closure_row(updated, "power_outage"))
            for item in evidence:
                if item.target_id not in {"*", runtime_target, group.module_id, actuator_id}:
                    continue
                assessment = assessments.get(HealthDeriver.subject_id(item))
                state = HealthState.UNKNOWN if assessment is None else assessment.state
                if item.event_domain == EventDomain.ACTUATOR_CHANNEL and state in {
                    HealthState.STALE,
                    HealthState.MISSING,
                    HealthState.FAILED,
                    HealthState.UNKNOWN,
                }:
                    relevant_channel_failure = True
                if item.event_domain == EventDomain.ASSET_AVAILABILITY and state in {
                    HealthState.MISSING,
                    HealthState.FAILED,
                    HealthState.UNKNOWN,
                }:
                    relevant_channel_failure = True
                if (
                    item.event_domain == EventDomain.ASSET_CONNECTION
                    and actuator.source_entity_type == "charger"
                    and item.connection == ConnectionState.DISCONNECTED
                ):
                    disconnected = True
            if relevant_channel_failure:
                updated = self._safe_idle(updated, "actuator_or_asset_unavailable")
                log.append(self._closure_row(updated, "actuator_or_asset_unavailable"))
            elif disconnected:
                updated = self._invalidate_modes(
                    updated,
                    {port.mode for port in updated.ports if port.mode != "IDLE"},
                    "asset_disconnected",
                )
                log.append(self._closure_row(updated, "asset_disconnected"))

            for action in actuator.actions:
                for dependency_path, outcomes in action.dependencies.items():
                    state = path_health.get(
                        (group.owner_agent_id, dependency_path),
                        HealthState.UNKNOWN,
                    )
                    if state == HealthState.HEALTHY:
                        continue
                    raw_outcome = outcomes.get(state.value)
                    effect = (
                        str(raw_outcome.get("effect", "allow"))
                        if isinstance(raw_outcome, Mapping)
                        else str(raw_outcome or "allow")
                    )
                    updated = self._apply_dependency_effect(
                        updated,
                        action,
                        effect,
                        f"{dependency_path}:{state.value}",
                    )
                    if effect not in {"allow", "degraded_input"}:
                        log.append(
                            self._closure_row(
                                updated,
                                effect,
                                dependency_path=dependency_path,
                                health_state=state.value,
                            )
                        )

            # Domain defaults are explicit and conservative even if a manual
            # interface omitted an optional economic dependency.
            updated, default_rows = self._default_fail_safe(
                updated,
                actuator,
                interface,
                part_by_path,
                path_health,
            )
            log.extend(default_rows)
            result.append(updated)
        return tuple(result), tuple(log)

    def _update_mismatch_counts(self, frame: TypedRuntimeFrame) -> None:
        observed = set()
        for item in frame.execution_feedback:
            key = (item.agent_id, item.actuator_id)
            observed.add(key)
            comparable = item.requested_value is not None and item.applied_value is not None
            mismatch = comparable and abs(float(item.requested_value) - float(item.applied_value)) > 1.0e-3
            self._mismatch_counts[key] = self._mismatch_counts.get(key, 0) + 1 if mismatch else 0
        # Do not increase counters when feedback is absent. A good matching
        # feedback sample is required to clear an isolated group.

    def _default_fail_safe(
        self,
        group: ActionGroupInstance,
        actuator: ActuatorDefinition,
        interface: TypedAgentInterface,
        parts: Mapping[Tuple[str, str], ObservationPart],
        health: Mapping[Tuple[str, str], HealthState],
    ) -> Tuple[ActionGroupInstance, list[Mapping[str, Any]]]:
        rows = []
        safety_paths = []
        charger_paths = []
        for sensor in interface.sensors:
            for observation in sensor.observations:
                path = observation.path
                meter_name = observation.observation_id.lower()
                if sensor.sensor_type == "building_meter" and (
                    observation.channel_id == "grid"
                    or any(
                        token in meter_name
                        for token in (
                            "net_power",
                            "import_power",
                            "export_power",
                            "phase_power",
                        )
                    )
                ):
                    safety_paths.append(path)
                if sensor.sensor_id == actuator.actuator_id or (
                    actuator.source_entity_type == "charger"
                    and sensor.sensor_type in {"bidirectional_ev_charger", "ev_charger"}
                ):
                    charger_paths.append((path, observation.observation_id))

        unsafe_meter = any(
            health.get((group.owner_agent_id, path), HealthState.UNKNOWN)
            in {
                HealthState.STALE,
                HealthState.MISSING,
                HealthState.FAILED,
                HealthState.UNKNOWN,
            }
            for path in safety_paths
        )
        if unsafe_meter:
            group = self._safe_idle(group, "main_or_grid_meter_unavailable")
            rows.append(self._closure_row(group, "main_or_grid_meter_unavailable"))
            return group, rows

        if actuator.source_entity_type == "charger":
            connected_paths = [path for path, name in charger_paths if "connected" in name]
            connection_confirmed = any(
                (
                    parts.get((group.owner_agent_id, path)) is not None
                    and parts[(group.owner_agent_id, path)].values[0] > 0.5
                    and health.get((group.owner_agent_id, path)) == HealthState.HEALTHY
                )
                for path in connected_paths
            )
            if not connection_confirmed:
                group = self._invalidate_modes(
                    group,
                    {port.mode for port in group.ports if port.mode != "IDLE"},
                    "charger_connection_unconfirmed",
                )
                rows.append(self._closure_row(group, "charger_connection_unconfirmed"))
                return group, rows
            sensitive_names = ("soc", "required_soc", "departure", "schedule")
            uncertain_service = any(
                any(token in name for token in sensitive_names)
                and health.get((group.owner_agent_id, path), HealthState.UNKNOWN)
                in {
                    HealthState.DEGRADED,
                    HealthState.STALE,
                    HealthState.MISSING,
                    HealthState.FAILED,
                    HealthState.UNKNOWN,
                }
                for path, name in charger_paths
            )
            if uncertain_service:
                group = self._invalidate_modes(group, {"DISCHARGE_EV"}, "uncertain_ev_service")
                charge = next(
                    (port for port in group.ports if port.mode == "CHARGE_EV" and port.valid),
                    None,
                )
                if charge is not None:
                    group = replace(
                        group,
                        degraded_mode="URGENT_SAFE_CHARGE",
                        forced_mode="CHARGE_EV",
                        forced_fraction=1.0,
                        fallback_reason="uncertain_ev_service",
                    )
                rows.append(self._closure_row(group, "uncertain_ev_service"))

        if actuator.source_entity_type == "storage":
            storage_sensor_ids = {
                sensor.sensor_id
                for sensor in interface.sensors
                if sensor.sensor_type == "stationary_battery"
            }
            storage_soc_paths = [
                path
                for (agent_id, path), part in parts.items()
                if agent_id == group.owner_agent_id
                and part.sensor_id in storage_sensor_ids
                and "soc" in part.observation_id
            ]
            if not storage_soc_paths or any(
                health.get((group.owner_agent_id, path), HealthState.UNKNOWN)
                in {HealthState.MISSING, HealthState.FAILED, HealthState.UNKNOWN}
                for path in storage_soc_paths
            ):
                group = self._safe_idle(group, "stationary_storage_state_unavailable")
                rows.append(self._closure_row(group, "stationary_storage_state_unavailable"))

        community_parts = [
            part
            for (agent_id, _path), part in parts.items()
            if agent_id == group.owner_agent_id and part.scope == "community"
        ]
        if community_parts and not any(part.valid for part in community_parts):
            group = replace(
                group,
                degraded_mode=group.degraded_mode or "LOCAL_ONLY",
            )
            rows.append(self._closure_row(group, "community_coordination_unavailable"))
        price_parts = [
            part
            for part in community_parts
            if "price" in part.observation_id.lower()
            or "pricing" in part.observation_id.lower()
        ]
        if price_parts and not any(part.valid for part in price_parts):
            group = replace(
                group,
                degraded_mode=group.degraded_mode or "NO_PRICE_ARBITRAGE",
            )
            rows.append(self._closure_row(group, "price_optimization_unavailable"))
        return group, rows

    @staticmethod
    def _apply_dependency_effect(
        group: ActionGroupInstance,
        action: ActionDefinition,
        effect: str,
        reason: str,
    ) -> ActionGroupInstance:
        if effect in {"allow", "degraded_input", "remove_optimization"}:
            return replace(group, degraded_mode=("DEGRADED_INPUT" if effect != "allow" else group.degraded_mode))
        if effect in {"invalidate", "invalidate_port", "contract_zero", "no_v2g"}:
            return TypedInterfaceCompiler._invalidate_modes(group, {action.mode}, reason)
        if effect in {"disable_group", "safe_idle", "isolated_safe", "fallback_idle"}:
            return TypedInterfaceCompiler._safe_idle(group, reason)
        if effect in {"max_safe_charge", "urgent_safe_charge"}:
            charge_mode = "CHARGE_EV" if group.group_type == "ev_session" else "CHARGE_STATIONARY"
            group = TypedInterfaceCompiler._invalidate_modes(
                group,
                {
                    mode
                    for mode in (port.mode for port in group.ports)
                    if mode.startswith("DISCHARGE_")
                },
                reason,
            )
            charge = next((port for port in group.ports if port.mode == charge_mode and port.valid), None)
            if charge is None:
                return TypedInterfaceCompiler._safe_idle(group, reason)
            return replace(
                group,
                degraded_mode="URGENT_SAFE_CHARGE",
                forced_mode=charge_mode,
                forced_fraction=1.0,
                fallback_reason=reason,
            )
        raise ValueError(f"Unsupported TI-MARL dependency effect: {effect!r}")

    @staticmethod
    def _invalidate_modes(
        group: ActionGroupInstance,
        modes: set[str],
        reason: str,
    ) -> ActionGroupInstance:
        return replace(
            group,
            ports=tuple(
                replace(
                    port,
                    valid=False,
                    invalid_reasons=tuple(sorted(set(port.invalid_reasons) | {reason})),
                )
                if port.mode in modes
                else port
                for port in group.ports
            ),
            degraded_mode=group.degraded_mode or "LOCAL_FALLBACK",
        )

    @staticmethod
    def _safe_idle(group: ActionGroupInstance, reason: str) -> ActionGroupInstance:
        return replace(
            group,
            enabled=False,
            ports=tuple(
                port
                if port.mode == "IDLE"
                else replace(
                    port,
                    valid=False,
                    invalid_reasons=tuple(sorted(set(port.invalid_reasons) | {reason})),
                )
                for port in group.ports
            ),
            degraded_mode="ISOLATED_SAFE",
            forced_mode="IDLE",
            forced_fraction=0.0,
            fallback_reason=reason,
        )

    @staticmethod
    def _closure_row(
        group: ActionGroupInstance,
        consequence: str,
        **extra: Any,
    ) -> Mapping[str, Any]:
        return {
            "rule_id": "ti_marl_failsafe_v1",
            "target_id": group.group_id,
            "consequence": consequence,
            **extra,
        }

    def _constraints(
        self,
        frame: TypedRuntimeFrame,
        samples: Mapping[str, TypedObservationSample],
        groups: Sequence[ActionGroupInstance],
        parts: Sequence[ObservationPart],
    ) -> Tuple[LocalConstraint, ...]:
        constraints = []
        for agent_id in frame.active_agent_ids:
            interface = self.interface_registry.for_agent(agent_id)
            member_ids = tuple(group.group_id for group in groups if group.owner_agent_id == agent_id)
            charge_headroom = self._headroom(parts, agent_id, export=False)
            export_headroom = self._headroom(parts, agent_id, export=True)
            configured_grid = interface.constraints.get("grid_import", {})
            configured_max = (
                float(configured_grid.get("max"))
                if isinstance(configured_grid, Mapping) and configured_grid.get("max") is not None
                else None
            )
            configured_export = interface.constraints.get("grid_export", {})
            configured_export_max = (
                float(configured_export.get("max"))
                if isinstance(configured_export, Mapping)
                and configured_export.get("max") is not None
                else None
            )
            if charge_headroom is None:
                charge_headroom = 0.0
            if configured_max is not None:
                charge_headroom = min(charge_headroom, max(configured_max, 0.0))
            if export_headroom is None:
                export_headroom = 0.0
            if configured_export_max is not None:
                export_headroom = min(
                    export_headroom,
                    max(configured_export_max, 0.0),
                )
            constraints.extend(
                (
                    LocalConstraint(
                        constraint_id=f"{agent_id}:charging_headroom_kw",
                        owner_agent_id=agent_id,
                        constraint_type="charging_headroom_kw",
                        upper_bound=charge_headroom,
                        member_group_ids=member_ids,
                    ),
                    LocalConstraint(
                        constraint_id=f"{agent_id}:export_headroom_kw",
                        owner_agent_id=agent_id,
                        constraint_type="export_headroom_kw",
                        upper_bound=export_headroom,
                        member_group_ids=member_ids,
                    ),
                )
            )
        return tuple(constraints)

    @staticmethod
    def _headroom(
        parts: Sequence[ObservationPart],
        agent_id: str,
        *,
        export: bool,
    ) -> Optional[float]:
        candidates = []
        for part in parts:
            if (
                part.owner_agent_id != agent_id
                or not part.valid
                or part.scope != "local"
                or part.unit != "kW"
            ):
                continue
            name = part.observation_id.lower()
            if "headroom" not in name:
                continue
            is_export = "export" in name or "discharge" in name
            if is_export == export:
                candidates.append(max(float(part.values[0]), 0.0))
        return min(candidates) if candidates else None

    def _dependencies(self) -> Tuple[Dependency, ...]:
        result = []
        for interface in self.interface_registry.interfaces.values():
            for actuator in interface.actuators:
                for action in actuator.actions:
                    for path in action.dependencies:
                        result.append(
                            Dependency(
                                dependency_id=f"{interface.agent_id}:{actuator.actuator_id}:{action.action_id}:{path}",
                                source_kind="observation",
                                source_type=path,
                                target_group_type=actuator.group_type,
                                target_semantic_type=None,
                                consequence="typed_outcome",
                                condition_states=tuple(
                                    HealthState(state)
                                    for state in ("DEGRADED", "STALE", "MISSING", "FAILED", "UNKNOWN")
                                ),
                            )
                        )
        return tuple(sorted(result, key=lambda item: item.dependency_id))
