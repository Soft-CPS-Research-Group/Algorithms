"""Discovery, binding, health evaluation and snapshot compilation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from algorithms.ti_marl.compiler.closure import apply_closure, validate_dependency_graph
from algorithms.ti_marl.compiler.health import HealthDeriver
from algorithms.ti_marl.contracts.compatibility import CompatibilitySignature
from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    HealthState,
    QualityState,
)
from algorithms.ti_marl.contracts.models import (
    ActionGroupInstance,
    ActionPortInstance,
    AgentSchema,
    Dependency,
    EntityInstance,
    FaultEvidence,
    InterfaceSnapshot,
    LocalConstraint,
    ModuleInstance,
    ObservationPart,
    SharedResource,
)


COMPILER_VERSION = "tic_v1"


def load_versioned_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"TI-MARL contract file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or not str(payload.get("version", "")).strip():
        raise ValueError(f"TI-MARL contract file must contain a non-empty version: {resolved}")
    return payload


class TypedInterfaceCompiler:
    """Compile ``entity_v1 + runtime_status_v1`` into ``ti_marl_v1``."""

    def __init__(
        self,
        *,
        contract_version: str,
        agent_schema_path: str,
        type_registry_path: str,
        health_rules_path: str,
    ) -> None:
        self.contract_version = str(contract_version)
        if self.contract_version != "ti_marl_v1":
            raise ValueError(f"Unsupported TI-MARL contract version: {self.contract_version!r}")
        self.agent_schema_config = load_versioned_yaml(agent_schema_path)
        self.type_registry = load_versioned_yaml(type_registry_path)
        self.health_rules = load_versioned_yaml(health_rules_path)
        validate_dependency_graph(self.agent_schema_config.get("dependencies", []))
        self.health_deriver = HealthDeriver(self.health_rules)
        self.entity_specs: Dict[str, Any] = {}
        self.agent_schema = self._agent_schema()
        self.compatibility_signature = CompatibilitySignature.build(
            contract_version=self.contract_version,
            agent_schema=self.agent_schema_config,
            type_registry=self.type_registry,
            health_rules=self.health_rules,
            compiler_version=COMPILER_VERSION,
        )

    def attach_entity_specs(self, entity_specs: Mapping[str, Any]) -> None:
        specs = deepcopy(dict(entity_specs or {}))
        if str(specs.get("version")) != "entity_v1":
            raise ValueError("TIMARL requires Simulator entity_specs version='entity_v1'")
        status_contract = specs.get("runtime_status_contract", {})
        if str(status_contract.get("version")) != "runtime_status_v1":
            raise ValueError("TIMARL requires Simulator runtime_status_v1")
        execution_contract = specs.get("action_execution_contract", {})
        if str(execution_contract.get("version")) != "entity_action_execution_v1":
            raise ValueError("TIMARL requires Simulator entity_action_execution_v1")
        known_types = set(self.type_registry.get("entity_types", {}))
        emitted_types = {
            str(name)
            for name, table in specs.get("tables", {}).items()
            if len(table.get("ids", [])) > 0
        }
        unknown = sorted(emitted_types - known_types)
        if unknown:
            raise ValueError(f"TI-MARL type registry does not classify entity types: {unknown}")
        self.entity_specs = specs

    def snapshot_state(self) -> Mapping[str, Any]:
        return {
            "entity_specs": deepcopy(self.entity_specs),
            "health": self.health_deriver.snapshot_state(),
        }

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        self.entity_specs = deepcopy(dict(payload.get("entity_specs", {})))
        self.health_deriver.restore_state(payload.get("health", {}))

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Persist semantic runtime state without freezing one composition."""

        return {"health": self.health_deriver.snapshot_state()}

    def load_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        self.health_deriver.restore_state(payload.get("health", {}))

    def compile(self, payload: Mapping[str, Any]) -> InterfaceSnapshot:
        if not self.entity_specs:
            raise RuntimeError("TypedInterfaceCompiler.attach_entity_specs() must be called first")
        meta = payload.get("meta", {})
        if str(meta.get("spec_version")) != "entity_v1":
            raise ValueError("TIMARL received an observation that is not entity_v1")
        runtime_status = meta.get("runtime_status")
        if not isinstance(runtime_status, Mapping) or str(runtime_status.get("version")) != "runtime_status_v1":
            raise ValueError("TIMARL requires meta.runtime_status.version='runtime_status_v1'")
        if runtime_status.get("emits_health_state") is not False:
            raise ValueError("Simulator runtime_status must be facts-only (emits_health_state=false)")

        entities, owners = self._discover_entities(payload)
        discovered_agent_ids = {
            entity.entity_id
            for entity in entities
            if entity.entity_type == self.agent_schema.agent_entity_type
        }
        # Preserve the Simulator's canonical building order.  Action vectors,
        # rewards and action-execution entries all use this order; sorting IDs
        # lexicographically would silently associate Building_10's transition
        # with Building_2 on populations larger than nine agents.
        schema_agent_ids = tuple(
            str(item)
            for item in self.entity_specs.get("tables", {})
            .get(self.agent_schema.agent_entity_type, {})
            .get("ids", [])
        )
        agent_ids = tuple(
            agent_id for agent_id in schema_agent_ids if agent_id in discovered_agent_ids
        )
        if not agent_ids:
            raise ValueError("TIMARL discovery found no active building agents")
        modules = self._discover_modules(entities, owners)
        evidence = self._fault_evidence(runtime_status)
        modules = self._apply_module_facts(modules, evidence)
        nominal_subjects = self._nominal_health_subjects(entities, evidence)
        health = self.health_deriver.derive(
            evidence,
            time_step=int(meta.get("time_step", 0)),
            nominal_subjects=nominal_subjects,
        )
        dependencies = self._dependencies()
        parts = self._observation_parts(entities, agent_ids)
        groups = self._action_groups(entities, owners, payload)
        groups, parts, closure_log = apply_closure(
            groups=groups,
            parts=parts,
            evidence=evidence,
            health=health,
            dependencies=dependencies,
        )
        constraints = self._constraints(entities, groups)
        shared_resources = self._shared_resources(agent_ids)
        return InterfaceSnapshot(
            contract_version=self.contract_version,
            compiler_version=COMPILER_VERSION,
            topology_version=int(meta.get("topology_version", 0)),
            time_step=int(meta.get("time_step", 0)),
            agent_ids=agent_ids,
            modules=modules,
            entities=entities,
            fault_evidence=evidence,
            health=health,
            observation_parts=parts,
            action_groups=groups,
            dependencies=dependencies,
            constraints=constraints,
            shared_resources=shared_resources,
            closure_log=closure_log,
        )

    def _agent_schema(self) -> AgentSchema:
        cfg = self.agent_schema_config
        return AgentSchema(
            version=str(cfg["version"]),
            agent_entity_type=str(cfg.get("agent_entity_type", "building")),
            module_types=tuple(str(item) for item in cfg.get("module_types", [])),
            action_group_types=tuple(str(item) for item in cfg.get("action_group_types", [])),
            observation_semantic_types=tuple(
                str(item) for item in cfg.get("observation_semantic_types", [])
            ),
        )

    def _discover_entities(
        self,
        payload: Mapping[str, Any],
    ) -> Tuple[Tuple[EntityInstance, ...], Mapping[Tuple[str, int], str]]:
        tables = payload.get("tables", {})
        edges = payload.get("edges", {})
        table_specs = self.entity_specs.get("tables", {})
        building_ids = tuple(str(item) for item in table_specs.get("building", {}).get("ids", []))
        if len(set(building_ids)) != len(building_ids):
            raise ValueError("TI-MARL binding failed: duplicate building IDs")
        owners: Dict[Tuple[str, int], str] = {
            ("building", index): agent_id for index, agent_id in enumerate(building_ids)
        }
        relation_to_type = {
            "building_to_storage": "storage",
            "building_to_charger": "charger",
            "building_to_deferrable_appliance": "deferrable_appliance",
            "building_to_pv": "pv",
        }
        for relation, entity_type in relation_to_type.items():
            for source, target in self._edge_pairs(edges.get(relation)):
                if 0 <= source < len(building_ids):
                    self._bind_owner(owners, (entity_type, target), building_ids[source])

        # EV ownership follows actual or incoming charger relations.
        charger_owner = {
            row: owner for (kind, row), owner in owners.items() if kind == "charger"
        }
        for relation in ("charger_to_ev_connected", "charger_to_ev_incoming"):
            pairs = self._edge_pairs(edges.get(relation))
            mask = np.asarray(edges.get(f"{relation}_mask", []), dtype=np.float64).reshape(-1)
            for index, (charger_row, ev_row) in enumerate(pairs):
                if index < len(mask) and mask[index] <= 0.5:
                    continue
                if charger_row in charger_owner and ev_row >= 0:
                    self._bind_owner(owners, ("ev", ev_row), charger_owner[charger_row])

        entities = []
        for entity_type in sorted(table_specs):
            spec = table_specs[entity_type]
            ids = [str(item) for item in spec.get("ids", [])]
            features = tuple(str(item) for item in spec.get("features", []))
            values = np.asarray(tables.get(entity_type, []), dtype=np.float64)
            if values.ndim == 1 and values.size:
                values = values.reshape(1, -1)
            for row, entity_id in enumerate(ids):
                if values.ndim != 2 or row >= values.shape[0]:
                    continue
                owner = owners.get((entity_type, row))
                if entity_type == "district":
                    owner = None
                entities.append(
                    EntityInstance(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        owner_agent_id=owner,
                        row_index=row,
                        feature_names=features,
                        values=tuple(self._finite(value) for value in values[row, : len(features)]),
                    )
                )
        return tuple(sorted(entities, key=lambda item: (item.entity_type, item.entity_id))), owners

    @staticmethod
    def _bind_owner(
        owners: Dict[Tuple[str, int], str],
        key: Tuple[str, int],
        owner: str,
    ) -> None:
        previous = owners.get(key)
        if previous is not None and previous != owner:
            raise ValueError(
                f"TI-MARL ambiguous binding for {key[0]} row {key[1]}: "
                f"{previous!r} and {owner!r}"
            )
        owners[key] = owner

    def _discover_modules(
        self,
        entities: Sequence[EntityInstance],
        owners: Mapping[Tuple[str, int], str],
    ) -> Tuple[ModuleInstance, ...]:
        modules = []
        module_types = {"building", "storage", "charger", "deferrable_appliance", "pv"}
        for entity in entities:
            if entity.entity_type not in module_types:
                continue
            owner = entity.owner_agent_id or (
                entity.entity_id if entity.entity_type == "building" else None
            )
            if owner is None:
                continue
            modules.append(
                ModuleInstance(
                    module_id=entity.entity_id,
                    module_type=entity.entity_type,
                    owner_agent_id=owner,
                    entity_id=entity.entity_id,
                )
            )
        return tuple(sorted(modules, key=lambda item: item.module_id))

    def _fault_evidence(self, status: Mapping[str, Any]) -> Tuple[FaultEvidence, ...]:
        collection_domains = {
            "asset_connections": EventDomain.ASSET_CONNECTION,
            "asset_availability": EventDomain.ASSET_AVAILABILITY,
            "sensor_channels": EventDomain.SENSOR_CHANNEL,
            "actuator_channels": EventDomain.ACTUATOR_CHANNEL,
            "communication_links": EventDomain.COMMUNICATION_LINK,
            "value_quality": EventDomain.VALUE_QUALITY,
        }
        active_events = {
            str(row.get("event_id")): dict(row)
            for row in status.get("active_events", []) or []
            if row.get("event_id") is not None
        }
        evidence = []
        for collection, domain in collection_domains.items():
            for index, row in enumerate(status.get(collection, []) or []):
                if domain == EventDomain.ASSET_CONNECTION:
                    target_id = str(row.get("source_id") or row.get("target_id") or "*")
                    target_type = str(row.get("source_type") or row.get("target_type") or "*")
                    target_feature = str(row.get("relation") or row.get("target_feature") or "*")
                else:
                    target_id = str(row.get("target_id") or row.get("source_id") or "*")
                    target_type = str(row.get("target_type") or row.get("source_type") or "*")
                    target_feature = str(row.get("target_feature") or "*")
                event_ids = tuple(sorted(str(item) for item in row.get("event_ids", []) or []))
                raw_id = str(row.get("event_id") or (event_ids[0] if event_ids else f"{collection}:{index}"))
                active = active_events.get(raw_id, {})
                evidence.append(
                    FaultEvidence(
                        evidence_id=f"{domain.value}:{target_type}:{target_id}:{target_feature}",
                        event_domain=domain,
                        fault_mode=None if row.get("fault_mode") is None else str(row.get("fault_mode")),
                        target_type=target_type,
                        target_id=target_id,
                        target_feature=target_feature,
                        availability=self._enum(AvailabilityState, row.get("availability"), AvailabilityState.UNKNOWN),
                        connection=self._enum(ConnectionState, row.get("connection"), ConnectionState.UNKNOWN),
                        quality=self._enum(QualityState, row.get("quality"), QualityState.UNKNOWN),
                        start_time_step=self._optional_int(
                            row.get("start_time_step", active.get("start_time_step"))
                        ),
                        active_duration_steps=max(
                            int(row.get("active_duration_steps", active.get("active_duration_steps", 0)) or 0),
                            0,
                        ),
                        last_update_time_step=self._optional_int(row.get("last_update_time_step")),
                        last_fresh_time_step=self._optional_int(row.get("last_fresh_time_step")),
                        age_steps=self._optional_int(row.get("age_steps")),
                        event_ids=event_ids or (raw_id,),
                    )
                )
        return tuple(sorted(evidence, key=lambda item: item.evidence_id))

    @staticmethod
    def _apply_module_facts(
        modules: Sequence[ModuleInstance],
        evidence: Sequence[FaultEvidence],
    ) -> Tuple[ModuleInstance, ...]:
        availability = {
            item.target_id: item.availability
            for item in evidence
            if item.event_domain == EventDomain.ASSET_AVAILABILITY
        }
        connections = {
            item.target_id: item.connection
            for item in evidence
            if item.event_domain == EventDomain.ASSET_CONNECTION
        }
        return tuple(
            replace(
                module,
                available=availability.get(module.module_id, module.available),
                connected=connections.get(module.module_id, module.connected),
            )
            for module in modules
        )

    def _nominal_health_subjects(
        self,
        entities: Iterable[EntityInstance],
        evidence: Iterable[FaultEvidence],
    ) -> Mapping[str, Tuple[str, str]]:
        subjects = {}
        for entity in entities:
            semantic = str(
                self.type_registry.get("entity_types", {})
                .get(entity.entity_type, {})
                .get("semantic_type", "local_energy")
            )
            subjects[f"ENTITY:{entity.entity_type}:{entity.entity_id}:*"] = (
                semantic,
                "operational" if entity.entity_type != "district" else "advisory",
            )
        domain_semantics = {
            EventDomain.ASSET_CONNECTION: ("asset_connection", "operational"),
            EventDomain.ASSET_AVAILABILITY: ("asset_availability", "safety"),
            EventDomain.ACTUATOR_CHANNEL: ("actuator_channel", "safety"),
            EventDomain.COMMUNICATION_LINK: ("community_signal", "advisory"),
        }
        registry = self.type_registry.get("entity_types", {})
        for item in evidence:
            semantic, criticality = domain_semantics.get(
                item.event_domain,
                (
                    str(registry.get(item.target_type, {}).get("semantic_type", "local_energy")),
                    "operational",
                ),
            )
            subjects[
                f"{item.event_domain.value}:{item.target_type}:{item.target_id}:{item.target_feature or '*'}"
            ] = (semantic, criticality)
        return subjects

    def _observation_parts(
        self,
        entities: Sequence[EntityInstance],
        agent_ids: Sequence[str],
    ) -> Tuple[ObservationPart, ...]:
        parts = []
        registry = self.type_registry.get("entity_types", {})
        for entity in entities:
            type_config = registry.get(entity.entity_type)
            if type_config is None:
                continue
            selected = tuple(str(item) for item in type_config.get("features", []))
            index = {feature: position for position, feature in enumerate(entity.feature_names)}
            # Registry order is the stable typed slot layout. Optional fields
            # absent from one composition retain a zero slot instead of
            # shifting every later feature into a different network input.
            values = tuple(
                entity.values[index[feature]] if feature in index else 0.0
                for feature in selected
            )
            owners = agent_ids if entity.entity_type == "district" else (
                (entity.owner_agent_id,) if entity.owner_agent_id is not None else ()
            )
            for owner in owners:
                parts.append(
                    ObservationPart(
                        part_id=f"{owner}:{entity.entity_type}:{entity.entity_id}",
                        owner_agent_id=str(owner),
                        source_entity_id=entity.entity_id,
                        semantic_type=str(type_config.get("semantic_type", "local_energy")),
                        feature_names=selected,
                        values=values,
                        health=HealthState.HEALTHY,
                    )
                )
        return tuple(sorted(parts, key=lambda item: item.part_id))

    def _action_groups(
        self,
        entities: Sequence[EntityInstance],
        owners: Mapping[Tuple[str, int], str],
        payload: Mapping[str, Any],
    ) -> Tuple[ActionGroupInstance, ...]:
        by_type = {(item.entity_type, item.row_index): item for item in entities}
        groups = []
        action_types = self.type_registry.get("action_group_types", {})
        for group_type in sorted(action_types):
            config = action_types[group_type]
            entity_type = str(config["entity_type"])
            for (kind, row), owner in sorted(owners.items()):
                if kind != entity_type or (kind, row) not in by_type:
                    continue
                entity = by_type[(kind, row)]
                values = dict(zip(entity.feature_names, entity.values))
                charge_bound = self._available_bound(values, "charge")
                discharge_bound = self._available_bound(values, "discharge")
                if group_type == "deferrable":
                    start_bound = self._available_bound(values, "start")
                else:
                    start_bound = 1.0
                ports = []
                for mode in config.get("modes", []):
                    upper = 0.0 if str(mode) == "IDLE" else 1.0
                    if "CHARGE" in str(mode):
                        upper = charge_bound
                    elif "DISCHARGE" in str(mode):
                        upper = discharge_bound
                    elif str(mode) == "START":
                        upper = start_bound
                    ports.append(
                        ActionPortInstance(
                            port_id=f"{entity.entity_id}:{mode}",
                            mode=str(mode),
                            target_entity_id=entity.entity_id,
                            action_name=str(config["action_name"]),
                            lower_bound=0.0,
                            upper_bound=float(np.clip(upper, 0.0, 1.0)),
                            valid=str(mode) == "IDLE" or upper > 0.0,
                            invalid_reasons=() if str(mode) == "IDLE" or upper > 0.0 else ("zero_runtime_bound",),
                            contracted_by=() if upper >= 1.0 else ("runtime_available_action",),
                        )
                    )
                groups.append(
                    ActionGroupInstance(
                        group_id=f"{group_type}:{entity.entity_id}",
                        group_type=group_type,
                        owner_agent_id=owner,
                        module_id=entity.entity_id,
                        ports=tuple(ports),
                        max_charge_power_kw=self._power(values, charge=True),
                        max_discharge_power_kw=self._power(values, charge=False),
                    )
                )
        return tuple(sorted(groups, key=lambda item: item.group_id))

    def _constraints(
        self,
        entities: Sequence[EntityInstance],
        groups: Sequence[ActionGroupInstance],
    ) -> Tuple[LocalConstraint, ...]:
        building_entities = {item.entity_id: item for item in entities if item.entity_type == "building"}
        constraints = []
        for agent_id, entity in sorted(building_entities.items()):
            values = dict(zip(entity.feature_names, entity.values))
            member_ids = tuple(group.group_id for group in groups if group.owner_agent_id == agent_id)
            raw_charge_headroom = values.get("charging_building_headroom_kw")
            raw_export_headroom = values.get("charging_building_export_headroom_kw")
            charge_headroom = (
                None
                if raw_charge_headroom is None or not np.isfinite(raw_charge_headroom)
                else max(float(raw_charge_headroom), 0.0)
            )
            export_headroom = (
                None
                if raw_export_headroom is None or not np.isfinite(raw_export_headroom)
                else max(float(raw_export_headroom), 0.0)
            )
            constraints.extend(
                [
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
                ]
            )
        return tuple(constraints)

    def _dependencies(self) -> Tuple[Dependency, ...]:
        dependencies = []
        for row in self.agent_schema_config.get("dependencies", []):
            dependencies.append(
                Dependency(
                    dependency_id=str(row["dependency_id"]),
                    source_kind=str(row.get("source_kind", "*")),
                    source_type=str(row.get("source_type", "*")),
                    target_group_type=(
                        None
                        if row.get("target_group_type") is None
                        else str(row["target_group_type"])
                    ),
                    target_semantic_type=(
                        None
                        if row.get("target_semantic_type") is None
                        else str(row["target_semantic_type"])
                    ),
                    consequence=str(row.get("consequence", "")),
                    condition_states=tuple(HealthState(str(item)) for item in row.get("condition_states", [])),
                    parameter=None if row.get("parameter") is None else float(row["parameter"]),
                )
            )
        return tuple(sorted(dependencies, key=lambda item: item.dependency_id))

    def _shared_resources(self, agent_ids: Sequence[str]) -> Tuple[SharedResource, ...]:
        return tuple(
            SharedResource(
                resource_id=str(row["resource_id"]),
                resource_type=str(row["resource_type"]),
                member_agent_ids=tuple(agent_ids),
                observable_only=bool(row.get("observable_only", True)),
            )
            for row in self.agent_schema_config.get("shared_resources", [])
        )

    @staticmethod
    def _edge_pairs(values: Any) -> list[Tuple[int, int]]:
        array = np.asarray(values if values is not None else [], dtype=np.int64)
        if array.ndim != 2 or array.shape[1] < 2:
            return []
        return [(int(row[0]), int(row[1])) for row in array]

    @staticmethod
    def _finite(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        return result if np.isfinite(result) else 0.0

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _enum(enum_type, value: Any, default):
        try:
            return enum_type(str(value))
        except ValueError:
            return default

    @staticmethod
    def _available_bound(values: Mapping[str, float], direction: str) -> float:
        aliases = {
            "charge": ("available_charge_action_normalized", "available_charging_action_normalized"),
            "discharge": ("available_discharge_action_normalized", "available_discharging_action_normalized"),
            "start": ("available_start_action_normalized",),
        }
        for name in aliases[direction]:
            if name in values:
                return float(np.clip(values[name], 0.0, 1.0))
        if direction == "start" and "can_start" in values:
            return float(values["can_start"] > 0.5)
        return 1.0

    @staticmethod
    def _power(values: Mapping[str, float], *, charge: bool) -> float:
        aliases = (
            ("max_charge_power_kw", "max_charging_power_kw")
            if charge
            else ("max_discharge_power_kw", "max_discharging_power_kw")
        )
        for name in aliases:
            if name in values:
                return max(float(values[name]), 0.0)
        return 0.0
