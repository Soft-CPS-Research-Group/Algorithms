"""User-authored, single-file definition of a TI-MARL typed interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml


TYPED_INTERFACE_VERSION = "typed_interface_v1"


def _mapping(value: Any, path: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"TI-MARL {path} must be a mapping")
    return deepcopy(dict(value))


def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"TI-MARL {path} must be a list")
    items = [str(item) for item in value]
    if not allow_empty and not items:
        raise ValueError(f"TI-MARL {path} must not be empty")
    if len(items) != len(set(items)):
        raise ValueError(f"TI-MARL {path} contains duplicate values")
    return items


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"TI-MARL typed interface not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"TI-MARL typed interface must be a mapping: {resolved}")
    return deepcopy(dict(payload))


@dataclass(frozen=True)
class TypedInterfaceDefinition:
    """Validated public definition plus the compiler's three internal views.

    Users edit one document.  The split views remain an implementation detail
    so policy, health and checkpoint code do not need to understand file layout.
    """

    source_path: str
    raw: Mapping[str, Any]
    contract_version: str
    simulator_contracts: Mapping[str, str]
    agent_schema: Mapping[str, Any]
    type_registry: Mapping[str, Any]
    health_rules: Mapping[str, Any]
    required_features: Mapping[str, Tuple[str, ...]]
    catalog: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "TypedInterfaceDefinition":
        resolved = Path(path).expanduser().resolve()
        raw = _load_yaml(resolved)
        if str(raw.get("version")) != TYPED_INTERFACE_VERSION:
            raise ValueError(
                "TI-MARL typed interface requires "
                f"version='{TYPED_INTERFACE_VERSION}'"
            )
        allowed = {
            "version",
            "contract_version",
            "description",
            "simulator",
            "fixed",
            "observations",
            "actions",
            "health",
            "catalog",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"TI-MARL typed interface has unknown sections: {unknown}")

        contract_version = str(raw.get("contract_version", ""))
        if contract_version != "ti_marl_v1":
            raise ValueError("TI-MARL typed interface requires contract_version='ti_marl_v1'")

        simulator = _mapping(raw.get("simulator"), "simulator")
        simulator_contracts = {
            "entity": str(simulator.get("entity_contract", "")),
            "runtime_status": str(simulator.get("runtime_status_contract", "")),
            "action_execution": str(simulator.get("action_execution_contract", "")),
        }
        expected_contracts = {
            "entity": "entity_v1",
            "runtime_status": "runtime_status_v1",
            "action_execution": "entity_action_execution_v1",
        }
        if simulator_contracts != expected_contracts:
            raise ValueError(
                "TI-MARL typed interface has unsupported Simulator contracts: "
                f"{simulator_contracts}"
            )

        fixed = _mapping(raw.get("fixed"), "fixed")
        observations = _mapping(raw.get("observations"), "observations")
        actions = _mapping(raw.get("actions"), "actions")
        health = _mapping(raw.get("health"), "health")
        entities = _mapping(observations.get("entities"), "observations.entities")
        groups = _mapping(actions.get("groups"), "actions.groups")

        agent_entity_type = str(fixed.get("agent_entity_type", ""))
        if not agent_entity_type or agent_entity_type not in entities:
            raise ValueError(
                "TI-MARL fixed.agent_entity_type must name an observations.entities entry"
            )
        module_types = _string_list(fixed.get("module_types"), "fixed.module_types")
        semantic_types = _string_list(
            fixed.get("observation_semantic_types"),
            "fixed.observation_semantic_types",
        )
        relations = _string_list(fixed.get("relations"), "fixed.relations", allow_empty=True)

        required_features: Dict[str, Tuple[str, ...]] = {}
        normalized_entities: Dict[str, Dict[str, Any]] = {}
        for entity_type, raw_config in entities.items():
            config = _mapping(
                raw_config,
                f"observations.entities.{entity_type}",
            )
            features = _string_list(
                config.get("features"),
                f"observations.entities.{entity_type}.features",
                allow_empty=True,
            )
            required = _string_list(
                config.pop("required_features", []),
                f"observations.entities.{entity_type}.required_features",
                allow_empty=True,
            )
            missing_selection = sorted(set(required) - set(features))
            if missing_selection:
                raise ValueError(
                    f"TI-MARL required features for {entity_type!r} are not selected: "
                    f"{missing_selection}"
                )
            semantic_type = str(config.get("semantic_type", ""))
            if semantic_type not in semantic_types:
                raise ValueError(
                    f"TI-MARL entity {entity_type!r} uses unknown semantic type "
                    f"{semantic_type!r}"
                )
            config["features"] = features
            normalized_entities[str(entity_type)] = config
            required_features[str(entity_type)] = tuple(required)

        normalized_groups: Dict[str, Dict[str, Any]] = {}
        for group_type, raw_config in groups.items():
            config = _mapping(raw_config, f"actions.groups.{group_type}")
            entity_type = str(config.get("entity_type", ""))
            if entity_type not in normalized_entities:
                raise ValueError(
                    f"TI-MARL action group {group_type!r} targets unknown entity type "
                    f"{entity_type!r}"
                )
            modes = _string_list(
                config.get("modes"),
                f"actions.groups.{group_type}.modes",
            )
            if "IDLE" not in modes:
                raise ValueError(f"TI-MARL action group {group_type!r} must define IDLE")
            for field in ("action_table", "action_name"):
                if not str(config.get(field, "")).strip():
                    raise ValueError(
                        f"TI-MARL action group {group_type!r} requires {field}"
                    )
            config["modes"] = modes
            normalized_groups[str(group_type)] = config

        dependencies = list(fixed.get("dependencies", []) or [])
        local_constraints = list(fixed.get("local_constraints", []) or [])
        shared_resources = list(fixed.get("shared_resources", []) or [])
        for name, values in (
            ("fixed.dependencies", dependencies),
            ("fixed.local_constraints", local_constraints),
            ("fixed.shared_resources", shared_resources),
        ):
            if not all(isinstance(item, Mapping) for item in values):
                raise ValueError(f"TI-MARL {name} must contain mappings")

        schema_version = str(fixed.get("schema_version", ""))
        registry_version = str(observations.get("registry_version", ""))
        health_version = str(health.get("version", ""))
        if not schema_version or not registry_version or not health_version:
            raise ValueError(
                "TI-MARL fixed.schema_version, observations.registry_version and "
                "health.version are required"
            )

        agent_schema = {
            "version": schema_version,
            "agent_entity_type": agent_entity_type,
            "module_types": module_types,
            "observation_semantic_types": semantic_types,
            "action_group_types": list(normalized_groups),
            "dependencies": deepcopy(dependencies),
            "local_constraints": deepcopy(local_constraints),
            "shared_resources": deepcopy(shared_resources),
        }
        type_registry = {
            "version": registry_version,
            "feature_width": int(observations.get("feature_width", 16)),
            "entity_types": normalized_entities,
            "relation_types": relations,
            "action_group_types": normalized_groups,
        }
        if type_registry["feature_width"] < 1:
            raise ValueError("TI-MARL observations.feature_width must be positive")

        catalog = _mapping(raw.get("catalog", {}), "catalog")
        policy = str(catalog.get("policy", "compatible"))
        if catalog and policy not in {"compatible", "exact"}:
            raise ValueError("TI-MARL catalog.policy must be 'compatible' or 'exact'")

        return cls(
            source_path=str(resolved),
            raw=raw,
            contract_version=contract_version,
            simulator_contracts=simulator_contracts,
            agent_schema=agent_schema,
            type_registry=type_registry,
            health_rules=health,
            required_features=required_features,
            catalog=catalog,
        )

    def validate_entity_specs(self, entity_specs: Mapping[str, Any]) -> None:
        """Check a manual/generated interface against the live Simulator catalog."""

        specs = dict(entity_specs or {})
        actual_contracts = {
            "entity": str(specs.get("version", "")),
            "runtime_status": str(
                dict(specs.get("runtime_status_contract", {}) or {}).get("version", "")
            ),
            "action_execution": str(
                dict(specs.get("action_execution_contract", {}) or {}).get("version", "")
            ),
        }
        if actual_contracts != dict(self.simulator_contracts):
            raise ValueError(
                "TI-MARL typed interface and Simulator contract versions differ: "
                f"expected {dict(self.simulator_contracts)}, got {actual_contracts}"
            )

        tables = _mapping(specs.get("tables", {}), "entity_specs.tables")
        known_types = set(self.type_registry.get("entity_types", {}))
        unknown_types = sorted(set(tables) - known_types)
        if unknown_types:
            raise ValueError(
                "TI-MARL typed interface does not classify entity types: "
                f"{unknown_types}"
            )
        for entity_type, required in self.required_features.items():
            if entity_type not in tables:
                continue
            active_ids = list(tables[entity_type].get("ids", []) or [])
            if not active_ids:
                continue
            available = set(str(item) for item in tables[entity_type].get("features", []))
            missing = sorted(set(required) - available)
            if missing:
                raise ValueError(
                    f"TI-MARL Simulator is missing required {entity_type!r} observations: "
                    f"{missing}"
                )

        action_specs = _mapping(specs.get("actions", {}), "entity_specs.actions")
        for group_type, config in self.type_registry.get("action_group_types", {}).items():
            entity_type = str(config["entity_type"])
            # A generic interface may support optional module types that are
            # absent from this composition. Require their action contract as
            # soon as at least one such entity is active.
            active_ids = list(tables.get(entity_type, {}).get("ids", []) or [])
            if not active_ids:
                continue
            table = str(config["action_table"])
            action_name = str(config["action_name"])
            available = set(
                str(item) for item in action_specs.get(table, {}).get("features", [])
            )
            if action_name not in available:
                raise ValueError(
                    f"TI-MARL action group {group_type!r} requires Simulator action "
                    f"{table}.{action_name}"
                )

        if self.catalog:
            self._validate_catalog(tables, action_specs)

    def _validate_catalog(
        self,
        tables: Mapping[str, Any],
        action_specs: Mapping[str, Any],
    ) -> None:
        policy = str(self.catalog.get("policy", "compatible"))
        expected_observations = _mapping(
            self.catalog.get("observations", {}),
            "catalog.observations",
        )
        expected_actions = _mapping(self.catalog.get("actions", {}), "catalog.actions")
        for name, expected in expected_observations.items():
            expected_set = set(_string_list(expected, f"catalog.observations.{name}", allow_empty=True))
            actual_set = set(str(item) for item in tables.get(name, {}).get("features", []))
            if not expected_set.issubset(actual_set) or (
                policy == "exact" and expected_set != actual_set
            ):
                raise ValueError(
                    f"TI-MARL Simulator observation catalog differs for {name!r}"
                )
        for name, expected in expected_actions.items():
            expected_set = set(_string_list(expected, f"catalog.actions.{name}", allow_empty=True))
            actual_set = set(str(item) for item in action_specs.get(name, {}).get("features", []))
            if not expected_set.issubset(actual_set) or (
                policy == "exact" and expected_set != actual_set
            ):
                raise ValueError(f"TI-MARL Simulator action catalog differs for {name!r}")

    def with_simulator_catalog(
        self,
        entity_specs: Mapping[str, Any],
        *,
        source: str,
        policy: str = "compatible",
    ) -> Dict[str, Any]:
        """Return the same editable document with an auditable generated catalog."""

        if policy not in {"compatible", "exact"}:
            raise ValueError("catalog policy must be 'compatible' or 'exact'")
        self.validate_entity_specs(entity_specs)
        result = deepcopy(dict(self.raw))
        result["catalog"] = {
            "generated_from": str(source),
            "policy": policy,
            "observations": {
                str(name): [str(item) for item in spec.get("features", [])]
                for name, spec in sorted(entity_specs.get("tables", {}).items())
            },
            "actions": {
                str(name): [str(item) for item in spec.get("features", [])]
                for name, spec in sorted(entity_specs.get("actions", {}).items())
            },
        }
        return result
