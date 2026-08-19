"""Human-authored, deployment-neutral typed interfaces and atomic registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from algorithms.ti_marl.contracts.profile_registry import (
    CapabilityProfileRegistry,
    SUPPORTED_AGENT_TYPES,
    SUPPORTED_ROLES,
    SUPPORTED_UNITS,
)


TYPED_AGENT_INTERFACE_VERSION = "typed_agent_interface_v1"
CONTRACT_VERSION = "ti_marl_v1"
OBSERVATION_USES = {
    "policy_input",
    "safety_dependency",
    "runtime_bound",
    "trace_only",
    "excluded",
}
HEALTH_DEPENDENCY_STATES = ("DEGRADED", "STALE", "MISSING", "FAILED", "UNKNOWN")


def _mapping(value: Any, path: str, *, optional: bool = False) -> Dict[str, Any]:
    if value is None and optional:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"TI-MARL {path} must be a mapping")
    return deepcopy(dict(value))


def _identifier(value: Any, path: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"TI-MARL {path} must be a non-empty identifier")
    if any(character in result for character in ("/", "\\", ":")):
        raise ValueError(f"TI-MARL {path} contains a reserved character: {result!r}")
    return result


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"TI-MARL {path} has unknown fields: {unknown}")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"TI-MARL typed interface not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"TI-MARL typed interface must be a mapping: {path}")
    return deepcopy(dict(payload))


@dataclass(frozen=True)
class ObservationDefinition:
    sensor_id: str
    sensor_type: str
    scope: str
    channel_id: str
    observation_id: str
    source_feature: str
    unit: str
    dimensions: Mapping[str, Tuple[str, ...]]
    semantic_type: str
    use: str
    policy_input: bool
    criticality: str
    required: bool
    normalisation: str
    reason: Optional[str] = None

    @property
    def path(self) -> str:
        return f"{self.sensor_id}.{self.channel_id}.{self.observation_id}"

    def resolved(self) -> Mapping[str, Any]:
        result: Dict[str, Any] = {
            "source_feature": self.source_feature,
            "unit": self.unit,
            "semantic_type": self.semantic_type,
            "use": self.use,
            "policy_input": self.policy_input,
            "criticality": self.criticality,
            "required": self.required,
            "normalisation": self.normalisation,
        }
        if self.dimensions:
            result["dimensions"] = {
                key: list(values) for key, values in self.dimensions.items()
            }
        if self.reason is not None:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class SensorDefinition:
    sensor_id: str
    sensor_type: str
    scope: str
    profile: Optional[str]
    source_entity_id: Optional[str]
    observations: Tuple[ObservationDefinition, ...]


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    mode: str
    unit: str
    lower_bound: float
    upper_bound: float
    dependencies: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ActuatorDefinition:
    actuator_id: str
    actuator_type: str
    target_entity_id: Optional[str]
    group_type: str
    source_entity_type: str
    modes: Tuple[str, ...]
    actions: Tuple[ActionDefinition, ...]


@dataclass(frozen=True)
class TypedAgentInterface:
    source_path: str
    version: str
    contract_version: str
    description: str
    agent_id: str
    role: str
    agent_type: str
    sensors: Tuple[SensorDefinition, ...]
    actuators: Tuple[ActuatorDefinition, ...]
    constraints: Mapping[str, Any]
    fallback: Mapping[str, Any]
    raw: Mapping[str, Any]
    resolved_payload: Mapping[str, Any]
    interface_hash: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        profiles: CapabilityProfileRegistry | None = None,
    ) -> "TypedAgentInterface":
        registry = profiles or CapabilityProfileRegistry()
        resolved_path = Path(path).expanduser().resolve()
        raw = _load_yaml(resolved_path)
        version = str(raw.get("version", ""))
        if version == "typed_interface_v1":
            raise ValueError(
                "The global typed_interface_v1 format was retired; provide one "
                "typed_agent_interface_v1 YAML per registered agent"
            )
        if version != TYPED_AGENT_INTERFACE_VERSION:
            raise ValueError(
                "TI-MARL agent interface requires "
                f"version='{TYPED_AGENT_INTERFACE_VERSION}'"
            )
        allowed = {
            "version",
            "contract",
            "description",
            "agent",
            "sensors",
            "actuators",
            "constraints",
            "fallback",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"TI-MARL typed agent interface has unknown sections: {unknown}")
        contract = str(raw.get("contract", ""))
        if contract != CONTRACT_VERSION:
            raise ValueError(f"TI-MARL agent interface requires contract='{CONTRACT_VERSION}'")

        agent = _mapping(raw.get("agent"), "agent")
        _reject_unknown(agent, {"id", "role", "type"}, "agent")
        agent_id = _identifier(agent.get("id"), "agent.id")
        role = str(agent.get("role", "")).strip().lower()
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"TI-MARL agent.role must be one of {SUPPORTED_ROLES}")
        agent_type = str(agent.get("type", "")).strip().lower()
        if agent_type not in SUPPORTED_AGENT_TYPES:
            raise ValueError(f"TI-MARL agent.type must be one of {SUPPORTED_AGENT_TYPES}")

        sensors = cls._parse_sensors(raw.get("sensors"), registry)
        observation_paths = {
            observation.path
            for sensor in sensors
            for observation in sensor.observations
        }
        actuators = cls._parse_actuators(
            raw.get("actuators", {}),
            registry,
            observation_paths,
        )
        constraints = cls._parse_constraints(raw.get("constraints", {}))
        fallback = _mapping(raw.get("fallback", {}), "fallback")
        description = str(raw.get("description", "")).strip()

        resolved = {
            "version": TYPED_AGENT_INTERFACE_VERSION,
            "contract": contract,
            "description": description,
            "agent": {"id": agent_id, "role": role, "type": agent_type},
            "sensors": cls._resolved_sensors(sensors),
            "actuators": cls._resolved_actuators(actuators),
            "constraints": deepcopy(constraints),
            "fallback": deepcopy(fallback),
            "profile_registry_version": registry.version,
        }
        return cls(
            source_path=str(resolved_path),
            version=version,
            contract_version=contract,
            description=description,
            agent_id=agent_id,
            role=role,
            agent_type=agent_type,
            sensors=sensors,
            actuators=actuators,
            constraints=constraints,
            fallback=fallback,
            raw=raw,
            resolved_payload=resolved,
            interface_hash=_canonical_hash(resolved),
        )

    @staticmethod
    def _parse_sensors(
        raw_sensors: Any,
        profiles: CapabilityProfileRegistry,
    ) -> Tuple[SensorDefinition, ...]:
        sensors_payload = _mapping(raw_sensors, "sensors")
        if not sensors_payload:
            raise ValueError("TI-MARL agent interface must define at least one sensor")
        sensors = []
        for raw_sensor_id, raw_sensor in sensors_payload.items():
            sensor_id = _identifier(raw_sensor_id, "sensors.<id>")
            sensor = _mapping(raw_sensor, f"sensors.{sensor_id}")
            _reject_unknown(
                sensor,
                {"type", "scope", "profile", "channels"},
                f"sensors.{sensor_id}",
            )
            sensor_type = str(sensor.get("type", "")).strip()
            profiles.entity_type(sensor_type)
            scope = str(sensor.get("scope", "local")).strip().lower()
            if scope not in {"local", "community"}:
                raise ValueError(f"TI-MARL sensors.{sensor_id}.scope must be local or community")
            profile = None if sensor.get("profile") is None else str(sensor.get("profile"))
            expected_profile = f"{sensor_type}_v1"
            if profile is not None and profile != expected_profile:
                raise ValueError(
                    f"Unknown TI-MARL profile {profile!r} for sensor type "
                    f"{sensor_type!r}; expected {expected_profile!r}"
                )
            source_entity_id = None
            channels_payload = _mapping(sensor.get("channels"), f"sensors.{sensor_id}.channels")
            if not channels_payload:
                raise ValueError(f"TI-MARL sensor {sensor_id!r} must define channels")
            observations = []
            for raw_channel_id, raw_channel in channels_payload.items():
                channel_id = _identifier(raw_channel_id, f"sensors.{sensor_id}.channels.<id>")
                channel = _mapping(
                    raw_channel,
                    f"sensors.{sensor_id}.channels.{channel_id}",
                )
                _reject_unknown(
                    channel,
                    {"observations"},
                    f"sensors.{sensor_id}.channels.{channel_id}",
                )
                raw_observations = _mapping(
                    channel.get("observations"),
                    f"sensors.{sensor_id}.channels.{channel_id}.observations",
                )
                if not raw_observations:
                    raise ValueError(
                        f"TI-MARL channel {sensor_id}.{channel_id} must define observations"
                    )
                for raw_observation_id, raw_observation in raw_observations.items():
                    observation_id = _identifier(
                        raw_observation_id,
                        f"sensors.{sensor_id}.{channel_id}.observations.<id>",
                    )
                    observation = _mapping(
                        raw_observation,
                        f"sensors.{sensor_id}.{channel_id}.observations.{observation_id}",
                    )
                    _reject_unknown(
                        observation,
                        {
                            "unit",
                            "dimensions",
                            "semantic_type",
                            "use",
                            "policy_input",
                            "criticality",
                            "required",
                            "normalisation",
                            "reason",
                        },
                        f"sensors.{sensor_id}.{channel_id}.observations.{observation_id}",
                    )
                    defaults = profiles.observation_defaults(
                        sensor_type=sensor_type,
                        channel=channel_id,
                        observation=observation_id,
                        scope=scope,
                        unit=observation.get("unit"),
                    )
                    merged = {**defaults, **observation}
                    semantic_type = str(merged.get("semantic_type", "local_energy"))
                    if semantic_type not in profiles.supported_semantic_types:
                        raise ValueError(
                            f"TI-MARL observation {sensor_id}.{channel_id}."
                            f"{observation_id} uses unknown semantic type "
                            f"{semantic_type!r}"
                        )
                    use = str(merged.get("use", ""))
                    if use not in OBSERVATION_USES:
                        raise ValueError(
                            f"TI-MARL observation {sensor_id}.{channel_id}.{observation_id} "
                            f"has invalid use {use!r}"
                        )
                    policy_input = bool(merged.get("policy_input", use == "policy_input"))
                    if use == "excluded" and policy_input:
                        raise ValueError("Excluded TI-MARL observations cannot be policy inputs")
                    reason = merged.get("reason")
                    if use == "excluded" and not str(reason or "").strip():
                        raise ValueError(
                            "Every excluded TI-MARL observation requires a justification"
                        )
                    dimensions_payload = _mapping(
                        merged.get("dimensions", {}),
                        f"sensors.{sensor_id}.{channel_id}.{observation_id}.dimensions",
                    )
                    dimensions: Dict[str, Tuple[str, ...]] = {}
                    for dimension, values in dimensions_payload.items():
                        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                            raise ValueError("TI-MARL observation dimensions must contain lists")
                        dimensions[str(dimension)] = tuple(str(item) for item in values)
                    unit = str(merged.get("unit", "scalar"))
                    if unit not in SUPPORTED_UNITS:
                        raise ValueError(
                            f"TI-MARL observation {sensor_id}.{channel_id}."
                            f"{observation_id} uses unknown unit {unit!r}"
                        )
                    observations.append(
                        ObservationDefinition(
                            sensor_id=sensor_id,
                            sensor_type=sensor_type,
                            scope=scope,
                            channel_id=channel_id,
                            observation_id=observation_id,
                            source_feature=observation_id,
                            unit=unit,
                            dimensions=dimensions,
                            semantic_type=semantic_type,
                            use=use,
                            policy_input=policy_input,
                            criticality=str(merged.get("criticality", "operational")),
                            required=bool(merged.get("required", False)),
                            normalisation=str(merged.get("normalisation", "signed_log1p")),
                            reason=None if reason is None else str(reason),
                        )
                    )
            sensors.append(
                SensorDefinition(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    scope=scope,
                    profile=profile,
                    source_entity_id=source_entity_id,
                    observations=tuple(sorted(observations, key=lambda item: item.path)),
                )
            )
        return tuple(sorted(sensors, key=lambda item: item.sensor_id))

    @staticmethod
    def _parse_actuators(
        raw_actuators: Any,
        profiles: CapabilityProfileRegistry,
        observation_paths: set[str],
    ) -> Tuple[ActuatorDefinition, ...]:
        actuators_payload = _mapping(raw_actuators, "actuators")
        actuators = []
        for raw_actuator_id, raw_actuator in actuators_payload.items():
            actuator_id = _identifier(raw_actuator_id, "actuators.<id>")
            actuator = _mapping(raw_actuator, f"actuators.{actuator_id}")
            _reject_unknown(
                actuator,
                {"type", "actions"},
                f"actuators.{actuator_id}",
            )
            actuator_type = str(actuator.get("type", "")).strip()
            profile = profiles.action_profile(actuator_type)
            target_entity_id = None
            raw_actions = _mapping(
                actuator.get("actions"),
                f"actuators.{actuator_id}.actions",
            )
            if not raw_actions:
                raise ValueError(f"TI-MARL actuator {actuator_id!r} must define actions")
            actions = []
            for raw_action_id, raw_action in raw_actions.items():
                action_id = _identifier(raw_action_id, f"actuators.{actuator_id}.actions.<id>")
                action = _mapping(raw_action, f"actuators.{actuator_id}.actions.{action_id}")
                _reject_unknown(
                    action,
                    {"parameter", "dependencies", "mode"},
                    f"actuators.{actuator_id}.actions.{action_id}",
                )
                parameter = _mapping(
                    action.get("parameter"),
                    f"actuators.{actuator_id}.actions.{action_id}.parameter",
                )
                _reject_unknown(
                    parameter,
                    {"unit", "bounds"},
                    f"actuators.{actuator_id}.actions.{action_id}.parameter",
                )
                bounds = parameter.get("bounds")
                if (
                    not isinstance(bounds, Sequence)
                    or isinstance(bounds, (str, bytes))
                    or len(bounds) != 2
                ):
                    raise ValueError("TI-MARL action parameter.bounds must contain [min, max]")
                lower, upper = float(bounds[0]), float(bounds[1])
                if lower > upper:
                    raise ValueError("TI-MARL action lower bound exceeds upper bound")
                dependencies = _mapping(
                    action.get("dependencies", {}),
                    f"actuators.{actuator_id}.actions.{action_id}.dependencies",
                )
                normalized_dependencies: Dict[str, Mapping[str, Any]] = {}
                for observation_path, raw_outcomes in dependencies.items():
                    reference = str(observation_path)
                    if reference not in observation_paths:
                        raise ValueError(
                            f"TI-MARL action dependency references unknown observation {reference!r}"
                        )
                    outcomes = _mapping(raw_outcomes, f"dependency {reference}")
                    missing_states = sorted(set(HEALTH_DEPENDENCY_STATES) - set(outcomes))
                    if missing_states:
                        raise ValueError(
                            f"TI-MARL dependency {reference!r} must declare outcomes for "
                            f"{missing_states}"
                        )
                    normalized_dependencies[reference] = outcomes
                unit = str(parameter.get("unit", "fraction"))
                if unit not in SUPPORTED_UNITS:
                    raise ValueError(
                        f"TI-MARL action {actuator_id}.{action_id} uses unknown "
                        f"unit {unit!r}"
                    )
                mode_lookup = {
                    "charge": "CHARGE_EV" if "charger" in actuator_type else "CHARGE_STATIONARY",
                    "discharge": "DISCHARGE_EV" if "charger" in actuator_type else "DISCHARGE_STATIONARY",
                    "start": "START",
                    "idle": "IDLE",
                }
                mode = str(action.get("mode", mode_lookup.get(action_id.lower(), action_id.upper())))
                if mode not in profile["modes"]:
                    raise ValueError(
                        f"TI-MARL action {actuator_id}.{action_id} uses unsupported mode {mode!r}"
                    )
                actions.append(
                    ActionDefinition(
                        action_id=action_id,
                        mode=mode,
                        unit=unit,
                        lower_bound=lower,
                        upper_bound=upper,
                        dependencies=normalized_dependencies,
                    )
                )
            modes = tuple(dict.fromkeys(("IDLE", *(item.mode for item in actions))))
            actuators.append(
                ActuatorDefinition(
                    actuator_id=actuator_id,
                    actuator_type=actuator_type,
                    target_entity_id=target_entity_id,
                    group_type=str(profile["group_type"]),
                    source_entity_type=str(profile["entity_type"]),
                    modes=modes,
                    actions=tuple(sorted(actions, key=lambda item: item.action_id)),
                )
            )
        return tuple(sorted(actuators, key=lambda item: item.actuator_id))

    @staticmethod
    def _parse_constraints(raw_constraints: Any) -> Mapping[str, Any]:
        constraints = _mapping(raw_constraints, "constraints")
        result: Dict[str, Any] = {}
        for raw_constraint_id, raw_constraint in constraints.items():
            constraint_id = _identifier(raw_constraint_id, "constraints.<id>")
            constraint = _mapping(
                raw_constraint,
                f"constraints.{constraint_id}",
            )
            _reject_unknown(
                constraint,
                {"unit", "min", "max"},
                f"constraints.{constraint_id}",
            )
            unit = str(constraint.get("unit", ""))
            if unit not in SUPPORTED_UNITS:
                raise ValueError(
                    f"TI-MARL constraint {constraint_id!r} uses unknown unit {unit!r}"
                )
            if "min" not in constraint and "max" not in constraint:
                raise ValueError(
                    f"TI-MARL constraint {constraint_id!r} requires min or max"
                )
            normalized: Dict[str, Any] = {"unit": unit}
            for bound_name in ("min", "max"):
                if bound_name not in constraint:
                    continue
                value = constraint[bound_name]
                if isinstance(value, Mapping):
                    normalized[bound_name] = {
                        _identifier(key, f"constraints.{constraint_id}.{bound_name}.<id>"): float(item)
                        for key, item in value.items()
                    }
                else:
                    normalized[bound_name] = float(value)
            result[constraint_id] = normalized
        return result

    @staticmethod
    def _resolved_sensors(sensors: Sequence[SensorDefinition]) -> Mapping[str, Any]:
        result: Dict[str, Any] = {}
        for sensor in sensors:
            channels: Dict[str, Any] = {}
            for observation in sensor.observations:
                channel = channels.setdefault(observation.channel_id, {"observations": {}})
                channel["observations"][observation.observation_id] = observation.resolved()
            result[sensor.sensor_id] = {
                "type": sensor.sensor_type,
                "scope": sensor.scope,
                **({"profile": sensor.profile} if sensor.profile is not None else {}),
                **(
                    {"source_entity_id": sensor.source_entity_id}
                    if sensor.source_entity_id is not None
                    else {}
                ),
                "channels": channels,
            }
        return result

    @staticmethod
    def _resolved_actuators(actuators: Sequence[ActuatorDefinition]) -> Mapping[str, Any]:
        result: Dict[str, Any] = {}
        for actuator in actuators:
            actions = {}
            for action in actuator.actions:
                actions[action.action_id] = {
                    "mode": action.mode,
                    "parameter": {
                        "unit": action.unit,
                        "bounds": [action.lower_bound, action.upper_bound],
                    },
                    "dependencies": deepcopy(dict(action.dependencies)),
                }
            result[actuator.actuator_id] = {
                "type": actuator.actuator_type,
                **(
                    {"target_entity_id": actuator.target_entity_id}
                    if actuator.target_entity_id is not None
                    else {}
                ),
                "group_type": actuator.group_type,
                "source_entity_type": actuator.source_entity_type,
                "actions": actions,
            }
        return result

    @property
    def observation_index(self) -> Mapping[str, ObservationDefinition]:
        return {
            item.path: item
            for sensor in self.sensors
            for item in sensor.observations
        }

    @property
    def compatibility_shape(self) -> Mapping[str, Any]:
        """Instance-ID-free shape used for composition-compatible checkpoints."""

        return {
            "role": self.role,
            "agent_type": self.agent_type,
            "sensor_types": sorted(sensor.sensor_type for sensor in self.sensors),
            "observation_types": sorted(
                (
                    item.sensor_type,
                    item.channel_id,
                    item.semantic_type,
                    item.unit,
                    tuple(sorted(item.dimensions)),
                    item.use,
                )
                for sensor in self.sensors
                for item in sensor.observations
            ),
            "actuator_types": sorted(
                (item.actuator_type, item.group_type, tuple(item.modes))
                for item in self.actuators
            ),
            "constraint_types": sorted(self.constraints),
        }


@dataclass(frozen=True)
class RegistryDelta:
    added_agent_ids: Tuple[str, ...] = ()
    removed_agent_ids: Tuple[str, ...] = ()
    changed_agent_ids: Tuple[str, ...] = ()


class InterfaceRegistry:
    """Immutable-at-decision-time registry with transactional directory reloads."""

    def __init__(
        self,
        typed_interfaces_dir: str | Path,
        *,
        polling_enabled: bool = False,
        profiles: CapabilityProfileRegistry | None = None,
    ) -> None:
        self.directory = Path(typed_interfaces_dir).expanduser().resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"TI-MARL typed_interfaces_dir does not exist: {self.directory}"
            )
        self.polling_enabled = bool(polling_enabled)
        self.profiles = profiles or CapabilityProfileRegistry()
        self._interfaces: Mapping[str, TypedAgentInterface] = {}
        self._fingerprint = ""
        self.reload_interfaces()

    @property
    def interfaces(self) -> Mapping[str, TypedAgentInterface]:
        return self._interfaces

    @property
    def agent_ids(self) -> Tuple[str, ...]:
        return tuple(self._interfaces)

    @property
    def registry_hash(self) -> str:
        return _canonical_hash(
            {
                agent_id: interface.interface_hash
                for agent_id, interface in self._interfaces.items()
            }
        )

    def for_agent(self, agent_id: str) -> TypedAgentInterface:
        try:
            return self._interfaces[str(agent_id)]
        except KeyError as exc:
            raise KeyError(f"No typed interface registered for agent {agent_id!r}") from exc

    def _files(self) -> Tuple[Path, ...]:
        return tuple(
            sorted(
                path
                for path in self.directory.glob("*.yaml")
                if not path.name.endswith(".resolved.yaml")
            )
        )

    def _directory_fingerprint(self, files: Sequence[Path]) -> str:
        return _canonical_hash(
            [
                (path.name, path.stat().st_mtime_ns, path.stat().st_size)
                for path in files
            ]
        )

    def reload_interfaces(self) -> RegistryDelta:
        """Parse and validate into a candidate, then swap atomically on success."""

        files = self._files()
        if not files:
            raise ValueError(
                f"TI-MARL typed_interfaces_dir contains no agent YAML files: {self.directory}"
            )
        initial_fingerprint = self._directory_fingerprint(files)
        candidate: Dict[str, TypedAgentInterface] = {}
        for path in files:
            interface = TypedAgentInterface.load(path, profiles=self.profiles)
            if path.stem != interface.agent_id:
                raise ValueError(
                    f"TI-MARL interface filename {path.name!r} must match agent.id "
                    f"{interface.agent_id!r}"
                )
            if interface.agent_id in candidate:
                raise ValueError(
                    f"Duplicate TI-MARL registered agent ID: {interface.agent_id!r}"
                )
            candidate[interface.agent_id] = interface
        # A concurrent editor may have replaced, added or removed a file while
        # the candidate was parsed.  Reject that generation in full and let a
        # later poll/reload retry it; never publish a mixed directory view.
        final_files = self._files()
        final_fingerprint = self._directory_fingerprint(final_files)
        if final_files != files or final_fingerprint != initial_fingerprint:
            raise RuntimeError(
                "TI-MARL interface directory changed during atomic reload"
            )
        old = self._interfaces
        added = tuple(sorted(set(candidate) - set(old)))
        removed = tuple(sorted(set(old) - set(candidate)))
        changed = tuple(
            sorted(
                agent_id
                for agent_id in set(old) & set(candidate)
                if old[agent_id].interface_hash != candidate[agent_id].interface_hash
            )
        )
        # No code below this line can fail: this is the atomic commit point.
        self._interfaces = dict(sorted(candidate.items()))
        self._fingerprint = final_fingerprint
        return RegistryDelta(added, removed, changed)

    def maybe_reload(self) -> RegistryDelta:
        if not self.polling_enabled:
            return RegistryDelta()
        files = self._files()
        if self._directory_fingerprint(files) == self._fingerprint:
            return RegistryDelta()
        return self.reload_interfaces()

    def resolved_bundle(self) -> Mapping[str, Any]:
        return {
            "version": "typed_interface_registry_v1",
            "contract": CONTRACT_VERSION,
            "profile_registry": self.profiles.canonical_payload(),
            "registry_hash": self.registry_hash,
            "compatibility_signature": _canonical_hash(
                sorted(
                    {
                        _canonical_hash(interface.compatibility_shape)
                        for interface in self._interfaces.values()
                    }
                )
            ),
            "interfaces": {
                agent_id: deepcopy(dict(interface.resolved_payload))
                for agent_id, interface in self._interfaces.items()
            },
        }
