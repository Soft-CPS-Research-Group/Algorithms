"""Deterministic dependency closure over a compiled interface snapshot."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from algorithms.ti_marl.contracts.enums import EventDomain, HealthState
from algorithms.ti_marl.contracts.models import (
    ActionGroupInstance,
    ActionPortInstance,
    Dependency,
    FaultEvidence,
    HealthAssessment,
    ObservationPart,
)


def validate_dependency_graph(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject explicit dependency cycles before runtime."""

    valid_consequences = {
        "disable_group",
        "invalidate_non_idle_ports",
        "remove_observation_part",
        "contract_bound",
        "modify_constraint",
        "enter_degraded_mode",
        "activate_fallback",
    }
    graph: Dict[str, set[str]] = {}
    consequences: Dict[tuple[str, str], str] = {}
    for row in rows:
        source = str(row.get("source_kind", ""))
        group_target = row.get("target_group_type")
        semantic_target = row.get("target_semantic_type")
        if (group_target is None) == (semantic_target is None):
            raise ValueError(
                "TI-MARL dependency must declare exactly one of "
                "target_group_type or target_semantic_type"
            )
        target = str(group_target or semantic_target or "")
        if source and target:
            key = (source, target)
            consequence = str(row.get("consequence", ""))
            if consequence not in valid_consequences:
                raise ValueError(
                    f"TI-MARL dependency has unsupported consequence: {consequence!r}"
                )
            previous = consequences.get(key)
            if previous is not None and previous != consequence:
                raise ValueError(
                    "TI-MARL dependency conflict for "
                    f"{source!r} -> {target!r}: {previous!r} versus {consequence!r}"
                )
            consequences[key] = consequence
            graph.setdefault(source, set()).add(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"TI-MARL dependency cycle detected at {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node)


def apply_closure(
    *,
    groups: Iterable[ActionGroupInstance],
    parts: Iterable[ObservationPart],
    evidence: Iterable[FaultEvidence],
    health: Iterable[HealthAssessment],
    dependencies: Iterable[Dependency],
) -> Tuple[Tuple[ActionGroupInstance, ...], Tuple[ObservationPart, ...], Tuple[Mapping[str, Any], ...]]:
    """Propagate only declared v1 consequences, preserving cause evidence."""

    health_by_subject = {item.subject_id: item for item in health}
    evidence_rows = tuple(evidence)
    dependency_rows = tuple(dependencies)
    log: list[Mapping[str, Any]] = []
    final_groups = []

    for group in sorted(groups, key=lambda item: item.group_id):
        relevant = [item for item in evidence_rows if _targets_group(item, group)]
        disable = False
        invalidate_non_idle = False
        reasons: list[str] = []
        for item in relevant:
            subject = _subject_id(item)
            state = health_by_subject.get(subject)
            derived = state.state if state is not None else HealthState.UNKNOWN
            matching = (
                dependency
                for dependency in dependency_rows
                if dependency.target_group_type in {group.group_type, "*"}
                and _matches_source(dependency, item)
                and derived in dependency.condition_states
            )
            for dependency in matching:
                if dependency.consequence == "disable_group":
                    disable = True
                    reasons.append("asset_unavailable")
                elif dependency.consequence == "invalidate_non_idle_ports":
                    invalidate_non_idle = True
                    reasons.append(
                        "asset_disconnected"
                        if item.event_domain == EventDomain.ASSET_CONNECTION
                        else "actuator_channel_invalid"
                    )
                else:
                    raise ValueError(
                        "TI-MARL v1 group closure does not implement dependency "
                        f"consequence {dependency.consequence!r}"
                    )
                log.append(
                    _closure_row(
                        dependency.dependency_id,
                        item,
                        group.group_id,
                        dependency.consequence,
                    )
                )

        ports = []
        for port in group.ports:
            if port.mode == "IDLE":
                ports.append(port)
                continue
            invalid = disable or invalidate_non_idle
            ports.append(
                replace(
                    port,
                    valid=port.valid and not invalid,
                    invalid_reasons=tuple(sorted(set(port.invalid_reasons) | set(reasons))) if invalid else port.invalid_reasons,
                )
            )
        final_groups.append(
            replace(
                group,
                enabled=group.enabled and not disable,
                ports=tuple(ports),
                degraded_mode=("LOCAL_FALLBACK" if disable or invalidate_non_idle else group.degraded_mode),
            )
        )

    final_parts = []
    for part in sorted(parts, key=lambda item: item.part_id):
        relevant = [item for item in evidence_rows if _targets_part(item, part)]
        states = [
            health_by_subject.get(_subject_id(item)).state
            for item in relevant
            if health_by_subject.get(_subject_id(item)) is not None
        ]
        state = max(states, key=_health_order) if states else part.health
        invalid = False
        for item in relevant:
            assessment = health_by_subject.get(_subject_id(item))
            derived = (
                HealthState.UNKNOWN if assessment is None else assessment.state
            )
            for dependency in dependency_rows:
                if (
                    dependency.target_semantic_type
                    not in {part.semantic_type, "*"}
                    or not _matches_source(dependency, item)
                    or derived not in dependency.condition_states
                ):
                    continue
                if dependency.consequence != "remove_observation_part":
                    raise ValueError(
                        "TI-MARL v1 observation closure does not implement dependency "
                        f"consequence {dependency.consequence!r}"
                    )
                invalid = True
                log.append(
                    _closure_row(
                        dependency.dependency_id,
                        item,
                        part.part_id,
                        dependency.consequence,
                    )
                )
        final_parts.append(replace(part, health=state, valid=part.valid and not invalid))

    unique_log = {
        (
            str(row["rule_id"]),
            str(row["evidence_id"]),
            str(row["target_id"]),
            str(row["consequence"]),
        ): row
        for row in log
    }
    return tuple(final_groups), tuple(final_parts), tuple(unique_log[key] for key in sorted(unique_log))


def _targets_group(item: FaultEvidence, group: ActionGroupInstance) -> bool:
    return item.target_id in {group.module_id, group.group_id, group.owner_agent_id, "*"} or (
        item.target_id and item.target_id in group.module_id
    )


def _targets_part(item: FaultEvidence, part: ObservationPart) -> bool:
    if item.event_domain == EventDomain.COMMUNICATION_LINK and part.semantic_type == "community_signal":
        return True
    return item.target_id in {part.source_entity_id, part.owner_agent_id, "*"} and (
        item.target_feature in {"", "*", "both"} or item.target_feature in part.feature_names
    )


def _matches_source(dependency: Dependency, item: FaultEvidence) -> bool:
    return (
        dependency.source_kind in {item.event_domain.value.lower(), "*"}
        and dependency.source_type in {item.target_type, "*"}
    )


def _subject_id(item: FaultEvidence) -> str:
    return f"{item.event_domain.value}:{item.target_type}:{item.target_id}:{item.target_feature or '*'}"


def _health_order(state: HealthState) -> int:
    return {
        HealthState.HEALTHY: 0,
        HealthState.DEGRADED: 1,
        HealthState.STALE: 2,
        HealthState.MISSING: 3,
        HealthState.FAILED: 4,
        HealthState.UNKNOWN: 5,
    }[state]


def _closure_row(rule_id: str, item: FaultEvidence, target_id: str, consequence: str) -> Mapping[str, Any]:
    return {
        "rule_id": rule_id,
        "evidence_id": item.evidence_id,
        "fault_mode": item.fault_mode,
        "target_id": target_id,
        "consequence": consequence,
    }
