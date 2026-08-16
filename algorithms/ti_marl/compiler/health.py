"""Versioned health derivation from Simulator runtime facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from algorithms.ti_marl.contracts.enums import (
    AvailabilityState,
    ConnectionState,
    EventDomain,
    HealthState,
    QualityState,
)
from algorithms.ti_marl.contracts.models import FaultEvidence, HealthAssessment, HealthRule


@dataclass
class _RecoveryState:
    last_state: HealthState
    recovery_steps: int = 0
    since_time_step: Optional[int] = None
    semantic_type: str = "local_energy"
    criticality: str = "operational"


class HealthDeriver:
    """Derive control health without changing or translating ``fault_mode``."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.version = str(config.get("version", "unversioned_health_rules"))
        self._states: Dict[str, _RecoveryState] = {}

    def snapshot_state(self) -> Dict[str, Tuple[str, int, Optional[int], str, str]]:
        return {
            subject: (
                state.last_state.value,
                state.recovery_steps,
                state.since_time_step,
                state.semantic_type,
                state.criticality,
            )
            for subject, state in self._states.items()
        }

    def reset(self) -> None:
        self._states.clear()

    def restore_state(self, payload: Mapping[str, Tuple[Any, ...]]) -> None:
        restored = {}
        for subject, values in payload.items():
            # Accept the early local three-field prototype while preserving
            # full semantic identity in every newly written v1 checkpoint.
            semantic_type = str(values[3]) if len(values) > 3 else "local_energy"
            criticality = str(values[4]) if len(values) > 4 else "operational"
            restored[str(subject)] = _RecoveryState(
                HealthState(values[0]),
                int(values[1]),
                values[2],
                semantic_type,
                criticality,
            )
        self._states = restored

    def rule_for(self, semantic_type: str, criticality: Optional[str] = None) -> HealthRule:
        defaults = dict(self.config.get("defaults", {}))
        semantic = dict(self.config.get("semantic_types", {}).get(semantic_type, {}))
        selected_criticality = str(
            criticality or semantic.get("criticality") or "operational"
        )
        critical = dict(self.config.get("criticality", {}).get(selected_criticality, {}))
        merged = {**defaults, **critical, **semantic}
        return HealthRule(
            rule_id=f"{self.version}:{semantic_type}:{selected_criticality}",
            semantic_type=str(semantic_type),
            criticality=selected_criticality,
            degraded_after_steps=max(int(merged.get("degraded_after_steps", 1)), 0),
            stale_after_steps=max(int(merged.get("stale_after_steps", 4)), 0),
            missing_after_steps=max(int(merged.get("missing_after_steps", 8)), 0),
            recovery_hysteresis_steps=max(int(merged.get("recovery_hysteresis_steps", 2)), 0),
            cache_allowed=bool(merged.get("cache_allowed", True)),
        )

    def derive(
        self,
        evidence: Iterable[FaultEvidence],
        *,
        time_step: int,
        nominal_subjects: Mapping[str, Tuple[str, str]] | None = None,
    ) -> Tuple[HealthAssessment, ...]:
        grouped: Dict[str, list[FaultEvidence]] = {}
        for item in evidence:
            grouped.setdefault(item.evidence_id.split("#", 1)[0], []).append(item)

        descriptors = dict(nominal_subjects or {})
        for items in grouped.values():
            for item in items:
                subject = self.subject_id(item)
                descriptors.setdefault(subject, (self.semantic_type(item), self.criticality(item)))

        # Previously unhealthy subjects remain visible during recovery even
        # after the Simulator stops reporting the sparse fault record.
        for subject, previous in self._states.items():
            descriptors.setdefault(
                subject,
                (previous.semantic_type, previous.criticality),
            )

        by_subject: Dict[str, list[FaultEvidence]] = {}
        for item in evidence:
            by_subject.setdefault(self.subject_id(item), []).append(item)

        assessments = []
        next_states: Dict[str, _RecoveryState] = {}
        for subject in sorted(descriptors):
            semantic_type, criticality = descriptors[subject]
            rule = self.rule_for(semantic_type, criticality)
            items = by_subject.get(subject, [])
            if items:
                state, explanation = self._derive_active(items, rule)
                previous = self._states.get(subject)
                if (
                    state == HealthState.HEALTHY
                    and previous is not None
                    and previous.last_state != HealthState.HEALTHY
                    and previous.recovery_steps < rule.recovery_hysteresis_steps
                ):
                    recovery_steps = previous.recovery_steps + 1
                    state = HealthState.DEGRADED
                    explanation = "nominal evidence under recovery hysteresis"
                    since = previous.since_time_step
                    next_states[subject] = _RecoveryState(
                        previous.last_state,
                        recovery_steps,
                        since,
                        semantic_type,
                        rule.criticality,
                    )
                    recovery_pending = rule.recovery_hysteresis_steps - recovery_steps
                else:
                    since = (
                        previous.since_time_step
                        if previous is not None and previous.last_state == state
                        else int(time_step)
                    )
                    next_states[subject] = _RecoveryState(
                        state,
                        0,
                        since,
                        semantic_type,
                        rule.criticality,
                    )
                    recovery_pending = 0
                evidence_ids = tuple(sorted({event_id for item in items for event_id in item.event_ids} or {item.evidence_id for item in items}))
            else:
                previous = self._states.get(subject)
                if previous is not None and previous.last_state != HealthState.HEALTHY:
                    recovery_steps = previous.recovery_steps + 1
                    if recovery_steps <= rule.recovery_hysteresis_steps:
                        state = HealthState.DEGRADED
                        explanation = "recovery hysteresis"
                        next_states[subject] = _RecoveryState(
                            previous.last_state,
                            recovery_steps,
                            previous.since_time_step,
                            semantic_type,
                            rule.criticality,
                        )
                        recovery_pending = rule.recovery_hysteresis_steps - recovery_steps
                    else:
                        state = HealthState.HEALTHY
                        explanation = "recovered"
                        next_states[subject] = _RecoveryState(
                            state,
                            0,
                            int(time_step),
                            semantic_type,
                            rule.criticality,
                        )
                        recovery_pending = 0
                else:
                    state = HealthState.HEALTHY
                    explanation = "nominal sparse default"
                    next_states[subject] = _RecoveryState(
                        state,
                        0,
                        int(time_step),
                        semantic_type,
                        rule.criticality,
                    )
                    recovery_pending = 0
                evidence_ids = ()
                since = next_states[subject].since_time_step

            assessments.append(
                HealthAssessment(
                    subject_id=subject,
                    semantic_type=semantic_type,
                    criticality=rule.criticality,
                    state=state,
                    rule_id=rule.rule_id,
                    evidence_ids=evidence_ids,
                    since_time_step=since,
                    recovery_pending_steps=recovery_pending,
                    explanation=explanation,
                )
            )
        self._states = next_states
        return tuple(assessments)

    def _derive_active(
        self,
        items: list[FaultEvidence],
        rule: HealthRule,
    ) -> Tuple[HealthState, str]:
        states = [self._derive_one(item, rule) for item in items]
        order = {
            HealthState.HEALTHY: 0,
            HealthState.DEGRADED: 1,
            HealthState.STALE: 2,
            HealthState.MISSING: 3,
            HealthState.FAILED: 4,
            HealthState.UNKNOWN: 5,
        }
        state = max(states, key=lambda candidate: order[candidate])
        causes = ",".join(sorted({str(item.fault_mode or item.event_domain.value) for item in items}))
        return state, f"derived from {causes} using age, domain and {rule.criticality} criticality"

    def _derive_one(self, item: FaultEvidence, rule: HealthRule) -> HealthState:
        age = max(int(item.age_steps or 0), int(item.active_duration_steps or 0))

        if item.event_domain == EventDomain.ASSET_CONNECTION:
            return (
                HealthState.MISSING
                if item.connection == ConnectionState.DISCONNECTED
                else HealthState.HEALTHY
            )
        if item.event_domain == EventDomain.ASSET_AVAILABILITY:
            if item.availability == AvailabilityState.UNAVAILABLE:
                return HealthState.FAILED
        if item.event_domain == EventDomain.ACTUATOR_CHANNEL:
            if item.availability == AvailabilityState.UNAVAILABLE or item.quality == QualityState.INVALID:
                return HealthState.MISSING
        if item.event_domain in {EventDomain.SENSOR_CHANNEL, EventDomain.COMMUNICATION_LINK}:
            if item.availability == AvailabilityState.UNAVAILABLE or item.quality == QualityState.INVALID:
                if rule.cache_allowed and item.last_fresh_time_step is not None and age < rule.missing_after_steps:
                    return HealthState.STALE
                return HealthState.MISSING

        mode = str(item.fault_mode or "").lower()
        initial = str(self.config.get("fault_modes", {}).get(mode, {}).get("initial_state", "DEGRADED"))
        initial_state = HealthState(initial)
        if age >= rule.missing_after_steps:
            return HealthState.MISSING
        if age >= rule.stale_after_steps:
            return HealthState.STALE
        if age >= rule.degraded_after_steps:
            return initial_state
        return HealthState.HEALTHY

    @staticmethod
    def subject_id(item: FaultEvidence) -> str:
        feature = item.target_feature or "*"
        return f"{item.event_domain.value}:{item.target_type}:{item.target_id}:{feature}"

    @staticmethod
    def semantic_type(item: FaultEvidence) -> str:
        return {
            EventDomain.ASSET_CONNECTION: "asset_connection",
            EventDomain.ASSET_AVAILABILITY: "asset_availability",
            EventDomain.ACTUATOR_CHANNEL: "actuator_channel",
            EventDomain.COMMUNICATION_LINK: "community_signal",
            EventDomain.SENSOR_CHANNEL: "local_energy",
            EventDomain.VALUE_QUALITY: "local_energy",
        }[item.event_domain]

    @staticmethod
    def criticality(item: FaultEvidence) -> str:
        if item.event_domain in {EventDomain.ACTUATOR_CHANNEL, EventDomain.ASSET_AVAILABILITY}:
            return "safety"
        if item.event_domain == EventDomain.COMMUNICATION_LINK:
            return "advisory"
        return "operational"
