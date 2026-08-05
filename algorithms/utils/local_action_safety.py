"""Simulator-independent safety projection for one building's action vector.

The projector consumes semantic constraints expressed in physical units and
returns the action that may be executed.  It deliberately has no CityLearn or
agent dependencies: an integration layer is responsible for translating raw
observations and environment metadata into the dataclasses below.

Action values are assumed to use zero as the no-power command.  Positive
values import power (charge/start) and negative values export power
(discharge/V2G).  ``*_kw_per_action`` converts one action unit into total
three-phase power; ``phase_weights`` describes how that total is distributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Optional, Sequence, Tuple


_EPSILON = 1.0e-9


class ActionKind(str, Enum):
    """Semantic category of an action slot."""

    STORAGE = "storage"
    EV = "ev"
    DEFERRABLE = "deferrable"
    OTHER = "other"


class InterventionReason(str, Enum):
    """Why a proposed action was changed or constrained."""

    BOUNDS_LOW = "bounds_low"
    BOUNDS_HIGH = "bounds_high"
    ACTION_UNAVAILABLE = "action_unavailable"
    IMPORT_AVAILABILITY = "import_availability"
    EXPORT_AVAILABILITY = "export_availability"
    STORAGE_SOC_MIN = "storage_soc_min"
    STORAGE_SOC_MAX = "storage_soc_max"
    EV_DISCONNECTED = "ev_disconnected"
    EV_MINIMUM = "ev_minimum"
    DEFERRABLE_UNAVAILABLE = "deferrable_unavailable"
    DEFERRABLE_MUST_START = "deferrable_must_start"
    IMPORT_HEADROOM_TOTAL = "import_headroom_total"
    EXPORT_HEADROOM_TOTAL = "export_headroom_total"
    IMPORT_HEADROOM_PHASE = "import_headroom_phase"
    EXPORT_HEADROOM_PHASE = "export_headroom_phase"


class InfeasibleCode(str, Enum):
    """Hard requirements that the available action envelope cannot satisfy."""

    EV_MINIMUM_LOCAL_LIMIT = "ev_minimum_local_limit"
    EV_MINIMUM_HEADROOM = "ev_minimum_headroom"
    DEFERRABLE_MUST_START_LOCAL_LIMIT = "deferrable_must_start_local_limit"
    DEFERRABLE_MUST_START_HEADROOM = "deferrable_must_start_headroom"


@dataclass(frozen=True)
class ActionSpec:
    """Static definition of one action slot.

    ``priority`` is used only when multiple mandatory or discretionary actions
    compete for the same headroom.  Larger values are allocated first; vector
    order is the stable tie-breaker.
    """

    name: str
    kind: ActionKind
    low: float
    high: float
    import_kw_per_action: float = 0.0
    export_kw_per_action: Optional[float] = None
    phases: Tuple[str, ...] = ()
    phase_weights: Mapping[str, float] = field(default_factory=dict)
    priority: int = 0
    idle_action: float = 0.0

    def __post_init__(self) -> None:
        _require_finite(self.low, "ActionSpec.low")
        _require_finite(self.high, "ActionSpec.high")
        _require_finite(self.idle_action, "ActionSpec.idle_action")
        if self.high < self.low:
            raise ValueError(f"ActionSpec '{self.name}' has high < low.")
        if not self.low <= self.idle_action <= self.high:
            raise ValueError(f"ActionSpec '{self.name}' idle_action must be inside bounds.")
        if not math.isclose(self.idle_action, 0.0, rel_tol=0.0, abs_tol=_EPSILON):
            raise ValueError("ActionSpec.idle_action must be zero for power-safe projection.")
        _require_finite(self.import_kw_per_action, "ActionSpec.import_kw_per_action")
        if self.import_kw_per_action < 0.0:
            raise ValueError("ActionSpec.import_kw_per_action must be non-negative.")
        if self.export_kw_per_action is not None:
            _require_finite(self.export_kw_per_action, "ActionSpec.export_kw_per_action")
            if self.export_kw_per_action < 0.0:
                raise ValueError("ActionSpec.export_kw_per_action must be non-negative.")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError(f"ActionSpec '{self.name}' phases must be unique.")
        unknown = set(self.phase_weights) - set(self.phases)
        if unknown:
            raise ValueError(
                f"ActionSpec '{self.name}' has weights for unknown phases: {sorted(unknown)}."
            )
        for phase, value in self.phase_weights.items():
            _require_finite(value, f"ActionSpec.phase_weights[{phase!r}]")
            if value < 0.0:
                raise ValueError("ActionSpec.phase_weights must be non-negative.")
        if self.phase_weights and sum(self.phase_weights.values()) <= _EPSILON:
            raise ValueError(f"ActionSpec '{self.name}' phase weights sum to zero.")

    def power_scale(self, *, export: bool) -> float:
        if export and self.export_kw_per_action is not None:
            return float(self.export_kw_per_action)
        return float(self.import_kw_per_action)

    def normalized_phase_weights(self) -> Mapping[str, float]:
        if not self.phases:
            return {}
        if not self.phase_weights:
            equal_share = 1.0 / len(self.phases)
            return {phase: equal_share for phase in self.phases}
        weights = {phase: float(self.phase_weights.get(phase, 0.0)) for phase in self.phases}
        total = sum(weights.values())
        if total <= _EPSILON:
            raise ValueError(f"ActionSpec '{self.name}' phase weights sum to zero.")
        return {phase: value / total for phase, value in weights.items()}


@dataclass(frozen=True)
class StorageConstraint:
    """Dynamic storage envelope for the current control step."""

    soc: float
    min_soc: float = 0.0
    max_soc: float = 1.0
    available_charge_action: Optional[float] = None
    available_discharge_action: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_soc_envelope(self.soc, self.min_soc, self.max_soc, "StorageConstraint")
        _validate_optional_non_negative(
            self.available_charge_action, "StorageConstraint.available_charge_action"
        )
        _validate_optional_non_negative(
            self.available_discharge_action, "StorageConstraint.available_discharge_action"
        )


@dataclass(frozen=True)
class EVConstraint:
    """Dynamic EV availability and service requirement."""

    connected: bool
    min_required_action: float = 0.0
    available_charge_action: Optional[float] = None
    available_discharge_action: Optional[float] = None

    def __post_init__(self) -> None:
        _require_finite(self.min_required_action, "EVConstraint.min_required_action")
        if self.min_required_action < 0.0:
            raise ValueError("EVConstraint.min_required_action must be non-negative.")
        _validate_optional_non_negative(
            self.available_charge_action, "EVConstraint.available_charge_action"
        )
        _validate_optional_non_negative(
            self.available_discharge_action, "EVConstraint.available_discharge_action"
        )


@dataclass(frozen=True)
class DeferrableConstraint:
    """Current state of a start-command deferrable appliance."""

    pending: bool
    running: bool
    can_start: bool
    must_start: bool = False
    start_action: float = 1.0
    trigger_threshold: float = 0.5

    def __post_init__(self) -> None:
        _require_finite(self.start_action, "DeferrableConstraint.start_action")
        if self.start_action <= 0.0:
            raise ValueError("DeferrableConstraint.start_action must be positive.")
        _require_finite(self.trigger_threshold, "DeferrableConstraint.trigger_threshold")
        if not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError("DeferrableConstraint.trigger_threshold must be in [0, 1].")


@dataclass(frozen=True)
class ActionConstraint:
    """Dynamic constraints associated with one :class:`ActionSpec`."""

    available: bool = True
    storage: Optional[StorageConstraint] = None
    ev: Optional[EVConstraint] = None
    deferrable: Optional[DeferrableConstraint] = None


@dataclass(frozen=True)
class ElectricalHeadroom:
    """Power still available to flexible actions before this projection.

    ``None`` means that a total or phase limit is not known and is therefore
    not enforced.  All finite values are non-negative kW.
    """

    total_import_kw: Optional[float] = None
    total_export_kw: Optional[float] = None
    phase_import_kw: Mapping[str, float] = field(default_factory=dict)
    phase_export_kw: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_optional_non_negative(self.total_import_kw, "ElectricalHeadroom.total_import_kw")
        _validate_optional_non_negative(self.total_export_kw, "ElectricalHeadroom.total_export_kw")
        _validate_headroom_mapping(self.phase_import_kw, "ElectricalHeadroom.phase_import_kw")
        _validate_headroom_mapping(self.phase_export_kw, "ElectricalHeadroom.phase_export_kw")


@dataclass(frozen=True)
class LocalActionProjectionRequest:
    """Complete safety projection request for exactly one building."""

    proposed_actions: Sequence[float]
    action_specs: Sequence[ActionSpec]
    action_constraints: Sequence[ActionConstraint]
    headroom: ElectricalHeadroom = field(default_factory=ElectricalHeadroom)
    building_id: Optional[str] = None


@dataclass(frozen=True)
class ActionIntervention:
    action_index: int
    action_name: str
    proposed_action: float
    executed_action: float
    reason_codes: Tuple[InterventionReason, ...]


@dataclass(frozen=True)
class InfeasibleReason:
    code: InfeasibleCode
    action_index: int
    action_name: str
    required_action: float
    executed_action: float
    message: str


@dataclass(frozen=True)
class ProjectionResult:
    executed_actions: Tuple[float, ...]
    interventions: Tuple[ActionIntervention, ...]
    infeasible_reasons: Tuple[InfeasibleReason, ...]
    remaining_headroom: ElectricalHeadroom

    @property
    def feasible(self) -> bool:
        return not self.infeasible_reasons


@dataclass
class _PreparedAction:
    proposed: float
    target: float
    mandatory: float
    mandatory_code_local: Optional[InfeasibleCode] = None
    mandatory_code_headroom: Optional[InfeasibleCode] = None
    mandatory_label: Optional[str] = None
    reasons: list[InterventionReason] = field(default_factory=list)


class LocalActionSafetyProjector:
    """Project a complete local action vector into its safe feasible envelope."""

    def project(self, request: LocalActionProjectionRequest) -> ProjectionResult:
        specs, constraints, proposals = self._validate_request(request)
        prepared: list[_PreparedAction] = []
        infeasible: list[InfeasibleReason] = []

        for index, (proposed, spec, constraint) in enumerate(zip(proposals, specs, constraints)):
            row, local_infeasible = self._prepare_local_action(index, proposed, spec, constraint)
            prepared.append(row)
            infeasible.extend(local_infeasible)

        ledger = _HeadroomLedger(request.headroom)
        executed = [float(spec.idle_action) for spec in specs]
        allocation_order = sorted(range(len(specs)), key=lambda idx: (-specs[idx].priority, idx))

        # Mandatory service is reserved before any discretionary battery/EV use.
        for index in allocation_order:
            row = prepared[index]
            if row.mandatory <= specs[index].idle_action + _EPSILON:
                continue
            allocated, headroom_reasons = ledger.allocate_increment(
                specs[index], specs[index].idle_action, row.mandatory
            )
            executed[index] = allocated
            row.reasons.extend(headroom_reasons)
            if allocated + _EPSILON < row.mandatory and row.mandatory_code_headroom is not None:
                infeasible.append(
                    self._infeasible_reason(
                        row.mandatory_code_headroom,
                        index,
                        specs[index],
                        row.mandatory,
                        allocated,
                        row.mandatory_label or "mandatory action",
                    )
                )

        # Allocate the remaining request without undoing mandatory reservations.
        for index in allocation_order:
            row = prepared[index]
            allocated, headroom_reasons = ledger.allocate_increment(
                specs[index], executed[index], row.target
            )
            executed[index] = allocated
            row.reasons.extend(headroom_reasons)

        interventions: list[ActionIntervention] = []
        for index, (row, spec, value) in enumerate(zip(prepared, specs, executed)):
            reasons = _unique_reasons(row.reasons)
            if reasons or not math.isclose(row.proposed, value, rel_tol=0.0, abs_tol=_EPSILON):
                interventions.append(
                    ActionIntervention(
                        action_index=index,
                        action_name=spec.name,
                        proposed_action=row.proposed,
                        executed_action=value,
                        reason_codes=reasons,
                    )
                )

        return ProjectionResult(
            executed_actions=tuple(executed),
            interventions=tuple(interventions),
            infeasible_reasons=tuple(_deduplicate_infeasible(infeasible)),
            remaining_headroom=ledger.snapshot(),
        )

    @staticmethod
    def _validate_request(
        request: LocalActionProjectionRequest,
    ) -> tuple[tuple[ActionSpec, ...], tuple[ActionConstraint, ...], tuple[float, ...]]:
        specs = tuple(request.action_specs)
        constraints = tuple(request.action_constraints)
        proposals = tuple(float(value) for value in request.proposed_actions)
        if not (len(specs) == len(constraints) == len(proposals)):
            raise ValueError(
                "proposed_actions, action_specs and action_constraints must have equal length."
            )
        for index, value in enumerate(proposals):
            _require_finite(value, f"proposed_actions[{index}]")
        return specs, constraints, proposals

    def _prepare_local_action(
        self,
        index: int,
        proposed: float,
        spec: ActionSpec,
        constraint: ActionConstraint,
    ) -> tuple[_PreparedAction, list[InfeasibleReason]]:
        row = _PreparedAction(proposed=proposed, target=proposed, mandatory=spec.idle_action)
        infeasible: list[InfeasibleReason] = []

        row.target = _clip_with_reason(
            row.target,
            spec.low,
            spec.high,
            row.reasons,
            InterventionReason.BOUNDS_LOW,
            InterventionReason.BOUNDS_HIGH,
        )
        if not constraint.available:
            if not math.isclose(row.target, spec.idle_action, abs_tol=_EPSILON) or (
                constraint.ev is not None
                and constraint.ev.min_required_action > spec.idle_action + _EPSILON
            ) or (constraint.deferrable is not None and constraint.deferrable.must_start):
                row.reasons.append(InterventionReason.ACTION_UNAVAILABLE)
            row.target = spec.idle_action
            if constraint.ev is not None and constraint.ev.min_required_action > spec.idle_action + _EPSILON:
                infeasible.append(
                    self._infeasible_reason(
                        InfeasibleCode.EV_MINIMUM_LOCAL_LIMIT,
                        index,
                        spec,
                        constraint.ev.min_required_action,
                        spec.idle_action,
                        "EV minimum while action is unavailable",
                    )
                )
            if constraint.deferrable is not None and constraint.deferrable.must_start:
                infeasible.append(
                    self._infeasible_reason(
                        InfeasibleCode.DEFERRABLE_MUST_START_LOCAL_LIMIT,
                        index,
                        spec,
                        constraint.deferrable.start_action,
                        spec.idle_action,
                        "deferrable must-start while action is unavailable",
                    )
                )
            return row, infeasible

        if spec.kind == ActionKind.STORAGE and constraint.storage is not None:
            self._apply_storage_constraints(row, spec, constraint.storage)
        elif spec.kind == ActionKind.EV and constraint.ev is not None:
            infeasible.extend(self._apply_ev_constraints(index, row, spec, constraint.ev))
        elif spec.kind == ActionKind.DEFERRABLE and constraint.deferrable is not None:
            infeasible.extend(
                self._apply_deferrable_constraints(index, row, spec, constraint.deferrable)
            )
        return row, infeasible

    @staticmethod
    def _apply_storage_constraints(
        row: _PreparedAction,
        spec: ActionSpec,
        constraint: StorageConstraint,
    ) -> None:
        row.target = _apply_directional_availability(
            row.target,
            constraint.available_charge_action,
            constraint.available_discharge_action,
            row.reasons,
        )
        if constraint.soc >= constraint.max_soc - _EPSILON and row.target > spec.idle_action:
            row.target = spec.idle_action
            row.reasons.append(InterventionReason.STORAGE_SOC_MAX)
        if constraint.soc <= constraint.min_soc + _EPSILON and row.target < spec.idle_action:
            row.target = spec.idle_action
            row.reasons.append(InterventionReason.STORAGE_SOC_MIN)

    def _apply_ev_constraints(
        self,
        index: int,
        row: _PreparedAction,
        spec: ActionSpec,
        constraint: EVConstraint,
    ) -> list[InfeasibleReason]:
        infeasible: list[InfeasibleReason] = []
        required = float(constraint.min_required_action)
        if not constraint.connected:
            if not math.isclose(row.target, spec.idle_action, abs_tol=_EPSILON) or (
                required > spec.idle_action + _EPSILON
            ):
                row.reasons.append(InterventionReason.EV_DISCONNECTED)
            row.target = spec.idle_action
            if required > spec.idle_action + _EPSILON:
                infeasible.append(
                    self._infeasible_reason(
                        InfeasibleCode.EV_MINIMUM_LOCAL_LIMIT,
                        index,
                        spec,
                        required,
                        spec.idle_action,
                        "EV minimum while disconnected",
                    )
                )
            return infeasible

        row.target = _apply_directional_availability(
            row.target,
            constraint.available_charge_action,
            constraint.available_discharge_action,
            row.reasons,
        )
        local_maximum = spec.high
        if constraint.available_charge_action is not None:
            local_maximum = min(local_maximum, constraint.available_charge_action)
        feasible_required = min(required, local_maximum)
        if feasible_required + _EPSILON < required:
            infeasible.append(
                self._infeasible_reason(
                    InfeasibleCode.EV_MINIMUM_LOCAL_LIMIT,
                    index,
                    spec,
                    required,
                    feasible_required,
                    "EV minimum exceeds bounds or charge availability",
                )
            )
        row.mandatory = max(spec.idle_action, feasible_required)
        row.mandatory_code_local = InfeasibleCode.EV_MINIMUM_LOCAL_LIMIT
        row.mandatory_code_headroom = InfeasibleCode.EV_MINIMUM_HEADROOM
        row.mandatory_label = "EV minimum"
        if row.target + _EPSILON < row.mandatory:
            row.target = row.mandatory
            row.reasons.append(InterventionReason.EV_MINIMUM)
        return infeasible

    def _apply_deferrable_constraints(
        self,
        index: int,
        row: _PreparedAction,
        spec: ActionSpec,
        constraint: DeferrableConstraint,
    ) -> list[InfeasibleReason]:
        infeasible: list[InfeasibleReason] = []
        may_start = constraint.pending and constraint.can_start and not constraint.running
        if not may_start:
            if not math.isclose(row.target, spec.idle_action, abs_tol=_EPSILON) or constraint.must_start:
                row.reasons.append(InterventionReason.DEFERRABLE_UNAVAILABLE)
            row.target = spec.idle_action
            if constraint.must_start:
                infeasible.append(
                    self._infeasible_reason(
                        InfeasibleCode.DEFERRABLE_MUST_START_LOCAL_LIMIT,
                        index,
                        spec,
                        constraint.start_action,
                        spec.idle_action,
                        "deferrable must-start while unavailable",
                    )
                )
            return infeasible

        # CityLearn interprets this action as a binary start command, not a
        # scalable power request.  Canonicalizing here keeps the projected
        # action and the headroom reservation consistent with what the
        # simulator will execute.
        row.target = (
            float(constraint.start_action)
            if row.target > constraint.trigger_threshold
            else float(spec.idle_action)
        )

        if not constraint.must_start:
            return infeasible

        required = float(constraint.start_action)
        feasible_required = min(max(required, spec.low), spec.high)
        if not math.isclose(feasible_required, required, abs_tol=_EPSILON):
            infeasible.append(
                self._infeasible_reason(
                    InfeasibleCode.DEFERRABLE_MUST_START_LOCAL_LIMIT,
                    index,
                    spec,
                    required,
                    feasible_required,
                    "deferrable start action exceeds bounds",
                )
            )
        row.mandatory = max(spec.idle_action, feasible_required)
        row.mandatory_code_local = InfeasibleCode.DEFERRABLE_MUST_START_LOCAL_LIMIT
        row.mandatory_code_headroom = InfeasibleCode.DEFERRABLE_MUST_START_HEADROOM
        row.mandatory_label = "deferrable must-start"
        if row.target + _EPSILON < row.mandatory:
            row.target = row.mandatory
            row.reasons.append(InterventionReason.DEFERRABLE_MUST_START)
        return infeasible

    @staticmethod
    def _infeasible_reason(
        code: InfeasibleCode,
        index: int,
        spec: ActionSpec,
        required: float,
        executed: float,
        label: str,
    ) -> InfeasibleReason:
        return InfeasibleReason(
            code=code,
            action_index=index,
            action_name=spec.name,
            required_action=float(required),
            executed_action=float(executed),
            message=f"{label} for '{spec.name}' requires {required:.6g}, feasible {executed:.6g}.",
        )


class _HeadroomLedger:
    def __init__(self, headroom: ElectricalHeadroom) -> None:
        self.total_import = headroom.total_import_kw
        self.total_export = headroom.total_export_kw
        self.phase_import = {str(key): float(value) for key, value in headroom.phase_import_kw.items()}
        self.phase_export = {str(key): float(value) for key, value in headroom.phase_export_kw.items()}

    def allocate_increment(
        self,
        spec: ActionSpec,
        current_action: float,
        target_action: float,
    ) -> tuple[float, list[InterventionReason]]:
        if math.isclose(current_action, target_action, rel_tol=0.0, abs_tol=_EPSILON):
            return current_action, []

        # Mandatory allocations and normal projections only move away from idle
        # in one direction.  Supporting reversal defensively keeps the API safe
        # for future integrations without silently double-counting headroom.
        if current_action * target_action < -_EPSILON:
            self._release(spec, current_action)
            current_action = spec.idle_action

        if abs(target_action) <= abs(current_action) + _EPSILON and current_action * target_action >= 0.0:
            self._release(spec, current_action - target_action)
            return target_action, []

        increment = target_action - current_action
        export = increment < 0.0
        requested_magnitude = abs(increment)
        scale = spec.power_scale(export=export)
        if scale <= _EPSILON:
            return target_action, []

        requested_power = requested_magnitude * scale
        allowed_power, reasons = self._allowed_power(spec, export=export)
        if spec.kind == ActionKind.DEFERRABLE and requested_power > allowed_power + _EPSILON:
            # A start command is atomic.  A fractional allocation would still
            # cross (or fail to cross) the simulator trigger threshold while
            # reserving the wrong amount of electrical headroom.
            return current_action, reasons
        allocated_power = min(requested_power, allowed_power)
        allocated_magnitude = allocated_power / scale
        allocated_action = current_action + (-allocated_magnitude if export else allocated_magnitude)
        self._reserve(spec, allocated_power, export=export)
        if requested_power <= allowed_power + _EPSILON:
            return target_action, []
        return allocated_action, reasons

    def _allowed_power(
        self,
        spec: ActionSpec,
        *,
        export: bool,
    ) -> tuple[float, list[InterventionReason]]:
        total = self.total_export if export else self.total_import
        phases = self.phase_export if export else self.phase_import
        total_reason = (
            InterventionReason.EXPORT_HEADROOM_TOTAL
            if export
            else InterventionReason.IMPORT_HEADROOM_TOTAL
        )
        phase_reason = (
            InterventionReason.EXPORT_HEADROOM_PHASE
            if export
            else InterventionReason.IMPORT_HEADROOM_PHASE
        )
        candidates: list[tuple[float, InterventionReason]] = []
        if total is not None:
            candidates.append((max(total, 0.0), total_reason))
        for phase, weight in spec.normalized_phase_weights().items():
            if weight > _EPSILON and phase in phases:
                candidates.append((max(phases[phase], 0.0) / weight, phase_reason))
        if not candidates:
            return math.inf, []
        allowed = min(value for value, _ in candidates)
        reasons = [reason for value, reason in candidates if value <= allowed + _EPSILON]
        return allowed, list(_unique_reasons(reasons))

    def _reserve(self, spec: ActionSpec, power_kw: float, *, export: bool) -> None:
        if power_kw <= _EPSILON:
            return
        if export:
            if self.total_export is not None:
                self.total_export = max(0.0, self.total_export - power_kw)
            phases = self.phase_export
        else:
            if self.total_import is not None:
                self.total_import = max(0.0, self.total_import - power_kw)
            phases = self.phase_import
        for phase, weight in spec.normalized_phase_weights().items():
            if phase in phases:
                phases[phase] = max(0.0, phases[phase] - power_kw * weight)

    def _release(self, spec: ActionSpec, action_delta: float) -> None:
        if abs(action_delta) <= _EPSILON:
            return
        export = action_delta < 0.0
        power_kw = abs(action_delta) * spec.power_scale(export=export)
        if export:
            if self.total_export is not None:
                self.total_export += power_kw
            phases = self.phase_export
        else:
            if self.total_import is not None:
                self.total_import += power_kw
            phases = self.phase_import
        for phase, weight in spec.normalized_phase_weights().items():
            if phase in phases:
                phases[phase] += power_kw * weight

    def snapshot(self) -> ElectricalHeadroom:
        return ElectricalHeadroom(
            total_import_kw=self.total_import,
            total_export_kw=self.total_export,
            phase_import_kw=dict(self.phase_import),
            phase_export_kw=dict(self.phase_export),
        )


def project_local_actions(request: LocalActionProjectionRequest) -> ProjectionResult:
    """Convenience functional API for stateless integrations."""

    return LocalActionSafetyProjector().project(request)


def _apply_directional_availability(
    action: float,
    available_import_action: Optional[float],
    available_export_action: Optional[float],
    reasons: list[InterventionReason],
) -> float:
    if action > 0.0 and available_import_action is not None and action > available_import_action:
        reasons.append(InterventionReason.IMPORT_AVAILABILITY)
        return float(available_import_action)
    if action < 0.0 and available_export_action is not None and -action > available_export_action:
        reasons.append(InterventionReason.EXPORT_AVAILABILITY)
        return -float(available_export_action)
    return action


def _clip_with_reason(
    value: float,
    low: float,
    high: float,
    reasons: list[InterventionReason],
    low_reason: InterventionReason,
    high_reason: InterventionReason,
) -> float:
    if value < low:
        reasons.append(low_reason)
        return low
    if value > high:
        reasons.append(high_reason)
        return high
    return value


def _unique_reasons(reasons: Sequence[InterventionReason]) -> Tuple[InterventionReason, ...]:
    return tuple(dict.fromkeys(reasons))


def _deduplicate_infeasible(reasons: Sequence[InfeasibleReason]) -> list[InfeasibleReason]:
    deduplicated: list[InfeasibleReason] = []
    seen: set[tuple[InfeasibleCode, int, str]] = set()
    for reason in reasons:
        key = (reason.code, reason.action_index, reason.action_name)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(reason)
    return deduplicated


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite.")


def _validate_optional_non_negative(value: Optional[float], field_name: str) -> None:
    if value is None:
        return
    _require_finite(value, field_name)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")


def _validate_headroom_mapping(values: Mapping[str, float], field_name: str) -> None:
    for phase, value in values.items():
        _validate_optional_non_negative(value, f"{field_name}[{phase!r}]")


def _validate_soc_envelope(soc: float, min_soc: float, max_soc: float, prefix: str) -> None:
    _require_finite(soc, f"{prefix}.soc")
    _require_finite(min_soc, f"{prefix}.min_soc")
    _require_finite(max_soc, f"{prefix}.max_soc")
    if max_soc < min_soc:
        raise ValueError(f"{prefix}.max_soc must be >= min_soc.")
