"""Versioned, dependency-free boundaries shared by jobs and agent implementations.

Numerical learning remains owned by the agents. These records describe what the
runtime may call and preserve the distinction between a policy proposal, an action
submitted to the simulator and an executed action reported by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

CONTRACT_VERSION = "algorithms.agent.v1"
LearningKind = Literal["none", "replay", "rollout", "offline"]
ObservationKind = Literal["raw", "encoded"]
SignalKind = Literal["none", "price", "direction", "actions", "legacy"]
ActionRows = Sequence[Sequence[float]]


@dataclass(frozen=True)
class AgentCapabilities:
    learning: LearningKind
    observations: ObservationKind
    interfaces: tuple[str, ...] = ("flat", "entity")
    dynamic_topology: bool = False
    final_stage_only: bool = False
    standalone_only: bool = False
    single_controller: bool = False
    single_building: bool = False
    checkpoint_save: bool = False
    checkpoint_load: bool = False
    export_formats: tuple[str, ...] = ()
    input_signal: SignalKind = "none"
    output_signal: SignalKind = "actions"
    dependencies: tuple[str, ...] = ("training",)
    status: Literal["existing", "experimental"] = "existing"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    implementation: str
    configuration: str
    capabilities: AgentCapabilities
    environment_validator: str | None = None


@dataclass(frozen=True)
class StageContext:
    """A manager hand-off; values retain the numerical representation agents use."""

    value: Any
    source_stage: int | None = None
    kind: SignalKind = "legacy"
    member_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdateSchedule:
    global_step: int
    update: bool
    update_targets: bool
    exploration_done: bool


@dataclass(frozen=True)
class Transition:
    observations: Any
    next_observations: Any
    rewards: Sequence[float]
    submitted_actions: ActionRows
    terminated: bool
    truncated: bool
    proposed_actions: ActionRows | None = None
    executed_actions: ActionRows | None = None
    member_ids: tuple[str, ...] = ()
    duration_seconds: float | None = None
    info: Mapping[str, Any] | None = None

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class SimulatorStep:
    observations: Any
    rewards: Sequence[float]
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    duration_seconds: float | None

    def as_tuple(self) -> tuple[Any, Sequence[float], bool, bool, Mapping[str, Any]]:
        return self.observations, self.rewards, self.terminated, self.truncated, self.info
