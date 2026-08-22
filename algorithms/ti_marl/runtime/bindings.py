"""Technology-specific routing kept outside public typed interfaces."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


SIMULATOR_BINDINGS_VERSION = "ti_marl_simulator_bindings_v1"


class SimulatorBindingMap:
    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        raw = deepcopy(dict(payload or {}))
        if raw and str(raw.get("version")) != SIMULATOR_BINDINGS_VERSION:
            raise ValueError(
                f"Simulator bindings require version='{SIMULATOR_BINDINGS_VERSION}'"
            )
        self.payload = raw
        self.agents = dict(raw.get("agents", {}) or {})

    @classmethod
    def load(cls, path: str | Path | None) -> "SimulatorBindingMap":
        if path is None:
            return cls()
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"TI-MARL Simulator bindings not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("TI-MARL Simulator bindings must be a mapping")
        return cls(payload)

    def sensor(self, agent_id: str, sensor_id: str) -> Mapping[str, Any]:
        return dict(
            dict(self.agents.get(str(agent_id), {}) or {})
            .get("sensors", {})
            .get(str(sensor_id), {})
            or {}
        )

    def actuator(self, agent_id: str, actuator_id: str) -> Mapping[str, Any]:
        return dict(
            dict(self.agents.get(str(agent_id), {}) or {})
            .get("actuators", {})
            .get(str(actuator_id), {})
            or {}
        )

    def sensor_entity_id(self, agent_id: str, sensor_id: str) -> Optional[str]:
        value = self.sensor(agent_id, sensor_id).get("entity_id")
        return None if value is None else str(value)

    def actuator_entity_id(self, agent_id: str, actuator_id: str) -> Optional[str]:
        value = self.actuator(agent_id, actuator_id).get("entity_id")
        return None if value is None else str(value)

    def observation_feature(
        self,
        agent_id: str,
        sensor_id: str,
        observation_id: str,
    ) -> str:
        observations = dict(self.sensor(agent_id, sensor_id).get("observations", {}) or {})
        return str(observations.get(observation_id, observation_id))
