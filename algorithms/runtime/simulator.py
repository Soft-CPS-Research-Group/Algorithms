"""Shared simulator service extracted from the original job entry point."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from algorithms.contracts import SimulatorStep
from algorithms.registry import ENCODED_OBSERVATION_ALGORITHMS


class SimulatorAdapter:
    """Normalize the environment boundary, independently of the learner."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self, **kwargs: Any) -> tuple[Any, Mapping[str, Any]]:
        value = self.env.reset(**kwargs)
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
            return value
        return value, {}

    def step(self, actions: Any) -> SimulatorStep:
        value = self.env.step(actions)
        if not isinstance(value, tuple) or len(value) not in (4, 5):
            raise ValueError(
                "Simulator step must return observations, rewards, termination flags and info"
            )
        if len(value) == 5:
            observations, rewards, terminated, truncated, info = value
        else:
            observations, rewards, done, info = value
            info = info or {}
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated
        info = dict(info or {})
        duration = info.get("duration_seconds", getattr(self.env, "seconds_per_time_step", None))
        duration = float(duration) if duration is not None else None
        if duration is not None and (not np.isfinite(duration) or duration <= 0):
            raise ValueError("Simulator step duration_seconds must be finite and positive")
        return SimulatorStep(
            observations, rewards, bool(terminated), bool(truncated), info, duration
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def _community_market_overlay(simulator_cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(simulator_cfg, Mapping):
        return None
    community_market = simulator_cfg.get("community_market")
    if not isinstance(community_market, Mapping):
        return None

    overlay = dict(community_market)
    if overlay.get("intra_community_sell_ratio") is None:
        overlay["intra_community_sell_ratio"] = overlay.get("local_price_ratio_to_grid_import", 0.8)
    return overlay


def _load_electrical_service_overrides(
    simulator_cfg: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    """Load an explicit simulation-only electrical-service scenario overlay.

    The source dataset remains immutable.  These values describe an experimental
    or deployment contract and are deliberately kept outside the public dataset
    schema so provenance cannot be confused with measured dataset metadata.
    """

    if not isinstance(simulator_cfg, Mapping):
        return {}
    raw_path = simulator_cfg.get("electrical_service_overrides_path")
    if raw_path in (None, ""):
        return {}

    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Electrical-service overrides not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load electrical-service overrides: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Electrical-service overrides must be a mapping")
    if str(payload.get("version", "")) != "electrical_service_overrides_v1":
        raise ValueError(
            "Electrical-service overrides require version='electrical_service_overrides_v1'"
        )
    buildings = payload.get("buildings")
    if not isinstance(buildings, Mapping) or not buildings:
        raise ValueError("Electrical-service overrides require a non-empty buildings mapping")

    result: dict[str, Mapping[str, Any]] = {}
    for raw_agent_id, raw_service in buildings.items():
        agent_id = str(raw_agent_id)
        if not isinstance(raw_service, Mapping) or not raw_service:
            raise ValueError(
                f"Electrical-service override for {agent_id!r} must be a non-empty mapping"
            )
        result[agent_id] = deepcopy(dict(raw_service))
    return result


def _apply_simulation_schema_overlays(
    payload: dict[str, Any],
    simulator_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply declared scenario overlays to an in-memory schema copy only."""

    if not isinstance(simulator_cfg, Mapping):
        return payload
    buildings = payload.get("buildings")
    if not isinstance(buildings, Mapping):
        if simulator_cfg.get("building_ids") or simulator_cfg.get(
            "electrical_service_overrides_path"
        ):
            raise ValueError("Schema overlays require a buildings mapping")
        return payload

    mutable_buildings = dict(buildings)
    for agent_id, service in _load_electrical_service_overrides(simulator_cfg).items():
        if agent_id not in mutable_buildings:
            raise ValueError(
                f"Electrical-service override references unknown building {agent_id!r}"
            )
        building = deepcopy(dict(mutable_buildings[agent_id] or {}))
        existing = building.get("electrical_service")
        if existing not in (None, {}) and existing != service:
            raise ValueError(
                f"Electrical-service override refuses to replace dataset limits for {agent_id!r}"
            )
        building["electrical_service"] = deepcopy(dict(service))
        mutable_buildings[agent_id] = building

    raw_ids = simulator_cfg.get("building_ids")
    if raw_ids is not None:
        selected = [str(item) for item in raw_ids]
        if not selected:
            raise ValueError("simulator.building_ids must not be empty")
        if len(selected) != len(set(selected)):
            raise ValueError("simulator.building_ids must contain unique IDs")
        unknown = [agent_id for agent_id in selected if agent_id not in mutable_buildings]
        if unknown:
            raise ValueError(
                "simulator.building_ids contains unknown buildings: " + ", ".join(unknown)
            )
        mutable_buildings = {agent_id: mutable_buildings[agent_id] for agent_id in selected}

    payload["buildings"] = mutable_buildings
    if simulator_cfg.get("terminal_observation_padding") is not None:
        payload["terminal_observation_padding"] = bool(
            simulator_cfg["terminal_observation_padding"]
        )
    return payload


def _resolve_citylearn_schema_input(
    dataset_path_value: Any,
    simulator_cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Prefer local schema payloads to avoid remote dataset-index lookups.

    CityLearn may query remote dataset registries when receiving a string schema
    identifier/path. If the provided path exists locally, loading and passing the
    JSON payload avoids those external calls (important on restricted HPC nodes).
    """
    if not isinstance(dataset_path_value, str):
        return dataset_path_value

    candidate = Path(dataset_path_value)
    if not candidate.exists():
        return dataset_path_value

    try:
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return dataset_path_value

    if isinstance(payload, dict):
        # Prefer the schema directory for portability, but preserve an explicit
        # relative root when it demonstrably resolves local dataset assets. This
        # is required for canonical variant schemas stored under ``schemas/``
        # with ``root_directory: ..``. Absolute or stale copied paths still
        # fall back to the directory containing the local schema.
        local_root = candidate.resolve().parent
        declared_root = payload.get("root_directory")
        if isinstance(declared_root, str) and declared_root.strip():
            declared_path = Path(declared_root).expanduser()
            if not declared_path.is_absolute():
                resolved_declared_root = (local_root / declared_path).resolve()
                referenced_files = []
                for building in (payload.get("buildings") or {}).values():
                    if not isinstance(building, Mapping):
                        continue
                    for key in ("energy_simulation", "weather", "pricing", "carbon_intensity"):
                        value = building.get(key)
                        if isinstance(value, str) and value.strip():
                            referenced_files.append(value)
                if any(
                    (resolved_declared_root / relative).is_file() for relative in referenced_files
                ):
                    local_root = resolved_declared_root
        payload["root_directory"] = str(local_root)
        community_market = _community_market_overlay(simulator_cfg)
        if community_market is not None:
            payload["community_market"] = community_market
        return _apply_simulation_schema_overlays(payload, simulator_cfg)

    return dataset_path_value


def _resolve_agent_observation_dimensions(wrapper: Any, algorithm_name: Optional[str]) -> list[int]:
    """Resolve the observation dimensions seen by the agent, after preprocessing.

    The wrapper owns preprocessing, but neural agents build networks before the
    first rollout. For agents that consume encoded observations, use the encoded
    observation shape exposed by the wrapper's encoder method without changing
    wrapper behavior.
    """
    fallback = list(getattr(wrapper, "observation_dimension", []) or [])
    if str(algorithm_name or "").strip() not in ENCODED_OBSERVATION_ALGORITHMS:
        return fallback

    get_encoded = getattr(wrapper, "get_encoded_observations", None)
    if not callable(get_encoded):
        return fallback

    observation_space = list(getattr(wrapper, "observation_space", []) or [])
    observation_names = list(getattr(wrapper, "observation_names", []) or [])
    dimensions: list[int] = []

    try:
        for index, space in enumerate(observation_space):
            if hasattr(space, "low"):
                sample = np.asarray(space.low, dtype=np.float64).reshape(-1)
            elif index < len(observation_names):
                sample = np.zeros(len(observation_names[index]), dtype=np.float64)
            else:
                sample = np.array([], dtype=np.float64)

            encoded = np.asarray(get_encoded(index, sample), dtype=np.float64).reshape(-1)
            dimensions.append(int(encoded.shape[0]))
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise ValueError(
            f"Could not resolve encoded observation dimensions for {algorithm_name}: {exc}"
        ) from exc

    return dimensions if dimensions else fallback


def _validate_dynamic_entity_schema_input(
    schema_input: Any,
    *,
    interface: str,
    topology_mode: str,
) -> None:
    """Fail fast when dynamic entity mode is requested with incompatible schema payload."""
    if (
        str(interface).strip().lower() != "entity"
        or str(topology_mode).strip().lower() != "dynamic"
    ):
        return

    if not isinstance(schema_input, dict):
        raise ValueError(
            "simulator.interface='entity' with simulator.topology_mode='dynamic' requires a local JSON schema payload."
        )

    schema_interface = str(schema_input.get("interface", "")).strip().lower()
    if schema_interface and schema_interface != "entity":
        raise ValueError("Dynamic topology schema must declare interface='entity'.")

    topology_events = schema_input.get("topology_events")
    if not isinstance(topology_events, list):
        raise ValueError("Dynamic topology schema must include a topology_events list.")


def create_environment(config, simulation_data_dir, *, env_factory=None):
    """Construct the environment once, with the same settings for jobs and preflight."""
    if env_factory is None:
        from citylearn.citylearn import CityLearnEnv

        env_factory = CityLearnEnv
    from reward_function.registry import REWARD_FUNCTION_MAP, get_available_reward_function_names

    simulator_cfg = config["simulator"]
    export_cfg = simulator_cfg.get("export", {})
    reward_key = simulator_cfg["reward_function"]
    reward_cls = REWARD_FUNCTION_MAP.get(reward_key)
    if reward_cls is None:
        raise ValueError(
            f"Unsupported reward function {reward_key!r}; available: {get_available_reward_function_names()}"
        )
    schema_input = _resolve_citylearn_schema_input(simulator_cfg["dataset_path"], simulator_cfg)
    interface_mode = simulator_cfg.get("interface", "flat")
    topology_mode = simulator_cfg.get("topology_mode", "static")
    _validate_dynamic_entity_schema_input(
        schema_input, interface=interface_mode, topology_mode=topology_mode
    )
    path_info = {"simulation_data_dir": simulation_data_dir}
    env_kwargs = {
        "schema": schema_input,
        "central_agent": simulator_cfg["central_agent"],
        "interface": interface_mode,
        "topology_mode": topology_mode,
        "reward_function": reward_cls,
        "offline": True,
        "render_mode": export_cfg.get("mode", "none"),
        "export_kpis_on_episode_end": export_cfg.get("export_kpis_on_episode_end", False),
        "export_only_final_episode": export_cfg.get("final_episode_only", True),
        "render_directory": str(path_info["simulation_data_dir"]),
    }
    if simulator_cfg.get("random_seed") is not None:
        env_kwargs["random_seed"] = int(simulator_cfg["random_seed"])
    reward_function_kwargs = simulator_cfg.get("reward_function_kwargs")
    if isinstance(reward_function_kwargs, dict) and reward_function_kwargs:
        env_kwargs["reward_function_kwargs"] = reward_function_kwargs
    if export_cfg.get("session_name"):
        env_kwargs["render_session_name"] = export_cfg["session_name"]

    for key in ("simulation_start_time_step", "simulation_end_time_step", "episode_time_steps"):
        value = simulator_cfg.get(key)
        if value is not None:
            env_kwargs[key] = value

    return env_factory(**env_kwargs)
