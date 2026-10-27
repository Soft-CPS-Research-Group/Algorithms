"""Explicit migration and override precedence for job configuration v1."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Mapping

SCHEMA_VERSION = "algorithms.config.v1"

# These keys existed in templates but were discarded by the original schema.
# Preserve the effective legacy behaviour; enabling their requested values is an
# explicit v1 configuration change, never a side effect of consolidation.
LEGACY_IGNORED_RBC_KEYS = frozenset(
    {
        "ev_v2g_buffer_soc",
        "ev_subhour_service_guard_steps",
        "ev_subhour_deadline_buffer_steps",
        "ev_subhour_v2g_guard_steps",
        "ev_subhour_min_service_margin_rate",
        "ev_subhour_v2g_reserve_soc",
        "deferrable_import_block_threshold_kw",
        "deferrable_community_import_block_threshold_kw",
        "ev_subhour_tight_feasibility_ratio",
        "ev_service_soc_tolerance",
    }
)


class ConfigurationMigrationWarning(UserWarning):
    """A legacy value needs an explicit migration decision."""


def migrate_config(
    raw: Mapping[str, Any], *, legacy_algorithm: bool = False
) -> tuple[dict[str, Any], list[str]]:
    """Return an independent v1 document and an audit of changed syntax/values."""
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration must be a mapping")
    payload = deepcopy(dict(raw))
    version = payload.get("schema_version")
    if version not in (None, SCHEMA_VERSION):
        raise ValueError(f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
    notes: list[str] = []
    if "algorithm" in payload:
        if "pipeline" in payload:
            raise ValueError("algorithm and pipeline are ambiguous; retain only pipeline")
        if not legacy_algorithm:
            raise ValueError(
                "Deprecated top-level 'algorithm': migrate to a 'pipeline' list with --migrate-config"
            )
        algorithm = payload.pop("algorithm")
        if not isinstance(algorithm, dict) or not algorithm.get("name"):
            raise ValueError("Legacy algorithm must be a mapping with a name")
        algorithm["algorithm"] = algorithm.pop("name")
        payload["pipeline"] = [algorithm]
        notes.append("Moved algorithm.name and parameters into pipeline[0].")
    if version is None:
        for index, stage in enumerate(payload.get("pipeline") or []):
            if not isinstance(stage, dict):
                continue
            name = str(stage.get("algorithm", ""))
            if name in {"PPO", "IPPO", "MAPPO", "HAPPO"}:
                from algorithms.configuration.extensions import PARAMETER_KEYS

                exploration = stage.get("exploration") or {}
                params = exploration.get("params") if isinstance(exploration, dict) else None
                if isinstance(params, dict):
                    unused = (PARAMETER_KEYS["MADDPG"] - PARAMETER_KEYS["PPO"]) & params.keys()
                    for key in sorted(unused):
                        params.pop(key)
                        notes.append(
                            f"pipeline[{index}].exploration.params.{key} was unused by {name}; removed."
                        )
            if name == "CommunityCoordinator":
                parameters = stage.get("hyperparameters") or {}
                for key in ("w_factor", "w_smoothness"):
                    if key in parameters:
                        parameters.pop(key)
                        notes.append(
                            f"pipeline[{index}].hyperparameters.{key} was unused by CommunityCoordinator; removed."
                        )
            if name in {"MADDPG", "MATD3", "TD3", "MASAC"}:
                exploration = stage.get("exploration") or {}
                params = exploration.get("params") if isinstance(exploration, dict) else None
                if isinstance(params, dict) and "actor_behavior_cloning_replay_capacity" in params:
                    params.pop("actor_behavior_cloning_replay_capacity")
                    notes.append(
                        f"pipeline[{index}].exploration.params.actor_behavior_cloning_replay_capacity was unused by {name}; removed."
                    )
            if "RBC" not in name and name not in {
                "NormalPolicy",
                "NormalNoBatteryPolicy",
                "RuleBasedPolicy",
                "RandomPolicy",
            }:
                continue
            parameters = stage.get("hyperparameters")
            if not isinstance(parameters, dict):
                continue
            for key in sorted(LEGACY_IGNORED_RBC_KEYS & parameters.keys()):
                value = parameters.pop(key)
                notes.append(
                    f"pipeline[{index}].hyperparameters.{key}={value!r} was ignored by the legacy schema; retained the agent default. Set schema_version={SCHEMA_VERSION!r} explicitly to use this parameter."
                )
    if legacy_algorithm and "simulator" in payload:
        from algorithms.configuration.episodes import episode_plan

        plan = episode_plan(payload)
        payload.setdefault("training", {})["episodes"] = plan["training"]
        payload.setdefault("evaluation", {})["episodes"] = plan["evaluation"]
        payload["simulator"].pop("episodes", None)
        payload["simulator"].pop("deterministic_finish", None)
        notes.append(
            f"Preserved {plan['training']} training and {plan['evaluation']} evaluation episodes explicitly."
        )
    payload["schema_version"] = SCHEMA_VERSION
    return payload, notes


def normalize_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload, notes = migrate_config(raw)
    for note in notes:
        warnings.warn(note, ConfigurationMigrationWarning, stacklevel=2)
    return payload


def apply_overrides(raw: Mapping[str, Any], overrides: list[str]) -> dict[str, Any]:
    """CLI values override the source, then the same schemas validate the result."""
    import yaml

    if not isinstance(raw, Mapping):
        raise ValueError("Configuration must be a mapping")
    payload = deepcopy(dict(raw))
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must be path=value: {expression!r}")
        path, value = expression.split("=", 1)
        parts = path.split(".")
        if not all(parts):
            raise ValueError(f"Invalid override path: {path!r}")
        target: Any = payload
        try:
            for component in parts[:-1]:
                if isinstance(target, list):
                    if int(component) < 0:
                        raise IndexError(component)
                    target = target[int(component)]
                elif isinstance(target, dict):
                    target = target.setdefault(component, {})
                else:
                    raise ValueError(f"Override crosses a scalar at {path!r}")
            if not isinstance(target, (list, dict)):
                raise ValueError(f"Override target is not a mapping/list: {path!r}")
            if isinstance(target, list):
                index = int(parts[-1])
                if index < 0:
                    raise IndexError(index)
                target[index] = yaml.safe_load(value)
            else:
                target[parts[-1]] = yaml.safe_load(value)
        except (IndexError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid override target: {path!r}") from error
    return payload
