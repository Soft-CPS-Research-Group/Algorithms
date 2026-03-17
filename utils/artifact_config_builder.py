"""Helpers to auto-populate artifact config from dataset electrical topology."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_PHASES = ["L1", "L2", "L3"]


def build_auto_artifact_config(
    *,
    context: Mapping[str, Any],
    agent_index: int,
) -> Dict[str, Any]:
    """Build optional artifact config fields from dataset schema and agent context."""
    dataset_path = _resolve_dataset_path(context)
    if dataset_path is None:
        return {}

    schema = _load_schema(dataset_path)
    buildings = schema.get("buildings")
    if not isinstance(buildings, Mapping) or not buildings:
        return {}

    building_name = _resolve_building_name(context=context, agent_index=agent_index, buildings=buildings)
    if building_name is None:
        return {}

    building = buildings.get(building_name)
    if not isinstance(building, Mapping):
        return {}

    action_names = _resolve_action_names(context=context, agent_index=agent_index)

    line_limits = _extract_line_limits(building)
    max_board_kw = _extract_max_board_kw(building)
    available_lines = list(line_limits.keys()) if line_limits else list(DEFAULT_PHASES)
    charger_phase_map = _extract_charger_phase_map(building, line_limits)

    chargers_cfg: Dict[str, Dict[str, Any]] = {}
    selected_charger_ids: List[str] = []
    selected_charger_set: set[str] = set()
    raw_chargers = building.get("chargers")
    if isinstance(raw_chargers, Mapping):
        charger_ids = [str(charger_id) for charger_id in raw_chargers.keys()]
        selected_charger_ids = _resolve_selected_charger_ids(
            action_names=action_names,
            charger_ids=charger_ids,
        )
        selected_charger_set = set(selected_charger_ids)
        for charger_id, charger_data in raw_chargers.items():
            cid = str(charger_id)
            if selected_charger_set and cid not in selected_charger_set:
                continue
            if not isinstance(charger_data, Mapping):
                continue

            attrs = charger_data.get("attributes")
            attrs = attrs if isinstance(attrs, Mapping) else {}
            max_kw = _safe_float(
                attrs.get("max_charging_power"),
                _safe_float(attrs.get("nominal_power"), 0.0),
            )
            min_kw = _safe_float(attrs.get("min_charging_power"), 0.0)

            phases = charger_phase_map.get(cid) or _resolve_phases_from_connection(
                attrs.get("phase_connection"),
                available_lines=available_lines,
            )
            phases = _normalize_phases(phases)

            meta: Dict[str, Any] = {
                "min_kw": float(min_kw),
                "max_kw": float(max_kw),
                "allow_flex_when_ev": True,
            }
            if len(phases) == 1:
                meta["line"] = phases[0]
            elif len(phases) > 1:
                meta["phases"] = phases
            chargers_cfg[cid] = meta

    if not chargers_cfg:
        return {}

    _reconcile_line_limits(
        line_limits=line_limits,
        chargers=chargers_cfg,
        filter_to_actions=bool(selected_charger_set),
    )

    action_order = [charger_id for charger_id in selected_charger_ids if charger_id in chargers_cfg]
    if not action_order:
        action_order = sorted(chargers_cfg.keys())

    config: Dict[str, Any] = {"chargers": chargers_cfg}
    if line_limits:
        config["line_limits"] = line_limits
    if action_order:
        config["action_order"] = action_order
    if max_board_kw is not None:
        config["max_board_kw"] = float(max_board_kw)
    return config


def _resolve_dataset_path(context: Mapping[str, Any]) -> Optional[Path]:
    simulator = ((context.get("config") or {}).get("simulator") or {})
    raw = simulator.get("dataset_path")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.exists():
        return None
    return path.resolve()


@lru_cache(maxsize=16)
def _load_schema(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_building_name(
    *,
    context: Mapping[str, Any],
    agent_index: int,
    buildings: Mapping[str, Any],
) -> Optional[str]:
    env = context.get("environment") or {}
    building_names = env.get("building_names")
    if isinstance(building_names, list) and building_names:
        if 0 <= agent_index < len(building_names):
            candidate = str(building_names[agent_index])
            if candidate in buildings:
                return candidate
        candidate = str(building_names[0])
        if candidate in buildings:
            return candidate

    ordered = list(buildings.keys())
    if not ordered:
        return None
    if 0 <= agent_index < len(ordered):
        return str(ordered[agent_index])
    return str(ordered[-1])


def _resolve_action_names(*, context: Mapping[str, Any], agent_index: int) -> List[str]:
    env = context.get("environment") or {}
    by_agent = env.get("action_names_by_agent")
    if isinstance(by_agent, Mapping):
        values = by_agent.get(str(agent_index))
        if values is None:
            values = by_agent.get(agent_index)
        if isinstance(values, list):
            return [str(item) for item in values]
    if isinstance(by_agent, list):
        if 0 <= agent_index < len(by_agent) and isinstance(by_agent[agent_index], list):
            return [str(item) for item in by_agent[agent_index]]
        if len(by_agent) == 1 and isinstance(by_agent[0], list):
            return [str(item) for item in by_agent[0]]

    action_names = env.get("action_names")
    if isinstance(action_names, list):
        return [str(item) for item in action_names]
    return []


def _resolve_selected_charger_ids(
    *,
    action_names: List[str],
    charger_ids: List[str],
) -> List[str]:
    if not action_names or not charger_ids:
        return []

    ordered: List[str] = []
    seen: set[str] = set()
    for action_name in action_names:
        matched = _match_action_to_charger_id(action_name=action_name, charger_ids=charger_ids)
        if matched is None or matched in seen:
            continue
        ordered.append(matched)
        seen.add(matched)
    return ordered


def _match_action_to_charger_id(*, action_name: str, charger_ids: List[str]) -> Optional[str]:
    action = str(action_name).strip()
    if not action:
        return None
    for charger_id in charger_ids:
        if action == charger_id:
            return charger_id
    for charger_id in charger_ids:
        if action.endswith(f"_{charger_id}") or action.endswith(charger_id):
            return charger_id
    return None


def _extract_max_board_kw(building: Mapping[str, Any]) -> Optional[float]:
    constraints = building.get("charging_constraints")
    if isinstance(constraints, Mapping):
        board_limit = _maybe_float(constraints.get("building_limit_kw"))
        if board_limit is not None:
            return board_limit

    electrical_service = building.get("electrical_service")
    if isinstance(electrical_service, Mapping):
        limits = electrical_service.get("limits")
        if isinstance(limits, Mapping):
            total = limits.get("total")
            if isinstance(total, Mapping):
                import_limit = _maybe_float(total.get("import_kw"))
                if import_limit is not None:
                    return import_limit
    return None


def _extract_line_limits(building: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    limits: Dict[str, Dict[str, Any]] = {}

    constraints = building.get("charging_constraints")
    if isinstance(constraints, Mapping):
        phases = constraints.get("phases")
        if isinstance(phases, list):
            for raw_phase in phases:
                if not isinstance(raw_phase, Mapping):
                    continue
                name = str(raw_phase.get("name") or "").strip()
                if not name:
                    continue
                phase_limit = _maybe_float(raw_phase.get("limit_kw"))
                chargers_raw = raw_phase.get("chargers")
                chargers = (
                    [str(item) for item in chargers_raw if item is not None]
                    if isinstance(chargers_raw, list)
                    else []
                )
                entry: Dict[str, Any] = {"chargers": chargers}
                if phase_limit is not None:
                    entry["limit_kw"] = float(phase_limit)
                limits[name] = entry
            if limits:
                return limits

    electrical_service = building.get("electrical_service")
    if isinstance(electrical_service, Mapping):
        svc_limits = electrical_service.get("limits")
        if isinstance(svc_limits, Mapping):
            per_phase = svc_limits.get("per_phase")
            if isinstance(per_phase, Mapping):
                for name, raw_cfg in per_phase.items():
                    line_name = str(name).strip()
                    if not line_name:
                        continue
                    cfg = raw_cfg if isinstance(raw_cfg, Mapping) else {}
                    phase_limit = _maybe_float(cfg.get("import_kw"))
                    entry: Dict[str, Any] = {"chargers": []}
                    if phase_limit is not None:
                        entry["limit_kw"] = float(phase_limit)
                    limits[line_name] = entry

    return limits


def _extract_charger_phase_map(
    building: Mapping[str, Any],
    line_limits: Mapping[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for line_name, info in line_limits.items():
        chargers = info.get("chargers")
        if not isinstance(chargers, list):
            continue
        for charger_id in chargers:
            cid = str(charger_id)
            mapping.setdefault(cid, [])
            if line_name not in mapping[cid]:
                mapping[cid].append(line_name)
    return mapping


def _resolve_phases_from_connection(
    raw_connection: Any,
    *,
    available_lines: List[str],
) -> List[str]:
    phases = []
    if isinstance(raw_connection, str):
        token = raw_connection.strip()
        if token:
            lowered = token.lower()
            if lowered in {"all", "all_phases", "three_phase", "three-phases"}:
                phases = list(available_lines or DEFAULT_PHASES)
            else:
                phases = [token]
    elif isinstance(raw_connection, list):
        phases = [str(item) for item in raw_connection if item is not None]
    return _normalize_phases(phases)


def _normalize_phases(phases: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    for phase in phases:
        token = str(phase).strip()
        if not token or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    return normalized


def _reconcile_line_limits(
    *,
    line_limits: Dict[str, Dict[str, Any]],
    chargers: Mapping[str, Dict[str, Any]],
    filter_to_actions: bool,
) -> None:
    for info in line_limits.values():
        raw = info.get("chargers")
        known = [cid for cid in (raw if isinstance(raw, list) else []) if cid in chargers]
        info["chargers"] = known

    for cid, meta in chargers.items():
        phases = []
        if isinstance(meta.get("line"), str):
            phases = [meta["line"]]
        elif isinstance(meta.get("phases"), list):
            phases = [str(item) for item in meta["phases"]]
        for line_name in _normalize_phases(phases):
            line_cfg = line_limits.setdefault(line_name, {"chargers": []})
            line_chargers = line_cfg.get("chargers")
            if not isinstance(line_chargers, list):
                line_cfg["chargers"] = [cid]
                continue
            if cid not in line_chargers:
                line_chargers.append(cid)

    if filter_to_actions:
        for line_name, info in list(line_limits.items()):
            chargers_on_line = info.get("chargers")
            if isinstance(chargers_on_line, list) and not chargers_on_line:
                del line_limits[line_name]


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    parsed = _maybe_float(value)
    return parsed if parsed is not None else default
