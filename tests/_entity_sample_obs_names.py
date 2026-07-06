"""Synthesize per-building observation names from the bundled entity payload
sample without requiring a CityLearn env.

Mirrors the emission ordering in ``utils/entity_adapter.py`` for the first
building (``Building_1``) of the bundled tokenizer fixture.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List


_SAMPLE_PATH = Path("configs/tokenizers/fixtures/entity_obs_sample.json")


def _table_features(payload: dict, table: str) -> list[str]:
    return list(payload["tables"][table]["features"])


def _table_row_ids(payload: dict, table: str) -> list[str]:
    return [row["id"] for row in payload["tables"][table]["rows"]]


def _edge_pairs(payload: dict, edge_name: str) -> list[tuple[int, int]]:
    edges = payload["edges"][edge_name]["edges"]
    return [(e["source_index"], e["target_index"]) for e in edges]


def _edge_targets_for_source(
    payload: dict, edge_name: str, source_index: int
) -> list[int]:
    return [
        target
        for src, target in _edge_pairs(payload, edge_name)
        if src == source_index and target >= 0
    ]


def load_sample_observation_names_for_first_building() -> List[str]:
    """Return ``observation_names[0]`` for ``Building_1`` of the bundled sample.

    Emission order (matches ``utils/entity_adapter.py``):
      1. district block (prefixed ``district__``)
      2. building block (unprefixed)
      3. per-storage attached to building (prefixed ``storage::<id>::``)
      4. per-PV attached to building (prefixed ``pv::<id>::``)
      5. per-charger attached to building, each producing:
            charger features (``charger::<id>::``)
            connected_ev features (``charger::<id>::connected_ev::``)
            incoming_ev features (``charger::<id>::incoming_ev::``)
      6. operational counters (3 unprefixed)
      7. RBC alias features OR fallback charger-state features
      8. ``electric_vehicle_is_flexible``
    """
    payload = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))
    district_features = _table_features(payload, "district")
    building_features = _table_features(payload, "building")
    storage_features = _table_features(payload, "storage")
    pv_features = _table_features(payload, "pv")
    charger_features = _table_features(payload, "charger")
    ev_features = _table_features(payload, "ev")

    storage_ids = _table_row_ids(payload, "storage")
    pv_ids = _table_row_ids(payload, "pv")
    charger_ids = _table_row_ids(payload, "charger")

    building_index = 0  # Building_1
    storage_targets = _edge_targets_for_source(
        payload, "building_to_storage", building_index
    )
    pv_targets = _edge_targets_for_source(
        payload, "building_to_pv", building_index
    )
    charger_targets = _edge_targets_for_source(
        payload, "building_to_charger", building_index
    )

    names: List[str] = []
    # 1. district
    for f in district_features:
        names.append(f"district__{f}")
    # 2. building
    for f in building_features:
        names.append(f)
    # 3. per-storage
    for row in storage_targets:
        sid = storage_ids[row]
        for f in storage_features:
            names.append(f"storage::{sid}::{f}")
    # 4. per-pv
    for row in pv_targets:
        pid = pv_ids[row]
        for f in pv_features:
            names.append(f"pv::{pid}::{f}")
    # 5. per-charger (+ connected_ev, + incoming_ev)
    for row in charger_targets:
        cid = charger_ids[row]
        for f in charger_features:
            names.append(f"charger::{cid}::{f}")
        for f in ev_features:
            names.append(f"charger::{cid}::connected_ev::{f}")
        for f in ev_features:
            names.append(f"charger::{cid}::incoming_ev::{f}")

    # 6. operational counters
    names.append("active_chargers_count")
    names.append("active_storages_count")
    names.append("active_pvs_count")

    # 7. RBC charger aliases (only when chargers are attached); see
    #    utils/entity_adapter.py:301-319. Building_1 has chargers attached,
    #    so the alias branch is taken.
    if charger_targets:
        names.append("electric_vehicle_charger_state")
        names.append("electric_vehicle_soc")
        names.append("electric_vehicle_required_soc_departure")
        names.append("electric_vehicle_departure_time")
    else:
        names.extend(
            [
                "electric_vehicle_charger_state",
                "electric_vehicle_soc",
                "electric_vehicle_required_soc_departure",
                "electric_vehicle_departure_time",
            ]
        )

    # 8. flexibility flag
    names.append("electric_vehicle_is_flexible")
    return names
