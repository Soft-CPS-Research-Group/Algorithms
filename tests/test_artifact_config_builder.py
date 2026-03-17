from __future__ import annotations

import json
from pathlib import Path

from utils.artifact_config_builder import build_auto_artifact_config


def _write_schema(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_auto_artifact_config_from_charging_constraints(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_schema(
        schema_path,
        {
            "buildings": {
                "Building_A": {
                    "chargers": {
                        "AC0001": {"attributes": {"max_charging_power": 4.6, "min_charging_power": 0.0}},
                        "BB0001": {"attributes": {"nominal_power": 9.0, "min_charging_power": 0.0}},
                    },
                    "charging_constraints": {
                        "building_limit_kw": 33.0,
                        "phases": [
                            {"name": "L1", "limit_kw": 11.0, "chargers": ["AC0001", "BB0001"]},
                            {"name": "L2", "limit_kw": 11.0, "chargers": ["BB0001"]},
                            {"name": "L3", "limit_kw": 11.0, "chargers": []},
                        ],
                    },
                }
            }
        },
    )
    context = {
        "config": {"simulator": {"dataset_path": str(schema_path)}},
        "environment": {
            "building_names": ["Building_A"],
            "action_names_by_agent": {"0": ["AC0001", "BB0001", "b_1"]},
        },
    }

    cfg = build_auto_artifact_config(context=context, agent_index=0)

    assert cfg["max_board_kw"] == 33.0
    assert cfg["action_order"] == ["AC0001", "BB0001"]
    assert cfg["chargers"]["AC0001"]["line"] == "L1"
    assert cfg["chargers"]["BB0001"]["phases"] == ["L1", "L2"]
    assert cfg["line_limits"]["L1"]["limit_kw"] == 11.0
    assert cfg["line_limits"]["L1"]["chargers"] == ["AC0001", "BB0001"]
    assert cfg["line_limits"]["L2"]["chargers"] == ["BB0001"]
    assert "L3" not in cfg["line_limits"]


def test_build_auto_artifact_config_from_electrical_service(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_schema(
        schema_path,
        {
            "buildings": {
                "Building_B": {
                    "chargers": {
                        "CHG1": {
                            "attributes": {
                                "max_charging_power": 4.0,
                                "min_charging_power": 0.0,
                                "phase_connection": "L1",
                            }
                        },
                        "CHG2": {
                            "attributes": {
                                "max_charging_power": 7.0,
                                "min_charging_power": 0.0,
                                "phase_connection": "all_phases",
                            }
                        },
                    },
                    "electrical_service": {
                        "limits": {
                            "total": {"import_kw": 12.0},
                            "per_phase": {
                                "L1": {"import_kw": 7.0},
                                "L2": {"import_kw": 5.0},
                            },
                        }
                    },
                }
            }
        },
    )
    context = {
        "config": {"simulator": {"dataset_path": str(schema_path)}},
        "environment": {
            "building_names": ["Building_B"],
            "action_names_by_agent": {"0": ["CHG2"]},
        },
    }

    cfg = build_auto_artifact_config(context=context, agent_index=0)

    assert cfg["max_board_kw"] == 12.0
    assert cfg["action_order"] == ["CHG2"]
    assert cfg["chargers"] == {
        "CHG2": {
            "min_kw": 0.0,
            "max_kw": 7.0,
            "allow_flex_when_ev": True,
            "phases": ["L1", "L2"],
        }
    }
    assert cfg["line_limits"]["L1"]["limit_kw"] == 7.0
    assert cfg["line_limits"]["L1"]["chargers"] == ["CHG2"]
    assert cfg["line_limits"]["L2"]["limit_kw"] == 5.0
    assert cfg["line_limits"]["L2"]["chargers"] == ["CHG2"]


def test_build_auto_artifact_config_matches_prefixed_action_names(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    _write_schema(
        schema_path,
        {
            "buildings": {
                "Building_C": {
                    "chargers": {
                        "CHG2": {
                            "attributes": {
                                "max_charging_power": 7.0,
                                "min_charging_power": 0.0,
                                "phase_connection": "L1",
                            }
                        }
                    },
                    "electrical_service": {
                        "limits": {
                            "total": {"import_kw": 12.0},
                            "per_phase": {"L1": {"import_kw": 7.0}},
                        }
                    },
                }
            }
        },
    )
    context = {
        "config": {"simulator": {"dataset_path": str(schema_path)}},
        "environment": {
            "building_names": ["Building_C"],
            "action_names_by_agent": {"0": ["electrical_storage", "electric_vehicle_storage_CHG2"]},
        },
    }

    cfg = build_auto_artifact_config(context=context, agent_index=0)

    assert cfg["action_order"] == ["CHG2"]
    assert cfg["chargers"]["CHG2"]["line"] == "L1"
    assert cfg["line_limits"]["L1"]["limit_kw"] == 7.0
    assert cfg["line_limits"]["L1"]["chargers"] == ["CHG2"]


def test_build_auto_artifact_config_returns_empty_without_dataset(tmp_path: Path) -> None:
    context = {
        "config": {"simulator": {"dataset_path": str(tmp_path / "missing_schema.json")}},
        "environment": {"building_names": ["Building_A"]},
    }

    assert build_auto_artifact_config(context=context, agent_index=0) == {}
