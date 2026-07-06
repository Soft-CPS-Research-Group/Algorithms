"""Regenerate the tokenizer validation fixture from a live simulator.

The fixture ``configs/tokenizers/fixtures/entity_obs_sample.json`` is a
JSON snapshot of the simulator's entity schema (per-table feature names
and row ids, plus edge index pairs). It is loaded by
``utils.entity_tokenizer_schema._load_default_sample()`` and used by the
5 hard-fail tokenizer validation rules (see docs/transformer_ppo_spec.md §13.4).

Regenerate whenever the simulator schema changes (feature added/removed,
new asset type, adapter emission order changed).

Usage:

    python scripts/dump_entity_obs_sample.py \\
        --config configs/templates/dynamic/rule_based_entity_dynamic_assets_only_local.yaml \\
        --output configs/tokenizers/fixtures/entity_obs_sample.json

Any entity-mode config works. Row values are dropped — only ``id`` is
kept — because validation reads ``tables.<X>.features`` and ``edges``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from citylearn.citylearn import CityLearnEnv
from reward_function.registry import REWARD_FUNCTION_MAP
from run_experiment import (
    _resolve_citylearn_schema_input,
    _validate_dynamic_entity_schema_input,
)
from utils.config_schema import validate_config


def _build_env(config: Mapping[str, Any]) -> CityLearnEnv:
    sim = dict(config["simulator"])
    interface = str(sim.get("interface", "flat")).strip().lower()
    topology = str(sim.get("topology_mode", "static")).strip().lower()
    if interface != "entity":
        raise ValueError(
            f"Fixture dump requires simulator.interface='entity'; got {interface!r}"
        )
    schema_input = _resolve_citylearn_schema_input(sim["dataset_path"])
    _validate_dynamic_entity_schema_input(
        schema_input, interface=interface, topology_mode=topology
    )
    reward_cls = REWARD_FUNCTION_MAP.get(sim["reward_function"])
    if reward_cls is None:
        raise ValueError(f"Unknown reward function {sim['reward_function']!r}")
    return CityLearnEnv(
        schema=schema_input,
        central_agent=sim["central_agent"],
        interface=interface,
        topology_mode=topology,
        reward_function=reward_cls,
        offline=True,
        render_mode="none",
        export_kpis_on_episode_end=False,
    )


def _edge_pairs(raw_edges: Any) -> list[dict[str, Any]]:
    """Convert simulator edge output (numpy Nx2 array or list of pairs) to
    the fixture format: ``[{source_index, target_index, source_id, target_id}, ...]``.
    """
    if hasattr(raw_edges, "tolist"):
        raw_edges = raw_edges.tolist()
    out: list[dict[str, Any]] = []
    for pair in raw_edges or []:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            src, tgt = int(pair[0]), int(pair[1])
        elif isinstance(pair, Mapping):
            src = int(pair.get("source_index", -1))
            tgt = int(pair.get("target_index", -1))
        else:
            continue
        out.append(
            {
                "source_index": src,
                "target_index": tgt,
                "source_id": None,
                "target_id": None,
            }
        )
    return out


def _build_fixture(env: CityLearnEnv, payload: Mapping[str, Any]) -> dict[str, Any]:
    specs = env.entity_specs
    tables: dict[str, dict[str, Any]] = {}
    for name, spec in specs["tables"].items():
        tables[name] = {
            "features": list(spec.get("features", [])),
            "rows": [{"id": rid} for rid in spec.get("ids", [])],
        }

    edges: dict[str, dict[str, Any]] = {}
    for name, spec in specs.get("edges", {}).items():
        raw = payload.get("edges", {}).get(name)
        edges[name] = {
            "source_table": spec.get("source"),
            "target_table": spec.get("target"),
            "edges": _edge_pairs(raw),
        }
    # Include mask edges from the payload (present even if not in specs.edges).
    for name, raw in payload.get("edges", {}).items():
        if name in edges:
            continue
        edges[name] = {
            "source_table": None,
            "target_table": None,
            "edges": _edge_pairs(raw),
        }

    meta_in = payload.get("meta") or {}
    return {
        "tables": tables,
        "edges": edges,
        "meta": {
            "spec_version": meta_in.get("spec_version", "entity_v1"),
            "topology_version": int(meta_in.get("topology_version", 0)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an entity-mode YAML config")
    parser.add_argument(
        "--output",
        default="configs/tokenizers/fixtures/entity_obs_sample.json",
        help="Where to write the fixture",
    )
    args = parser.parse_args()

    raw_cfg = yaml.safe_load(Path(args.config).read_text())
    config = validate_config(raw_cfg).to_dict()

    env = _build_env(config)
    payload, _ = env.reset()

    fixture = _build_fixture(env, payload)
    Path(args.output).write_text(json.dumps(fixture, indent=2) + "\n")

    summary = ", ".join(
        f"{name}={len(t['features'])}f/{len(t['rows'])}r"
        for name, t in fixture["tables"].items()
    )
    print(f"wrote {args.output}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
