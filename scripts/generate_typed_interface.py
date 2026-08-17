"""Generate one editable TI-MARL interface with a Simulator catalog.

The control meaning comes from ``--base`` because the Simulator cannot infer
dependencies, health policy or action semantics. Observation/action names are
read from either a saved ``entity_specs`` document or a live entity-mode run.

Examples:

    python scripts/generate_typed_interface.py \
      --entity-specs /tmp/entity_specs.yaml \
      --output /tmp/my_typed_interface.yaml

    python scripts/generate_typed_interface.py \
      --config /path/to/entity_run.yaml \
      --output /tmp/my_typed_interface.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ti_marl.contracts.interface_definition import TypedInterfaceDefinition


def _load_document(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    payload = json.loads(text) if resolved.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {resolved}")
    return payload


def _specs_from_config(path: str | Path) -> Mapping[str, Any]:
    # Reuse the established entity fixture path so config resolution and
    # Simulator construction stay identical across developer tools.
    from scripts.dump_entity_obs_sample import _build_env
    from utils.config_schema import validate_config

    raw = _load_document(path)
    config = validate_config(raw).to_dict()
    env = _build_env(config)
    try:
        env.reset()
        return env.entity_specs
    finally:
        env.close()


def generate(
    *,
    base_path: str | Path,
    entity_specs: Mapping[str, Any],
    output_path: str | Path,
    source: str,
    policy: str = "compatible",
) -> Path:
    definition = TypedInterfaceDefinition.load(base_path)
    payload = definition.with_simulator_catalog(
        entity_specs,
        source=source,
        policy=policy,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    # Re-read what was actually written, then validate it against the same
    # Simulator contract. A generated file is immediately usable or generation
    # fails loudly.
    TypedInterfaceDefinition.load(output).validate_entity_specs(entity_specs)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--entity-specs", help="JSON/YAML env.entity_specs document")
    source.add_argument("--config", help="Entity-mode Algorithms run configuration")
    parser.add_argument(
        "--base",
        default="configs/ti_marl/typed_interface_v1.yaml",
        help="Editable semantic base interface",
    )
    parser.add_argument("--output", required=True, help="Generated YAML path")
    parser.add_argument(
        "--catalog-policy",
        choices=("compatible", "exact"),
        default="compatible",
    )
    args = parser.parse_args()

    if args.entity_specs:
        specs = _load_document(args.entity_specs)
        source_label = str(Path(args.entity_specs).expanduser().resolve())
    else:
        specs = _specs_from_config(args.config)
        source_label = str(Path(args.config).expanduser().resolve())
    output = generate(
        base_path=args.base,
        entity_specs=specs,
        output_path=args.output,
        source=source_label,
        policy=args.catalog_policy,
    )
    print(f"wrote validated typed interface: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
