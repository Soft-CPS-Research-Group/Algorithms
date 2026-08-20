#!/usr/bin/env python3
"""Create local TI-MARL evaluation evidence and select a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_protocol import (
    build_confirmation_report,
    build_evaluation_record,
    load_json_records,
    select_checkpoint,
    verify_selected_checkpoint,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="Create one deterministic replay record")
    record.add_argument("--candidate-id", required=True)
    record.add_argument("--role", choices=("candidate", "reference"), required=True)
    record.add_argument("--config", required=True)
    record.add_argument("--exported-kpis", required=True)
    record.add_argument("--checkpoint")
    record.add_argument(
        "--result",
        help="Runner result.json containing the authoritative remote pairing fingerprint",
    )
    record.add_argument("--simulator-version")
    record.add_argument("--output", required=True)

    select = commands.add_parser("select", help="Promote from development records only")
    select.add_argument("--reference", action="append", required=True)
    select.add_argument("--candidate", action="append", required=True)
    select.add_argument("--rules", required=True)
    select.add_argument("--output", required=True)

    confirm = commands.add_parser(
        "confirm", help="Aggregate a frozen candidate on confirmation records"
    )
    confirm.add_argument("--reference", action="append", required=True)
    confirm.add_argument("--candidate", action="append", required=True)
    confirm.add_argument("--rules", required=True)
    confirm.add_argument("--selection", required=True)
    confirm.add_argument("--output", required=True)

    verify = commands.add_parser("verify", help="Verify a promoted checkpoint hash")
    verify.add_argument("--selection", required=True)
    verify.add_argument("--checkpoint", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "record":
        pairing = None
        if args.result:
            result_payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
            pairing = result_payload.get("pairing_fingerprint")
            if not isinstance(pairing, dict):
                raise ValueError("--result does not contain pairing_fingerprint")
        payload = build_evaluation_record(
            candidate_id=args.candidate_id,
            role=args.role,
            config=_load_yaml(args.config),
            exported_kpis_path=args.exported_kpis,
            checkpoint_path=args.checkpoint,
            simulator_version=args.simulator_version,
            pairing_fingerprint=pairing,
        )
        _write_json(args.output, payload)
        return 0
    if args.command == "select":
        payload = select_checkpoint(
            references=load_json_records(args.reference),
            candidates=load_json_records(args.candidate),
            rules=_load_yaml(args.rules),
        )
        _write_json(args.output, payload)
        return 0 if payload["status"] == "selected" else 2
    if args.command == "confirm":
        payload = build_confirmation_report(
            references=load_json_records(args.reference),
            candidates=load_json_records(args.candidate),
            rules=_load_yaml(args.rules),
            selection=json.loads(Path(args.selection).read_text(encoding="utf-8")),
        )
        _write_json(args.output, payload)
        return 0 if payload["status"] == "confirmed" else 2

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    return 0 if verify_selected_checkpoint(selection, args.checkpoint) else 1


if __name__ == "__main__":
    raise SystemExit(main())
