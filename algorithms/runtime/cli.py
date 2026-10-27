"""Lightweight CLI: help, catalogue and configuration validation need no simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or validate an Algorithms job")
    parser.add_argument("--config", help="Path to YAML job configuration")
    parser.add_argument("--job_id", help="Unique output job identifier")
    parser.add_argument("--base-dir", help="Output root (OPEVA_BASE_DIR or ./runs)")
    parser.add_argument(
        "--list-agents", action="store_true", help="Print the capability catalogue as JSON"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration without loading a simulator or model",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset/environment compatibility without training",
    )
    mode.add_argument(
        "--migrate-config",
        metavar="OUTPUT",
        help="Write an explicit v1 migration; do not run a job",
    )
    parser.add_argument("--profile", choices=["train", "evaluate", "smoke"])
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE"
    )
    return parser


def cli_main(default_base: Optional[str] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.list_agents:
        from algorithms.catalogue import AGENT_SPECS

        print(json.dumps([asdict(spec) for spec in AGENT_SPECS.values()], indent=2))
        return
    if not args.config:
        parser.error("--config is required unless --list-agents is used")
    import yaml

    from algorithms.configuration.migration import apply_overrides, migrate_config
    from algorithms.configuration.project import validate_config

    try:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        raw = apply_overrides(raw, args.overrides)
        if args.profile:
            if not isinstance(raw.setdefault("job", {}), dict):
                raise ValueError("job must be a mapping")
            raw["job"]["profile"] = args.profile
        if args.migrate_config:
            migrated, notes = migrate_config(raw, legacy_algorithm=True)
            resolved = validate_config(migrated).to_dict()
            output = Path(args.migrate_config)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            print(
                json.dumps({"status": "migrated", "output": str(output), "notes": notes}, indent=2)
            )
            return
        resolved = validate_config(raw).to_dict()
        if args.validate:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "schema_version": resolved["schema_version"],
                        "profile": resolved["job"]["profile"],
                        "agents": [s["algorithm"] for s in resolved["pipeline"]],
                    },
                    indent=2,
                )
            )
            return
        if args.dry_run:
            with redirect_stdout(sys.stderr):
                from algorithms.runtime.preflight import preflight_job

                report = preflight_job(resolved)
            print(json.dumps(report, indent=2))
            return
        from algorithms.runtime.job import run_experiment

        base = Path(
            args.base_dir or default_base or os.environ.get("OPEVA_BASE_DIR", "./runs")
        ).resolve()
        run_experiment(resolved, args.job_id, base)
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(2, f"Job validation/execution failed: {error}\n")
