"""Replay a dataset through SimulatorAdapter and TIC without training.

This is a binding/runtime acceptance tool. It deliberately emits no campaign
configuration or result into the repository; callers choose a local output.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from scripts.dump_entity_obs_sample import _build_env
from utils.config_schema import validate_config


def _zero_actions(env):
    tables = {}
    table_spaces = env.action_space.spaces.get("tables")
    for name, space in table_spaces.spaces.items():
        tables[name] = np.zeros(space.shape, dtype=space.dtype)
    return {"tables": tables}


def replay(
    config_path: Path,
    interfaces_dir: Path,
    *,
    simulator_bindings_path: Path | None = None,
    max_steps: int | None = None,
    measure_memory: bool = False,
    progress_every: int = 0,
):
    config = validate_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ).to_dict()
    env = _build_env(config)
    compiler = TypedInterfaceCompiler(
        contract_version="ti_marl_v1",
        typed_interfaces_dir=interfaces_dir,
        simulator_bindings_path=simulator_bindings_path,
    )
    compiler.attach_entity_specs(
        env.entity_specs,
        seconds_per_time_step=float(env.seconds_per_time_step),
    )
    payload, _ = env.reset()
    latencies = []
    topology_versions = set()
    topology_events = []
    frame_count = 0
    if measure_memory:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        while True:
            before = time.perf_counter()
            snapshot = compiler.compile(payload)
            latencies.append((time.perf_counter() - before) * 1000.0)
            frame_count += 1
            topology_versions.add(snapshot.topology_version)
            topology_events.extend(snapshot.topology_events)
            if progress_every > 0 and frame_count % progress_every == 0:
                print(
                    f"TI-MARL replay progress: {frame_count} frames",
                    file=sys.stderr,
                    flush=True,
                )
            if max_steps is not None and frame_count >= max_steps:
                break
            outcome = env.step(_zero_actions(env))
            payload, _reward, terminated, truncated, info = outcome
            topology_events.extend(
                item
                for item in info.get("topology_events_applied", []) or []
                if isinstance(item, dict)
            )
            # Dynamic topology changes rebuild the public catalog.
            if env.entity_specs != compiler.entity_specs:
                compiler.attach_entity_specs(
                    env.entity_specs,
                    seconds_per_time_step=float(env.seconds_per_time_step),
                )
            if terminated or truncated:
                # The terminal observation is still part of the dataset and
                # must pass adapter/TIC validation even though no action is
                # selected from it.  This makes a 35,040-row annual dataset
                # report exactly 35,040 validated frames.
                before = time.perf_counter()
                snapshot = compiler.compile(payload)
                latencies.append((time.perf_counter() - before) * 1000.0)
                frame_count += 1
                topology_versions.add(snapshot.topology_version)
                topology_events.extend(snapshot.topology_events)
                if progress_every > 0:
                    print(
                        f"TI-MARL replay progress: {frame_count} frames (terminal)",
                        file=sys.stderr,
                        flush=True,
                    )
                break
        peak_memory = (
            tracemalloc.get_traced_memory()[1]
            if measure_memory
            else None
        )
    finally:
        elapsed = time.perf_counter() - started
        if measure_memory:
            tracemalloc.stop()
        env.close()
    return {
        "version": "ti_marl_interface_replay_v1",
        "compiler_version": snapshot.compiler_version,
        "simulator_version": _simulator_version(),
        "registry_hash": compiler.interface_registry.registry_hash,
        "dataset": str(config["simulator"]["dataset_path"]),
        "seconds_per_time_step": float(env.seconds_per_time_step),
        "frames": frame_count,
        "elapsed_seconds": elapsed,
        "mean_compile_ms": statistics.fmean(latencies),
        "p95_compile_ms": float(np.percentile(latencies, 95)),
        "max_compile_ms": max(latencies),
        "peak_traced_memory_mb": (
            None if peak_memory is None else peak_memory / (1024.0 * 1024.0)
        ),
        "structure_recompilations": compiler.structure_recompilations,
        "topology_versions": sorted(topology_versions),
        "topology_event_ids": sorted(
            {
                str(item.get("event_id", item.get("id", "")))
                for item in topology_events
                if isinstance(item, dict)
            }
        ),
        "registered_agents": len(compiler.interface_registry.agent_ids),
        "final_active_agents": len(snapshot.agent_ids),
        "binding_errors": 0,
    }


def _simulator_version() -> str:
    try:
        return package_version("softcpsrecsimulator")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--interfaces-dir", required=True, type=Path)
    parser.add_argument("--simulator-bindings", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Write a progress line to stderr every N validated frames.",
    )
    args = parser.parse_args()
    result = replay(
        args.config.expanduser().resolve(),
        args.interfaces_dir.expanduser().resolve(),
        simulator_bindings_path=(
            None
            if args.simulator_bindings is None
            else args.simulator_bindings.expanduser().resolve()
        ),
        max_steps=args.max_steps,
        measure_memory=args.measure_memory,
        progress_every=max(int(args.progress_every), 0),
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
