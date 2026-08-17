"""High-frequency TI-MARL topology/latency smoke on the 15-second dataset."""

from __future__ import annotations

import json
from pathlib import Path
import time

from citylearn.citylearn import CityLearnEnv
import numpy as np

from algorithms.ti_marl.compiler import TypedInterfaceCompiler
from scripts.generate_typed_interfaces import (
    augment_dynamic_assets,
    generate,
    generate_simulator_bindings,
    write_generated,
)


def _zero_actions(env):
    return {
        "tables": {
            name: np.zeros(space.shape, dtype=space.dtype)
            for name, space in env.action_space.spaces["tables"].spaces.items()
        }
    }


def test_ti_marl_15s_handles_all_shifted_asset_events_with_bounded_overhead(tmp_path):
    dataset = Path(
        "datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet"
    ).resolve()
    schema = json.loads((dataset / "schema.json").read_text(encoding="utf-8"))
    for index, event in enumerate(schema["topology_events"], start=2):
        event["time_step"] = index
    env = CityLearnEnv(
        schema,
        root_directory=dataset,
        interface="entity",
        topology_mode="dynamic",
        central_agent=False,
        episode_time_steps=12,
        random_seed=19,
    )
    try:
        payload, _ = env.reset()
        interfaces, coverage = generate(env.entity_specs, payload)
        future_bindings = augment_dynamic_assets(interfaces, schema, env.entity_specs)
        simulator_bindings = generate_simulator_bindings(
            env.entity_specs,
            payload,
            interfaces,
            future_bindings,
        )
        interfaces_dir = tmp_path / "interfaces"
        write_generated(
            interfaces_dir,
            interfaces,
            coverage,
            source="15-second stress fixture",
            simulator_bindings=simulator_bindings,
        )
        compiler = TypedInterfaceCompiler(
            contract_version="ti_marl_v1",
            typed_interfaces_dir=interfaces_dir,
            simulator_bindings_path=(
                interfaces_dir / "technology_bindings" / "simulator.yaml"
            ),
        )
        assert compiler.interface_registry.for_agent("Building_6").role == "prosumer"
        compiler.attach_entity_specs(env.entity_specs, seconds_per_time_step=15)
        versions = set()
        latencies = []
        event_ids = set()
        seen_groups = set()
        for _ in range(11):
            started = time.perf_counter()
            snapshot = compiler.compile(payload)
            latencies.append((time.perf_counter() - started) * 1000.0)
            versions.add(snapshot.topology_version)
            seen_groups.update(
                (group.owner_agent_id, group.module_id)
                for group in snapshot.action_groups
            )
            payload, _reward, terminated, truncated, info = env.step(_zero_actions(env))
            event_ids.update(
                str(item.get("event_id", item.get("id", "")))
                for item in info.get("topology_events_applied", []) or []
            )
            compiler.attach_entity_specs(
                env.entity_specs,
                seconds_per_time_step=15,
            )
            if terminated or truncated:
                break
        assert versions == set(range(7))
        assert event_ids == {event["id"] for event in schema["topology_events"]}
        assert compiler.structure_recompilations >= 7
        assert ("Building_2", "charger_1") in seen_groups
        assert ("Building_3", "battery_1") in seen_groups
        assert np.percentile(latencies, 95) < 500.0
    finally:
        env.close()
