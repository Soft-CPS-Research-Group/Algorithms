"""Collect contemporary RBC transitions and relabel a fixed data set.

The reward tap records physical quantities *inside* reward.calculate, after the
action has been executed and before the simulator advances. Reading next-state
observations for this purpose would risk a one-interval shift. No policy,
simulator or base reward implementation is changed by this collector.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any


POLICIES = {
    "basic": "RBCBasicPolicy",
    "smart": "RBCSmartLocalPolicy",
    "community": "RBCCommunityPolicy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RampState:
    """Executed community exchange; import is positive and exports stay signed."""

    previous_power_kw: float | None = None

    def observe(self, energy_kwh: float, duration_seconds: float) -> dict[str, float]:
        if not math.isfinite(energy_kwh) or not math.isfinite(duration_seconds):
            raise ValueError("Non-finite community energy or interval duration")
        if duration_seconds <= 0:
            raise ValueError("Interval duration must be positive")
        power = energy_kwh * 3600.0 / duration_seconds
        previous = self.previous_power_kw
        delta = 0.0 if previous is None else power - previous
        self.previous_power_kw = power
        return {
            "community_net_energy_kwh": energy_kwh,
            "community_net_power_kw": power,
            "previous_community_net_power_kw": power if previous is None else previous,
            "has_previous_executed_interval": int(previous is not None),
            "ramp_abs_kw": abs(delta),
            "ramp_up_kw": max(delta, 0.0),
            "duration_seconds": duration_seconds,
        }


class RewardCapture:
    """Transparent delegate that captures reward-time physical evidence."""

    def __init__(self, reward: Any, duration_seconds: float):
        self.reward = reward
        self.duration_seconds = float(duration_seconds)
        self.ramp = RampState()
        self.calls = 0
        self.last = None

    def __getattr__(self, name):
        return getattr(self.reward, name)

    def calculate(self, observations):
        energy = sum(float(obs["net_electricity_consumption"]) for obs in observations)
        result = self.reward.calculate(observations=observations)
        components = self.reward.get_last_components()
        per_agent = deepcopy(components["per_agent"])
        if len(per_agent) != len(observations):
            raise ValueError("Reward component count differs from building count")
        self.last = {
            **self.ramp.observe(energy, self.duration_seconds),
            "components": per_agent,
            "community_components": deepcopy(components["community"]),
            "building_net_energy_kwh": [
                float(obs["net_electricity_consumption"]) for obs in observations
            ],
        }
        self.calls += 1
        return result


def validate_windows(windows: list[dict]) -> None:
    """Training/validation are explicit time partitions of 2023, never fake seeds."""
    if not windows:
        raise ValueError("At least one explicit collection window is required")
    seen = set()
    intervals = []
    validation = [(31 * 96, 45 * 96), (120 * 96, 134 * 96),
                  (212 * 96, 226 * 96), (304 * 96, 318 * 96)]
    for window in windows:
        name = str(window["id"])
        if name in seen or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Window IDs must be unique simple path-safe names")
        seen.add(name)
        start, count = int(window["start_step"]), int(window["transitions"])
        end = start + count
        split = window["split"]
        if start < 0 or count <= 0 or end > 365 * 96:
            raise ValueError("Collection must use positive windows wholly within 2023")
        if split not in {"train", "validation", "smoke"}:
            raise ValueError("Unsupported collection split")
        if split == "train" and any(start < b and end > a for a, b in validation):
            raise ValueError("Training window overlaps a reserved validation interval")
        if split == "validation" and not any(a <= start and end <= b for a, b in validation):
            raise ValueError("Validation window is outside the reserved intervals")
        if any(start < b and end > a for a, b in intervals):
            raise ValueError("Collection windows overlap")
        intervals.append((start, end))


def relabel_rewards(base_reward, ramp_abs_kw, *, weight: float, ramp_scale_kw: float):
    """Apply the same normalised community penalty to every participant.

    Weight is in base-reward units per normalised community ramp. It is not
    divided by cohort size or building count, so grouping cannot change it.
    """
    import numpy as np

    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Ramping weight must be finite and non-negative")
    if not math.isfinite(ramp_scale_kw) or ramp_scale_kw <= 0:
        raise ValueError("Training-derived ramp scale must be finite and positive")
    reward = np.asarray(base_reward, dtype=np.float64)
    ramp = np.asarray(ramp_abs_kw, dtype=np.float64)
    if not np.isfinite(reward).all() or not np.isfinite(ramp).all() or (ramp < 0).any():
        raise ValueError("Invalid reward or ramp values")
    return reward - weight * ramp / ramp_scale_kw


def submitted_actions(proposed, spaces, *, policy="clip"):
    """Mirror the shared runtime's strict shape/finite/bounds checks."""
    import numpy as np

    if len(proposed) != len(spaces):
        raise ValueError("Action agent count mismatch")
    result = []
    for action, space in zip(proposed, spaces):
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1 or action.shape != space.low.shape:
            raise ValueError("Action dimension mismatch")
        if not np.isfinite(action).all():
            raise ValueError("Non-finite proposed action")
        clipped = np.clip(action, space.low, space.high)
        if policy == "reject" and not np.array_equal(clipped, action):
            raise ValueError("Action outside bounds")
        result.append(clipped)
    return result


def verify_environment(config: dict, *, expected_simulator: str, expected_schema_sha256: str):
    installed = importlib.metadata.version("softcpsrecsimulator")
    if installed != expected_simulator:
        raise ValueError(f"Simulator mismatch: expected {expected_simulator}, installed {installed}")
    schema = Path(config["simulator"]["dataset_path"])
    if sha256(schema) != expected_schema_sha256:
        raise ValueError("Dataset schema SHA256 does not match the frozen protocol")
    simulator = config["simulator"]
    if simulator.get("interface") != "entity" or simulator.get("central_agent"):
        raise ValueError("Collection requires decentralised entity observations")
    if simulator.get("topology_mode", "static") != "static":
        raise ValueError("This campaign requires a static topology")
    if simulator["reward_function"] != "CostServiceCommunityFeasiblePrecisionRewardV46":
        raise ValueError("The fixed base reward must be V46; record any protocol change explicitly")
    if simulator.get("reward_function_kwargs", {}).get("community_ramping_penalty", 0) != 0:
        raise ValueError("Base reward must have zero additional community ramping penalty")
    if config.get("tracking", {}).get("mlflow_enabled") is not False:
        raise ValueError("Collection requires tracking.mlflow_enabled=false explicitly")


def collection_entity_adapter(env, simulator: dict):
    """Use the public runtime's configured observation encoding for saved vectors."""
    from utils.entity_adapter import EntityContractAdapter

    encoding = simulator.get("entity_encoding", {}) or {}
    return EntityContractAdapter(
        env,
        normalization_enabled=bool(encoding.get("enabled", True)),
        clip=bool(encoding.get("clip", True)),
        encoding_profile=encoding.get("normalization", "minmax_space"),
    )


def collect_window(config: dict, window: dict, *, policy_name: str, output: Path) -> dict:
    """Stream one explicit episode using the normal simulator construction path."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from algorithms.agents import baseline_policies
    from algorithms.catalogue import AGENT_SPECS
    from algorithms.offline_rl.entity_layout import canonicalize_entity_names, ENTITY_NAME_CONVENTION
    from algorithms.runtime.simulator import create_environment
    local = deepcopy(config)
    simulator = local["simulator"]
    start, count = int(window["start_step"]), int(window["transitions"])
    simulator.update(simulation_start_time_step=start, simulation_end_time_step=start + count,
                     episode_time_steps=count + 1)
    if "simulator_seed" in window:
        simulator["random_seed"] = int(window["simulator_seed"])
    simulator["export"] = {**simulator.get("export", {}), "mode": "none"}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".partial.parquet").exists():
        raise FileExistsError(f"Refusing to overwrite collection output: {output}")
    env = create_environment(local, output.parent / f"{output.stem}_simulator")
    writer = None
    rows = []
    collected = 0
    began = time.monotonic()
    partial = output.with_suffix(".partial.parquet")
    try:
        if float(env.seconds_per_time_step) != 900:
            raise ValueError("Core30 campaign requires 900-second intervals")
        payload, _ = env.reset()
        capture = RewardCapture(env.reward_function, env.seconds_per_time_step)
        env.reward_function = capture
        adapter = collection_entity_adapter(env, simulator)
        raw, raw_names, raw_spaces = adapter.to_agent_observations(payload)
        encoded, _, _ = adapter.to_agent_encoded_observations(payload)
        names = adapter.encoded_observation_names(raw_names)
        action_names = [list(group) for group in env.action_names]
        original_names = names
        canonical = [canonicalize_entity_names(obs, act) for obs, act in zip(names, action_names)]
        names = [item[0] for item in canonical]
        dataset_action_names = [item[1] for item in canonical]
        spaces = list(env.flat_action_space)
        layouts = {}
        members = {}
        for i, (obs, acts) in enumerate(zip(names, dataset_action_names)):
            key = f"obs{len(obs)}_act{len(acts)}"
            layout = (sorted(obs), sorted(acts))
            if key in layouts and layouts[key] != layout:
                raise ValueError(f"Same dimensions have incompatible semantic layouts: {key}")
            layouts[key] = layout
            members.setdefault(key, []).append(str(env.buildings[i].name))
        supplied = local.get("pipeline", [])
        selected_spec = AGENT_SPECS.get(supplied[0]["algorithm"]) if len(supplied) == 1 else None
        if selected_spec is None or selected_spec.implementation != (
            "algorithms.agents.baseline_policies:" + POLICIES[policy_name]
        ):
            raise ValueError("Explicit config policy differs from selected collection policy")
        policy = getattr(baseline_policies, POLICIES[policy_name])(
            config={**local, "algorithm": {"name": POLICIES[policy_name],
                    "hyperparameters": supplied[0].get("hyperparameters", {})}})
        policy.attach_environment(observation_names=raw_names, action_names=action_names,
                                  action_space=spaces, observation_space=raw_spaces,
                                  metadata={
            "building_names": [b.name for b in env.buildings],
            "seconds_per_time_step": env.seconds_per_time_step,
            "community_market": {
                "enabled": bool(env.community_market_enabled),
                "local_price_ratio_to_grid_import": env.community_market_sell_ratio,
                "grid_export_price": env.community_market_grid_export_price,
            },
        })
        terminated = truncated = False
        schema = None
        while not (terminated or truncated):
            if collected >= count:
                raise RuntimeError("Simulator did not terminate at the declared window boundary")
            proposed = policy.predict(raw, deterministic=True)
            submitted = submitted_actions(proposed, spaces,
                                          policy=simulator.get("action_policy", "clip"))
            next_payload, reward, terminated, truncated, _ = env.step(
                adapter.to_entity_actions(submitted, action_names))
            if capture.calls != collected + 1:
                raise ValueError("Expected exactly one physical reward capture per transition")
            physical = capture.last
            next_raw, _, _ = adapter.to_agent_observations(next_payload)
            next_encoded, _, _ = adapter.to_agent_encoded_observations(next_payload)
            for i, (obs, nxt, action) in enumerate(zip(encoded, next_encoded, submitted)):
                if len(obs) != len(names[i]) or len(nxt) != len(names[i]):
                    raise ValueError("Encoded observation layout changed inside a static episode")
                if not np.isfinite(obs).all() or not np.isfinite(nxt).all():
                    raise ValueError("Non-finite encoded observation in transition")
                row = {
                    "seed": int(simulator.get("random_seed", local["training"]["seed"])),
                    "episode": str(window["id"]), "split": window["split"],
                    "behaviour_policy": POLICIES[policy_name], "timestep": collected,
                    "source_step": start + collected, "agent_idx": i,
                    "building_id": str(env.buildings[i].name),
                    "obs_dim": len(obs), "action_dim": len(action),
                    "reward": float(reward[i]), "base_reward": float(reward[i]),
                    "terminated": int(bool(terminated)), "truncated": int(bool(truncated)),
                    **{key: value for key, value in physical.items()
                       if key not in {"components", "community_components", "building_net_energy_kwh"}},
                    "building_net_energy_kwh": physical["building_net_energy_kwh"][i],
                    "reward_components_json": json.dumps(physical["components"][i], sort_keys=True),
                    "community_components_json": json.dumps(physical["community_components"], sort_keys=True),
                }
                row.update({f"obs__{name}": float(value) for name, value in zip(names[i], obs)})
                row.update({f"next_obs__{name}": float(value) for name, value in zip(names[i], nxt)})
                row.update({f"action__{name}": float(value) for name, value in zip(dataset_action_names[i], action)})
                rows.append(row)
            raw, encoded = next_raw, next_encoded
            collected += 1
            if collected % 64 == 0 or terminated or truncated:
                table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(partial, schema, compression="zstd")
                if table.schema != schema:
                    table = table.cast(schema)
                writer.write_table(table)
                rows.clear()
        if collected != count:
            raise ValueError(f"Expected {count} transitions, got {collected}")
        writer.close()
        writer = None
        partial.rename(output)
        return {"window": window, "policy": POLICIES[policy_name],
                "transitions": collected, "rows": collected * len(names),
                "groups": members, "layouts": layouts, "seconds": time.monotonic() - began,
                "name_convention": ENTITY_NAME_CONVENTION,
                "member_name_bindings": [
                    {"building": str(building.name),
                     "observations": dict(zip(original, canonical_obs)),
                     "actions": dict(zip(original_act, canonical_act))}
                    for building, original, canonical_obs, original_act, canonical_act
                    in zip(env.buildings, original_names, names, action_names, dataset_action_names)
                ],
                "file": output.name, "sha256": sha256(output)}
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-simulator", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--source-commit", help="Explicit source commit for images without .git")
    args = parser.parse_args(argv)
    import yaml
    from algorithms.configuration.project import validate_config

    raw = yaml.safe_load(args.config.read_text())
    config = validate_config(raw).model_dump(exclude_none=True)
    verify_environment(config, expected_simulator=args.expected_simulator,
                       expected_schema_sha256=args.expected_schema_sha256)
    windows = json.loads(args.windows.read_text())
    validate_windows(windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"{args.policy}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    source_commit = args.source_commit
    if source_commit is None:
        try:
            source_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = "not_exposed"
    from algorithms.offline_rl import entity_layout
    manifest = {
        "schema_version": "indin.core30.transitions.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulator_version": args.expected_simulator,
        "schema_sha256": args.expected_schema_sha256,
        "config": config, "config_sha256": sha256(args.config),
        "observation_preprocessing": {
            "entity_encoding": deepcopy(config["simulator"].get("entity_encoding", {})),
            "standardiser": "not applied during collection; IQL fits train rows only",
        },
        "windows": windows, "windows_sha256": sha256(args.windows),
        "collector_sha256": sha256(Path(__file__)),
        "entity_layout_sha256": sha256(Path(entity_layout.__file__)),
        "source_commit": source_commit,
        "physical_alignment": "reward.calculate after action, before next_time_step",
        "ramp_convention": "signed net kW; first executed interval excluded; reset per window",
        "split_convention": "explicit chronological train/validation directories; seed is simulator RNG seed",
        "episodes": [], "complete": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for window in windows:
        seed = int(window.get("simulator_seed", config["simulator"].get("random_seed", config["training"]["seed"])))
        path = args.output_dir / window["split"] / f"seed_{seed}_{args.policy}_{window['id']}.parquet"
        manifest["episodes"].append(collect_window(config, window, policy_name=args.policy, output=path))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "episodes": len(windows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
