"""Run a small controlled MADDPG learning diagnostic.

This script intentionally does not change the wrapper or MADDPG implementation.
It builds a temporary in-memory subset of an existing dataset schema, then runs
the current MADDPG stack against that smaller environment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from citylearn.citylearn import CityLearnEnv
from loguru import logger
from torch.nn.utils import parameters_to_vector

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import build_execution_unit
from reward_function.registry import REWARD_FUNCTION_MAP
from run_experiment import _resolve_agent_observation_dimensions
from utils.config_schema import validate_config
from utils.wrapper_citylearn import Wrapper_CityLearn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-2 controlled MADDPG diagnostic.")
    parser.add_argument(
        "--config",
        default="configs/templates/maddpg/maddpg_local.yaml",
        help="Base MADDPG config to mutate for the diagnostic.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to runs/maddpg_diagnostics/<timestamp>.",
    )
    parser.add_argument(
        "--building",
        action="append",
        default=[],
        help="Building to keep in the in-memory schema. Can be repeated. Defaults to Building_1.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Training episodes.")
    parser.add_argument("--steps", type=int, default=256, help="Steps per episode.")
    parser.add_argument("--start", type=int, default=0, help="Simulation start time step.")
    parser.add_argument("--profile", default="maddpg_v3_operational", help="Entity encoding profile.")
    parser.add_argument("--batch-size", type=int, default=64, help="Replay batch size.")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity.")
    parser.add_argument("--actor-layers", default="128,64", help="Comma-separated actor hidden layers.")
    parser.add_argument("--critic-layers", default="256,128", help="Comma-separated critic hidden layers.")
    parser.add_argument("--actor-lr", type=float, default=5.0e-5, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=5.0e-4, help="Critic learning rate.")
    parser.add_argument("--random-exploration-steps", type=int, default=64, help="Initial random exploration steps.")
    parser.add_argument("--sigma", type=float, default=0.15, help="Gaussian exploration sigma after warm-up.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def _parse_layers(raw: str) -> list[int]:
    layers = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not layers:
        raise ValueError("Network layers cannot be empty.")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"Network layers must be positive integers: {layers}")
    return layers


def _load_subset_schema(base_schema_path: Path, selected_buildings: list[str]) -> dict[str, Any]:
    with base_schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    buildings = schema.get("buildings")
    if not isinstance(buildings, dict) or not buildings:
        raise ValueError(f"Schema does not contain buildings: {base_schema_path}")

    missing = [name for name in selected_buildings if name not in buildings]
    if missing:
        raise ValueError(f"Selected buildings not present in schema: {missing}")

    schema["root_directory"] = str(base_schema_path.resolve().parent)
    for building_name, building_cfg in buildings.items():
        if isinstance(building_cfg, dict):
            building_cfg["include"] = building_name in selected_buildings

    return schema


def _mutate_config(
    raw_config: dict[str, Any],
    *,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = validate_config(raw_config).to_dict()

    config["runtime"]["log_dir"] = str(output_dir / "logs")
    config["runtime"]["job_dir"] = str(output_dir)

    metadata = config.setdefault("metadata", {})
    metadata["experiment_name"] = "maddpg_phase2_controlled_diagnostic"
    metadata["run_name"] = "MADDPG Phase 2 Controlled Diagnostic"
    metadata["description"] = "Small controlled MADDPG run for implementation/reward/exploration diagnosis."

    tracking = config.setdefault("tracking", {})
    tracking["mlflow_enabled"] = False
    tracking["log_level"] = "INFO"
    tracking["log_frequency"] = 16
    tracking["mlflow_step_sample_interval"] = 16
    tracking["progress_updates_enabled"] = True
    tracking["progress_update_interval"] = 16
    tracking["system_metrics_enabled"] = False

    checkpointing = config.setdefault("checkpointing", {})
    checkpointing["checkpoint_interval"] = max(args.steps, 1)
    checkpointing["require_update_step"] = True
    checkpointing["require_initial_exploration_done"] = True

    simulator = config.setdefault("simulator", {})
    simulator["episodes"] = max(args.episodes, 1)
    simulator["simulation_start_time_step"] = max(args.start, 0)
    simulator["simulation_end_time_step"] = max(args.start, 0) + max(args.steps, 1) - 1
    simulator["episode_time_steps"] = max(args.steps, 1)
    simulator["interface"] = "entity"
    simulator["topology_mode"] = "static"
    simulator.setdefault("entity_encoding", {})
    simulator["entity_encoding"]["enabled"] = True
    simulator["entity_encoding"]["normalization"] = "minmax_space"
    simulator["entity_encoding"]["profile"] = args.profile
    simulator["entity_encoding"]["clip"] = True
    simulator.setdefault("export", {})
    simulator["export"]["mode"] = "none"
    simulator["export"]["export_kpis_on_episode_end"] = False
    simulator["export"]["session_name"] = None

    training = config.setdefault("training", {})
    training["seed"] = int(args.seed)
    training["steps_between_training_updates"] = 1
    training["target_update_interval"] = 1

    algorithm = config.setdefault("algorithm", {})
    algorithm["name"] = "MADDPG"
    algorithm.setdefault("hyperparameters", {})
    algorithm["hyperparameters"]["gamma"] = float(algorithm["hyperparameters"].get("gamma", 0.995))
    algorithm.setdefault("networks", {})
    algorithm["networks"]["actor"] = {
        "class": "Actor",
        "layers": _parse_layers(args.actor_layers),
        "lr": float(args.actor_lr),
    }
    algorithm["networks"]["critic"] = {
        "class": "Critic",
        "layers": _parse_layers(args.critic_layers),
        "lr": float(args.critic_lr),
    }
    algorithm["replay_buffer"] = {
        "class": "MultiAgentReplayBuffer",
        "capacity": int(args.buffer_capacity),
        "batch_size": int(args.batch_size),
    }
    exploration_params = dict((algorithm.get("exploration") or {}).get("params") or {})
    exploration_params.update(
        {
            "bias": 0.0,
            "sigma": float(args.sigma),
            "decay": 0.999,
            "min_sigma": 0.03,
            "noise_clip": 0.3,
            "tau": 0.001,
            "gamma": float(algorithm["hyperparameters"]["gamma"]),
            "end_initial_exploration_time_step": int(args.random_exploration_steps),
            "random_exploration_steps": int(args.random_exploration_steps),
        }
    )
    algorithm["exploration"] = {"strategy": "GaussianNoise", "params": exploration_params}

    config["topology"] = {
        "num_agents": None,
        "observation_dimensions": None,
        "action_dimensions": None,
        "action_space": None,
    }

    return validate_config(config).to_dict()


def _build_env(config: Mapping[str, Any], schema_payload: Mapping[str, Any]) -> CityLearnEnv:
    simulator = config["simulator"]
    reward_cls = REWARD_FUNCTION_MAP[simulator["reward_function"]]
    env_kwargs: dict[str, Any] = {
        "schema": deepcopy(schema_payload),
        "central_agent": simulator["central_agent"],
        "interface": simulator["interface"],
        "topology_mode": simulator["topology_mode"],
        "reward_function": reward_cls,
        "offline": True,
        "simulation_start_time_step": simulator["simulation_start_time_step"],
        "simulation_end_time_step": simulator["simulation_end_time_step"],
        "episode_time_steps": simulator["episode_time_steps"],
    }
    reward_kwargs = simulator.get("reward_function_kwargs")
    if isinstance(reward_kwargs, dict) and reward_kwargs:
        env_kwargs["reward_function_kwargs"] = reward_kwargs
    return CityLearnEnv(**env_kwargs)


def _actor_vectors(agent: Any) -> list[torch.Tensor]:
    vectors: list[torch.Tensor] = []
    for actor in getattr(agent, "actors", []):
        vectors.append(parameters_to_vector(actor.parameters()).detach().cpu().clone())
    return vectors


def _actor_delta_l2(before: list[torch.Tensor], after: list[torch.Tensor]) -> list[float]:
    values: list[float] = []
    for initial, final in zip(before, after):
        values.append(float(torch.linalg.vector_norm(final - initial).item()))
    return values


def _read_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"records": 0, "episode_reward_sums": []}

    episode_reward_sums: list[dict[str, Any]] = []
    step_reward_count = 0
    grouped_metrics: dict[str, dict[str, list[float]]] = {
        "reward_components": {},
        "action_diagnostics": {},
        "maddpg_diagnostics": {},
    }
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics") or {}
            if any(key.endswith("_Reward") for key in metrics):
                step_reward_count += 1
            episode_values = {
                key: value
                for key, value in metrics.items()
                if key.endswith("_Episode_Reward_Sum")
            }
            if episode_values:
                episode_reward_sums.append({"step": record.get("step"), "values": episode_values})

            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if key.startswith("RewardComponent/"):
                    bucket = grouped_metrics["reward_components"]
                elif key.startswith("Action/") or key.startswith("Deferrable/"):
                    bucket = grouped_metrics["action_diagnostics"]
                elif key.startswith("MADDPG/"):
                    bucket = grouped_metrics["maddpg_diagnostics"]
                else:
                    continue
                bucket.setdefault(key, []).append(float(value))

    return {
        "records": step_reward_count + len(episode_reward_sums),
        "step_reward_records": step_reward_count,
        "episode_reward_sums": episode_reward_sums,
        **{
            group_name: _summarize_metric_group(values)
            for group_name, values in grouped_metrics.items()
        },
    }


def _summarize_metric_group(group: Mapping[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, values in sorted(group.items()):
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "records": int(array.shape[0]),
            "mean": float(np.mean(array)),
            "abs_mean": float(np.mean(np.abs(array))),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "last": float(array[-1]),
        }

    top_abs = sorted(
        (
            {"metric": key, "abs_mean": stats["abs_mean"], "mean": stats["mean"], "last": stats["last"]}
            for key, stats in summary.items()
        ),
        key=lambda row: row["abs_mean"],
        reverse=True,
    )
    return {
        "metric_count": len(summary),
        "top_abs_mean": top_abs[:20],
        "metrics": summary,
    }


def _read_training_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"update_records": 0}

    critic_losses: list[float] = []
    actor_losses: list[float] = []
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        rf"Avg Critic Loss: (?P<critic>{number_pattern}), Avg Actor Loss: (?P<actor>{number_pattern})"
    )
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            critic_losses.append(float(match.group("critic")))
            actor_losses.append(float(match.group("actor")))

    if not critic_losses:
        return {"update_records": 0}

    return {
        "update_records": len(critic_losses),
        "critic_loss_first": critic_losses[0],
        "critic_loss_last": critic_losses[-1],
        "critic_loss_mean": float(np.mean(critic_losses)),
        "actor_loss_first": actor_losses[0],
        "actor_loss_last": actor_losses[-1],
        "actor_loss_mean": float(np.mean(actor_losses)),
    }


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    training_log = summary.get("training_log") or {}
    lines = [
        "# MADDPG Phase 2 Diagnostic",
        "",
        "Small controlled run using the current MADDPG implementation without algorithm changes.",
        "",
        "## Summary",
        "",
        f"- Buildings: `{summary['selected_buildings']}`",
        f"- Episodes: `{summary['episodes']}`",
        f"- Steps per episode: `{summary['steps_per_episode']}`",
        f"- Global steps: `{summary['global_step']}`",
        f"- Num agents: `{summary['num_agents']}`",
        f"- Action dimensions: `{summary['action_dimensions']}`",
        f"- Observation dimensions: `{summary['observation_dimensions']}`",
        f"- Replay buffer size: `{summary['replay_buffer_size']}`",
        f"- Actor delta L2: `{summary['actor_delta_l2']}`",
        f"- Updates likely happened: `{summary['updates_likely_happened']}`",
        f"- Update records in log: `{training_log.get('update_records', 0)}`",
    ]
    reward_components = ((summary.get("metrics") or {}).get("reward_components") or {}).get("top_abs_mean") or []
    if reward_components:
        lines.extend(["", "## Top Reward Components", ""])
        for row in reward_components[:10]:
            lines.append(
                f"- `{row['metric']}`: abs_mean={row['abs_mean']:.6g}, "
                f"mean={row['mean']:.6g}, last={row['last']:.6g}"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `diagnostic_config.resolved.yaml`: exact config used by the wrapper/agent.",
            "- `diagnostic_schema.json`: generated in-memory schema snapshot.",
            "- `logs/metrics.jsonl`: local reward/episode metrics.",
            "- `progress.json`: progress tracker output.",
            "- `summary.json`: machine-readable diagnostic summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / "maddpg_diagnostics" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    for path in (
        output_dir / "diagnostic.log",
        output_dir / "logs" / "metrics.jsonl",
        output_dir / "progress.json",
        output_dir / "summary.json",
        output_dir / "README.md",
    ):
        if path.exists():
            path.unlink()

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "diagnostic.log", level="INFO", mode="w")

    selected_buildings = args.building or ["Building_1"]
    base_config_path = Path(args.config)
    raw_config = _load_yaml(base_config_path)
    config = _mutate_config(raw_config, output_dir=output_dir, args=args)

    base_schema_path = Path(config["simulator"]["dataset_path"])
    schema_payload = _load_subset_schema(base_schema_path, selected_buildings)

    _write_yaml(output_dir / "diagnostic_config.resolved.yaml", config)
    (output_dir / "diagnostic_schema.json").write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    env = _build_env(config, schema_payload)
    wrapper = Wrapper_CityLearn(
        env=env,
        config=config,
        job_id=output_dir.name,
        progress_path=str(output_dir / "progress.json"),
    )
    config["topology"]["observation_dimensions"] = _resolve_agent_observation_dimensions(wrapper, "MADDPG")
    config["topology"]["action_dimensions"] = list(wrapper.action_dimension)
    config["topology"]["num_agents"] = len(wrapper.action_space)
    _write_yaml(output_dir / "diagnostic_config.resolved.yaml", config)

    agent = build_execution_unit(config)
    wrapper.set_model(agent)
    actor_before = _actor_vectors(agent)

    started_at = time.time()
    wrapper.learn(episodes=int(config["simulator"]["episodes"]))
    duration_seconds = time.time() - started_at

    actor_after = _actor_vectors(agent)
    actor_delta = _actor_delta_l2(actor_before, actor_after)
    replay_size = len(getattr(agent, "replay_buffer", []))
    metrics_summary = _read_metrics(output_dir / "logs" / "metrics.jsonl")
    training_log_summary = _read_training_log(output_dir / "diagnostic.log")
    env_info = wrapper.describe_environment()

    updates_likely_happened = bool(replay_size >= int(args.batch_size) and any(delta > 0.0 for delta in actor_delta))
    summary: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
        "base_config": str(base_config_path),
        "base_schema": str(base_schema_path),
        "selected_buildings": selected_buildings,
        "episodes": int(config["simulator"]["episodes"]),
        "steps_per_episode": int(config["simulator"]["episode_time_steps"]),
        "global_step": int(wrapper.global_step),
        "duration_seconds": duration_seconds,
        "num_agents": int(config["topology"]["num_agents"]),
        "observation_dimensions": list(config["topology"]["observation_dimensions"]),
        "action_dimensions": list(config["topology"]["action_dimensions"]),
        "action_names_by_agent": env_info.get("action_names_by_agent"),
        "building_names": env_info.get("building_names"),
        "replay_buffer_size": int(replay_size),
        "batch_size": int(args.batch_size),
        "random_exploration_steps": int(args.random_exploration_steps),
        "final_sigma": float(getattr(agent, "sigma", 0.0)),
        "actor_delta_l2": actor_delta,
        "updates_likely_happened": updates_likely_happened,
        "metrics": metrics_summary,
        "training_log": training_log_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_readme(output_dir / "README.md", summary)
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_diagnostic(args)
    print(json.dumps({
        "output_dir": summary["output_dir"],
        "num_agents": summary["num_agents"],
        "global_step": summary["global_step"],
        "replay_buffer_size": summary["replay_buffer_size"],
        "actor_delta_l2": summary["actor_delta_l2"],
        "updates_likely_happened": summary["updates_likely_happened"],
    }, indent=2))


if __name__ == "__main__":
    main()
