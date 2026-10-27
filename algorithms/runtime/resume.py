"""Explicit episode-boundary restoration; legacy files remain weight initializers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import citylearn

from algorithms.runtime.randomness import capture_rng, restore_rng

# Exact continuation is tested only for these state-complete serializers.
# Other families remain loadable as initializers, never advertised as exact resume.
EPISODE_RESUME_AGENTS = {"MADDPG", "AgentTransformerPPO", "AgentTransformerMATD3", "TIMARL"}


def leaves(unit, path="root"):
    from algorithms.pipeline import Ensemble, Pipeline

    children = (
        unit.stages
        if isinstance(unit, Pipeline)
        else unit.agents
        if isinstance(unit, Ensemble)
        else None
    )
    if children is None:
        yield path, unit
    else:
        for index, child in enumerate(children):
            yield from leaves(child, f"{path}/{index}")


def resume_signature(config):
    simulator = deepcopy(config["simulator"])
    for key in ("episodes", "deterministic_finish", "export"):
        simulator.pop(key, None)
    training = deepcopy(config.get("training", {}))
    training.pop("episodes", None)
    payload = {
        "simulator": simulator,
        "training": training,
        "pipeline": config["pipeline"],
        "inputs": config.get("runtime", {}).get("input_identity"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def capture_runtime(wrapper):
    units = list(leaves(wrapper.model))
    boundary = bool(getattr(wrapper, "_at_episode_boundary", False))
    static = wrapper.config["simulator"].get("topology_mode", "static") == "static"
    checkpoint = wrapper.config.get("checkpointing", {})
    complete = checkpoint.get("checkpoint_mode", "full") == "full" and all(
        getattr(unit, "agent_name", type(unit).__name__) in EPISODE_RESUME_AGENTS
        for _, unit in units
    )
    tracker = getattr(wrapper.env.unwrapped, "episode_tracker", None)
    return {
        "version": "algorithms.resume.v1",
        "runtime_versions": {
            key: wrapper.config.get("runtime", {}).get("randomness", {}).get(key)
            for key in ("python", "numpy", "torch", "cuda", "device_available")
        },
        "simulator_version": citylearn.__version__,
        "signature": resume_signature(wrapper.config),
        "boundary": boundary,
        "resume_capability": "episode" if boundary and static and complete else "weights",
        "global_step": wrapper.global_step,
        "learning_step": getattr(wrapper, "learning_step", wrapper.global_step),
        "next_episode": getattr(wrapper, "_next_episode", 0),
        "episode_history": wrapper.get_episode_execution_history(),
        "tracker_state": deepcopy(vars(tracker)) if tracker is not None else None,
        "rng": capture_rng(),
        "streams": {
            path: deepcopy(unit.random_streams.states)
            for path, unit in units
            if hasattr(unit, "random_streams")
        },
        "agents": [
            {
                "path": path,
                "name": getattr(unit, "agent_name", type(unit).__name__),
                "identity": getattr(unit, "runtime_identity", None),
            }
            for path, unit in units
        ],
    }


def restore_runtime(wrapper, state):
    if (
        state.get("version") != "algorithms.resume.v1"
        or state.get("resume_capability") != "episode"
        or not state.get("boundary")
    ):
        raise ValueError(
            "Checkpoint does not support complete episode-boundary resume; use resume_mode=weights"
        )
    if state.get("simulator_version") != citylearn.__version__:
        raise ValueError("Episode resume requires the same Simulator version")
    versions = {
        key: wrapper.config.get("runtime", {}).get("randomness", {}).get(key)
        for key in ("python", "numpy", "torch", "cuda", "device_available")
    }
    if state.get("runtime_versions") != versions:
        raise ValueError("Episode resume requires the same Python/NumPy/PyTorch/CUDA runtime")
    if state["signature"] != resume_signature(wrapper.config):
        raise ValueError("Episode resume configuration or dataset identity differs from checkpoint")
    tracker = getattr(wrapper.env.unwrapped, "episode_tracker", None)
    if state["next_episode"] >= wrapper.config["simulator"]["episodes"]:
        raise ValueError("Episode resume requires a remaining episode in the requested budget")
    if tracker is None or state.get("tracker_state") is None:
        raise ValueError("Simulator cannot restore its episode tracker")
    # Version-pinned adapter: no mid-episode simulator state is fabricated.
    vars(tracker).update(deepcopy(state["tracker_state"]))
    wrapper.global_step = state["global_step"]
    wrapper.learning_step = state["learning_step"]
    wrapper._resume_episode = state["next_episode"]
    wrapper._episode_execution_history = deepcopy(state["episode_history"])
    for path, unit in leaves(wrapper.model):
        if path in state["streams"]:
            unit.random_streams.states = deepcopy(state["streams"][path])
    restore_rng(state["rng"])
