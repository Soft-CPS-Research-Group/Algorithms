"""Stable named streams isolate initialization, exploration and replay/learning.

Legacy agents use process-global generators. Scopes swap their generator states
only for the duration of a synchronous hook, preserving the simulator's stream.
This runtime executes hooks serially; concurrent hooks in one process are unsupported.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
from contextlib import contextmanager, nullcontext

import numpy as np
import torch


def derived_seed(seed: int, *labels: object) -> int:
    identity = ":".join(map(str, (seed, *labels)))
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def seeded(seed: int):
    previous = capture_rng()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(previous)


class RandomStreams:
    def __init__(self, seed: int):
        self.seed = seed
        self.states = {}
        for name in ("initialization", "exploration", "learning", "evaluation"):
            with seeded(derived_seed(seed, name)):
                self.states[name] = capture_rng()

    @contextmanager
    def scope(self, name: str):
        previous = capture_rng()
        try:
            restore_rng(self.states[name])
            yield
        finally:
            self.states[name] = capture_rng()
            restore_rng(previous)


def agent_scope(unit, name):
    streams = getattr(unit, "random_streams", None)
    return streams.scope(name) if streams is not None else nullcontext()


def call_hook(unit, name, *args, stream="learning", **kwargs):
    with agent_scope(unit, stream):
        return getattr(unit, name)(*args, **kwargs)


def initialize_job(config):
    training = config.get("training", {})
    seed = int(training.get("seed", 22))
    # Set before CUDA initialization; strict kernels fail explicitly if unsupported.
    strict = training.get("determinism", "seeded") == "strict"
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    seed_all(seed)
    simulator = config["simulator"]
    if simulator.get("random_seed") is None:
        simulator["random_seed"] = derived_seed(seed, "simulator") % 65536
    if (
        simulator.get("random_episode_split")
        and int(simulator["random_seed"]) * (int(simulator.get("episodes", 1)) + 1) >= 2**32
    ):
        raise ValueError(
            "Simulator random episode splitting requires a smaller random_seed for this episode budget"
        )
    return {
        "version": "algorithms.random.v1",
        "training_seed": seed,
        "simulator_seed": simulator["random_seed"],
        "determinism": training.get("determinism", "seeded"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device_available": torch.cuda.is_available(),
        "comparison": "Exact repetition is tested on the pinned CPU runtime; cross-device equality is not promised.",
    }
