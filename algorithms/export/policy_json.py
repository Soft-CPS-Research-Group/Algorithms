"""Executable JSON state for registered deterministic rule/signal policies.

No arbitrary imports, pickle, expression evaluation or unknown-policy fallback.
This format deliberately differs from the serving service's generic rule JSON.
"""

import torch

from algorithms.export.portable import decode, encode, read_json
from algorithms.registry import agent_spec, import_symbol


def policy_state(unit):
    excluded = set(vars(torch.nn.Module())) | {
        "_config",
        "config",
        "_dataset_path",
        "random_streams",
        "capabilities",
        "_training_progress_callback",
        "training_progress_callback",
    }
    state = encode(
        {
            key: value
            for key, value in vars(unit).items()
            if key not in excluded and not key.startswith("_export_")
        }
    )
    # Unsupported helper records must remain explicit research dependencies.
    try:
        decode(state)
    except (ValueError, KeyError, TypeError) as exc:
        raise TypeError(f"Unsupported executable policy state: {exc}") from exc
    return state


def load_policy(path):
    payload = read_json(path)
    if payload.get("format") != "algorithms_policy_v1":
        raise ValueError("Unsupported executable policy format")
    spec = agent_spec(payload["algorithm"])
    if spec.capabilities.learning != "none":
        raise ValueError("JSON policies only support registered non-learning implementations")
    cls = import_symbol(spec.implementation)
    policy = cls.__new__(cls)
    torch.nn.Module.__init__(policy)
    state = decode(payload["state"])
    for key, value in state.items():
        if key.startswith("__") or callable(getattr(cls, key, None)):
            raise ValueError(f"Invalid policy state field: {key}")
        setattr(policy, key, value)
    policy._dataset_path = None
    policy._config = {}
    return policy
