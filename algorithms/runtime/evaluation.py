"""Evaluation mode and an identity for the policy associated with reported KPIs."""

from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager

import torch


def model_modules(unit):
    """Include legacy plain lists and per-building dataclasses, not only ModuleList."""
    visited, modules = set(), []

    def walk(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.nn.Module):
            modules.append(value)
            for child in vars(value).values():
                walk(child)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                walk(getattr(value, field.name))

    walk(unit)
    # ExecutionUnit is not itself a torch Module: visit its stored networks.
    if hasattr(unit, "__dict__"):
        walk(vars(unit))
    for name in ("stages", "agents"):
        for child in getattr(unit, name, []):
            modules.extend(model_modules(child))
    return list({id(module): module for module in modules}.values())


@contextmanager
def evaluation_mode(unit):
    modules = model_modules(unit)
    modes = [(module, module.training) for module in modules]
    try:
        for module in modules:
            module.training = False
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def policy_digest(unit):
    import json

    digest = hashlib.sha256()
    digest.update(type(unit).__name__.encode())
    digest.update(
        json.dumps(
            getattr(unit, "config", {}).get("algorithm", {}), sort_keys=True, default=str
        ).encode()
    )
    seen = set()
    for module in model_modules(unit):
        for name, tensor in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
