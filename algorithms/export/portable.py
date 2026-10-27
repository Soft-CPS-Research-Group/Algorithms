"""Strict JSON representations and confined files shared by bundle producers/consumers."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np


def encode(value):
    """Preserve infinity/unavailable numeric semantics, never replace them with zero."""
    if isinstance(value, np.ndarray):
        return {"$array": encode(value.tolist()), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"$number": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, np.random.Generator):
        return {"$rng": encode(value.bit_generator.state)}
    if dataclasses.is_dataclass(value):
        return {"$record": type(value).__name__, "fields": encode(vars(value))}
    if isinstance(value, tuple):
        return {"$tuple": [encode(item) for item in value]}
    if isinstance(value, set):
        return {"$set": [encode(item) for item in sorted(value)]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) and not key.startswith("$") for key in value):
            return {key: encode(item) for key, item in value.items()}
        return {"$mapping": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"No portable representation for {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$number" in value:
        return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[value["$number"]]
    if "$array" in value:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not a portable policy state")
        return np.asarray(decode(value["$array"]), dtype=dtype)
    if "$mapping" in value:
        return {decode(key): decode(item) for key, item in value["$mapping"]}
    if "$tuple" in value:
        return tuple(decode(item) for item in value["$tuple"])
    if "$set" in value:
        return set(decode(item) for item in value["$set"])
    if "$rng" in value:
        rng = np.random.default_rng()
        rng.bit_generator.state = decode(value["$rng"])
        return rng
    if "$record" in value:
        from algorithms.agents.rbc_agent import ChargerInfo, ElectricalServiceInfo, StorageInfo

        records = {cls.__name__: cls for cls in (ChargerInfo, ElectricalServiceInfo, StorageInfo)}
        if value["$record"] not in records:
            raise ValueError("Unsupported portable policy record")
        return records[value["$record"]](**decode(value["fields"]))
    if any(key.startswith("$") for key in value):
        raise ValueError("Unknown portable JSON tag")
    return {key: decode(item) for key, item in value.items()}


def confined_path(root, raw):
    """Bundle paths are POSIX relative files; reject even in-root symlinks/absolute paths."""
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"Invalid portable bundle path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or PureWindowsPath(raw).drive or ".." in path.parts or str(path) != raw:
        raise ValueError(f"Non-portable bundle path: {raw!r}")
    root = Path(root).resolve()
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Symlink is not allowed in a bundle: {raw!r}")
    if not target.resolve().is_relative_to(root) or not target.is_file():
        raise ValueError(f"Missing or escaping bundle file: {raw!r}")
    return target


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-standard JSON number: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def manifest_values(value):
    """Keep ordinary manifest arrays/objects; tag only non-finite numeric values."""
    if isinstance(value, np.ndarray):
        return manifest_values(value.tolist())
    if isinstance(value, np.generic):
        return manifest_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return encode(value)
    if isinstance(value, (list, tuple)):
        return [manifest_values(item) for item in value]
    if isinstance(value, dict):
        return {key: manifest_values(item) for key, item in value.items()}
    return value
