"""Checkpoint discovery and integrity independent of agent serialization format."""

from __future__ import annotations

import json
from pathlib import Path

from algorithms.runtime.atomic import file_hash

INDEX_VERSION = "algorithms.checkpoints.v1"


def resolve_index(path: Path, *, verify=True):
    """Return the latest committed entry, its agent payload and generation directory."""
    path = path.resolve()
    candidates = [path / "index.json", path / "checkpoints" / "index.json"] if path.is_dir() else []
    if path.name == "index.json":
        candidates.insert(0, path)
    for index_path in candidates:
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text())
        if index.get("version") != INDEX_VERSION:
            raise ValueError("Unsupported checkpoint index version")
        entry = next(e for e in index["entries"] if e["id"] == index["latest"])
        root = index_path.parent
        generation = (root / entry["directory"]).resolve()
        if (
            not generation.is_relative_to(root.resolve())
            or not (generation / ".complete").is_file()
        ):
            raise ValueError("Checkpoint generation is outside its root or incomplete")
        for record in entry["files"]:
            file = (generation / record["path"]).resolve()
            if not file.is_relative_to(generation):
                raise ValueError("Checkpoint index contains an escaping path")
            if verify and (not file.is_file() or file_hash(file) != record["sha256"]):
                raise ValueError(f"Checkpoint integrity mismatch: {record['path']}")
        artifact = (generation / entry["artifact"]).resolve()
        if not artifact.is_relative_to(generation) or not artifact.exists():
            raise ValueError("Checkpoint artifact is missing or outside its generation")
        return entry, artifact, generation
    return None


def discover_legacy(path: Path, artifact="latest_checkpoint.pth") -> Path:
    """Read old layouts, including TPPO's nested numbered .pt files."""
    indexed = resolve_index(path)
    if indexed:
        return indexed[1]
    if path.is_file():
        return path
    direct = path / artifact
    if direct.is_file():
        return direct
    candidates = list(path.glob("transformer_ppo_step*.pt")) + list(
        path.glob("checkpoints/transformer_ppo_step*.pt")
    )
    if candidates:
        return max(candidates, key=lambda p: int(p.stem.split("step")[-1]))
    return path
