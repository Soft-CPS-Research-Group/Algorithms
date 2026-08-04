#!/usr/bin/env python3
"""Build a compact, inference-safe checkpoint pack for a frozen PPO ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = "ppo_frozen_inference_v1"
CHECKPOINT_NAME = "latest_checkpoint.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _agent_index(path: Path) -> int:
    match = re.fullmatch(r"agent_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint member directory: {path}")
    return int(match.group(1))


def _compact_payload(checkpoint: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    actor_keys = sorted(key for key in checkpoint if key.startswith("actor_state_dict_"))
    value_keys = sorted(key for key in checkpoint if key.startswith("value_state_dict_"))
    if actor_keys != ["actor_state_dict_0"] or value_keys != ["value_state_dict_0"]:
        raise ValueError(
            "Each distributed PPO member must contain exactly actor_state_dict_0 "
            "and value_state_dict_0."
        )
    return {
        "checkpoint_format": FORMAT_VERSION,
        "source_sha256": source_sha256,
        "source_step": int(checkpoint.get("step", 0)),
        "actor_state_dict_0": checkpoint["actor_state_dict_0"],
        "value_state_dict_0": checkpoint["value_state_dict_0"],
    }


def package_checkpoint_root(
    *,
    source_root: Path,
    output_root: Path,
    seed: int,
    source_job: str,
    expected_agents: int = 17,
) -> dict[str, Any]:
    source_root_label = source_root.as_posix()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")

    members = sorted(
        (path for path in source_root.glob("agent_*") if path.is_dir()),
        key=_agent_index,
    )
    indices = [_agent_index(path) for path in members]
    expected = list(range(expected_agents))
    if indices != expected:
        raise ValueError(f"Expected agent indices {expected}, found {indices}")

    manifest_members: list[dict[str, Any]] = []
    output_root.mkdir(parents=True)
    for source_member in members:
        index = _agent_index(source_member)
        source_path = source_member / CHECKPOINT_NAME
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {source_path}")
        source_hash = _sha256(source_path)
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint root must be a mapping: {source_path}")

        member_dir = output_root / f"agent_{index}"
        member_dir.mkdir()
        output_path = member_dir / CHECKPOINT_NAME
        torch.save(_compact_payload(checkpoint, source_sha256=source_hash), output_path)
        manifest_members.append(
            {
                "agent_index": index,
                "path": str(output_path.relative_to(output_root)),
                "sha256": _sha256(output_path),
                "source_sha256": source_hash,
                "bytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "format": FORMAT_VERSION,
        "seed": int(seed),
        "source_job": str(source_job),
        "source_root": source_root_label,
        "member_count": len(manifest_members),
        "total_bytes": sum(int(member["bytes"]) for member in manifest_members),
        "omitted_state": [
            "actor_optimizers",
            "value_optimizers",
            "rollout",
            "behavior_cloning_replay",
            "exploration_state",
            "rng_state",
        ],
        "required_loader_flags": {
            "restore_optimizers": False,
            "restore_replay_buffer": False,
            "restore_exploration_state": False,
        },
        "members": manifest_members,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--expected-agents", type=int, default=17)
    args = parser.parse_args()
    manifest = package_checkpoint_root(
        source_root=args.source_root,
        output_root=args.output_root,
        seed=args.seed,
        source_job=args.source_job,
        expected_agents=args.expected_agents,
    )
    print(json.dumps({
        "output_root": str(args.output_root),
        "member_count": manifest["member_count"],
        "total_bytes": manifest["total_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
