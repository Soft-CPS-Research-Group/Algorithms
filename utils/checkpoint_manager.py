"""Versioned, transactional checkpoint publication around agent-owned serializers."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger

from algorithms.exceptions import DeferredCheckpointError
from algorithms.execution_unit import ExecutionUnit
from algorithms.runtime.atomic import atomic_json, file_inventory
from algorithms.runtime.checkpoint_index import INDEX_VERSION
from algorithms.runtime.tracking import mlflow


class CheckpointManager:
    def __init__(self, base_dir: Optional[str], interval: Optional[int],
                 log_to_mlflow: bool = False, require_update_step: bool = True,
                 require_initial_exploration_done: bool = True,
                 checkpoint_on_episode_end: bool = False, keep_episode_checkpoints: bool = False,
                 save_final: bool = True, min_free_disk_mb: float = 64):
        self.base_dir = Path(base_dir) if base_dir else None
        self.interval = interval if interval and interval > 0 else None
        self.log_to_mlflow = log_to_mlflow
        self.require_update_step = require_update_step
        self.require_initial_exploration_done = require_initial_exploration_done
        self.checkpoint_on_episode_end = checkpoint_on_episode_end
        self.keep_episode_checkpoints = keep_episode_checkpoints
        self._save_final_enabled = save_final
        self.min_free_disk_mb = min_free_disk_mb
        self.runtime_state = None
        self.latest_entry = None
        self.total_seconds = 0.0

    def maybe_save(self, agent, step, initial_exploration_done, update_step):
        if not self.interval or not self.base_dir or step <= 0:
            return None
        if self.require_initial_exploration_done and not initial_exploration_done:
            return None
        if self.require_update_step and not update_step:
            return None
        if step % self.interval:
            return None
        return self.save(agent, step, reason="periodic")

    def save_final(self, agent, step):
        if not self._save_final_enabled or not self.base_dir:
            return None
        return self.save(agent, step, reason="final")

    def save_episode_end(self, agent, *, step, episode, deterministic):
        if not self.checkpoint_on_episode_end or not self.base_dir or deterministic:
            return None
        return self.save(agent, step, reason="episode_end", episode=episode)

    def save(self, agent: ExecutionUnit, step: int, *, reason="explicit", episode=None):
        if not self.base_dir or not callable(getattr(agent, "save_checkpoint", None)):
            return None
        started = time.perf_counter()
        root = self.base_dir / "checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < self.min_free_disk_mb * 1024**2:
            raise OSError("Checkpoint free disk space is below checkpointing.min_free_disk_mb")
        # Every request gets a generation, even at the same step. No old file is overwritten.
        identity = f"step_{int(step):012d}_{uuid.uuid4().hex[:12]}"
        staging = root / f".{identity}.partial"
        published = root / identity
        staging.mkdir()
        try:
            path = agent.save_checkpoint(str(staging), step)
            if path is None:
                shutil.rmtree(staging)
                return None
            artifact = Path(path).resolve()
            if not artifact.is_relative_to(staging.resolve()) or not artifact.exists():
                raise ValueError("Agent checkpoint must exist inside the supplied directory")
            relative = artifact.relative_to(staging.resolve())
            runtime = self.runtime_state() if callable(self.runtime_state) else None
            if runtime is not None:
                import torch
                torch.save(runtime, staging / "runtime_state.pt")
            files = file_inventory(staging)
            # Sync payloads before publishing the generation marker and index.
            for record in files:
                with (staging / record["path"]).open("rb") as handle:
                    os.fsync(handle.fileno())
            (staging / ".complete").touch()
            staging.replace(published)
            entry = {"id": identity, "directory": identity, "artifact": str(relative),
                     "step": int(step), "episode": episode, "reason": reason,
                     "agent": type(agent).__name__, "agents": runtime.get("agents", []) if runtime else [], "files": files,
                     "resume_capability": runtime.get("resume_capability", "weights") if runtime else "weights",
                     "boundary": runtime.get("boundary") if runtime else None}
            index_path = root / "index.json"
            index = json.loads(index_path.read_text()) if index_path.exists() else {
                "version": INDEX_VERSION, "entries": [], "best": None}
            index["entries"].append(entry)
            index["latest"] = identity
            atomic_json(index_path, index)
            self.latest_entry = entry
            # Legacy file readers retain a relative alias; the index is authoritative.
            target = published / relative
            if target.is_file():
                alias = root / target.name
                temporary_alias = root / f".{target.name}.{uuid.uuid4().hex}.link"
                temporary_alias.symlink_to(target.relative_to(root))
                temporary_alias.replace(alias)
            if self.log_to_mlflow and mlflow.active_run():
                mlflow.log_artifacts(str(published), artifact_path=f"checkpoints/{identity}")
                mlflow.log_artifact(str(index_path), artifact_path="checkpoints")
            return target
        except (NotImplementedError, DeferredCheckpointError) as error:
            if staging.exists():
                shutil.rmtree(staging)
            logger.debug("Checkpoint unavailable/deferred: {}", error)
            return None
        finally:
            # Failed serializers are retained as .partial for diagnosis, never indexed.
            self.total_seconds += time.perf_counter() - started
