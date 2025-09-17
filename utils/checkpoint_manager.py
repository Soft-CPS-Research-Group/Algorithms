"""Utilities for saving and logging training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow
from loguru import logger

from algorithms.agents.base_agent import BaseAgent


class CheckpointManager:
    """Handle periodic checkpointing and logging."""

    def __init__(
        self,
        base_dir: Optional[str],
        interval: Optional[int],
        log_to_mlflow: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else None
        self.interval = interval if interval and interval > 0 else None
        self.log_to_mlflow = log_to_mlflow

    def maybe_save(
        self,
        agent: BaseAgent,
        step: int,
        initial_exploration_done: bool,
        update_step: bool,
    ) -> Optional[Path]:
        if not self.interval or not self.base_dir:
            return None
        if step <= 0 or not initial_exploration_done or not update_step:
            return None
        if step % self.interval != 0:
            return None
        return self.save(agent, step)

    def save(self, agent: BaseAgent, step: int) -> Optional[Path]:
        if not self.base_dir:
            logger.debug("Checkpoint directory is not set; skipping save.")
            return None

        checkpoint_dir = self.base_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path = agent.save_checkpoint(str(checkpoint_dir), step)
        except NotImplementedError:
            logger.debug("Agent does not implement checkpoint saving; skipping.")
            return None

        if checkpoint_path and self.log_to_mlflow and mlflow.active_run():
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            logger.info("Checkpoint logged to MLflow: {}", checkpoint_path)
        return Path(checkpoint_path) if checkpoint_path else None
