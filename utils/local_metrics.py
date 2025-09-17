"""Utilities for persisting metrics locally when MLflow is disabled."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class LocalMetricsLogger:
    def __init__(self, output_dir: Optional[str]) -> None:
        self._path: Optional[Path] = Path(output_dir) / "metrics.jsonl" if output_dir else None
        if self._path:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Could not create metrics directory %s: %s", self._path.parent, exc)
                self._path = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if not self._path:
            return
        record = {"step": step, "metrics": metrics}
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            logger.debug("Local metrics logged at step %s", step)
        except Exception as exc:
            logger.warning("Failed to write local metrics to %s: %s", self._path, exc)
