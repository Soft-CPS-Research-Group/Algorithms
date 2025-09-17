"""Progress file writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger


class ProgressTracker:
    """Writes incremental progress information for external observers."""

    def __init__(self, progress_path: Optional[str]) -> None:
        self.progress_path = Path(progress_path) if progress_path else None

    def update(
        self,
        episode: int,
        step: int,
        global_step: int,
        rewards: Optional[Sequence[float]] = None,
    ) -> None:
        if not self.progress_path:
            return

        payload = {
            "episode": episode,
            "step": step,
            "global_step": global_step,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if rewards is not None:
            payload["rewards"] = list(rewards)

        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to write progress file %s: %s", self.progress_path, exc)
