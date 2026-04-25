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
        *,
        episode_total: Optional[int] = None,
        step_total: Optional[int] = None,
        global_step_total: Optional[int] = None,
        status: Optional[str] = None,
        force_complete: bool = False,
    ) -> None:
        if not self.progress_path:
            return

        episode_current = max(0, episode) + 1
        step_current = max(0, step) + 1
        payload = {
            "episode": episode,
            "episode_current": episode_current,
            "step": step,
            "step_current": step_current,
            "global_step": global_step,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if episode_total is not None and episode_total > 0:
            payload["episode_total"] = episode_total
            payload["episode_current"] = min(episode_current, episode_total)

        if step_total is not None and step_total > 0:
            payload["step_total"] = step_total
            payload["step_current"] = min(step_current, step_total)

        progress_pct: Optional[float] = None
        if global_step_total is not None and global_step_total > 0:
            payload["global_step_total"] = global_step_total
            progress_pct = (max(0, global_step) / global_step_total) * 100.0
        elif (
            episode_total is not None
            and episode_total > 0
            and step_total is not None
            and step_total > 0
        ):
            bounded_step_current = min(step_current, step_total)
            step_fraction = bounded_step_current / step_total
            progress_pct = ((max(0, episode) + step_fraction) / episode_total) * 100.0

        if progress_pct is not None:
            payload["progress_pct"] = round(min(100.0, max(0.0, progress_pct)), 4)

        if status:
            payload["status"] = status

        if force_complete or (isinstance(status, str) and status.lower() == "completed"):
            payload["progress_pct"] = 100.0

        if rewards is not None:
            payload["rewards"] = list(rewards)

        try:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to write progress file {}: {}", self.progress_path, exc)
