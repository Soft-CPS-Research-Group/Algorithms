from __future__ import annotations

import json

from utils.progress_tracker import ProgressTracker


def test_progress_tracker_writes_totals_and_global_percentage(tmp_path):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path))

    tracker.update(
        episode=1,
        step=4,
        global_step=37,
        rewards=[1.5, -0.2],
        episode_total=3,
        step_total=24,
        global_step_total=72,
        status="running",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["episode"] == 1
    assert payload["episode_current"] == 2
    assert payload["episode_total"] == 3
    assert payload["step"] == 4
    assert payload["step_current"] == 5
    assert payload["step_total"] == 24
    assert payload["global_step"] == 37
    assert payload["global_step_total"] == 72
    assert payload["progress_pct"] == 51.3889
    assert payload["status"] == "running"
    assert payload["rewards"] == [1.5, -0.2]


def test_progress_tracker_fallback_percentage_without_global_total(tmp_path):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path))

    tracker.update(
        episode=1,
        step=24,
        global_step=999,
        episode_total=2,
        step_total=24,
        status="completed",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert "global_step_total" not in payload
    assert payload["step_current"] == 24
    assert payload["progress_pct"] == 100.0
    assert payload["status"] == "completed"
