from __future__ import annotations

import json

from utils import progress_tracker as progress_tracker_module
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


def test_progress_tracker_completed_status_forces_100_with_global_total(tmp_path):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path))

    tracker.update(
        episode=0,
        step=9,
        global_step=50,
        episode_total=1,
        step_total=100,
        global_step_total=100,
        status="completed",
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["progress_pct"] == 100.0
    assert payload["status"] == "completed"


def test_progress_tracker_writes_extra_runtime_fields(tmp_path):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path))

    tracker.update(
        episode=0,
        step=2,
        global_step=3,
        status="running",
        extra={
            "phase": "env_step_start",
            "process_rss_mb": 123.4,
        },
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "env_step_start"
    assert payload["process_rss_mb"] == 123.4


def test_progress_tracker_atomically_replaces_previous_payload(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress" / "progress.json"
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text('{"global_step": 7}', encoding="utf-8")
    tracker = ProgressTracker(str(progress_path))

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(progress_tracker_module.os, "replace", fail_replace)

    tracker.update(episode=0, step=8, global_step=8, status="running")

    assert json.loads(progress_path.read_text(encoding="utf-8")) == {"global_step": 7}
    assert list(progress_path.parent.glob(".progress.json.*.tmp")) == []


def test_progress_tracker_does_not_fsync_best_effort_telemetry_by_default(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path))
    monkeypatch.setattr(
        progress_tracker_module.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(AssertionError("unexpected fsync")),
    )

    tracker.update(episode=0, step=0, global_step=1, status="running")

    assert json.loads(progress_path.read_text(encoding="utf-8"))["global_step"] == 1


def test_progress_tracker_can_opt_into_durable_writes(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress" / "progress.json"
    tracker = ProgressTracker(str(progress_path), durable=True)
    calls = []
    monkeypatch.setattr(progress_tracker_module.os, "fsync", lambda fd: calls.append(fd))

    tracker.update(episode=0, step=0, global_step=1, status="running")

    assert len(calls) == 1
