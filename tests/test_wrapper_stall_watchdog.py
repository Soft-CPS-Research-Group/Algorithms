from __future__ import annotations

import json
import sys

import numpy as np

from utils import wrapper_citylearn as wrapper_module
from utils.wrapper_citylearn import Wrapper_CityLearn


class _Space:
    def __init__(self, low, high):
        self.low = np.array(low, dtype=np.float64)
        self.high = np.array(high, dtype=np.float64)


class _DummyEnv:
    def __init__(self):
        self.observation_names = [["obs_0"]]
        self.observation_space = [_Space([0.0], [1.0])]
        self.action_space = [_Space([-1.0], [1.0])]
        self.action_names = [["action_0"]]
        self.reward_function = type("reward", (), {})()
        self.time_steps = 1
        self.seconds_per_time_step = 3600
        self.time_step_ratio = 1.0
        self.random_seed = 0
        self.episode_tracker = type("tracker", (), {"episode_time_steps": self.time_steps})()
        self.unwrapped = self

    def reset(self):
        return [np.array([0.0], dtype=np.float64)], {}

    def get_metadata(self):
        return {"buildings": [{}]}


def test_stall_watchdog_arms_and_cancels_independent_of_phase_progress(tmp_path, monkeypatch):
    calls = []

    def fake_cancel():
        calls.append({"event": "cancel"})

    def fake_dump_traceback_later(timeout, *, repeat=False, file=None, exit=False):
        calls.append(
            {
                "event": "arm",
                "timeout": timeout,
                "repeat": repeat,
                "exit": exit,
                "file_name": getattr(file, "name", None),
            }
        )

    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", fake_cancel)
    monkeypatch.setattr(wrapper_module.faulthandler, "dump_traceback_later", fake_dump_traceback_later)

    log_dir = tmp_path / "logs"
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(log_dir)},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "progress_phase_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
                "stall_watchdog_exit_on_timeout": False,
                "stall_watchdog_repeat": True,
            },
        },
        job_id="watchdog-test",
    )
    wrapper.global_step = 42

    wrapper._write_phase_progress(
        phase="episode_reset_start",
        episode=1,
        step=2,
        episode_total=3,
        step_total=10,
        global_step_total=30,
    )

    arm_call = next(call for call in calls if call["event"] == "arm")
    assert arm_call["timeout"] == 123.0
    assert arm_call["repeat"] is True
    assert arm_call["exit"] is False
    assert arm_call["file_name"] == sys.stderr.name

    context_path = log_dir / "watchdog-test_stall_watchdog.log.context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "episode_reset_start"
    assert payload["global_step"] == 42
    assert payload["episode_current"] == 2
    assert payload["step_current"] == 3

    wrapper._write_phase_progress(
        phase="episode_reset_end",
        episode=1,
        step=2,
        episode_total=3,
        step_total=10,
        global_step_total=30,
    )

    assert calls[-1]["event"] == "cancel"


def test_stall_watchdog_cancels_only_after_completion_progress_write(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "cancel_dump_traceback_later",
        lambda: events.append("cancel"),
    )
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: events.append("arm"),
    )

    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": True,
                "progress_phase_updates_enabled": True,
                "progress_update_interval": 1,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
            },
        },
        job_id="watchdog-order-test",
    )
    wrapper.global_step = 1

    wrapper._write_phase_progress(
        phase="episode_export_start",
        episode=0,
        step=0,
        episode_total=1,
        step_total=1,
        global_step_total=1,
    )
    original_update = wrapper.progress_tracker.update

    def observe_update(*args, **kwargs):
        events.append("progress")
        assert events[-2] == "arm"
        original_update(*args, **kwargs)

    wrapper.progress_tracker.update = observe_update
    wrapper._write_phase_progress(
        phase="episode_export_end",
        episode=0,
        step=0,
        episode_total=1,
        step_total=1,
        global_step_total=1,
    )

    assert events[-2:] == ["progress", "cancel"]


def test_lightweight_step_progress_updates_without_phase_heartbeats(tmp_path):
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": True,
                "progress_phase_updates_enabled": False,
                "progress_update_interval": 4,
            },
        },
        job_id="lightweight-progress-test",
    )
    updates = []
    wrapper.progress_tracker.update = lambda **payload: updates.append(payload)

    wrapper.global_step = 3
    wrapper._write_step_progress(
        episode=0,
        step=2,
        episode_total=2,
        step_total=8,
        global_step_total=16,
        rewards=[-1.0],
        step_duration=0.5,
        normal_step_duration=0.4,
        update_duration=0.1,
    )
    assert updates == []

    wrapper.global_step = 4
    wrapper._write_step_progress(
        episode=0,
        step=3,
        episode_total=2,
        step_total=8,
        global_step_total=16,
        rewards=[-0.5],
        step_duration=0.6,
        normal_step_duration=0.45,
        update_duration=0.15,
    )

    assert len(updates) == 1
    assert updates[0]["global_step"] == 4
    assert updates[0]["status"] == "running"
    assert updates[0]["extra"]["phase"] == "step_end"
    assert updates[0]["extra"]["update_duration_seconds"] == 0.15


def test_lightweight_step_progress_avoids_duplicate_phase_write(tmp_path):
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": True,
                "progress_phase_updates_enabled": True,
                "progress_update_interval": 1,
            },
        },
        job_id="phase-progress-dedup-test",
    )
    updates = []
    wrapper.progress_tracker.update = lambda **payload: updates.append(payload)
    wrapper.global_step = 1

    wrapper._write_step_progress(
        episode=0,
        step=0,
        episode_total=1,
        step_total=1,
        global_step_total=1,
        rewards=[0.0],
        step_duration=0.1,
        normal_step_duration=0.1,
        update_duration=0.0,
    )

    assert updates == []


def test_step_end_keeps_watchdog_armed_across_loop_boundary(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "cancel_dump_traceback_later",
        lambda: events.append("cancel"),
    )
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: events.append("arm"),
    )
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "progress_phase_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
            },
        },
        job_id="watchdog-loop-boundary-test",
    )

    wrapper._write_phase_progress(
        phase="step_start",
        episode=0,
        step=0,
        episode_total=1,
        step_total=2,
        global_step_total=2,
    )
    wrapper._write_phase_progress(
        phase="step_end",
        episode=0,
        step=0,
        episode_total=1,
        step_total=2,
        global_step_total=2,
    )

    assert events[-1] == "arm"


def test_stall_watchdog_refreshes_once_per_step_window(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "cancel_dump_traceback_later",
        lambda: events.append("cancel"),
    )
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: events.append("arm"),
    )
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
                "stall_watchdog_context_interval_steps": 64,
            },
        },
        job_id="watchdog-window-test",
    )

    for global_step in (1, 2, 63, 64, 65):
        wrapper.global_step = global_step
        wrapper._write_phase_progress(
            phase="step_start",
            episode=0,
            step=global_step - 1,
            episode_total=1,
            step_total=128,
            global_step_total=128,
        )
        for phase in ("predict_start", "env_step_start", "model_update_start"):
            wrapper._write_phase_progress(
                phase=phase,
                episode=0,
                step=global_step - 1,
                episode_total=1,
                step_total=128,
                global_step_total=128,
            )

    assert events.count("arm") == 2


def test_stall_watchdog_context_writes_are_step_throttled(tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: None,
    )

    log_dir = tmp_path / "logs"
    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(log_dir)},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "progress_phase_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
                "stall_watchdog_context_interval_steps": 64,
            },
        },
        job_id="watchdog-test",
    )

    context_path = log_dir / "watchdog-test_stall_watchdog.log.context.json"

    wrapper.global_step = 0
    wrapper._write_phase_progress(
        phase="step_start",
        episode=0,
        step=0,
        episode_total=1,
        step_total=128,
        global_step_total=128,
    )
    assert json.loads(context_path.read_text(encoding="utf-8"))["global_step"] == 0

    wrapper.global_step = 1
    wrapper._write_phase_progress(
        phase="step_start",
        episode=0,
        step=1,
        episode_total=1,
        step_total=128,
        global_step_total=128,
    )
    assert json.loads(context_path.read_text(encoding="utf-8"))["global_step"] == 0

    wrapper.global_step = 64
    wrapper._write_phase_progress(
        phase="step_start",
        episode=0,
        step=64,
        episode_total=1,
        step_total=128,
        global_step_total=128,
    )
    assert json.loads(context_path.read_text(encoding="utf-8"))["global_step"] == 64


def test_stall_watchdog_context_throttles_subphases_in_the_same_step(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: None,
    )

    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "progress_phase_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
                "stall_watchdog_context_interval_steps": 64,
            },
        },
        job_id="watchdog-throttle-test",
    )
    original_write = wrapper._write_stall_watchdog_context

    def observe_write(context):
        events.append(context["phase"])
        original_write(context)

    monkeypatch.setattr(wrapper, "_write_stall_watchdog_context", observe_write)
    wrapper.global_step = 64
    for phase in ("step_start", "predict_start", "env_step_start", "model_update_start"):
        wrapper._write_phase_progress(
            phase=phase,
            episode=0,
            step=63,
            episode_total=1,
            step_total=128,
            global_step_total=128,
        )

    assert events == ["step_start"]


def test_stall_watchdog_arms_before_context_io(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(wrapper_module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(
        wrapper_module.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat=False, file=None, exit=False: events.append("arm"),
    )

    wrapper = Wrapper_CityLearn(
        env=_DummyEnv(),
        config={
            "runtime": {"log_dir": str(tmp_path / "logs")},
            "training": {},
            "checkpointing": {},
            "tracking": {
                "progress_updates_enabled": False,
                "stall_watchdog_enabled": True,
                "stall_watchdog_timeout_seconds": 123.0,
            },
        },
        job_id="watchdog-order-test",
    )
    monkeypatch.setattr(
        wrapper,
        "_write_stall_watchdog_context",
        lambda context: events.append("context"),
    )

    wrapper._write_phase_progress(
        phase="step_start",
        episode=0,
        step=0,
        episode_total=1,
        step_total=1,
        global_step_total=1,
    )

    assert events == ["arm", "context"]
