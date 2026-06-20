from __future__ import annotations

import json

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
        phase="env_step_start",
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
    assert arm_call["file_name"].endswith("watchdog-test_stall_watchdog.log")

    context_path = log_dir / "watchdog-test_stall_watchdog.log.context.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "env_step_start"
    assert payload["global_step"] == 42
    assert payload["episode_current"] == 2
    assert payload["step_current"] == 3

    wrapper._write_phase_progress(
        phase="env_step_end",
        episode=1,
        step=2,
        episode_total=3,
        step_total=10,
        global_step_total=30,
    )

    assert calls[-1]["event"] == "cancel"


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
