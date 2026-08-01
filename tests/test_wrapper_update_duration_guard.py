from __future__ import annotations

import pytest

from utils import wrapper_citylearn as wrapper_module
from utils.wrapper_citylearn import Wrapper_CityLearn


def _wrapper_for_update_duration_guard(max_update_seconds: float) -> tuple[Wrapper_CityLearn, list[dict]]:
    wrapper = object.__new__(Wrapper_CityLearn)
    wrapper.max_update_seconds = max_update_seconds
    wrapper.global_step = 42
    progress_updates: list[dict] = []
    wrapper._write_phase_progress = lambda **kwargs: progress_updates.append(kwargs)
    wrapper._cancel_stall_watchdog = lambda: None
    return wrapper, progress_updates


def test_update_duration_guard_allows_duration_below_limit():
    wrapper, progress_updates = _wrapper_for_update_duration_guard(max_update_seconds=240.0)

    wrapper._enforce_update_duration_guard(
        update_duration=239.0,
        episode=1,
        step=2,
        episode_total=3,
        step_total=10,
        global_step_total=30,
        rewards=[1.0],
    )

    assert progress_updates == []


def test_update_duration_guard_reports_failure_and_raises_exact_message():
    wrapper, progress_updates = _wrapper_for_update_duration_guard(max_update_seconds=2400.0)

    with pytest.raises(
        TimeoutError,
        match=r"^Update duration 2400\.100s exceeded configured limit 2400\.000s at global step 42\.$",
    ):
        wrapper._enforce_update_duration_guard(
            update_duration=2400.1,
            episode=1,
            step=2,
            episode_total=3,
            step_total=10,
            global_step_total=30,
            rewards=[1.0],
        )

    assert progress_updates[0]["extra"]["error_type"] == "UpdateDurationGuardError"


def test_update_timing_synchronizes_cuda_model_before_and_after(monkeypatch):
    wrapper = object.__new__(Wrapper_CityLearn)
    wrapper.model = type("Model", (), {"device": type("Device", (), {"type": "cuda"})()})()
    synchronize_calls = []
    monkeypatch.setattr(wrapper_module.torch.cuda, "synchronize", lambda: synchronize_calls.append(None))

    wrapper._synchronize_model_cuda_for_timing()
    wrapper._synchronize_model_cuda_for_timing()

    assert synchronize_calls == [None, None]


def test_update_timing_does_not_synchronize_cpu_model(monkeypatch):
    wrapper = object.__new__(Wrapper_CityLearn)
    wrapper.model = type("Model", (), {"device": type("Device", (), {"type": "cpu"})()})()
    monkeypatch.setattr(
        wrapper_module.torch.cuda,
        "synchronize",
        lambda: pytest.fail("CPU model must not synchronize CUDA"),
    )

    wrapper._synchronize_model_cuda_for_timing()
