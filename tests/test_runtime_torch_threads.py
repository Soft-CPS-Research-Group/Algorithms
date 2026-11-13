"""Thread-setting and validation contracts; no model or simulator execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from algorithms.configuration.common import RuntimeConfig


@pytest.mark.parametrize("value", [1, 2, 8])
def test_thread_count_accepts_positive_integers(value):
    assert RuntimeConfig(torch_num_threads=value).torch_num_threads == value


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, "1", [], {}])
def test_thread_count_rejects_invalid_or_coerced_values(value):
    with pytest.raises(ValidationError):
        RuntimeConfig(torch_num_threads=value)


def test_default_and_explicit_null_preserve_normalized_runtime():
    assert "torch_num_threads" not in RuntimeConfig().model_dump(exclude_none=True)
    assert "torch_num_threads" not in RuntimeConfig(torch_num_threads=None).model_dump(
        exclude_none=True
    )


def test_public_validation_remains_lightweight(tmp_path):
    """The public validation path must not import Torch or create an environment."""
    config = {
        "schema_version": "algorithms.config.v1",
        "metadata": {"experiment_name": "thread-contract", "run_name": "validation"},
        "runtime": {"torch_num_threads": 1},
        "simulator": {
            "dataset_name": "unopened-dataset",
            "dataset_path": "unopened-schema.json",
            "reward_function": "RewardFunction",
            "episodes": 1,
        },
        "pipeline": [{"algorithm": "RBCBasicPolicy", "count": 1}],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    probe = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'citylearn'}:
        raise AssertionError('Numerical dependency imported during validation: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from algorithms.runtime.cli import cli_main
sys.argv = ['algorithms-job', '--config', sys.argv[1], '--validate']
cli_main()
"""
    source = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        cwd=source,
        env={"PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"


@pytest.mark.parametrize("runtime", [{}, {"torch_num_threads": None}])
def test_unset_thread_count_never_changes_torch_setting(monkeypatch, runtime):
    from algorithms.runtime import randomness

    previous_determinism = randomness.torch.are_deterministic_algorithms_enabled()
    previous_benchmark = randomness.torch.backends.cudnn.benchmark
    previous_cudnn = randomness.torch.backends.cudnn.deterministic

    def unexpected(*_args):
        raise AssertionError("An unset thread option changed or queried Torch threads")

    monkeypatch.setattr(randomness.torch, "set_num_threads", unexpected)
    monkeypatch.setattr(randomness.torch, "get_num_threads", unexpected)
    monkeypatch.setattr(randomness.torch, "set_num_interop_threads", unexpected)
    monkeypatch.setattr(randomness, "seed_all", lambda _seed: None)
    config = {"runtime": runtime, "training": {"seed": 123}, "simulator": {"random_seed": 6763}}
    try:
        metadata = randomness.initialize_job(config)
        assert "torch_num_threads" not in metadata
        assert metadata["training_seed"] == 123
        assert metadata["simulator_seed"] == 6763
    finally:
        randomness.torch.use_deterministic_algorithms(previous_determinism)
        randomness.torch.backends.cudnn.benchmark = previous_benchmark
        randomness.torch.backends.cudnn.deterministic = previous_cudnn


def test_explicit_thread_count_applies_before_seed_and_records_effective_value(monkeypatch):
    from algorithms.runtime import randomness

    torch = randomness.torch
    original = torch.get_num_threads()
    interop = torch.get_num_interop_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn = torch.backends.cudnn.deterministic
    seen = []
    monkeypatch.setattr(randomness, "seed_all", lambda seed: seen.append((seed, torch.get_num_threads())))
    config = {"runtime": {"torch_num_threads": 1}, "training": {"seed": 123}, "simulator": {"random_seed": 6763}}
    try:
        metadata = randomness.initialize_job(config)
        assert seen == [(123, 1)]
        assert torch.get_num_threads() == metadata["torch_num_threads"] == 1
        assert torch.get_num_interop_threads() == interop
        # This is the same metadata assignment made by the shared job entry point.
        config["runtime"]["randomness"] = metadata
        resolved_runtime = RuntimeConfig.model_validate(config["runtime"]).model_dump(exclude_none=True)
        assert resolved_runtime["torch_num_threads"] == 1
        assert resolved_runtime["randomness"]["torch_num_threads"] == 1
    finally:
        torch.set_num_threads(original)
        torch.use_deterministic_algorithms(previous_determinism)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn
