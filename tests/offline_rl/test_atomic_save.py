# tests/offline_rl/test_atomic_save.py
"""Tests for algorithms.offline_rl.checkpoint_utils (atomic save helpers).

These helpers underpin trainer resume: every checkpoint write must be either
fully successful or leave the previous (or absent) state intact — never
partial.  Tested via mocked failures because real-world failures
(kill -9, disk full) can't be reproduced in-process.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# atomic_save (torch)
# ---------------------------------------------------------------------------


def test_atomic_save_writes_payload(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import atomic_save

    path = tmp_path / "state.pt"
    atomic_save({"value": 42}, path)
    assert path.exists()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded == {"value": 42}


def test_atomic_save_overwrites_existing(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import atomic_save

    path = tmp_path / "state.pt"
    atomic_save({"v": 1}, path)
    atomic_save({"v": 2}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded == {"v": 2}


def test_atomic_save_leaves_original_on_failure(tmp_path):
    """If torch.save fails mid-write, the original file is untouched.

    The .tmp scratch file may or may not exist; what matters is that ``path``
    still points to the original payload (no partial write).
    """
    from algorithms.offline_rl.checkpoint_utils import atomic_save

    path = tmp_path / "state.pt"
    atomic_save({"v": 1}, path)

    with mock.patch(
        "algorithms.offline_rl.checkpoint_utils.torch.save",
        side_effect=RuntimeError("disk full"),
    ):
        with pytest.raises(RuntimeError):
            atomic_save({"v": 2}, path)

    # Original content preserved
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded == {"v": 1}


def test_atomic_save_no_path_created_on_failure_when_no_prior_file(tmp_path):
    """If save fails on first write, ``path`` itself never appears."""
    from algorithms.offline_rl.checkpoint_utils import atomic_save

    path = tmp_path / "state.pt"
    with mock.patch(
        "algorithms.offline_rl.checkpoint_utils.torch.save",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            atomic_save({"v": 1}, path)

    assert not path.exists()


def test_atomic_save_cleans_tmp_on_success(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import atomic_save

    path = tmp_path / "state.pt"
    atomic_save({"v": 1}, path)
    # No stray .tmp left behind
    assert not (tmp_path / "state.pt.tmp").exists()


# ---------------------------------------------------------------------------
# write_status (JSON)
# ---------------------------------------------------------------------------


def test_write_status_writes_json(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import write_status

    path = tmp_path / "status.json"
    write_status(path, {"stage": "collect", "status": "done"})
    assert path.exists()
    data = json.loads(path.read_text())
    assert data == {"stage": "collect", "status": "done"}


def test_write_status_overwrites_existing(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import write_status

    path = tmp_path / "status.json"
    write_status(path, {"a": 1})
    write_status(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}


def test_write_status_leaves_original_on_failure(tmp_path):
    """JSON dump failure must not corrupt the existing file."""
    from algorithms.offline_rl.checkpoint_utils import write_status

    path = tmp_path / "status.json"
    write_status(path, {"a": 1})

    # Mock json.dumps to raise — simulates an un-serialisable payload making
    # it past ``default=str`` (e.g. a circular reference).
    with mock.patch(
        "algorithms.offline_rl.checkpoint_utils.json.dumps",
        side_effect=ValueError("circular reference"),
    ):
        with pytest.raises(ValueError):
            write_status(path, {"a": 2})

    assert json.loads(path.read_text()) == {"a": 1}


def test_write_status_cleans_tmp_on_success(tmp_path):
    from algorithms.offline_rl.checkpoint_utils import write_status

    path = tmp_path / "status.json"
    write_status(path, {"a": 1})
    assert not (tmp_path / "status.json.tmp").exists()
