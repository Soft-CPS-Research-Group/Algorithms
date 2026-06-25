"""CLI flag tests for ``train_iql_entity`` and ``train_cql_entity``.

Phase 5: verifies that ``--force`` is exposed as a top-level CLI flag and
wired through to the trainer's ``force=True`` kwarg.  Without this, the
orchestrator's ``--force train-iql`` / ``--force train-cql`` cannot
override per-seed ``seed.done`` sentinels.

Refactoring contract: both ``parse_args`` and ``main`` accept an optional
``argv`` list (``None`` falls back to ``sys.argv`` like before), mirroring
the orchestrator's ``_build_parser`` pattern.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# IQL CLI
# ---------------------------------------------------------------------------


def test_train_iql_entity_parse_args_force_default_false(tmp_path):
    from scripts.train_iql_entity import parse_args

    args = parse_args(["--output", str(tmp_path)])
    assert args.force is False


def test_train_iql_entity_parse_args_force_explicit(tmp_path):
    from scripts.train_iql_entity import parse_args

    args = parse_args(["--output", str(tmp_path), "--force"])
    assert args.force is True


def test_train_iql_entity_main_forwards_force_to_all_groups(monkeypatch, tmp_path):
    """args.force=True must reach train_all_groups(force=True) in the default
    (all-groups) path."""
    import scripts.train_iql_entity as m

    captured: dict = {}

    def fake_all_groups(**kwargs):
        captured["force"] = kwargs.get("force", "MISSING")
        return {}

    monkeypatch.setattr(m, "train_all_groups", fake_all_groups)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
        "--force",
    ])
    assert rc == 0
    assert captured["force"] is True


def test_train_iql_entity_main_default_force_false(monkeypatch, tmp_path):
    """When --force is omitted, train_all_groups must receive force=False."""
    import scripts.train_iql_entity as m

    captured: dict = {}

    def fake_all_groups(**kwargs):
        captured["force"] = kwargs.get("force", "MISSING")
        return {}

    monkeypatch.setattr(m, "train_all_groups", fake_all_groups)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
    ])
    assert rc == 0
    assert captured["force"] is False


def test_train_iql_entity_main_forwards_force_to_multi_seed(monkeypatch, tmp_path):
    """When --groups subset is requested, force must reach train_entity_multi_seed."""
    import scripts.train_iql_entity as m

    captured: list = []

    def fake_multi_seed(**kwargs):
        captured.append(kwargs.get("force", "MISSING"))
        return {
            "best_val_policy_mse_mean": 0.0,
            "best_val_policy_mse_std": 0.0,
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(m, "train_entity_multi_seed", fake_multi_seed)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
        "--groups", "706:2",
        "--force",
    ])
    assert rc == 0
    assert captured == [True], f"expected [True]; got {captured}"


# ---------------------------------------------------------------------------
# CQL CLI
# ---------------------------------------------------------------------------


def test_train_cql_entity_parse_args_force_default_false(tmp_path):
    from scripts.train_cql_entity import parse_args

    args = parse_args(["--output", str(tmp_path)])
    assert args.force is False


def test_train_cql_entity_parse_args_force_explicit(tmp_path):
    from scripts.train_cql_entity import parse_args

    args = parse_args(["--output", str(tmp_path), "--force"])
    assert args.force is True


def test_train_cql_entity_main_forwards_force_to_all_groups(monkeypatch, tmp_path):
    import scripts.train_cql_entity as m

    captured: dict = {}

    def fake_all_groups(**kwargs):
        captured["force"] = kwargs.get("force", "MISSING")
        return {}

    monkeypatch.setattr(m, "train_all_groups", fake_all_groups)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
        "--force",
    ])
    assert rc == 0
    assert captured["force"] is True


def test_train_cql_entity_main_default_force_false(monkeypatch, tmp_path):
    import scripts.train_cql_entity as m

    captured: dict = {}

    def fake_all_groups(**kwargs):
        captured["force"] = kwargs.get("force", "MISSING")
        return {}

    monkeypatch.setattr(m, "train_all_groups", fake_all_groups)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
    ])
    assert rc == 0
    assert captured["force"] is False


def test_train_cql_entity_main_forwards_force_to_multi_seed(monkeypatch, tmp_path):
    """When --groups subset is requested, force must reach train_cql_multi_seed."""
    import scripts.train_cql_entity as m

    captured: list = []

    def fake_multi_seed(**kwargs):
        captured.append(kwargs.get("force", "MISSING"))
        return {
            "best_val_policy_mse_mean": 0.0,
            "best_val_policy_mse_std": 0.0,
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(m, "train_cql_multi_seed", fake_multi_seed)

    rc = m.main([
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "out"),
        "--seeds", "22",
        "--val-seeds", "23",
        "--groups", "706:2",
        "--force",
    ])
    assert rc == 0
    assert captured == [True], f"expected [True]; got {captured}"
