# tests/scripts/test_run_entity_pipeline_resume.py
"""Phase 7 — orchestrator-level resume / failure-recovery contract.

The trainer-level resume contract (identical final weights after interruption)
is covered by tests/offline_rl/test_iql_entity_resume.py and
test_cql_entity_resume.py.  This module covers the *orchestrator*'s share of
the contract:

  * On subprocess failure: no ``.{stage}.done`` sentinel is written.
  * The failed stage is recorded as ``status="failed"`` in ``status.json``
    (NEW behaviour — without this, the viewer shows stale ``running``).
  * Stages completed before the failure keep their sentinel + ``status="done"``.
  * Re-invoking the orchestrator with the same args:
      - Skips the completed-and-sentinelled stages.
      - Re-runs the failed stage *without* passing ``--force`` so the
        trainer's own ``checkpoint_latest.pt`` resume kicks in.
  * After a successful re-run, the prior ``failed`` status is overwritten.
  * Full pipeline runs are idempotent: re-invocation after total success
    is a no-op (all stages skipped).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers — mirror Phase 5 test patterns but allow per-cmd failure injection.
# ---------------------------------------------------------------------------


def _install_run_mock(
    monkeypatch,
    *,
    fail_if: Optional[Callable[[List[str]], bool]] = None,
) -> List[List[str]]:
    """Patch run_entity_pipeline._run; record argv; raise SystemExit if fail_if.

    The real ``_run`` calls ``sys.exit(returncode)`` on subprocess failure;
    we mimic that semantics so the orchestrator's exception-handling path
    is exercised honestly.
    """
    import scripts.run_entity_pipeline as m

    captured: List[List[str]] = []

    def fake_run(cmd):
        as_strs = [str(c) for c in cmd]
        captured.append(as_strs)
        if fail_if is not None and fail_if(as_strs):
            raise SystemExit(1)

    monkeypatch.setattr(m, "_run", fake_run)
    return captured


def _cmds_for(captured: List[List[str]], script_name: str) -> List[List[str]]:
    return [c for c in captured if any(script_name in part for part in c)]


def _run_main(argv: List[str]):
    import scripts.run_entity_pipeline as m
    return m.main(argv)


# ---------------------------------------------------------------------------
# 1. Failed stage does NOT write a sentinel.
# ---------------------------------------------------------------------------


def test_failed_train_iql_does_not_write_sentinel(monkeypatch, tmp_path):
    """When the trainer subprocess fails, no .train-iql.done is written."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])
    assert not (tmp_path / ".train-iql.done").exists(), (
        f"sentinel must NOT exist after failed stage; "
        f"dir contents={list(tmp_path.iterdir())}"
    )


def test_failed_collect_does_not_write_sentinel(monkeypatch, tmp_path):
    """Same contract for the collect stage."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("collect_rbcsmart_dataset" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "collect"])
    assert not (tmp_path / ".collect.done").exists()


# ---------------------------------------------------------------------------
# 2. status.json records failed stage as status="failed" (NEW behaviour).
# ---------------------------------------------------------------------------


def test_failed_stage_marks_status_failed_in_status_json(monkeypatch, tmp_path):
    """status.json[stages][train-iql].status == 'failed' after subprocess crash."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    status_path = tmp_path / "status.json"
    assert status_path.exists(), "status.json must be written even on failure"
    status = json.loads(status_path.read_text())
    entry = status["stages"].get("train-iql")
    assert entry is not None, f"train-iql missing from status: {status}"
    assert entry["status"] == "failed", (
        f"expected status='failed' for crashed stage; got {entry!r}"
    )


def test_failed_stage_records_failed_at_timestamp(monkeypatch, tmp_path):
    """The 'failed_at' wall-clock stamp is recorded for the viewer to surface."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    status = json.loads((tmp_path / "status.json").read_text())
    entry = status["stages"]["train-iql"]
    assert "failed_at" in entry, f"failed_at missing from entry: {entry}"
    assert entry["failed_at"], "failed_at must be a non-empty timestamp"


def test_failed_stage_records_duration_seconds(monkeypatch, tmp_path):
    """duration_seconds is recorded so the viewer can show how long the run took."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    status = json.loads((tmp_path / "status.json").read_text())
    entry = status["stages"]["train-iql"]
    assert "duration_seconds" in entry
    assert entry["duration_seconds"] >= 0


# ---------------------------------------------------------------------------
# 3. SystemExit is re-raised after status update (failure must propagate).
# ---------------------------------------------------------------------------


def test_failed_stage_propagates_systemexit(monkeypatch, tmp_path):
    """The orchestrator must re-raise the SystemExit so callers see the failure."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])
    # Surface the original exit code (1 in our mock).
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# 4. Stages completed before failure keep their sentinel + status=done.
# ---------------------------------------------------------------------------


def test_prior_completed_stage_keeps_sentinel_after_later_failure(monkeypatch, tmp_path):
    """collect completes → sentinel exists; then train-iql fails → collect's sentinel persists."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql"])
    assert (tmp_path / ".collect.done").exists(), (
        "collect sentinel must survive a later stage's failure"
    )


def test_prior_completed_stage_keeps_status_done(monkeypatch, tmp_path):
    """status.json for the earlier successful stage remains 'done' after later failure."""
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql"])
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stages"]["collect"]["status"] == "done"


# ---------------------------------------------------------------------------
# 5. Subsequent run: failed stage is re-invoked; completed stage is skipped.
# ---------------------------------------------------------------------------


def test_subsequent_run_reinvokes_failed_stage(monkeypatch, tmp_path):
    """After a failed first run, second run re-invokes the failed stage's subprocess."""
    # First run: fail train-iql.
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    # Second run: succeed (no fail_if).
    captured = _install_run_mock(monkeypatch)
    rc = _run_main(["--output", str(tmp_path), "--steps", "train-iql"])
    assert rc == 0
    assert _cmds_for(captured, "train_iql_entity"), (
        "second run must re-invoke trainer after prior failure"
    )


def test_subsequent_run_skips_prior_completed_stage(monkeypatch, tmp_path):
    """Stage that completed in run 1 (sentinel present) is skipped on run 2."""
    # First run: collect succeeds, train-iql fails.
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql"])

    # Second run: collect must be SKIPPED, train-iql must RE-RUN.
    captured = _install_run_mock(monkeypatch)
    rc = _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql"])
    assert rc == 0
    assert not _cmds_for(captured, "collect_rbcsmart_dataset"), (
        "collect must be skipped on retry; sentinel was present"
    )
    assert _cmds_for(captured, "train_iql_entity"), (
        "train-iql must re-run on retry; sentinel was missing"
    )


def test_resume_does_not_pass_force_to_trainer(monkeypatch, tmp_path):
    """Resume must NOT pass --force; trainer relies on checkpoint_latest.pt to resume.

    Passing --force would wipe the in-progress checkpoint and start over,
    defeating the entire resume contract.
    """
    # First run: fail.
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    # Second run: capture and inspect.
    captured = _install_run_mock(monkeypatch)
    rc = _run_main(["--output", str(tmp_path), "--steps", "train-iql"])
    assert rc == 0
    iql_cmds = _cmds_for(captured, "train_iql_entity")
    assert iql_cmds, "trainer must be re-invoked"
    assert "--force" not in iql_cmds[0], (
        f"resume must NOT pass --force (would clobber checkpoint); cmd={iql_cmds[0]}"
    )


# ---------------------------------------------------------------------------
# 6. Successful re-run overwrites prior 'failed' status with 'done'.
# ---------------------------------------------------------------------------


def test_successful_rerun_overwrites_failed_status(monkeypatch, tmp_path):
    """After resume succeeds, status.json reflects status='done', not the old 'failed'."""
    # First run: fail.
    _install_run_mock(
        monkeypatch,
        fail_if=lambda cmd: any("train_iql_entity" in p for p in cmd),
    )
    with pytest.raises(SystemExit):
        _run_main(["--output", str(tmp_path), "--steps", "train-iql"])

    pre_status = json.loads((tmp_path / "status.json").read_text())
    assert pre_status["stages"]["train-iql"]["status"] == "failed"

    # Second run: succeed.
    _install_run_mock(monkeypatch)
    rc = _run_main(["--output", str(tmp_path), "--steps", "train-iql"])
    assert rc == 0

    post_status = json.loads((tmp_path / "status.json").read_text())
    assert post_status["stages"]["train-iql"]["status"] == "done"
    assert "completed_at" in post_status["stages"]["train-iql"]


# ---------------------------------------------------------------------------
# 7. Idempotency after full success — third run is a complete no-op.
# ---------------------------------------------------------------------------


def test_third_run_after_full_success_skips_every_stage(monkeypatch, tmp_path):
    """Running the orchestrator three times in a row after full success is idempotent."""
    # Run 1: full success.
    _install_run_mock(monkeypatch)
    rc1 = _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql,train-cql"])
    assert rc1 == 0

    # Run 2: no work expected.
    captured2 = _install_run_mock(monkeypatch)
    rc2 = _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql,train-cql"])
    assert rc2 == 0
    for script in ("collect_rbcsmart_dataset", "train_iql_entity", "train_cql_entity"):
        assert not _cmds_for(captured2, script), f"{script} must be skipped on run 2"

    # Run 3: also no work.
    captured3 = _install_run_mock(monkeypatch)
    rc3 = _run_main(["--output", str(tmp_path), "--steps", "collect,train-iql,train-cql"])
    assert rc3 == 0
    for script in ("collect_rbcsmart_dataset", "train_iql_entity", "train_cql_entity"):
        assert not _cmds_for(captured3, script), f"{script} must be skipped on run 3"
