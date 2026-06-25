# tests/scripts/test_show_pipeline_status.py
"""Phase 6 — scripts.show_pipeline_status viewer CLI.

Reads ``runs/<output>/status.json`` (written by the orchestrator in Phase 5)
and renders a compact unicode table summarising stage progress.  Falls back
to scanning ``.{stage}.done`` sentinels when ``status.json`` is missing.

Tests cover:
  * Pure helpers: format_duration, render_table glyph selection.
  * status.json loading + corruption handling.
  * Sentinel fallback.
  * collect_rows aggregation + source attribution.
  * CLI main() exit codes and stdout rendering (with --ascii).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module import (will fail in RED phase until the script exists).
# ---------------------------------------------------------------------------


def _mod():
    """Import and return the module under test."""
    import scripts.show_pipeline_status as m
    return m


# ---------------------------------------------------------------------------
# Pure helpers — format_duration
# ---------------------------------------------------------------------------


def test_format_duration_none_renders_dash():
    """None duration renders as em-dash placeholder."""
    m = _mod()
    assert m.format_duration(None) == "—"


def test_format_duration_sub_minute_shows_seconds():
    """A duration under 60s includes 's' suffix and the value."""
    m = _mod()
    out = m.format_duration(45.3)
    assert "s" in out
    assert "45" in out


def test_format_duration_minutes_shows_m_and_s():
    """A 125-second duration renders as '2m 5s'-style text."""
    m = _mod()
    out = m.format_duration(125)
    assert "2m" in out
    assert "5s" in out


def test_format_duration_hours_shows_h_and_m():
    """A 3725-second duration (~1h2m5s) renders as '1h 02m'-style text."""
    m = _mod()
    out = m.format_duration(3725)
    assert "1h" in out
    assert "02m" in out


# ---------------------------------------------------------------------------
# read_status_file — happy path, missing, corrupt
# ---------------------------------------------------------------------------


def test_read_status_file_returns_none_when_missing(tmp_path):
    """No status.json present → returns None (caller falls back to sentinels)."""
    m = _mod()
    assert m.read_status_file(tmp_path) is None


def test_read_status_file_returns_dict_when_present(tmp_path):
    """Valid status.json → returns parsed dict."""
    m = _mod()
    payload = {"stages": {"collect": {"status": "done", "duration_seconds": 1.5}}}
    (tmp_path / "status.json").write_text(json.dumps(payload))
    out = m.read_status_file(tmp_path)
    assert out == payload


def test_read_status_file_returns_none_on_corrupt_json(tmp_path):
    """Corrupt JSON should not crash — return None so we fall back to sentinels."""
    m = _mod()
    (tmp_path / "status.json").write_text("{not: valid json")
    assert m.read_status_file(tmp_path) is None


# ---------------------------------------------------------------------------
# derive_from_sentinels — fallback when status.json is absent
# ---------------------------------------------------------------------------


def test_derive_from_sentinels_picks_up_done_sentinel(tmp_path):
    """A .collect.done sentinel is reported as status=done."""
    m = _mod()
    payload = {
        "stage": "collect",
        "completed_at": "2025-06-20T11:00:00+00:00",
        "duration_seconds": 12.5,
    }
    (tmp_path / ".collect.done").write_text(json.dumps(payload))
    out = m.derive_from_sentinels(tmp_path)
    assert "stages" in out
    assert out["stages"]["collect"]["status"] == "done"
    assert out["stages"]["collect"]["duration_seconds"] == 12.5
    assert out["stages"]["collect"]["completed_at"] == "2025-06-20T11:00:00+00:00"


def test_derive_from_sentinels_ignores_missing_sentinels(tmp_path):
    """Stages without sentinels are absent from the returned dict (→ pending)."""
    m = _mod()
    (tmp_path / ".collect.done").write_text("{}")
    out = m.derive_from_sentinels(tmp_path)
    assert "collect" in out["stages"]
    assert "train-iql" not in out["stages"]
    assert "benchmark" not in out["stages"]


def test_derive_from_sentinels_tolerates_corrupt_sentinel(tmp_path):
    """Malformed sentinel content should not crash; stage still reported as done."""
    m = _mod()
    (tmp_path / ".collect.done").write_text("not json at all")
    out = m.derive_from_sentinels(tmp_path)
    assert out["stages"]["collect"]["status"] == "done"


# ---------------------------------------------------------------------------
# collect_rows — aggregation + source attribution
# ---------------------------------------------------------------------------


def test_collect_rows_returns_one_row_per_stage_in_canonical_order(tmp_path):
    """All STAGE_ORDER stages appear, in order, regardless of status.json content."""
    m = _mod()
    (tmp_path / "status.json").write_text(json.dumps({"stages": {}}))
    rows, _source = m.collect_rows(tmp_path)
    assert [r.stage for r in rows] == m.STAGE_ORDER


def test_collect_rows_marks_unknown_stages_pending(tmp_path):
    """Stages not present in status.json default to status='pending'."""
    m = _mod()
    payload = {"stages": {"collect": {"status": "done"}}}
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rows, _ = m.collect_rows(tmp_path)
    statuses = {r.stage: r.status for r in rows}
    assert statuses["collect"] == "done"
    assert statuses["train-iql"] == "pending"
    assert statuses["benchmark"] == "pending"


def test_collect_rows_carries_duration_and_timestamps(tmp_path):
    """Fields from status.json are propagated to StageRow attributes."""
    m = _mod()
    payload = {
        "stages": {
            "collect": {
                "status": "done",
                "duration_seconds": 42.0,
                "completed_at": "2025-06-20T11:00:00+00:00",
                "started_at": "2025-06-20T10:59:18+00:00",
            }
        }
    }
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rows, _ = m.collect_rows(tmp_path)
    row = next(r for r in rows if r.stage == "collect")
    assert row.duration_seconds == 42.0
    assert row.started_at == "2025-06-20T10:59:18+00:00"
    assert row.completed_at == "2025-06-20T11:00:00+00:00"


def test_collect_rows_source_status_json_when_file_present(tmp_path):
    """source='status.json' when the file is loaded successfully."""
    m = _mod()
    (tmp_path / "status.json").write_text(json.dumps({"stages": {}}))
    _, source = m.collect_rows(tmp_path)
    assert source == "status.json"


def test_collect_rows_source_sentinels_when_only_sentinels_exist(tmp_path):
    """source='sentinels' when status.json is missing but at least one sentinel exists."""
    m = _mod()
    (tmp_path / ".collect.done").write_text("{}")
    _, source = m.collect_rows(tmp_path)
    assert source == "sentinels"


def test_collect_rows_source_empty_when_nothing_present(tmp_path):
    """source='empty' when neither status.json nor any sentinel exists."""
    m = _mod()
    _, source = m.collect_rows(tmp_path)
    assert source == "empty"


def test_collect_rows_falls_back_to_sentinels_on_corrupt_status(tmp_path):
    """Corrupt status.json + valid sentinel → sentinel-derived rows."""
    m = _mod()
    (tmp_path / "status.json").write_text("not json")
    (tmp_path / ".collect.done").write_text("{}")
    rows, source = m.collect_rows(tmp_path)
    assert source == "sentinels"
    assert next(r for r in rows if r.stage == "collect").status == "done"


# ---------------------------------------------------------------------------
# render_table — glyph selection + content
# ---------------------------------------------------------------------------


def test_render_table_contains_all_stage_names():
    """Every stage in STAGE_ORDER appears in the rendered output."""
    m = _mod()
    rows = [
        m.StageRow(
            stage=s, status="pending", duration_seconds=None,
            started_at=None, completed_at=None, skipped_at=None,
        )
        for s in m.STAGE_ORDER
    ]
    out = m.render_table(rows)
    for stage in m.STAGE_ORDER:
        assert stage in out, f"stage {stage!r} missing from rendered table:\n{out}"


def test_render_table_default_uses_unicode_box_drawing():
    """Default rendering uses ┌ ┐ │ ─ box-drawing characters."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="collect", status="done", duration_seconds=1.0,
            started_at=None, completed_at="2025-06-20T11:00:00+00:00", skipped_at=None,
        )
    ]
    out = m.render_table(rows)
    assert "┌" in out
    assert "┐" in out
    assert "│" in out
    assert "─" in out


def test_render_table_ascii_mode_uses_plain_characters():
    """unicode=False → plain ASCII (+, |, -) only, no box-drawing chars."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="collect", status="done", duration_seconds=1.0,
            started_at=None, completed_at="2025-06-20T11:00:00+00:00", skipped_at=None,
        )
    ]
    out = m.render_table(rows, unicode=False)
    assert "┌" not in out
    assert "│" not in out
    assert "─" not in out
    assert "+" in out
    assert "|" in out


def test_render_table_done_status_shows_glyph_in_unicode():
    """Unicode rendering shows the ✓ glyph for done."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="collect", status="done", duration_seconds=1.0,
            started_at=None, completed_at=None, skipped_at=None,
        )
    ]
    out = m.render_table(rows)
    assert "✓" in out


def test_render_table_skipped_status_renders_label():
    """Skipped stages show 'skipped' label in the Status column."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="collect", status="skipped", duration_seconds=None,
            started_at=None, completed_at=None, skipped_at="2025-06-20T11:00:00+00:00",
            failed_at=None,
        )
    ]
    out = m.render_table(rows)
    assert "skipped" in out


# ---------------------------------------------------------------------------
# Phase 7 — failed status surfacing
# ---------------------------------------------------------------------------


def test_collect_rows_carries_failed_at_timestamp(tmp_path):
    """failed_at from status.json is propagated to StageRow.failed_at."""
    m = _mod()
    payload = {
        "stages": {
            "train-iql": {
                "status": "failed",
                "duration_seconds": 3.0,
                "failed_at": "2025-06-20T12:00:00+00:00",
            }
        }
    }
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rows, _ = m.collect_rows(tmp_path)
    row = next(r for r in rows if r.stage == "train-iql")
    assert row.status == "failed"
    assert row.failed_at == "2025-06-20T12:00:00+00:00"


def test_render_table_failed_status_uses_x_glyph():
    """Failed stages render with the ✗ glyph in unicode mode."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="train-iql", status="failed", duration_seconds=3.0,
            started_at=None, completed_at=None, skipped_at=None,
            failed_at="2025-06-20T12:00:00+00:00",
        )
    ]
    out = m.render_table(rows)
    assert "✗" in out
    assert "failed" in out


def test_render_table_failed_at_surfaced_in_last_update_column():
    """The 'Last update' column shows 'failed <timestamp>' for a failed stage."""
    m = _mod()
    rows = [
        m.StageRow(
            stage="train-iql", status="failed", duration_seconds=3.0,
            started_at="2025-06-20T11:59:57+00:00",
            completed_at=None, skipped_at=None,
            failed_at="2025-06-20T12:00:00+00:00",
        )
    ]
    out = m.render_table(rows)
    # Must show the failed_at value, not just the started_at (priority).
    assert "2025-06-20T12:00:00+00:00" in out


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------


def test_main_exits_nonzero_when_output_dir_missing(tmp_path, capsys):
    """Missing output dir → exit code 1 and stderr message."""
    m = _mod()
    missing = tmp_path / "does-not-exist"
    rc = m.main([str(missing)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "does-not-exist" in err or "not exist" in err.lower()


def test_main_returns_zero_for_valid_output(tmp_path, capsys):
    """Valid output dir + status.json → exit 0 and stdout includes path + stages."""
    m = _mod()
    payload = {"stages": {"collect": {"status": "done", "duration_seconds": 1.0}}}
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rc = m.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(tmp_path) in out
    for stage in m.STAGE_ORDER:
        assert stage in out


def test_main_returns_zero_when_only_sentinels_present(tmp_path, capsys):
    """No status.json but sentinels present → still exit 0; output mentions sentinels."""
    m = _mod()
    (tmp_path / ".collect.done").write_text("{}")
    rc = m.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sentinels" in out.lower()


def test_main_returns_zero_when_run_is_empty(tmp_path, capsys):
    """Empty output dir → exit 0 with 'empty' source label and all-pending table."""
    m = _mod()
    rc = m.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # All five canonical stage names appear, even with nothing on disk.
    for stage in m.STAGE_ORDER:
        assert stage in out


def test_main_ascii_flag_disables_unicode_glyphs(tmp_path, capsys):
    """--ascii suppresses box-drawing characters in the rendered table."""
    m = _mod()
    payload = {"stages": {"collect": {"status": "done", "duration_seconds": 1.0}}}
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rc = m.main([str(tmp_path), "--ascii"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "┌" not in out
    assert "│" not in out


def test_main_prints_source_attribution(tmp_path, capsys):
    """Output includes a source-attribution line ('status.json' / 'sentinels' / 'empty')."""
    m = _mod()
    payload = {"stages": {}}
    (tmp_path / "status.json").write_text(json.dumps(payload))
    rc = m.main([str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "status.json" in out
