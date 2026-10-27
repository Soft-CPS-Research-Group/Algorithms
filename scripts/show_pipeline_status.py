"""Pipeline status viewer for ``runs/<output>/status.json``.

Renders a compact unicode table summarising stage progress for the
``scripts.run_entity_pipeline`` orchestrator (Phase 6 of the IQL+CQL
15-min initiative).

When ``status.json`` is absent (e.g. an interrupted pre-Phase-5 run, or a
run that crashed before its first stage transition) the viewer falls back
to scanning ``.{stage}.done`` sentinel files written by individual stages.

Usage
-----
::

    .venv/bin/python -m scripts.show_pipeline_status runs/offline_iql_cql_initiative_15min
    .venv/bin/python -m scripts.show_pipeline_status runs/foo --ascii   # plain output

Output (truncated)
------------------
::

    [pipeline status] runs/foo
    [source] status.json
    ┌──────────────────┬──────────┬──────────┬───────────────────────────┐
    │ Stage            │ Status   │ Duration │ Last update               │
    ├──────────────────┼──────────┼──────────┼───────────────────────────┤
    │ collect          │ ✓ done   │   2m 14s │ 2025-06-20T11:32:14+00:00 │
    │ train-iql        │ ▶ running│   1h 25m │ started 2025-06-20T13:00… │
    │ train-cql        │ ○ pending│        — │ —                         │
    │ benchmark        │ ○ pending│        — │ —                         │
    │ feature-analysis │ ○ pending│        — │ —                         │
    └──────────────────┴──────────┴──────────┴───────────────────────────┘
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Canonical stage order — kept in sync with scripts.run_entity_pipeline.STAGES.
STAGE_ORDER: List[str] = [
    "collect",
    "train-iql",
    "train-cql",
    "benchmark",
    "feature-analysis",
]

# (glyph, label) per status. ASCII variants substitute box-drawing-free
# equivalents for environments (CI logs, redirected files) that mangle
# wide / non-ASCII glyphs.
_STATUS_GLYPHS_UNICODE = {
    "done": ("✓", "done"),
    "running": ("▶", "running"),
    "skipped": ("⊝", "skipped"),
    "pending": ("○", "pending"),
    "failed": ("✗", "failed"),
}
_STATUS_GLYPHS_ASCII = {
    "done": ("v", "done"),
    "running": (">", "running"),
    "skipped": ("-", "skipped"),
    "pending": (".", "pending"),
    "failed": ("x", "failed"),
}


@dataclass
class StageRow:
    """A single rendered row of the status table."""

    stage: str
    status: str
    duration_seconds: Optional[float]
    started_at: Optional[str]
    completed_at: Optional[str]
    skipped_at: Optional[str]
    failed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: Optional[float]) -> str:
    """Render a duration in seconds as a compact human-friendly string.

    Examples
    --------
    >>> format_duration(None)
    '—'
    >>> format_duration(45.3)
    '45.3s'
    >>> format_duration(125)
    '2m 5s'
    >>> format_duration(3725)
    '1h 02m'
    """
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes_total = seconds / 60
    if minutes_total < 60:
        whole_minutes = int(minutes_total)
        rem_seconds = int(round(seconds - whole_minutes * 60))
        return f"{whole_minutes}m {rem_seconds}s"
    hours = int(minutes_total // 60)
    rem_minutes = int(round(minutes_total - hours * 60))
    return f"{hours}h {rem_minutes:02d}m"


def _format_last_update(row: StageRow) -> str:
    """Pick the most relevant timestamp for the 'Last update' column.

    Priority: completed_at (terminal success) > failed_at (terminal failure)
    > skipped_at (no-op) > started_at (in-flight) > dash placeholder.
    """
    if row.completed_at:
        return str(row.completed_at)
    if row.failed_at:
        return f"failed {row.failed_at}"
    if row.skipped_at:
        return f"skipped {row.skipped_at}"
    if row.started_at:
        return f"started {row.started_at}"
    return "—"


# ---------------------------------------------------------------------------
# Source loading: status.json (preferred) → sentinels (fallback) → empty
# ---------------------------------------------------------------------------


def read_status_file(output: Path) -> Optional[dict]:
    """Return parsed ``<output>/status.json`` or ``None`` if missing/corrupt.

    Corruption is treated as 'missing' so that the viewer transparently
    falls back to sentinel scanning instead of crashing on a partially-
    written file.
    """
    path = output / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def derive_from_sentinels(output: Path) -> dict:
    """Build a status-like dict from ``.{stage}.done`` sentinels only.

    Each present sentinel maps to ``{"status": "done"}`` plus any
    ``completed_at`` / ``duration_seconds`` recovered from the sentinel
    payload.  A missing sentinel means the stage is absent from the
    returned dict (collect_rows then renders it as 'pending').
    """
    stages: dict = {}
    for stage in STAGE_ORDER:
        sentinel = output / f".{stage}.done"
        if not sentinel.exists():
            continue
        try:
            payload: Any = json.loads(sentinel.read_text())
        except (json.JSONDecodeError, OSError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        stages[stage] = {
            "status": "done",
            "completed_at": payload.get("completed_at"),
            "duration_seconds": payload.get("duration_seconds"),
        }
    return {"stages": stages}


def collect_rows(output: Path) -> Tuple[List[StageRow], str]:
    """Aggregate per-stage rows from status.json (preferred) or sentinels.

    Returns
    -------
    rows : list[StageRow]
        One row per stage in :data:`STAGE_ORDER` (stages missing from the
        source are rendered as ``status='pending'``).
    source : str
        ``'status.json'`` | ``'sentinels'`` | ``'empty'`` — used by the
        CLI to annotate the rendered output.
    """
    data = read_status_file(output)
    if data is not None:
        source = "status.json"
    else:
        data = derive_from_sentinels(output)
        source = "sentinels" if data.get("stages") else "empty"

    stages_dict = data.get("stages", {}) or {}
    rows: List[StageRow] = []
    for stage in STAGE_ORDER:
        entry = stages_dict.get(stage, {}) or {}
        rows.append(
            StageRow(
                stage=stage,
                status=str(entry.get("status", "pending")),
                duration_seconds=entry.get("duration_seconds"),
                started_at=entry.get("started_at"),
                completed_at=entry.get("completed_at"),
                skipped_at=entry.get("skipped_at"),
                failed_at=entry.get("failed_at"),
            )
        )
    return rows, source


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_table(rows: List[StageRow], *, unicode: bool = True) -> str:
    """Render ``rows`` as a bordered table.

    Parameters
    ----------
    rows : list[StageRow]
        One per stage; produced by :func:`collect_rows`.
    unicode : bool, default True
        When True, use box-drawing characters and status glyphs.  Set to
        False (``--ascii`` on the CLI) for plain ``+``/``|``/``-`` output
        suitable for CI logs and dumb terminals.
    """
    glyph_table = _STATUS_GLYPHS_UNICODE if unicode else _STATUS_GLYPHS_ASCII

    headers = ["Stage", "Status", "Duration", "Last update"]
    body: List[List[str]] = []
    for row in rows:
        glyph, label = glyph_table.get(row.status, ("?", row.status))
        body.append(
            [
                row.stage,
                f"{glyph} {label}",
                format_duration(row.duration_seconds),
                _format_last_update(row),
            ]
        )

    # Column widths: max(header_len, max(cell_len))
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in body), default=0))
        for i in range(len(headers))
    ]

    if unicode:
        h_sep, v_sep = "─", "│"
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        ml, mr = "├", "┤"
        t_join, b_join, cross = "┬", "┴", "┼"
    else:
        h_sep, v_sep = "-", "|"
        tl = tr = bl = br = "+"
        ml = mr = "+"
        t_join = b_join = cross = "+"

    def _fmt_row(cells: List[str]) -> str:
        padded = (f" {c.ljust(w)} " for c, w in zip(cells, widths))
        return v_sep + v_sep.join(padded) + v_sep

    def _horizontal(left: str, mid: str, right: str) -> str:
        return left + mid.join(h_sep * (w + 2) for w in widths) + right

    lines = [
        _horizontal(tl, t_join, tr),
        _fmt_row(headers),
        _horizontal(ml, cross, mr),
        *[_fmt_row(r) for r in body],
        _horizontal(bl, b_join, br),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="show_pipeline_status",
        description=(
            "Show stage-by-stage progress for a run_entity_pipeline output "
            "directory.  Reads status.json (Phase 5) or falls back to "
            ".{stage}.done sentinels."
        ),
    )
    p.add_argument(
        "output",
        type=Path,
        help="Pipeline output directory (e.g. runs/offline_iql_cql_initiative_15min).",
    )
    p.add_argument(
        "--ascii",
        action="store_true",
        help="Render with plain ASCII characters (no box-drawing / glyphs).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output: Path = args.output

    if not output.exists():
        print(
            f"[show-status] ERROR: output directory does not exist: {output}",
            file=sys.stderr,
        )
        return 1

    rows, source = collect_rows(output)

    print(f"[pipeline status] {output}")
    print(f"[source] {source}")
    print(render_table(rows, unicode=not args.ascii))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
