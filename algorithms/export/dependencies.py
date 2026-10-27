"""Portable dependencies supplied by specialised agent exporters."""

from pathlib import Path

from utils.training_provenance import file_digest


def export_schedule(schedule, output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "schedule.json"
    path.write_text(schedule.to_json(), encoding="utf-8")
    return path.name, file_digest(path)
