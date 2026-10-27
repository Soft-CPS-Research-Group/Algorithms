"""Resolve bundled configuration assets while retaining explicit local overrides."""

from pathlib import Path


def configuration_asset(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_absolute():
        return requested
    if requested.parts and requested.parts[0] == "configs" and ".." not in requested.parts:
        packaged = Path(__file__).resolve().parents[2] / requested
        if packaged.is_file():
            return packaged
    return requested
