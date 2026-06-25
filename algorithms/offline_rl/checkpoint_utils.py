"""Atomic save/write helpers for trainer checkpointing and status reporting.

These primitives underpin Phase 2 of the IQL+CQL 15-min initiative: every
checkpoint write must be either fully successful or leave the prior
(or absent) state intact — never partial.  We achieve this by writing to
``<path>.tmp`` first and then ``os.replace(<path>.tmp, <path>)``.

Two flavours:

* :func:`atomic_save`  — wraps :func:`torch.save` for model / optimiser state.
* :func:`write_status` — wraps :func:`json.dumps` for ``status.json`` updates.

Both guarantee:

1. ``<path>`` either holds the new payload or is unchanged.
2. The ``.tmp`` scratch file is cleaned up on success.
3. On failure: the scratch file may linger (best-effort cleanup) but
   ``<path>`` is never corrupted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _tmp_for(path: Path) -> Path:
    """Return the scratch path used by atomic writes for ``path``."""
    return path.with_suffix(path.suffix + ".tmp")


def atomic_save(obj: Any, path: Path) -> None:
    """Persist ``obj`` to ``path`` via ``torch.save`` atomically.

    The contract:
    * On success: ``path`` contains the serialised ``obj`` and no ``.tmp``
      scratch file is left behind.
    * On failure: ``path`` is unchanged (or absent if it didn't exist).
      Any partially-written ``.tmp`` scratch file is best-effort removed.

    Raises whatever exception ``torch.save`` raises.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)

    try:
        torch.save(obj, tmp)
    except Exception:
        # Best-effort cleanup of partial scratch file.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    # os.replace is atomic on POSIX and Windows.
    os.replace(tmp, path)


def write_status(path: Path, payload: Any) -> None:
    """Serialise ``payload`` as JSON to ``path`` atomically.

    The payload is fully serialised (via ``json.dumps``) before any file write,
    so non-serialisable objects raise *before* touching the filesystem.

    Same contract as :func:`atomic_save`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise first: if payload is invalid, raise without touching disk.
    text = json.dumps(payload, indent=2, default=str)

    tmp = _tmp_for(path)
    try:
        tmp.write_text(text)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    os.replace(tmp, path)


__all__ = ["atomic_save", "write_status"]
