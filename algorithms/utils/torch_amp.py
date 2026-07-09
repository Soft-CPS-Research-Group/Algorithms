from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

try:
    from torch.amp import GradScaler as GradScaler
    from torch.amp import autocast as _autocast

    _SUPPORTS_DEVICE_TYPE = True
except ImportError:  # pragma: no cover - exercised on older Jetson PyTorch builds.
    from torch.cuda.amp import GradScaler as GradScaler
    from torch.cuda.amp import autocast as _autocast

    _SUPPORTS_DEVICE_TYPE = False


@contextmanager
def autocast(*, device_type: str, enabled: bool) -> Iterator[None]:
    if _SUPPORTS_DEVICE_TYPE:
        with _autocast(device_type=device_type, enabled=enabled):
            yield
    else:
        with _autocast(enabled=enabled and device_type == "cuda"):
            yield
