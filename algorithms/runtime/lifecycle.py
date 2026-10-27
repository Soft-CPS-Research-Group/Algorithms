"""Cancellation belongs to the attempt and must run its normal cleanup."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager


class JobCancelled(BaseException):
    pass


@contextmanager
def cancellation_signals():
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(signum, frame):
        raise JobCancelled(f"Job cancelled by signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
