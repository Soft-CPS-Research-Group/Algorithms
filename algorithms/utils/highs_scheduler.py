"""Serialize native HiGHS solves that share its process-wide thread pool."""

from contextlib import contextmanager
from threading import RLock

_lock = RLock()
_threads = None


@contextmanager
def highs_scheduler(highspy, threads, *, timeout=None):
    global _threads
    acquired = _lock.acquire() if timeout is None else _lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError("Timed out waiting for the shared HiGHS scheduler")
    try:
        if highspy is not None and threads != _threads:
            highspy.Highs.resetGlobalScheduler(True)
            _threads = threads
        yield
    finally:
        _lock.release()
