"""Offline policies and trainer implementations for the common job runtime.

The prepare_training hook owns offline optimization; algorithms.runtime owns
job configuration, simulation, evaluation and published artifacts. Collection
and multi-seed research commands remain in scripts; see docs/runtime/README.md.
"""
