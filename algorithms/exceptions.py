"""Exceptions shared across algorithm and training-runtime boundaries."""


class DeferredCheckpointError(ValueError):
    """Signal that checkpoint persistence is valid but must be deferred."""
