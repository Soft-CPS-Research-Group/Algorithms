"""Running value normalization shared by Transformer agents."""

from __future__ import annotations

from typing import Dict, Union

import torch


class RunningValueNormalizer:
    """Maintain scalar running statistics for value normalization."""

    _EPSILON = 1e-8

    def __init__(self) -> None:
        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

    def update(self, values: torch.Tensor) -> None:
        """Update running population statistics from a tensor of values."""
        if values.numel() == 0:
            return

        samples = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        batch_count = samples.numel()
        batch_mean = samples.mean().item()
        batch_variance = samples.var(unbiased=False).item()

        if self.count == 0:
            self.mean = batch_mean
            self.variance = batch_variance
            self.count = batch_count
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        combined_m2 = (
            self.variance * self.count
            + batch_variance * batch_count
            + delta * delta * self.count * batch_count / total_count
        )
        self.mean += delta * batch_count / total_count
        self.variance = combined_m2 / total_count
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values with the current running statistics."""
        mean = values.new_tensor(self.mean)
        scale = values.new_tensor((max(self.variance, 0.0) + self._EPSILON) ** 0.5)
        return (values - mean) / scale

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Restore normalized values to the original scale."""
        mean = values.new_tensor(self.mean)
        scale = values.new_tensor((max(self.variance, 0.0) + self._EPSILON) ** 0.5)
        return values * scale + mean

    def state_dict(self) -> Dict[str, Union[float, int]]:
        """Return device-agnostic scalar statistics."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }

    def load_state_dict(self, state: Dict[str, Union[float, int]]) -> None:
        """Restore scalar statistics from :meth:`state_dict`."""
        self.mean = float(state["mean"])
        self.variance = float(state["variance"])
        self.count = int(state["count"])


__all__ = ["RunningValueNormalizer"]
