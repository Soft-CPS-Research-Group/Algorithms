"""Shared Torch runtime selection and diagnostics."""

from __future__ import annotations

import torch
from loguru import logger


def select_torch_device(
    *, require_cuda: bool = False, algorithm_name: str = "MADDPG"
) -> torch.device:
    """Select the Torch device and fail early when CUDA is required."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            f"{algorithm_name} requires CUDA, but torch.cuda.is_available() is false."
        )
    return torch.device("cpu")


def log_torch_runtime(device: torch.device) -> None:
    """Log the selected Torch runtime details."""
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    logger.info(
        "Torch runtime: torch_version={}, torch_cuda_version={}, cuda_available={}, cuda_device_count={}",
        torch.__version__,
        torch.version.cuda,
        cuda_available,
        cuda_device_count,
    )
    if cuda_available:
        logger.info("CUDA device selected: {}", torch.cuda.get_device_name(device))
