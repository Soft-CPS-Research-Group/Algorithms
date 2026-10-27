"""Offline trainer settings, shared by CLI validation and numerical trainers."""

from __future__ import annotations

import dataclasses
import math
from typing import Annotated, List

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid"))
class BCTrainingConfig:
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.1
    learning_rate: Annotated[float, Field(gt=0)] = 3e-4
    weight_decay: Annotated[float, Field(ge=0)] = 1e-5
    gradient_clip_norm: Annotated[float, Field(gt=0)] = 1.0
    batch_size: Annotated[int, Field(gt=0)] = 256
    epochs: Annotated[int, Field(gt=0)] = 50
    val_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.1
    device: str = "cpu"
    num_workers: Annotated[int, Field(ge=0)] = 0  # in-memory dataset; workers add overhead
    log_every_n_epochs: Annotated[int, Field(gt=0)] = 1


@dataclass(config=ConfigDict(extra="forbid"))
class IQLTrainingConfig:
    # Architecture
    hidden_layers: List[int] = dataclasses.field(default_factory=lambda: [256, 256])
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.1
    log_std_init: float = math.log(0.1)
    # IQL hyperparameters
    tau_expectile: Annotated[float, Field(gt=0, lt=1)] = 0.7
    beta_advantage: float = 3.0
    advantage_clip: float = 100.0
    gamma: Annotated[float, Field(gt=0, le=1)] = 0.99
    tau_target: Annotated[float, Field(gt=0, le=1)] = 0.005
    # Optimisation
    learning_rate: Annotated[float, Field(gt=0)] = 3e-4
    weight_decay: Annotated[float, Field(ge=0)] = 1e-5
    gradient_clip_norm: Annotated[float, Field(gt=0)] = 1.0
    batch_size: Annotated[int, Field(gt=0)] = 256
    gradient_steps: Annotated[int, Field(gt=0)] = 150_000
    eval_every_n_steps: Annotated[int, Field(gt=0)] = 2_500
    checkpoint_every_n_steps: Annotated[int, Field(ge=0)] = 5_000
    """Persist ``checkpoint_latest.pt`` every N gradient steps (Phase 2 resume)."""
    val_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.1
    device: str = "cpu"


@dataclass(config=ConfigDict(extra="forbid"))
class CQLTrainingConfig(IQLTrainingConfig):
    """Hyperparameters for CQL training.

    Inherits all IQL fields; adds conservative Q-learning params.
    """

    # CQL-specific
    cql_alpha: Annotated[float, Field(ge=0)] = 0.2
    """Weight of the conservative Q penalty."""

    cql_n_random_actions: Annotated[int, Field(gt=0)] = 10
    """Random actions sampled per state for the logsumexp approximation."""

    cql_min_q_weight: Annotated[float, Field(ge=0)] = 0.0
    """If > 0, used as a lower bound on the CQL penalty (target action gap).
    Set to 0.0 to disable (standard CQL without gap target)."""


OFFLINE_SETTINGS = {
    "BCAgent": BCTrainingConfig,
    "IQLAgent": IQLTrainingConfig,
    "IQLEntityAgent": IQLTrainingConfig,
    "CQLEntityAgent": CQLTrainingConfig,
}


def validate_offline_parameters(name: str, values: dict) -> dict:
    adapter = TypeAdapter(OFFLINE_SETTINGS[name])
    validated = adapter.validate_python(values)
    # Keep implicit device selection and all other defaults in the trainer.
    return {key: getattr(validated, key) for key in values}
