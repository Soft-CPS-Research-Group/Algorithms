"""Lightweight contract for the receding-horizon community reference."""

from typing import Literal

from pydantic import Field

from algorithms.configuration.model import StrictModel


class MILPHyperparameters(StrictModel):
    horizon_hours: float = Field(default=24.0, gt=0, le=24, allow_inf_nan=False)
    terminal_storage_soc: float = Field(default=0.5, ge=0, le=0.99, allow_inf_nan=False)
    time_limit_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    solver_threads: int = Field(default=1, ge=1)
    mip_relative_gap: float = Field(default=0.001, ge=0, le=0.01, allow_inf_nan=False)
    forecast: Literal["perfect_24h"] = "perfect_24h"
    physics: Literal["conservative", "simulator_linear"] = "conservative"


class MILPStageConfig(StrictModel):
    algorithm: Literal["MILP"]
    count: Literal[1] = 1
    frozen: bool = False
    hyperparameters: MILPHyperparameters = Field(default_factory=MILPHyperparameters)
