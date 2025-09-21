"""Configuration schema definitions and helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal


class MetadataConfig(BaseModel):
    experiment_name: str = Field(..., min_length=1, description="Name registered in MLflow")
    run_name: str = Field(..., min_length=1, description="Friendly name for the MLflow run")


class RuntimeConfig(BaseModel):
    log_dir: Optional[str] = Field(default=None, description="Resolved at runtime; path for log files")
    mlflow_uri: Optional[str] = Field(default=None, description="Resolved at runtime; MLflow tracking URI")


class TrackingConfig(BaseModel):
    mlflow_enabled: bool = Field(default=True, description="If false, skips MLflow tracking")
    log_level: str = Field(default="INFO", description="Loguru log level")
    log_frequency: int = Field(default=1, ge=1, description="Log metrics every N environment steps")


class CheckpointingConfig(BaseModel):
    resume_training: bool = False
    checkpoint_run_id: Optional[str] = None
    checkpoint_artifact: str = Field(default="latest_checkpoint.pth")
    use_best_checkpoint_artifact: bool = False
    reset_replay_buffer: bool = False
    freeze_pretrained_layers: bool = False
    fine_tune: bool = False
    checkpoint_interval: Optional[int] = Field(default=None, ge=1)


class SimulatorConfig(BaseModel):
    dataset_name: str
    dataset_path: str
    central_agent: bool = False
    reward_function: str


class TrainingConfig(BaseModel):
    seed: int = 22
    end_initial_exploration_time_step: int = Field(default=0, ge=0)
    end_exploration_time_step: int = Field(default=0, ge=0)
    steps_between_training_updates: int = Field(default=1, ge=1)
    target_update_interval: int = Field(default=0, ge=0)


class NetworkConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    layers: List[int]
    lr: float = Field(gt=0)

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("layers must contain at least one hidden dimension")
        if any(layer <= 0 for layer in value):
            raise ValueError("layers must be positive integers")
        return value


class AlgorithmNetworks(BaseModel):
    actor: NetworkConfig
    critic: NetworkConfig


class ReplayBufferConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    capacity: int = Field(ge=1)
    batch_size: int = Field(ge=1)


class ExplorationParams(BaseModel):
    strategy: str
    params: Dict[str, float]


class AlgorithmHyperparameters(BaseModel):
    gamma: float = Field(gt=0)


class RuleBasedHyperparameters(BaseModel):
    pv_charge_threshold: float = Field(default=0.0, ge=0)
    flexibility_hours: float = Field(default=3.0, ge=0)
    emergency_hours: float = Field(default=1.0, ge=0)
    pv_preferred_charge_rate: float = Field(default=0.6, ge=0)
    flex_trickle_charge: float = Field(default=0.0, ge=0)
    min_charge_rate: float = Field(default=0.0, ge=0)
    emergency_charge_rate: float = Field(default=1.0, ge=0)
    energy_epsilon: float = Field(default=1e-3, ge=0)
    default_capacity_kwh: float = Field(default=60.0, ge=0)
    non_flexible_chargers: List[str] = Field(default_factory=list)


class TopologyConfig(BaseModel):
    num_agents: Optional[int] = None
    observation_dimensions: Optional[List[int]] = None
    action_dimensions: Optional[List[int]] = None
    action_space: Optional[Any] = None


class MADDPGAlgorithmConfig(BaseModel):
    name: Literal["MADDPG"]
    hyperparameters: AlgorithmHyperparameters
    networks: AlgorithmNetworks
    replay_buffer: ReplayBufferConfig
    exploration: ExplorationParams


class RuleBasedAlgorithmConfig(BaseModel):
    name: Literal["RuleBasedPolicy"]
    hyperparameters: RuleBasedHyperparameters = RuleBasedHyperparameters()
    networks: Optional[AlgorithmNetworks] = None
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None


class SingleAgentRLAlgorithmConfig(BaseModel):
    name: Literal["SingleAgentRL"]
    hyperparameters: AlgorithmHyperparameters
    policy: Optional[str] = Field(default=None, description="Identifier for the policy architecture")
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None


class ProjectConfig(BaseModel):
    metadata: MetadataConfig
    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    simulator: SimulatorConfig
    training: TrainingConfig = TrainingConfig()
    topology: TopologyConfig = TopologyConfig()
    algorithm: Union[MADDPGAlgorithmConfig, RuleBasedAlgorithmConfig, SingleAgentRLAlgorithmConfig]

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary using original key names (aliases)."""
        return self.model_dump(by_alias=True)


def validate_config(raw_config: Dict[str, Any]) -> ProjectConfig:
    """Validate a raw configuration dictionary and return the structured model."""
    return ProjectConfig.model_validate(raw_config)
