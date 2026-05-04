"""Configuration schema definitions and helpers."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal


class MetadataConfig(BaseModel):
    experiment_name: str = Field(..., min_length=1, description="Name registered in MLflow")
    run_name: str = Field(..., min_length=1, description="Friendly name for the MLflow run")
    community_name: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional community/site identifier for the run",
    )
    description: Optional[str] = Field(default=None, description="Optional human-readable bundle description")
    bundle_version: Optional[str] = Field(default=None, description="Optional bundle version string")
    alias_mapping_path: Optional[str] = Field(
        default=None,
        description="Optional alias mapping path (relative to bundle root)",
    )


class RuntimeConfig(BaseModel):
    log_dir: Optional[str] = Field(default=None, description="Resolved at runtime; path for log files")
    job_dir: Optional[str] = Field(default=None, description="Resolved at runtime; job root directory")
    mlflow_uri: Optional[str] = Field(default=None, description="Resolved at runtime; MLflow tracking URI")
    tracking_uri: Optional[str] = Field(default=None, description="Resolved at runtime; effective MLflow tracking URI")
    job_id: Optional[str] = Field(default=None, description="Resolved at runtime; orchestrator job identifier")
    run_id: Optional[str] = Field(default=None, description="Resolved at runtime; active run identifier")
    run_name: Optional[str] = Field(default=None, description="Resolved at runtime; active run display name")
    experiment_id: Optional[str] = Field(default=None, description="Resolved at runtime; MLflow experiment identifier")
    mlflow_run_url: Optional[str] = Field(default=None, description="Resolved at runtime; MLflow UI URL for the active run")


class TrackingConfig(BaseModel):
    mlflow_enabled: bool = Field(default=True, description="If false, skips MLflow tracking")
    log_level: str = Field(default="INFO", description="Loguru log level")
    log_frequency: int = Field(default=1, ge=1, description="Log metrics every N environment steps")
    mlflow_step_sample_interval: int = Field(
        default=10,
        ge=1,
        description="Sample MLflow step metrics every N steps to reduce logging overhead",
    )
    mlflow_artifacts_profile: Literal["minimal", "curated"] = Field(
        default="minimal",
        description="Artifact logging profile for MLflow",
    )
    progress_updates_enabled: bool = Field(
        default=True,
        description="Enable periodic progress.json updates while training",
    )
    progress_update_interval: int = Field(
        default=5,
        ge=1,
        description="Write progress.json every N steps when progress updates are enabled",
    )
    system_metrics_enabled: bool = Field(
        default=False,
        description="Collect CPU/RAM/GPU system metrics during training (debug-oriented)",
    )
    system_metrics_interval: int = Field(
        default=10,
        ge=1,
        description="Collect system metrics every N steps when enabled",
    )


class CheckpointingConfig(BaseModel):
    resume_training: bool = False
    checkpoint_run_id: Optional[str] = None
    checkpoint_artifact: str = Field(default="latest_checkpoint.pth")
    use_best_checkpoint_artifact: bool = False
    reset_replay_buffer: bool = False
    freeze_pretrained_layers: bool = False
    fine_tune: bool = False
    checkpoint_interval: Optional[int] = Field(default=None, ge=1)
    require_update_step: bool = True
    require_initial_exploration_done: bool = True


class SimulatorExportConfig(BaseModel):
    mode: Literal["none", "during", "end"] = "none"
    export_kpis_on_episode_end: bool = False
    session_name: Optional[str] = None


class WrapperRewardConfig(BaseModel):
    enabled: bool = False
    profile: Literal["cost_limits_v1"] = "cost_limits_v1"
    clip_enabled: bool = True
    clip_min: float = -10.0
    clip_max: float = 10.0
    squash: Literal["none", "tanh"] = "none"

    @model_validator(mode="after")
    def validate_clip_range(self) -> "WrapperRewardConfig":
        if self.clip_max < self.clip_min:
            raise ValueError("simulator.wrapper_reward.clip_max must be >= clip_min")
        return self


class EntityEncodingConfig(BaseModel):
    enabled: Optional[bool] = None
    normalization: Literal["minmax_space"] = "minmax_space"
    clip: bool = True


class SimulatorConfig(BaseModel):
    dataset_name: str
    dataset_path: str
    central_agent: bool = False
    interface: Literal["flat", "entity"] = "flat"
    topology_mode: Literal["static", "dynamic"] = "static"
    reward_function: str
    reward_function_kwargs: Dict[str, Any] = Field(default_factory=dict)
    episodes: int = Field(default=1, ge=1)
    simulation_start_time_step: Optional[int] = Field(default=None, ge=0)
    simulation_end_time_step: Optional[int] = Field(default=None, ge=0)
    episode_time_steps: Optional[Union[int, List[Tuple[int, int]]]] = None
    export: SimulatorExportConfig = SimulatorExportConfig()
    wrapper_reward: WrapperRewardConfig = WrapperRewardConfig()
    entity_encoding: EntityEncodingConfig = EntityEncodingConfig()

    @field_validator("episode_time_steps")
    @classmethod
    def validate_episode_time_steps(
        cls, value: Optional[Union[int, List[Tuple[int, int]]]]
    ) -> Optional[Union[int, List[Tuple[int, int]]]]:
        if value is None:
            return None
        if isinstance(value, int):
            if value < 1:
                raise ValueError("simulator.episode_time_steps must be >= 1")
            return value

        for start, end in value:
            if start < 0 or end < 0:
                raise ValueError("simulator.episode_time_steps ranges must be >= 0")
            if end < start:
                raise ValueError("simulator.episode_time_steps range end must be >= start")
        return value

    @model_validator(mode="after")
    def validate_time_window(self) -> "SimulatorConfig":
        if (
            self.simulation_start_time_step is not None
            and self.simulation_end_time_step is not None
            and self.simulation_end_time_step < self.simulation_start_time_step
        ):
            raise ValueError("simulator.simulation_end_time_step must be >= simulation_start_time_step")

        if self.topology_mode == "dynamic" and self.interface != "entity":
            raise ValueError("simulator.topology_mode='dynamic' requires simulator.interface='entity'")

        if self.entity_encoding.enabled is None:
            self.entity_encoding.enabled = self.interface == "entity"

        return self


class TrainingConfig(BaseModel):
    seed: int = 22
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


class MADDPGStageConfig(BaseModel):
    """Pipeline stage describing a MADDPG agent."""

    algorithm: Literal["MADDPG"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    hyperparameters: AlgorithmHyperparameters
    networks: AlgorithmNetworks
    replay_buffer: ReplayBufferConfig
    exploration: ExplorationParams


class RuleBasedStageConfig(BaseModel):
    """Pipeline stage describing a RuleBasedPolicy agent."""

    algorithm: Literal["RuleBasedPolicy"]
    count: int = Field(default=1, ge=1)
    hyperparameters: RuleBasedHyperparameters = RuleBasedHyperparameters()
    networks: Optional[AlgorithmNetworks] = None
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None


class SingleAgentRLStageConfig(BaseModel):
    """Pipeline stage placeholder for SingleAgentRL (no runtime impl yet)."""

    algorithm: Literal["SingleAgentRL"]
    count: int = Field(default=1, ge=1)
    hyperparameters: AlgorithmHyperparameters
    policy: Optional[str] = Field(default=None, description="Identifier for the policy architecture")
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None


PipelineStageConfig = Union[
    MADDPGStageConfig,
    RuleBasedStageConfig,
    SingleAgentRLStageConfig,
]


class DeucalionExecutionConfig(BaseModel):
    partition: Optional[str] = None
    account: Optional[str] = None
    time: Optional[str] = None
    cpus_per_task: Optional[int] = Field(default=None, ge=1)
    mem_gb: Optional[int] = Field(default=None, ge=1)
    gpus: Optional[int] = Field(default=None, ge=0)
    sif_path: Optional[str] = None
    sif_image: Optional[str] = None
    sif_version: Optional[str] = None
    modules: List[str] = Field(default_factory=list)
    required_paths: List[str] = Field(default_factory=list)
    command_mode: Literal["run", "exec"] = "run"
    datasets: List[str] = Field(default_factory=list)

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: List[str]) -> List[str]:
        validated: List[str] = []
        for raw in value:
            path = (raw or "").strip()
            if not path:
                raise ValueError("execution.deucalion.datasets entries must be non-empty")

            pure = PurePosixPath(path)
            if pure.is_absolute():
                raise ValueError(f"execution.deucalion.datasets must be relative paths, got: {path!r}")
            if ".." in pure.parts:
                raise ValueError(f"execution.deucalion.datasets cannot contain '..', got: {path!r}")

            normalized = str(pure)
            if normalized.startswith("./"):
                normalized = normalized[2:]
            if not normalized.startswith("datasets/"):
                raise ValueError(
                    f"execution.deucalion.datasets must start with 'datasets/', got: {path!r}"
                )
            validated.append(normalized)
        return validated


class ExecutionConfig(BaseModel):
    deucalion: Optional[DeucalionExecutionConfig] = None


class BundleConfig(BaseModel):
    bundle_version: Optional[str] = Field(default=None, description="Bundle version published in manifest metadata")
    description: Optional[str] = Field(default=None, description="Bundle description published in manifest metadata")
    alias_mapping_path: Optional[str] = Field(
        default=None,
        description="Optional alias mapping path published in manifest metadata",
    )
    require_observations_envelope: bool = Field(
        default=False,
        description="If true, inference expects features.observations envelope",
    )
    artifact_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-values merged into each exported artifact config",
    )
    per_agent_artifact_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent artifact config overrides keyed by agent_index as string",
    )

    @field_validator("per_agent_artifact_config")
    @classmethod
    def validate_per_agent_artifact_config(
        cls, value: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, cfg in value.items():
            if not isinstance(cfg, dict):
                raise ValueError(
                    "bundle.per_agent_artifact_config values must be objects "
                    f"(got {type(cfg).__name__} for key {key!r})"
                )
            normalized[str(key)] = dict(cfg)
        return normalized


class ProjectConfig(BaseModel):
    metadata: MetadataConfig
    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    simulator: SimulatorConfig
    training: TrainingConfig = TrainingConfig()
    topology: TopologyConfig = TopologyConfig()
    pipeline: List[PipelineStageConfig] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of execution stages. A single-element list represents "
            "a single agent (current default). Multi-element lists describe a "
            "vertical hierarchy (top stage feeds context to the next)."
        ),
    )
    execution: Optional[ExecutionConfig] = None
    bundle: BundleConfig = BundleConfig()

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_cross_constraints(self) -> "ProjectConfig":
        uses_maddpg = any(stage.algorithm == "MADDPG" for stage in self.pipeline)
        if (
            uses_maddpg
            and self.simulator.interface == "entity"
            and self.simulator.topology_mode == "dynamic"
        ):
            raise ValueError(
                "algorithm 'MADDPG' does not support simulator.interface='entity' "
                "with simulator.topology_mode='dynamic'."
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary using original key names (aliases)."""
        return self.model_dump(by_alias=True)


def validate_config(raw_config: Dict[str, Any]) -> ProjectConfig:
    """Validate a raw configuration dictionary and return the structured model."""
    if isinstance(raw_config, dict) and "algorithm" in raw_config and "pipeline" not in raw_config:
        raise ValueError(
            "Configuration uses the deprecated top-level 'algorithm' key. "
            "Migrate to a 'pipeline' list, e.g.:\n\n"
            "  pipeline:\n"
            "    - algorithm: \"<name>\"\n"
            "      count: 1\n"
            "      hyperparameters: { ... }\n"
            "      networks: { ... }   # if applicable\n"
            "      replay_buffer: { ... }   # if applicable\n"
            "      exploration: { ... }   # if applicable\n"
        )
    return ProjectConfig.model_validate(raw_config)
