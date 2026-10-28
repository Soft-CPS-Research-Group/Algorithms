"""Job, simulator and output service settings, independent of agent internals."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

from algorithms.configuration.model import StrictModel as BaseModel


class JobConfig(BaseModel):
    profile: Literal["train", "evaluate", "smoke"] = "train"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    # None migrates the existing bundle + checkpoint configuration unchanged.
    model_artifacts: Optional[Literal["none", "checkpoint", "bundle", "both"]] = None


class MetadataConfig(BaseModel):
    experiment_name: str = Field(..., min_length=1, description="Name registered in MLflow")
    run_name: str = Field(..., min_length=1, description="Friendly name for the MLflow run")
    community_name: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional community/site identifier for the run",
    )
    description: Optional[str] = Field(
        default=None, description="Optional human-readable bundle description"
    )
    bundle_version: Optional[str] = Field(
        default=None, description="Optional bundle version string"
    )
    alias_mapping_path: Optional[str] = Field(
        default=None,
        description="Optional alias mapping path (relative to bundle root)",
    )


class RuntimeConfig(BaseModel):
    randomness: Optional[Dict[str, Any]] = None
    episode_plan: Optional[Dict[str, Any]] = None
    input_identity: Optional[str] = None
    loaded_checkpoint: Optional[str] = None
    log_dir: Optional[str] = Field(
        default=None, description="Resolved at runtime; path for log files"
    )
    job_dir: Optional[str] = Field(
        default=None, description="Resolved at runtime; job root directory"
    )
    mlflow_uri: Optional[str] = Field(
        default=None, description="Resolved at runtime; MLflow tracking URI"
    )
    tracking_uri: Optional[str] = Field(
        default=None, description="Resolved at runtime; effective MLflow tracking URI"
    )
    job_id: Optional[str] = Field(
        default=None, description="Resolved at runtime; orchestrator job identifier"
    )
    run_id: Optional[str] = Field(
        default=None, description="Resolved at runtime; active run identifier"
    )
    run_name: Optional[str] = Field(
        default=None, description="Resolved at runtime; active run display name"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="Resolved at runtime; MLflow experiment identifier"
    )
    mlflow_run_url: Optional[str] = Field(
        default=None, description="Resolved at runtime; MLflow UI URL for the active run"
    )


class TrackingConfig(BaseModel):
    metrics_enabled: bool = True
    mlflow_enabled: bool = Field(default=False, description="If false, skips MLflow tracking")
    tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional run labels preserved in resolved configs and artifacts.",
    )
    log_level: str = Field(default="INFO", description="Loguru log level")
    log_frequency: int = Field(default=1, ge=1, description="Log metrics every N environment steps")
    mlflow_step_sample_interval: int = Field(
        default=10,
        ge=1,
        description="Sample MLflow step metrics every N steps to reduce logging overhead",
    )
    mlflow_failure_policy: Literal["local", "abort"] = "local"
    mlflow_upload_models: bool = False
    mlflow_upload_checkpoints: bool = False
    mlflow_artifacts_profile: Literal["minimal", "curated", "full"] = Field(
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
    action_diagnostics_enabled: bool = Field(
        default=False,
        description="Log compact action distribution diagnostics during rollouts",
    )
    action_diagnostics_detail: Literal["summary", "per_action"] = Field(
        default="summary",
        description="Action diagnostics detail level",
    )
    action_saturation_tolerance: float = Field(
        default=0.01,
        ge=0,
        description="Fraction of each action range considered near low/high bounds",
    )
    action_idle_tolerance: float = Field(
        default=0.02,
        ge=0,
        description="Absolute tolerance around zero for action idle diagnostics",
    )
    training_diagnostics_enabled: bool = Field(
        default=True,
        description="Log MADDPG internal training diagnostics such as Q stats and gradient norms",
    )
    training_diagnostics_detail: Literal["summary", "per_agent"] = Field(
        default="summary",
        description="MADDPG training diagnostics detail level",
    )
    reward_diagnostics_enabled: bool = Field(
        default=True,
        description="Log reward function component diagnostics when the reward exposes them",
    )
    reward_diagnostics_detail: Literal["summary", "per_agent"] = Field(
        default="summary",
        description="Reward component diagnostics detail level",
    )
    runtime_profiling_enabled: bool = Field(
        default=False,
        description="Log coarse runtime timings for wrapper and agent phases",
    )
    runtime_profiling_interval: int = Field(
        default=512,
        ge=1,
        description="Log runtime profiling metrics every N environment steps when enabled",
    )
    runtime_profiling_detail: Literal["summary", "detailed"] = Field(
        default="summary",
        description="Runtime profiling detail level",
    )
    progress_phase_updates_enabled: bool = Field(
        default=False,
        description="Write progress.json phase heartbeats around expensive wrapper phases",
    )
    progress_phase_start_step: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional first global step for detailed phase heartbeats",
    )
    progress_phase_end_step: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional last global step for detailed phase heartbeats",
    )
    max_step_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Abort training if a completed environment step exceeds this duration",
    )
    max_update_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Abort training if a completed model update exceeds this duration",
    )
    stall_watchdog_enabled: bool = Field(
        default=False,
        description="Arm a rolling faulthandler watchdog around episode boundaries and environment-step windows",
    )
    stall_watchdog_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Seconds without completing the current phase before dumping thread stacks",
    )
    stall_watchdog_exit_on_timeout: bool = Field(
        default=True,
        description="Exit the process after dumping stacks when the stall watchdog fires",
    )
    stall_watchdog_repeat: bool = Field(
        default=False,
        description="Repeat watchdog stack dumps when exit_on_timeout is false",
    )
    stall_watchdog_traceback_file: Optional[str] = Field(
        default=None,
        description="Optional path for stall watchdog stack dumps; defaults to captured stderr",
    )
    stall_watchdog_context_interval_steps: int = Field(
        default=1,
        ge=1,
        description="Write stall watchdog context at most every N environment steps to reduce remote I/O",
    )
    resource_guard_enabled: bool = Field(
        default=False,
        description="Abort training when configured process/system memory limits are crossed",
    )
    max_process_rss_mb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Abort when process resident memory exceeds this threshold",
    )
    min_available_ram_mb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Abort when system available RAM falls below this threshold",
    )

    @model_validator(mode="after")
    def validate_phase_window(self) -> "TrackingConfig":
        if (
            self.progress_phase_start_step is not None
            and self.progress_phase_end_step is not None
            and self.progress_phase_end_step < self.progress_phase_start_step
        ):
            raise ValueError(
                "tracking.progress_phase_end_step must be >= tracking.progress_phase_start_step"
            )
        return self


class CheckpointingConfig(BaseModel):
    resume_training: bool = False
    checkpoint_run_id: Optional[str] = None
    checkpoint_local_path: Optional[str] = None
    stage_checkpoint_local_paths: Dict[int, str] = Field(
        default_factory=dict,
        description=(
            "Optional pipeline-stage checkpoint roots keyed by zero-based stage "
            "index. Use this to initialise selected stages without restoring the "
            "entire pipeline."
        ),
    )
    checkpoint_artifact: str = Field(default="latest_checkpoint.pth")
    checkpoint_mode: Literal["full", "inference"] = Field(
        default="full",
        description=(
            "Persist the complete trainable state or an actor-only checkpoint "
            "intended for a frozen inference stage."
        ),
    )
    use_best_checkpoint_artifact: bool = False
    reset_replay_buffer: bool = False
    freeze_pretrained_layers: bool = False
    fine_tune: bool = False
    restore_optimizers: Optional[bool] = None
    restore_replay_buffer: Optional[bool] = None
    restore_exploration_state: bool = True
    restore_reward_normalizer: bool = True
    save_final: bool = True
    resume_mode: Literal["weights", "episode"] = "weights"
    min_free_disk_mb: float = Field(default=64.0, ge=0)
    checkpoint_interval: Optional[int] = Field(default=None, ge=1)
    checkpoint_on_episode_end: bool = Field(
        default=False,
        description=(
            "Save after each trainable episode, even when its exact step count "
            "does not align with checkpoint_interval."
        ),
    )
    keep_episode_checkpoints: bool = Field(
        default=False,
        description=(
            "Preserve a numbered copy of every episode-end checkpoint instead "
            "of retaining only the agent's latest artifact."
        ),
    )
    require_update_step: bool = True
    require_initial_exploration_done: bool = True

    @field_validator("stage_checkpoint_local_paths")
    @classmethod
    def validate_stage_checkpoint_local_paths(cls, value: Dict[int, str]) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        for raw_index, raw_path in value.items():
            index = int(raw_index)
            path = str(raw_path or "").strip()
            if index < 0:
                raise ValueError("checkpointing.stage_checkpoint_local_paths indices must be >= 0")
            if not path:
                raise ValueError(
                    "checkpointing.stage_checkpoint_local_paths values must be non-empty paths"
                )
            normalized[index] = path
        return normalized

    @model_validator(mode="after")
    def validate_checkpoint_source(self) -> "CheckpointingConfig":
        if self.resume_mode == "episode":
            if (
                not self.resume_training
                or not self.checkpoint_local_path
                or self.stage_checkpoint_local_paths
            ):
                raise ValueError(
                    "episode resume requires one local indexed checkpoint and resume_training=true"
                )
            if (
                self.checkpoint_mode != "full"
                or self.reset_replay_buffer
                or self.fine_tune
                or self.freeze_pretrained_layers
            ):
                raise ValueError("episode resume requires full state without reset/fine-tuning")
            if (
                self.restore_optimizers is False
                or self.restore_replay_buffer is False
                or not self.restore_exploration_state
                or not self.restore_reward_normalizer
            ):
                raise ValueError("episode resume cannot disable state restoration")
        if not self.stage_checkpoint_local_paths:
            return self
        if not self.resume_training:
            raise ValueError(
                "checkpointing.resume_training must be true when "
                "stage_checkpoint_local_paths is configured"
            )
        if (
            self.checkpoint_run_id
            or self.checkpoint_local_path
            or self.use_best_checkpoint_artifact
        ):
            raise ValueError(
                "checkpointing.stage_checkpoint_local_paths cannot be combined with "
                "checkpoint_run_id, checkpoint_local_path, or use_best_checkpoint_artifact"
            )
        return self


class SimulatorExportConfig(BaseModel):
    profile: Literal["final", "diagnostic"] = "final"
    mode: Literal["none", "during", "end"] = "none"
    export_kpis_on_episode_end: bool = True
    final_episode_only: bool = True
    kpis_final_episode_only: Optional[bool] = None
    timeseries_final_episode_only: Optional[bool] = None
    include_business_as_usual: bool = True
    export_business_as_usual_timeseries: bool = False
    kpi_round_decimals: Optional[int] = Field(default=None, ge=0)
    session_name: Optional[str] = None
    diagnostic_max_episodes: int = Field(default=3, ge=1, le=100)
    diagnostic_max_mb: float = Field(default=128, gt=0)

    @model_validator(mode="after")
    def resolve_export_policy(self):
        # Historical false flags require an explicit diagnostic profile now.
        # None inherits the common flag instead of silently becoming False.
        for name in ("kpis_final_episode_only", "timeseries_final_episode_only"):
            if getattr(self, name) is None:
                setattr(self, name, self.final_episode_only)
        if self.profile == "final":
            self.final_episode_only = True
            self.kpis_final_episode_only = True
            self.timeseries_final_episode_only = True
        return self


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
    profile: Literal[
        "minmax_space",
        "maddpg_v1",
        "maddpg_v2_compact",
        "maddpg_v3_operational",
        "maddpg_v3_realtime",
        "maddpg_v4_operational",
        "building_local_v1",
        "cc_level1",
        "cc_level2",
    ] = "minmax_space"
    clip: bool = True


class CommunityMarketKpisConfig(BaseModel):
    community_local_traded_enabled: bool = True
    community_self_consumption_enabled: bool = True


class CommunityMarketConfig(BaseModel):
    enabled: bool = True
    local_price_ratio_to_grid_import: float = Field(default=0.8, ge=0.0, le=1.0)
    intra_community_sell_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    grid_export_price: float = Field(default=0.0, ge=0.0)
    import_member_weights: Dict[str, float] = Field(default_factory=dict)
    kpis: CommunityMarketKpisConfig = CommunityMarketKpisConfig()

    @model_validator(mode="after")
    def default_sell_ratio(self) -> "CommunityMarketConfig":
        if self.intra_community_sell_ratio is None:
            self.intra_community_sell_ratio = self.local_price_ratio_to_grid_import
        return self


class SimulatorConfig(BaseModel):
    dataset_name: str
    dataset_path: str
    data_contract_path: Optional[str] = None
    building_ids: Optional[List[str]] = None
    electrical_service_overrides_path: Optional[str] = None
    central_agent: bool = False
    interface: Literal["flat", "entity"] = "flat"
    topology_mode: Literal["static", "dynamic"] = "static"
    action_policy: Literal["reject", "clip"] = "clip"
    reward_function: str
    reward_function_kwargs: Dict[str, Any] = Field(default_factory=dict)
    episodes: int = Field(default=1, ge=1)
    deterministic_finish: bool = False
    repeat_episode_scenario: bool = False
    random_seed: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Simulator/exogenous-process seed. This is deliberately separate "
            "from training.seed, which initializes the learning algorithm."
        ),
    )
    simulation_start_time_step: Optional[int] = Field(default=None, ge=0)
    simulation_end_time_step: Optional[int] = Field(default=None, ge=0)
    episode_time_steps: Optional[Union[int, List[Tuple[int, int]]]] = None
    terminal_observation_padding: bool = False
    export: SimulatorExportConfig = SimulatorExportConfig()
    wrapper_reward: WrapperRewardConfig = WrapperRewardConfig()
    entity_encoding: EntityEncodingConfig = EntityEncodingConfig()
    community_market: Optional[CommunityMarketConfig] = None

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

    @field_validator("building_ids")
    @classmethod
    def validate_building_ids(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        cleaned = [str(item).strip() for item in value]
        if not cleaned or any(not item for item in cleaned):
            raise ValueError("simulator.building_ids must contain non-empty IDs")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("simulator.building_ids must contain unique IDs")
        return cleaned

    @model_validator(mode="after")
    def validate_time_window(self) -> "SimulatorConfig":
        if (
            self.simulation_start_time_step is not None
            and self.simulation_end_time_step is not None
            and self.simulation_end_time_step < self.simulation_start_time_step
        ):
            raise ValueError(
                "simulator.simulation_end_time_step must be >= simulation_start_time_step"
            )

        if self.topology_mode == "dynamic" and self.interface != "entity":
            raise ValueError(
                "simulator.topology_mode='dynamic' requires simulator.interface='entity'"
            )

        if self.entity_encoding.enabled is None:
            self.entity_encoding.enabled = self.interface == "entity"

        return self


class EvaluationConfig(BaseModel):
    episodes: int = Field(default=1, ge=0, le=1)
    policy: Literal["final"] = "final"


class TrainingConfig(BaseModel):
    episodes: Optional[int] = Field(default=None, ge=0)
    seed: int = Field(default=22, ge=0, le=4294967295)
    determinism: Literal["seeded", "strict"] = "seeded"
    steps_between_training_updates: int = Field(default=1, ge=1)
    target_update_interval: int = Field(default=0, ge=0)


class TopologyConfig(BaseModel):
    num_agents: Optional[int] = None
    observation_dimensions: Optional[List[int]] = None
    action_dimensions: Optional[List[int]] = None
    action_space: Optional[Any] = None


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
                raise ValueError(
                    f"execution.deucalion.datasets must be relative paths, got: {path!r}"
                )
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
    verification_samples: int = Field(default=32, ge=2, le=1024)
    verification_atol: float = Field(default=1e-5, gt=0)
    verification_rtol: float = Field(default=1e-4, gt=0)
    require_deployable: bool = False
    bundle_version: Optional[str] = Field(
        default=None, description="Bundle version published in manifest metadata"
    )
    description: Optional[str] = Field(
        default=None, description="Bundle description published in manifest metadata"
    )
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


class ExperimentProtocolConfig(BaseModel):
    """Immutable provenance for train/development/confirmation separation."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["ti_marl_experiment_protocol_v1"] = "ti_marl_experiment_protocol_v1"
    protocol_id: str = Field(min_length=1)
    phase: Literal["train", "development", "confirmation"]
    role: Literal["candidate", "reference"] = "candidate"
    data_split: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    paired_reference_id: Optional[str] = Field(default=None, min_length=1)
    selection_rules_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    selection_record_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    selected_checkpoint_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_phase_evidence(self) -> "ExperimentProtocolConfig":
        if self.phase == "development" and self.selection_rules_sha256 is None:
            raise ValueError("development phase requires the pre-frozen selection_rules_sha256")
        if self.phase == "confirmation" and self.selection_record_sha256 is None:
            raise ValueError("confirmation phase requires selection_record_sha256")
        if (
            self.phase == "confirmation"
            and self.role == "candidate"
            and self.selected_checkpoint_sha256 is None
        ):
            raise ValueError("confirmation candidates require selected_checkpoint_sha256")
        if self.role == "candidate" and self.phase != "train" and not self.paired_reference_id:
            raise ValueError("evaluated candidates require paired_reference_id")
        return self
