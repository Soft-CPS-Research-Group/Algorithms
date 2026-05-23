"""Configuration schema definitions and helpers."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

# Imported here to avoid circular imports — registry imports from agents,
# agents do not import from config_schema.
from algorithms.registry import ENCODED_OBSERVATION_ALGORITHMS


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
    stall_watchdog_enabled: bool = Field(
        default=False,
        description="Arm a faulthandler watchdog around wrapper phases to diagnose stalled jobs",
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
        description="Optional path for stall watchdog stack dumps; defaults to the run log directory",
    )
    stall_watchdog_context_interval_steps: int = Field(
        default=1,
        ge=1,
        description="Write stall watchdog context every N step_start phases to reduce remote I/O",
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
    final_episode_only: bool = False
    kpis_final_episode_only: Optional[bool] = None
    timeseries_final_episode_only: Optional[bool] = None
    include_business_as_usual: bool = True
    export_business_as_usual_timeseries: bool = True
    kpi_round_decimals: Optional[int] = Field(default=None, ge=0)
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
    profile: Literal[
        "minmax_space",
        "maddpg_v1",
        "maddpg_v2_compact",
        "maddpg_v3_operational",
        "maddpg_v3_realtime",
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
    central_agent: bool = False
    interface: Literal["flat", "entity"] = "flat"
    topology_mode: Literal["static", "dynamic"] = "static"
    reward_function: str
    reward_function_kwargs: Dict[str, Any] = Field(default_factory=dict)
    episodes: int = Field(default=1, ge=1)
    deterministic_finish: bool = False
    simulation_start_time_step: Optional[int] = Field(default=None, ge=0)
    simulation_end_time_step: Optional[int] = Field(default=None, ge=0)
    episode_time_steps: Optional[Union[int, List[Tuple[int, int]]]] = None
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
    state_layers: Optional[List[int]] = None
    action_layers: Optional[List[int]] = None
    joint_layers: Optional[List[int]] = None
    head_layers: Optional[List[int]] = None

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("layers must contain at least one hidden dimension")
        if any(layer <= 0 for layer in value):
            raise ValueError("layers must be positive integers")
        return value

    @field_validator("state_layers", "action_layers", "joint_layers", "head_layers")
    @classmethod
    def validate_optional_layers(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if any(layer <= 0 for layer in value):
            raise ValueError("network tower layers must be positive integers")
        return value


class AlgorithmNetworks(BaseModel):
    actor: NetworkConfig
    critic: NetworkConfig


class ReplayBufferConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    capacity: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    priority_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    priority_alpha: Optional[float] = Field(default=None, ge=0.0)
    priority_epsilon: Optional[float] = Field(default=None, gt=0.0)
    priority_mode: Optional[Literal["abs_reward", "negative_reward", "positive_reward"]] = None
    priority_max: Optional[float] = Field(default=None, gt=0.0)
    behavior_action_priority_weight: Optional[float] = Field(default=None, ge=0.0)
    behavior_action_priority_mode: Optional[Literal["positive", "abs"]] = None
    behavior_action_priority_scope: Optional[Literal["all", "ev"]] = None
    observation_event_priority_weight: Optional[float] = Field(default=None, ge=0.0)
    observation_event_priority_mode: Optional[
        Literal["ev_departure_service", "ev_pv_price_peak", "combined"]
    ] = None


class ExplorationParams(BaseModel):
    strategy: str
    params: Dict[str, Any]


class AlgorithmHyperparameters(BaseModel):
    gamma: float = Field(gt=0)
    require_cuda: bool = Field(
        default=False,
        description="If true, MADDPG fails during initialization unless CUDA is available.",
    )


class RuleBasedHyperparameters(BaseModel):
    seed: Optional[int] = None
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
    control_storage: bool = True
    control_evs: bool = True
    control_deferrables: bool = True
    allow_v2g: bool = False
    deferrable_start_action: float = Field(default=1.0, ge=0)
    deferrable_urgency_threshold: float = Field(default=0.75, ge=0)
    deferrable_slack_threshold: float = Field(default=0.25, ge=0)
    deferrable_priority_threshold: float = Field(default=0.5, ge=0)
    deferrable_safety_margin_steps: float = Field(default=1.0, ge=0)
    storage_min_soc: float = Field(default=0.20, ge=0)
    storage_max_soc: float = Field(default=0.90, ge=0)
    storage_target_soc: float = Field(default=0.50, ge=0)
    storage_charge_rate: float = Field(default=0.35, ge=0)
    storage_discharge_rate: float = Field(default=0.35, ge=0)
    price_charge_rate: float = Field(default=0.60, ge=0)
    price_discharge_rate: float = Field(default=0.45, ge=0)
    pv_charge_rate: float = Field(default=0.75, ge=0)
    peak_discharge_rate: float = Field(default=0.65, ge=0)
    storage_price_charge_soc_ceiling: float = Field(default=0.90, ge=0)
    storage_price_discharge_soc_floor: float = Field(default=0.20, ge=0)
    storage_peak_discharge_soc_floor: float = Field(default=0.20, ge=0)
    normal_storage_discharge_import_threshold_kw: float = Field(default=0.25, ge=0)
    storage_discharge_import_threshold_kw: float = Field(default=0.25, ge=0)
    ev_normal_charge_rate: float = Field(default=1.0, ge=0)
    ev_normal_target_soc: float = Field(default=1.0, ge=0)
    ev_price_charge_rate: float = Field(default=0.70, ge=0)
    ev_pv_charge_rate: float = Field(default=0.85, ge=0)
    ev_v2g_discharge_rate: float = Field(default=0.30, ge=0)
    ev_community_charge_rate: float = Field(default=0.85, ge=0)
    community_v2g_discharge_rate: float = Field(default=0.30, ge=0)
    community_storage_charge_rate: float = Field(default=0.75, ge=0)
    community_storage_discharge_rate: float = Field(default=0.65, ge=0)
    community_surplus_charge_soc_ceiling: float = Field(default=0.90, ge=0)
    community_surplus_threshold_kw: float = Field(default=0.25, ge=0)
    community_import_threshold_kw: float = Field(default=7.0, ge=0)
    community_local_price_ratio: float = Field(default=0.8, ge=0)
    community_grid_export_price: float = Field(default=0.0, ge=0)
    pv_surplus_threshold_kw: float = Field(default=0.25, ge=0)
    import_peak_threshold_kw: float = Field(default=7.0, ge=0)
    low_headroom_threshold_kw: float = Field(default=2.0, ge=0)
    ev_v2g_reserve_soc: float = Field(default=0.15, ge=0)
    ev_service_margin_rate: float = Field(default=0.05, ge=0)
    ev_service_floor_rate: float = Field(default=0.25, ge=0)
    ev_service_lookahead_hours: float = Field(default=4.0, ge=0)
    ev_service_target_soc: float = Field(default=0.0, ge=0)
    ev_deadline_buffer_hours: float = Field(default=0.25, ge=0)
    ev_v2g_min_departure_hours: float = Field(default=2.0, ge=0)
    ev_v2g_service_margin_soc: float = Field(default=0.05, ge=0)


class TopologyConfig(BaseModel):
    num_agents: Optional[int] = None
    observation_dimensions: Optional[List[int]] = None
    action_dimensions: Optional[List[int]] = None
    action_space: Optional[Any] = None


class CCHyperparameters(BaseModel):
    num_steps:           int   = Field(default=2048,  gt=0)
    lr:                  float = Field(default=3e-4,  gt=0)
    gamma:               float = Field(default=0.99,  gt=0, lt=1)
    gae_lambda:          float = Field(default=0.95,  gt=0, le=1)
    num_epochs:          int   = Field(default=10,    gt=0)
    mini_batch_size:     int   = Field(default=64,    gt=0)
    clip_coef:           float = Field(default=0.2,   gt=0)
    vf_coef:             float = Field(default=0.5,   gt=0)
    ent_coef:            float = Field(default=0.01,  ge=0)
    max_grad_norm:       float = Field(default=0.5,   gt=0)
    target_kl:           Optional[float] = Field(default=0.02, gt=0)
    obs_dim:             int   = Field(default=9,     gt=0)
    cc_action_interval:  int   = Field(default=1,     gt=0)
    output_mode:         Literal["actions", "signal"] = "actions"


class BuildingAgentHyperparameters(BaseModel):
    gamma:                    float = Field(default=0.99,  ge=0, le=1)
    gae_lambda:               float = Field(default=0.95,  ge=0, le=1)
    num_epochs:               int   = Field(default=10,    ge=1)
    mini_batch_size:          int   = Field(default=64,    ge=1)
    clip_coef:                float = Field(default=0.2,   gt=0)
    vf_coef:                  float = Field(default=0.5,   ge=0)
    ent_coef:                 float = Field(default=0.01,  ge=0)
    max_grad_norm:            float = Field(default=0.5,   gt=0)
    target_kl:                Optional[float] = Field(default=0.02, gt=0)
    lr:                       float = Field(default=3e-4,  gt=0)
    obs_dim:                  int   = Field(default=0,     ge=0)   # 0 = auto from env
    action_dim:               int   = Field(default=0,     ge=0)   # 0 = auto from env
    num_steps:                int   = Field(default=2048,  ge=1)
    hidden_dims:              List[int] = Field(default=[64, 64])
    building_cost_weight:     float = Field(default=1.0,   ge=0)
    community_import_weight:  float = Field(default=0.3,   ge=0)
    constraint_penalty_weight: float = Field(default=0.5,  ge=0)


class CommunityCoordinatorAlgorithmConfig(BaseModel):
    algorithm: Literal["CommunityCoordinator"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    frozen: bool = False
    hyperparameters: CCHyperparameters = CCHyperparameters()


class BuildingAgentStageConfig(BaseModel):
    """Pipeline stage describing a BuildingAgent (per-building PPO worker)."""

    algorithm: Literal["BuildingAgent"]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    hyperparameters: BuildingAgentHyperparameters = BuildingAgentHyperparameters()
    networks: Optional[Any] = None
    replay_buffer: Optional[Any] = None
    exploration: Optional[Any] = None


class ActorCriticAlgorithmConfig(BaseModel):
    algorithm: Literal["MADDPG", "MATD3", "MASAC", "IPPO", "MAPPO", "HAPPO"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    frozen: bool = False
    hyperparameters: AlgorithmHyperparameters
    networks: AlgorithmNetworks
    replay_buffer: ReplayBufferConfig
    exploration: ExplorationParams


class RuleBasedAlgorithmConfig(BaseModel):
    algorithm: Literal[
        "RuleBasedPolicy",
        "RandomPolicy",
        "NormalPolicy",
        "NormalNoBatteryPolicy",
        "RBCBasicPolicy",
        "RBCCommunityPolicy",
        "RBCSmartPolicy",
    ]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    hyperparameters: RuleBasedHyperparameters = RuleBasedHyperparameters()
    networks: Optional[AlgorithmNetworks] = None
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None


class SingleAgentRLStageConfig(BaseModel):
    """Pipeline stage placeholder for SingleAgentRL (no runtime impl yet)."""

    algorithm: Literal["SingleAgentRL"]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    hyperparameters: AlgorithmHyperparameters
    policy: Optional[str] = Field(default=None, description="Identifier for the policy architecture")
    replay_buffer: Optional[ReplayBufferConfig] = None
    exploration: Optional[ExplorationParams] = None

    @model_validator(mode="after")
    def reject_placeholder(self) -> "SingleAgentRLStageConfig":
        raise ValueError(
            "Algorithm 'SingleAgentRL' is a schema placeholder and has no runtime "
            "implementation yet. Use one of: MADDPG, MATD3, MASAC, IPPO, MAPPO, HAPPO, "
            "RuleBasedPolicy, RBCBasicPolicy, RBCSmartPolicy, RandomPolicy, "
            "NormalPolicy, NormalNoBatteryPolicy."
        )
        return self  # unreachable; satisfies type checker


PipelineStageConfig = Union[
    BuildingAgentStageConfig,
    CommunityCoordinatorAlgorithmConfig,
    ActorCriticAlgorithmConfig,
    RuleBasedAlgorithmConfig,
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
        stage_names = {stage.algorithm for stage in self.pipeline}
        conflicting = stage_names & ENCODED_OBSERVATION_ALGORITHMS
        if (
            conflicting
            and self.simulator.interface == "entity"
            and self.simulator.topology_mode == "dynamic"
        ):
            raise ValueError(
                f"Pipeline stages {sorted(conflicting)} do not support simulator.interface='entity' "
                "with simulator.topology_mode='dynamic'."
            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary using original key names (aliases)."""
        payload = self.model_dump(by_alias=True)
        networks = payload.get("algorithm", {}).get("networks")
        if isinstance(networks, dict):
            for network in networks.values():
                if not isinstance(network, dict):
                    continue
                for key in ("state_layers", "action_layers", "joint_layers"):
                    if network.get(key) is None:
                        network.pop(key, None)
        return payload


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
