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
    def validate_stage_checkpoint_local_paths(
        cls, value: Dict[int, str]
    ) -> Dict[int, str]:
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
        if not self.stage_checkpoint_local_paths:
            return self
        if not self.resume_training:
            raise ValueError(
                "checkpointing.resume_training must be true when "
                "stage_checkpoint_local_paths is configured"
            )
        if self.checkpoint_run_id or self.checkpoint_local_path or self.use_best_checkpoint_artifact:
            raise ValueError(
                "checkpointing.stage_checkpoint_local_paths cannot be combined with "
                "checkpoint_run_id, checkpoint_local_path, or use_best_checkpoint_artifact"
            )
        return self


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
    building_ids: Optional[List[str]] = None
    electrical_service_overrides_path: Optional[str] = None
    central_agent: bool = False
    interface: Literal["flat", "entity"] = "flat"
    topology_mode: Literal["static", "dynamic"] = "static"
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
    behavior_action_stratified_sampling: Optional[bool] = None
    behavior_action_positive_threshold: Optional[float] = Field(default=None, ge=0.0)
    observation_event_priority_weight: Optional[float] = Field(default=None, ge=0.0)
    observation_event_priority_mode: Optional[
        Literal["ev_departure_service", "ev_pv_price_peak", "combined"]
    ] = None

    @model_validator(mode="after")
    def validate_behavior_action_stratified_sampling(self) -> "ReplayBufferConfig":
        if (
            self.behavior_action_stratified_sampling
            and self.behavior_action_priority_scope != "ev"
        ):
            raise ValueError(
                "behavior_action_stratified_sampling requires "
                "behavior_action_priority_scope='ev'"
            )
        return self


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
    schedule_path: Optional[str] = None
    repeat_schedule_for_training: bool = False
    local_action_safety_enabled: bool = True
    local_action_safety_fail_on_infeasible: bool = False
    local_action_safety_protect_ev_minimum: bool = True
    local_action_safety_ev_minimum_mode: Literal[
        "average", "deadline_feasible"
    ] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = True
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0)


class TopologyConfig(BaseModel):
    num_agents: Optional[int] = None
    observation_dimensions: Optional[List[int]] = None
    action_dimensions: Optional[List[int]] = None
    action_space: Optional[Any] = None


class ExperimentalPPOHyperparameters(BaseModel):
    """Shared schema for experimental hierarchical PPO agents.

    These agents are still evolving, so unknown hyperparameters are preserved
    instead of rejected. Core numeric fields are still checked to catch obvious
    template errors.
    """

    model_config = ConfigDict(extra="allow")

    num_steps: int = Field(default=2048, gt=0)
    lr: float = Field(default=3.0e-4, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    num_epochs: int = Field(default=10, ge=1)
    mini_batch_size: int = Field(default=64, ge=1)
    clip_coef: float = Field(default=0.2, gt=0)
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    target_kl: Optional[float] = Field(default=0.02, gt=0)
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 128])


class CommunityCoordinatorHyperparameters(ExperimentalPPOHyperparameters):
    output_mode: Literal["actions", "signal"] = "actions"
    c_dim: int = Field(default=12, gt=0)
    b_dim: int = Field(default=7, gt=0)
    num_buildings: int = Field(default=17, gt=0)
    cc_action_interval: int = Field(default=1, gt=0)
    net_weight: float = Field(default=0.01, ge=0)


class CCLevel1Hyperparameters(ExperimentalPPOHyperparameters):
    # Phase-1 market maker: emits a global price multiplier.
    c_dim: int = Field(default=17, gt=0)                # cc_level1 encoding width
    cc_action_interval: int = Field(default=4, gt=0)    # 4 × 15min = hourly
    price_min: float = Field(default=0.5, gt=0)         # min price multiplier
    price_max: float = Field(default=1.5, gt=0)         # max price multiplier
    initial_log_std: float = Field(default=0.0, ge=-5.0, le=1.0)
    reference_multiplier: Optional[float] = None
    policy_residual_scale: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_price_range(self) -> "CCLevel1Hyperparameters":
        if self.price_max <= self.price_min:
            raise ValueError("CCLevel1 price_max must be greater than price_min")
        if (
            self.reference_multiplier is not None
            and not self.price_min <= self.reference_multiplier <= self.price_max
        ):
            raise ValueError(
                "CCLevel1 reference_multiplier must lie within the configured price range"
            )
        return self


class CCLevel2Hyperparameters(ExperimentalPPOHyperparameters):
    """Per-building market maker with an auditable reference policy."""

    c_dim: int = Field(default=118, gt=0)
    num_buildings: int = Field(default=17, gt=0)
    cc_action_interval: int = Field(default=4, gt=0)
    price_min: float = Field(default=0.5, gt=0)
    price_max: float = Field(default=1.5, gt=0)
    initial_log_std: float = Field(default=-2.5, ge=-5.0, le=1.0)
    reference_multipliers: Optional[List[float]] = None
    policy_residual_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    policy_parameterization: Literal[
        "absolute_blend",
        "centered_residual",
        "sparse_centered_residual",
        "causal_active_only",
    ] = "absolute_blend"
    policy_deadband: float = Field(default=0.0, ge=0.0, lt=1.0)
    causal_initial_multiplier: float = Field(default=0.90, gt=0.0, le=1.0)
    causal_initial_multipliers: Optional[List[float]] = None
    causal_residual_scale: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    causal_use_physical_context: bool = False
    separate_value_encoder: bool = False
    include_community_headroom: bool = False
    bc_pretrain_enabled: bool = False
    bc_collect_steps: int = Field(default=336, ge=1)
    bc_train_steps: int = Field(default=2000, ge=1)
    bc_train_chunk_steps: int = Field(default=256, ge=1)
    bc_max_torch_threads: int = Field(default=1, ge=1)
    bc_progress_interval: int = Field(default=250, ge=1)
    bc_lr: float = Field(default=1.0e-3, gt=0)
    bc_use_physical_teacher_context: bool = False
    bc_teacher_mode: Literal[
        "continuous_score",
        "cheap_and_export",
        "oracle_storage_schedule",
    ] = (
        "continuous_score"
    )
    bc_collection_policy: Literal["teacher_rollout", "neutral_label_only"] = (
        "teacher_rollout"
    )
    bc_discount_multiplier: float = Field(default=0.90, gt=0.0, le=1.0)
    bc_export_activation_kw: float = Field(default=1.0e-9, ge=0.0)
    bc_oracle_schedule_path: Optional[str] = None
    bc_oracle_schedule_step_offset: int = Field(default=0, ge=0)
    bc_oracle_deadband_kw: float = Field(default=0.02, ge=0.0)
    bc_oracle_power_scale_kw: float = Field(default=1.0, gt=0.0)
    bc_anchor_weight: float = Field(default=0.0, ge=0.0)
    bc_anchor_min_weight: float = Field(default=0.0, ge=0.0)
    bc_anchor_decay_updates: int = Field(default=0, ge=0)
    bc_anchor_batch_size: int = Field(default=64, ge=1)
    w_factor: float = Field(default=0.3, ge=0)
    w_smoothness: float = Field(default=0.02, ge=0)
    credit_assignment: Literal["global", "member_decomposed"] = "global"
    team_reward_mix: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_normalization: Literal["running_zscore", "none"] = "running_zscore"
    neutral_baseline_enabled: bool = False
    neutral_warmup_episodes: int = Field(default=0, ge=0)
    counterfactual_baseline_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    training_episodes_per_validation: int = Field(default=0, ge=0)
    rollback_rejected_validation: bool = False
    restore_best_policy_for_deterministic: bool = False
    best_policy_min_improvement: float = Field(default=0.0, ge=0.0)
    train_log_std: bool = True

    @model_validator(mode="after")
    def validate_price_contract(self) -> "CCLevel2Hyperparameters":
        if self.price_max <= self.price_min:
            raise ValueError("CCLevel2 price_max must be greater than price_min")
        values = self.reference_multipliers
        if values is not None:
            if len(values) != self.num_buildings:
                raise ValueError(
                    "CCLevel2 reference_multipliers length must equal num_buildings"
                )
            if any(value < self.price_min or value > self.price_max for value in values):
                raise ValueError(
                    "CCLevel2 reference_multipliers must lie within the configured price range"
                )
        if self.bc_discount_multiplier < self.price_min:
            raise ValueError(
                "CCLevel2 bc_discount_multiplier must not be below price_min"
            )
        if (
            self.policy_parameterization != "sparse_centered_residual"
            and self.policy_deadband > 0.0
        ):
            raise ValueError(
                "CCLevel2 policy_deadband is only supported by "
                "sparse_centered_residual"
            )
        if (
            self.neutral_baseline_enabled
            and self.bc_pretrain_enabled
            and self.bc_collection_policy != "neutral_label_only"
        ):
            raise ValueError(
                "CCLevel2 neutral baseline collection and BC pretraining are "
                "compatible only with bc_collection_policy='neutral_label_only'"
            )
        if (
            self.bc_pretrain_enabled
            and self.bc_collection_policy == "neutral_label_only"
            and self.neutral_warmup_episodes < 1
        ):
            raise ValueError(
                "CCLevel2 neutral-label BC collection requires at least one "
                "neutral warm-up episode"
            )
        if self.bc_teacher_mode == "oracle_storage_schedule":
            if not self.bc_pretrain_enabled:
                raise ValueError(
                    "CCLevel2 oracle_storage_schedule requires bc_pretrain_enabled"
                )
            if not str(self.bc_oracle_schedule_path or "").strip():
                raise ValueError(
                    "CCLevel2 oracle_storage_schedule requires "
                    "bc_oracle_schedule_path"
                )
        if self.bc_anchor_min_weight > self.bc_anchor_weight:
            raise ValueError(
                "CCLevel2 bc_anchor_min_weight must not exceed bc_anchor_weight"
            )
        if self.neutral_warmup_episodes > 0 and not self.neutral_baseline_enabled:
            raise ValueError(
                "CCLevel2 neutral warm-up episodes require "
                "neutral_baseline_enabled"
            )
        if (
            self.training_episodes_per_validation > 0
            and not self.neutral_baseline_enabled
        ):
            raise ValueError(
                "CCLevel2 validation episodes require neutral_baseline_enabled"
            )
        if (
            self.restore_best_policy_for_deterministic
            and self.training_episodes_per_validation <= 0
        ):
            raise ValueError(
                "CCLevel2 best-policy restore requires deterministic validation "
                "episodes"
            )
        if (
            self.rollback_rejected_validation
            and self.training_episodes_per_validation <= 0
        ):
            raise ValueError(
                "CCLevel2 validation rollback requires deterministic "
                "validation episodes"
            )
        if self.policy_parameterization == "causal_active_only":
            if self.price_max > 1.0:
                raise ValueError(
                    "CCLevel2 causal_active_only requires price_max <= 1.0"
                )
            if not self.price_min <= self.causal_initial_multiplier <= self.price_max:
                raise ValueError(
                    "CCLevel2 causal_initial_multiplier must lie within the price range"
                )
            causal_values = self.causal_initial_multipliers
            if causal_values is not None:
                if len(causal_values) != self.num_buildings:
                    raise ValueError(
                        "CCLevel2 causal_initial_multipliers length must equal "
                        "num_buildings"
                    )
                if any(
                    value < self.price_min or value > self.price_max
                    for value in causal_values
                ):
                    raise ValueError(
                        "CCLevel2 causal_initial_multipliers must lie within "
                        "the price range"
                    )
            if self.bc_pretrain_enabled:
                raise ValueError(
                    "CCLevel2 causal_active_only does not support BC pretraining"
                )
        return self


class FixedPriceScheduleEntry(BaseModel):
    start_step: int = Field(ge=0)
    multiplier: float = Field(gt=0)


class FixedPriceVectorScheduleEntry(BaseModel):
    start_step: int = Field(ge=0)
    multipliers: List[float]

    @field_validator("multipliers")
    @classmethod
    def validate_multipliers(cls, values: List[float]) -> List[float]:
        if not values:
            raise ValueError(
                "FixedPriceSignal vector schedule multipliers must not be empty"
            )
        if any(value <= 0 for value in values):
            raise ValueError(
                "FixedPriceSignal vector schedule multipliers must all be positive"
            )
        return values


class FixedPriceSignalHyperparameters(BaseModel):
    multiplier: float = Field(default=1.0, gt=0)
    multipliers: Optional[List[float]] = None
    schedule: Optional[List[FixedPriceScheduleEntry]] = None
    vector_schedule: Optional[List[FixedPriceVectorScheduleEntry]] = None

    @field_validator("multipliers")
    @classmethod
    def validate_multiplier_vector(cls, values: Optional[List[float]]) -> Optional[List[float]]:
        if values is None:
            return None
        if not values:
            raise ValueError("FixedPriceSignal multipliers must not be empty")
        if any(value <= 0 for value in values):
            raise ValueError("FixedPriceSignal multipliers must all be positive")
        return values

    @model_validator(mode="after")
    def validate_schedule(self) -> "FixedPriceSignalHyperparameters":
        configured_modes = sum(
            value is not None
            for value in (self.multipliers, self.schedule, self.vector_schedule)
        )
        if configured_modes > 1:
            raise ValueError(
                "FixedPriceSignal multipliers, schedule and vector_schedule "
                "are mutually exclusive"
            )
        configured_schedule = (
            self.schedule if self.schedule is not None else self.vector_schedule
        )
        if configured_schedule is None:
            return self
        if not configured_schedule:
            raise ValueError("FixedPriceSignal schedule must not be empty")
        starts = [entry.start_step for entry in configured_schedule]
        if starts[0] != 0:
            raise ValueError("FixedPriceSignal schedule must start at step 0")
        if starts != sorted(set(starts)):
            raise ValueError(
                "FixedPriceSignal schedule start_step values must be strictly increasing"
            )
        if self.vector_schedule is not None:
            widths = {len(entry.multipliers) for entry in self.vector_schedule}
            if len(widths) != 1:
                raise ValueError(
                    "FixedPriceSignal vector_schedule entries must have equal widths"
                )
        return self


class CausalPriceSignalHyperparameters(BaseModel):
    neutral_multiplier: float = Field(default=1.0, gt=0)
    discount_multiplier: float = Field(default=0.95, gt=0)
    discount_multipliers: Optional[List[float]] = None
    vector_min_multiplier: float = Field(default=0.5, gt=0)
    vector_max_multiplier: float = Field(default=1.3, gt=0)
    cc_action_interval: int = Field(default=4, gt=0)
    community_export_threshold_kw: float = Field(default=1.0e-9, ge=0)
    forecast_mean_margin: float = Field(default=0.20, ge=0)
    forecast_min_margin: float = Field(default=0.10, ge=0)
    spread_floor_ratio: float = Field(default=0.05, gt=0)

    @model_validator(mode="after")
    def validate_discount_contract(self) -> "CausalPriceSignalHyperparameters":
        if self.discount_multiplier >= self.neutral_multiplier:
            raise ValueError(
                "CausalPriceSignal discount_multiplier must be below neutral_multiplier"
            )
        if self.vector_max_multiplier <= self.vector_min_multiplier:
            raise ValueError(
                "CausalPriceSignal vector_max_multiplier must be greater than "
                "vector_min_multiplier"
            )
        if self.discount_multipliers is not None:
            if not self.discount_multipliers:
                raise ValueError(
                    "CausalPriceSignal discount_multipliers must not be empty"
                )
            if any(
                value < self.vector_min_multiplier
                or value > self.vector_max_multiplier
                for value in self.discount_multipliers
            ):
                raise ValueError(
                    "CausalPriceSignal discount_multipliers must lie within "
                    "the configured vector multiplier range"
                )
        return self


class BuildingAgentHyperparameters(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    hidden_dims:              List[int] = Field(default_factory=lambda: [64, 64])
    building_cost_weight:     float = Field(default=1.0,   ge=0)
    community_import_weight:  float = Field(default=0.3,   ge=0)
    constraint_penalty_weight: float = Field(default=0.5,  ge=0)


class CommunityCoordinatorAlgorithmConfig(BaseModel):
    algorithm: Literal["CommunityCoordinator"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    frozen: bool = False
    hyperparameters: CommunityCoordinatorHyperparameters = Field(
        default_factory=CommunityCoordinatorHyperparameters
    )


class CCLevel1AlgorithmConfig(BaseModel):
    algorithm: Literal["CCLevel1"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    frozen: bool = False
    hyperparameters: CCLevel1Hyperparameters = Field(default_factory=CCLevel1Hyperparameters)


class FixedPriceSignalAlgorithmConfig(BaseModel):
    algorithm: Literal["FixedPriceSignal"]
    count: Literal[1] = 1
    frozen: bool = True
    hyperparameters: FixedPriceSignalHyperparameters = Field(
        default_factory=FixedPriceSignalHyperparameters
    )


class CausalPriceSignalAlgorithmConfig(BaseModel):
    algorithm: Literal["CausalPriceSignal"]
    count: Literal[1] = 1
    frozen: Literal[True] = True
    hyperparameters: CausalPriceSignalHyperparameters = Field(
        default_factory=CausalPriceSignalHyperparameters
    )


class CCLevel2AlgorithmConfig(BaseModel):
    algorithm: Literal["CCLevel2"]
    count: int = Field(default=1, ge=1, description="Number of identical agents at this level")
    frozen: bool = False
    hyperparameters: CCLevel2Hyperparameters = Field(default_factory=CCLevel2Hyperparameters)


class BuildingAgentStageConfig(BaseModel):
    """Pipeline stage describing a BuildingAgent (per-building PPO worker)."""

    algorithm: Literal["BuildingAgent"]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    hyperparameters: BuildingAgentHyperparameters = Field(default_factory=BuildingAgentHyperparameters)
    networks: Optional[Any] = None
    replay_buffer: Optional[Any] = None
    exploration: Optional[Any] = None


class ActorCriticAlgorithmConfig(BaseModel):
    algorithm: Literal["MADDPG", "MATD3", "MASAC", "PPO", "TD3", "IPPO", "MAPPO", "HAPPO"]
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
        "RBCSmartLocalPolicy",
        "RBCSmartPolicy",
        "SignalAwareRBC",
        "SignalAwareRBCSmartLocal",
        "FixedServiceOracleReplayPolicy",
        "TotalHomeOracleReplayPolicy",
        "TotalOracleReplayPolicy",
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
            "RuleBasedPolicy, RBCBasicPolicy, RBCSmartLocalPolicy, RBCSmartPolicy, SignalAwareRBC, "
            "SignalAwareRBCSmartLocal, "
            "RandomPolicy, NormalPolicy, NormalNoBatteryPolicy."
        )
        return self  # unreachable; satisfies type checker


class TransformerPPOTransformerConfig(BaseModel):
    d_model: int = Field(ge=1)
    nhead: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dim_feedforward: int = Field(ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_deterministic_representation(self) -> "TransformerPPOTransformerConfig":
        if self.dropout != 0.0:
            raise ValueError(
                "AgentTransformerPPO requires transformer.dropout=0.0 because PPO old/new probability ratios must use the same representation."
            )
        return self


class TransformerPPOHyperparameters(BaseModel):
    require_cuda: bool = Field(
        default=False,
        description="If true, AgentTransformerPPO fails during initialization unless CUDA is available.",
    )
    learning_rate: float = Field(gt=0)
    gamma: float = Field(gt=0, le=1.0)
    gae_lambda: float = Field(gt=0, le=1.0)
    clip_eps: float = Field(gt=0)
    ppo_epochs: int = Field(ge=1)
    minibatch_size: int = Field(ge=1)
    entropy_coeff: float = Field(ge=0)
    value_coeff: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    actor_log_std_init: float = -0.5
    local_action_safety_enabled: bool = False
    local_action_safety_fail_on_infeasible: bool = False
    local_action_safety_protect_ev_minimum: bool = True
    local_action_safety_ev_minimum_mode: Literal[
        "average", "deadline_feasible"
    ] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = False
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0)


class TransformerPPOBehaviorCloningTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["RBCSmartPolicy"] = "RBCSmartPolicy"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TransformerPPOBehaviorCloningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    demonstration_episodes: int = Field(default=1, ge=0)
    max_samples_per_building: int = Field(default=4096, ge=1)
    pretraining_epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=64, ge=1)
    weight: float = Field(default=0.0, ge=0.0)
    min_weight: float = Field(default=0.0, ge=0.0)
    decay_start_step: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=0, ge=0)
    ev_multiplier: float = Field(default=1.0, ge=0.0)
    storage_multiplier: float = Field(default=1.0, ge=0.0)
    teacher: TransformerPPOBehaviorCloningTeacherConfig = Field(
        default_factory=TransformerPPOBehaviorCloningTeacherConfig
    )

    @model_validator(mode="after")
    def require_demonstration_episode_when_enabled(self) -> "TransformerPPOBehaviorCloningConfig":
        if self.enabled and self.demonstration_episodes < 1:
            raise ValueError(
                "behavior_cloning.demonstration_episodes must be at least 1 when behavior_cloning.enabled=true."
            )
        if self.min_weight > self.weight:
            raise ValueError(
                "behavior_cloning.min_weight must be less than or equal to behavior_cloning.weight."
            )
        return self


class TransformerPPOStageConfig(BaseModel):
    algorithm: Literal["AgentTransformerPPO"]
    count: int = 1
    frozen: bool = False
    tokenizer_config_path: str = Field(min_length=1)
    transformer: TransformerPPOTransformerConfig
    hyperparameters: TransformerPPOHyperparameters
    behavior_cloning: Optional[TransformerPPOBehaviorCloningConfig] = None

    @field_validator("count")
    @classmethod
    def require_single_controller(cls, value: int) -> int:
        if value != 1:
            raise ValueError("AgentTransformerPPO pipeline stages require count=1")
        return value


class TIMARLBackboneConfig(BaseModel):
    name: Literal["ppo", "mappo"] = "mappo"


class TIMARLActorConfig(BaseModel):
    d_model: int = Field(default=128, ge=16)
    attention_heads: int = Field(default=4, ge=1)
    relation_layers: int = Field(default=2, ge=1)
    group_context_kind: Literal["local", "action_conditioned"] = "local"
    deterministic_mode_strategy: Literal["argmax", "expected_signed"] = "argmax"
    deterministic_mode_strategy_by_group_type: Dict[
        str, Literal["argmax", "expected_signed"]
    ] = Field(default_factory=dict)
    deterministic_expected_signed_gain_by_group_type: Dict[str, float] = Field(
        default_factory=dict
    )
    deterministic_expected_signed_deadband_by_group_type: Dict[str, float] = Field(
        default_factory=dict
    )
    deterministic_non_idle_logit_margin_by_group_type: Dict[str, float] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_attention_width(self) -> "TIMARLActorConfig":
        if self.d_model % self.attention_heads != 0:
            raise ValueError("TIMARL actor.d_model must be divisible by attention_heads")
        if any(
            gain < 0.0
            for gain in self.deterministic_expected_signed_gain_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic expected-signed gains must be "
                "non-negative"
            )
        if any(
            margin < 0.0
            for margin in self.deterministic_non_idle_logit_margin_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic non-idle logit margins must be "
                "non-negative"
            )
        if any(
            deadband < 0.0 or deadband > 1.0
            for deadband in self.deterministic_expected_signed_deadband_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic expected-signed deadbands must be "
                "between zero and one"
            )
        return self


class TransformerMATD3TransformerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(gt=0)
    nhead: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def require_compatible_attention_width(
        self,
    ) -> "TransformerMATD3TransformerConfig":
        if self.d_model % self.nhead != 0:
            raise ValueError("transformer.d_model must be divisible by nhead")
        return self


class TransformerMATD3Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_cuda: bool = False
    learning_rate: float = Field(gt=0.0)
    gamma: float = Field(gt=0.0, le=1.0)
    tau: float = Field(gt=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    buffer_capacity: int = Field(gt=0)
    max_grad_norm: float = Field(gt=0.0)
    n_step_returns: int = Field(default=1, gt=0)
    n_step_gamma: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    critic_team_reward_mix: float = Field(default=0.0, ge=0.0, le=1.0)
    critic_target_clip_abs: float = Field(default=0.0, ge=0.0)
    reward_normalization_enabled: bool = False
    reward_normalization_clip: float = Field(default=10.0, gt=0.0)
    target_policy_smoothing: bool = True
    target_policy_noise: float = Field(ge=0.0)
    target_policy_noise_clip: float = Field(ge=0.0)
    actor_update_interval: int = Field(default=2, gt=0)
    sigma: float = Field(ge=0.0)
    sigma_decay: float = Field(gt=0.0, le=1.0)
    min_sigma: float = Field(ge=0.0)
    bias: float
    noise_clip: Optional[float] = Field(default=None, ge=0.0)
    random_exploration_steps: int = Field(default=0, ge=0)
    end_initial_exploration_time_step: int = Field(default=0, ge=0)
    storage_exploration_noise_multiplier: float = Field(default=1.0, ge=0.0)
    ev_negative_exploration_noise_multiplier: float = Field(default=1.0, ge=0.0)
    deferrable_trigger_threshold: float = 0.0
    deferrable_on_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_policy_enabled: bool = False
    warm_start_policy_name: Optional[str] = None
    warm_start_policy_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    residual_action_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    residual_action_final_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    residual_action_scale_start_step: int = Field(default=0, ge=0)
    residual_action_scale_growth_steps: int = Field(default=0, ge=0)
    residual_storage_action_scale_multiplier: float = Field(default=1.0, ge=0.0)
    residual_ev_action_scale_multiplier: float = Field(default=1.0, ge=0.0)
    residual_deferrable_action_scale_multiplier: float = Field(default=1.0, ge=0.0)
    critic_action_input_mode: Literal["final"] = "final"
    residual_policy_runtime_only_export: bool = False
    local_action_safety_enabled: bool = False
    local_action_safety_fail_on_infeasible: bool = False
    local_action_safety_protect_ev_minimum: bool = True
    local_action_safety_ev_minimum_mode: Literal[
        "average", "deadline_feasible"
    ] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = False
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0.0)
    local_action_safety_runtime_only_export: bool = False
    local_price_conditioning_enabled: bool = False
    local_price_forecast_mode: Literal[
        "real_unmodified", "aligned_vector", "persist_current"
    ] = "real_unmodified"
    local_price_conditioning_runtime_only_export: bool = False

    @model_validator(mode="after")
    def validate_matd3_relationships(self) -> "TransformerMATD3Hyperparameters":
        if self.buffer_capacity < self.batch_size:
            raise ValueError(
                "buffer_capacity must be greater than or equal to batch_size"
            )
        if self.min_sigma > self.sigma:
            raise ValueError("min_sigma must be less than or equal to sigma")
        if self.residual_policy_enabled and not str(
            self.warm_start_policy_name or ""
        ).strip():
            raise ValueError(
                "residual_policy_enabled=true requires warm_start_policy_name"
            )
        if self.n_step_gamma is None:
            self.n_step_gamma = self.gamma
        return self


class TransformerMATD3ReplayBehaviorCloningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    teacher: Literal["warm_start", "replay_action", "external"] = "warm_start"
    weight: float = Field(default=0.0, ge=0.0)
    min_weight: float = Field(default=0.0, ge=0.0)
    decay_start_step: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=0, ge=0)
    ev_multiplier: float = Field(default=1.0, ge=0.0)
    storage_multiplier: float = Field(default=1.0, ge=0.0)
    deferrable_multiplier: float = Field(default=1.0, ge=0.0)
    extra_updates: Optional[int] = Field(default=None, ge=0)
    extra_update_start_step: int = Field(default=0, ge=0)
    extra_update_end_step: int = Field(default=0, ge=0)
    clip_target_to_residual_authority: bool = False
    offline_pretrain_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_weight_floor(
        self,
    ) -> "TransformerMATD3ReplayBehaviorCloningConfig":
        if self.min_weight > self.weight:
            raise ValueError("replay_based.min_weight must not exceed weight")
        if self.extra_updates is None:
            self.extra_updates = int(
                bool(self.extra_update_start_step or self.extra_update_end_step)
            )
        return self


class TIMARLCriticConfig(BaseModel):
    kind: Literal["local", "set"] = "set"


class TIMARLFeasibilityConfig(BaseModel):
    kind: Literal["analytic_projection"] = "analytic_projection"
    enforce_ev_service: bool = True
    ev_service_margin_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    ev_service_strategy: Literal[
        "average",
        "minimum_average",
        "just_in_time",
    ] = "average"
    ev_service_tolerance_ratio: float = Field(default=0.05, ge=0.0, le=0.2)
    headroom_reserve_kw: float = Field(default=0.0, ge=0.0)
    deferrable_service_margin_seconds: float = Field(default=0.0, ge=0.0)


class TIMARLTraceConfig(BaseModel):
    enabled: bool = True
    chunk_size: int = Field(default=256, ge=1)
    snapshot_interval: int = Field(default=256, ge=1)


class TIMARLBehaviorCloningTeacherConfig(BaseModel):
    policy: Literal["RBCSmartPolicy"] = "RBCSmartPolicy"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TransformerMATD3BehaviorCloningTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: Literal["RBCSmartPolicy"] = "RBCSmartPolicy"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TIMARLBehaviorCloningConfig(BaseModel):
    enabled: bool = True
    demonstration_episodes: int = Field(default=1, ge=1)
    max_samples: int = Field(default=4096, ge=1)
    pretraining_epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=3.0e-4, gt=0)
    balance_action_modes: bool = True
    mode_balance_exponent: float = Field(default=0.5, ge=0.0, le=1.0)
    max_mode_weight: float = Field(default=4.0, ge=1.0)
    balanced_loss_kind: Literal[
        "weighted",
        "hierarchical_mode_mean",
    ] = "weighted"
    calibration_epochs: int = Field(default=0, ge=0)
    calibration_learning_rate: Optional[float] = Field(default=None, gt=0)
    teacher: TIMARLBehaviorCloningTeacherConfig = Field(
        default_factory=TIMARLBehaviorCloningTeacherConfig
    )


class TIMARLEVPlanningConfig(BaseModel):
    """Causal auxiliary target for proactive low-level EV scheduling."""

    auxiliary_coeff: float = Field(default=0.0, ge=0.0)
    balance_targets: bool = True
    fraction_coeff: float = Field(default=0.25, ge=0.0)
    replay_capacity_per_reason: int = Field(default=16, ge=0)
    replay_samples_per_reason: int = Field(default=8, ge=0)
    charge_fraction: float = Field(default=0.95, gt=0.0, lt=1.0)
    discharge_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    service_tolerance_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    v2g_service_margin_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    price_tie_tolerance: float = Field(default=1.0e-6, ge=0.0)
    urgency_duty_ratio: float = Field(default=0.85, gt=0.0, le=1.0)
    minimum_price_spread: float = Field(default=0.0, ge=0.0)
    minimum_v2g_price_spread: float = Field(default=0.01, ge=0.0)
    minimum_v2g_departure_hours: float = Field(default=1.0, ge=0.0)


class TIMARLHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["ti_marl_v1"] = "ti_marl_v1"
    typed_interfaces_dir: str = Field(min_length=1)
    interface_polling: bool = False
    simulator_bindings_path: Optional[str] = Field(default=None, min_length=1)
    require_cuda: bool = False
    allow_checkpoint_compiler_migration: bool = False
    require_declared_electrical_service: bool = False
    backbone: TIMARLBackboneConfig = TIMARLBackboneConfig()
    actor: TIMARLActorConfig = TIMARLActorConfig()
    critic: TIMARLCriticConfig = TIMARLCriticConfig()
    feasibility: TIMARLFeasibilityConfig = TIMARLFeasibilityConfig()
    learning_rate: float = Field(default=3.0e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    discount_timebase_seconds: Optional[float] = Field(default=None, gt=0)
    clip_eps: float = Field(default=0.2, gt=0)
    ppo_epochs: int = Field(default=4, ge=1)
    entropy_coeff: float = Field(default=0.01, ge=0)
    entropy_coeff_by_group_type: Dict[str, float] = Field(default_factory=dict)
    advantage_normalization: Literal["global", "per_agent"] = "global"
    policy_credit_assignment: Literal["joint_agent", "typed_group"] = (
        "joint_agent"
    )
    policy_anchor_coeff: float = Field(default=0.0, ge=0)
    policy_anchor_reset_on_resume: bool = False
    exclude_intervened_actions_from_policy_loss: bool = False
    intervention_distillation_coeff: float = Field(default=0.0, ge=0)
    value_coeff: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    target_kl: Optional[float] = Field(default=0.03, gt=0)
    rollout_steps: int = Field(default=256, ge=1)
    normalize_value_targets: bool = True
    value_target_scale_floor: float = Field(default=1.0, gt=0)
    critic_loss: Literal["mse", "huber"] = "huber"
    trace: TIMARLTraceConfig = TIMARLTraceConfig()
    behavior_cloning: Optional[TIMARLBehaviorCloningConfig] = None
    ev_planning: TIMARLEVPlanningConfig = TIMARLEVPlanningConfig()

    @model_validator(mode="after")
    def validate_learning_architecture(self) -> "TIMARLHyperparameters":
        expected = {"ppo": "local", "mappo": "set"}[self.backbone.name]
        if self.critic.kind != expected:
            raise ValueError(
                f"TIMARL backbone.name={self.backbone.name!r} requires "
                f"critic.kind={expected!r}"
            )
        if any(value < 0.0 for value in self.entropy_coeff_by_group_type.values()):
            raise ValueError(
                "TIMARL entropy_coeff_by_group_type values must be non-negative"
            )
        if (
            self.exclude_intervened_actions_from_policy_loss
            and self.policy_credit_assignment != "typed_group"
        ):
            raise ValueError(
                "TIMARL exclude_intervened_actions_from_policy_loss requires "
                "policy_credit_assignment='typed_group'"
            )
        if (
            self.intervention_distillation_coeff > 0.0
            and not self.exclude_intervened_actions_from_policy_loss
        ):
            raise ValueError(
                "TIMARL intervention_distillation_coeff requires "
                "exclude_intervened_actions_from_policy_loss=true"
            )
        return self


class TIMARLStageConfig(BaseModel):
    algorithm: Literal["TIMARL"]
    count: Literal[1] = 1
    frozen: bool = False
    hyperparameters: TIMARLHyperparameters


class TransformerMATD3DemonstrationBehaviorCloningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    demonstration_episodes: int = Field(default=1, ge=0)
    max_samples_per_building: int = Field(default=4096, gt=0)
    pretraining_epochs: int = Field(default=4, gt=0)
    batch_size: int = Field(default=64, gt=0)
    weight: float = Field(default=0.0, ge=0.0)
    min_weight: float = Field(default=0.0, ge=0.0)
    decay_start_step: int = Field(default=0, ge=0)
    decay_steps: int = Field(default=0, ge=0)
    ev_multiplier: float = Field(default=1.0, ge=0.0)
    storage_multiplier: float = Field(default=1.0, ge=0.0)
    teacher: TransformerMATD3BehaviorCloningTeacherConfig = Field(
        default_factory=TransformerMATD3BehaviorCloningTeacherConfig
    )

    @model_validator(mode="after")
    def validate_demonstration_settings(
        self,
    ) -> "TransformerMATD3DemonstrationBehaviorCloningConfig":
        if self.enabled and self.demonstration_episodes < 1:
            raise ValueError(
                "demonstration_based.demonstration_episodes must be at least 1 when enabled"
            )
        if self.min_weight > self.weight:
            raise ValueError("demonstration_based.min_weight must not exceed weight")
        return self


class TransformerMATD3BehaviorCloningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_based: TransformerMATD3ReplayBehaviorCloningConfig = Field(
        default_factory=TransformerMATD3ReplayBehaviorCloningConfig
    )
    demonstration_based: TransformerMATD3DemonstrationBehaviorCloningConfig = Field(
        default_factory=TransformerMATD3DemonstrationBehaviorCloningConfig
    )


class TransformerMATD3StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["AgentTransformerMATD3"]
    count: Literal[1] = 1
    frozen: bool = False
    tokenizer_config_path: str = Field(min_length=1)
    transformer: TransformerMATD3TransformerConfig
    hyperparameters: TransformerMATD3Hyperparameters
    behavior_cloning: TransformerMATD3BehaviorCloningConfig = Field(
        default_factory=TransformerMATD3BehaviorCloningConfig
    )


PipelineStageConfig = Union[
    TIMARLStageConfig,
    BuildingAgentStageConfig,
    CCLevel1AlgorithmConfig,
    CausalPriceSignalAlgorithmConfig,
    FixedPriceSignalAlgorithmConfig,
    CCLevel2AlgorithmConfig,
    CommunityCoordinatorAlgorithmConfig,
    ActorCriticAlgorithmConfig,
    RuleBasedAlgorithmConfig,
    SingleAgentRLStageConfig,
    TransformerPPOStageConfig,
    TransformerMATD3StageConfig,
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


class ExperimentProtocolConfig(BaseModel):
    """Immutable provenance for train/development/confirmation separation."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["ti_marl_experiment_protocol_v1"] = (
        "ti_marl_experiment_protocol_v1"
    )
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
            raise ValueError(
                "development phase requires the pre-frozen selection_rules_sha256"
            )
        if self.phase == "confirmation" and self.selection_record_sha256 is None:
            raise ValueError(
                "confirmation phase requires selection_record_sha256"
            )
        if (
            self.phase == "confirmation"
            and self.role == "candidate"
            and self.selected_checkpoint_sha256 is None
        ):
            raise ValueError(
                "confirmation candidates require selected_checkpoint_sha256"
            )
        if self.role == "candidate" and self.phase != "train" and not self.paired_reference_id:
            raise ValueError(
                "evaluated candidates require paired_reference_id"
            )
        return self


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
    experiment_protocol: Optional[ExperimentProtocolConfig] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_cross_constraints(self) -> "ProjectConfig":
        for index, stage in enumerate(self.pipeline):
            if isinstance(
                stage, (TransformerPPOStageConfig, TransformerMATD3StageConfig)
            ) and index != len(self.pipeline) - 1:
                raise ValueError(
                    f"{stage.algorithm} must be the final pipeline stage because "
                    "it learns from its own executed actions."
                )
            if isinstance(stage, TIMARLStageConfig):
                if len(self.pipeline) != 1:
                    raise ValueError("TIMARL v1 must run as a standalone single-stage pipeline")
                if index != len(self.pipeline) - 1:
                    raise ValueError("TIMARL must be the final pipeline stage")
                if self.simulator.interface != "entity":
                    raise ValueError("TIMARL requires simulator.interface='entity'")
                if self.simulator.central_agent:
                    raise ValueError("TIMARL requires simulator.central_agent=false")

            if isinstance(stage, TransformerMATD3StageConfig):
                if self.simulator.interface != "entity":
                    raise ValueError(
                        "AgentTransformerMATD3 requires simulator.interface='entity'."
                    )
                if (
                    stage.hyperparameters.local_price_conditioning_enabled
                    and self.simulator.entity_encoding.profile != "minmax_space"
                ):
                    raise ValueError(
                        "AgentTransformerMATD3 local price conditioning requires "
                        "simulator.entity_encoding.profile='minmax_space'."
                    )

        stage_checkpoint_paths = self.checkpointing.stage_checkpoint_local_paths
        if stage_checkpoint_paths:
            if len(self.pipeline) < 2:
                raise ValueError(
                    "checkpointing.stage_checkpoint_local_paths requires a multi-stage pipeline"
                )
            invalid_indices = sorted(
                index for index in stage_checkpoint_paths if index >= len(self.pipeline)
            )
            if invalid_indices:
                raise ValueError(
                    "checkpointing.stage_checkpoint_local_paths contains stage indices "
                    f"outside pipeline range 0:{len(self.pipeline) - 1}: {invalid_indices}"
                )

        if self.simulator.interface == "entity" and self.simulator.topology_mode == "dynamic":
            from algorithms.registry import ALGORITHM_REGISTRY
            for stage in self.pipeline:
                agent_cls = ALGORITHM_REGISTRY.get(stage.algorithm)
                if agent_cls is None:
                    continue
                if not bool(getattr(agent_cls, "supports_dynamic_topology", False)):
                    if stage.algorithm == "MADDPG":
                        raise ValueError(
                            "algorithm.name='MADDPG' does not support simulator.interface='entity' "
                            "with simulator.topology_mode='dynamic'."
                        )
                    raise ValueError(
                        f"algorithm={stage.algorithm!r} does not support "
                        "simulator.topology_mode='dynamic' (supports_dynamic_topology=False)."
                    )

        protocol = self.experiment_protocol
        if protocol is not None:
            is_evaluation = protocol.phase in {"development", "confirmation"}
            if is_evaluation:
                if self.simulator.random_seed is None:
                    raise ValueError(
                        "development/confirmation requires explicit simulator.random_seed"
                    )
                if self.simulator.episodes != 1 or not self.simulator.deterministic_finish:
                    raise ValueError(
                        "development/confirmation must be one deterministic episode"
                    )
                export = self.simulator.export
                if not export.export_kpis_on_episode_end or not export.final_episode_only:
                    raise ValueError(
                        "development/confirmation must export final-episode KPIs"
                    )
                if (
                    self.checkpointing.checkpoint_interval is not None
                    or self.checkpointing.checkpoint_on_episode_end
                ):
                    raise ValueError(
                        "evaluation runs must not write training checkpoints"
                    )
                if protocol.role == "candidate":
                    has_source = bool(
                        self.checkpointing.checkpoint_local_path
                        or self.checkpointing.checkpoint_run_id
                    )
                    if not self.checkpointing.resume_training or not has_source:
                        raise ValueError(
                            "evaluated candidates require one explicit checkpoint source"
                        )
                    for stage in self.pipeline:
                        if isinstance(stage, TIMARLStageConfig) and not stage.frozen:
                            raise ValueError(
                                "TIMARL evaluation requires pipeline stage frozen=true"
                            )
            else:
                if protocol.role != "candidate":
                    raise ValueError("train phase only supports role='candidate'")
                if self.simulator.deterministic_finish:
                    if self.simulator.episodes < 2:
                        raise ValueError(
                            "protocol train phase deterministic_finish requires at "
                            "least one trainable episode before the diagnostic evaluation"
                        )
                    episode_windows = self.simulator.episode_time_steps
                    if (
                        not isinstance(episode_windows, list)
                        or len(episode_windows) != self.simulator.episodes
                    ):
                        raise ValueError(
                            "protocol train phase deterministic_finish requires one "
                            "explicit episode_time_steps window per episode"
                        )
                    export = self.simulator.export
                    if (
                        not export.export_kpis_on_episode_end
                        or not export.final_episode_only
                        or export.kpis_final_episode_only is False
                        or export.timeseries_final_episode_only is False
                    ):
                        raise ValueError(
                            "protocol train phase deterministic_finish must export "
                            "KPIs and timeseries only for the final diagnostic episode"
                        )
                if any(isinstance(stage, TIMARLStageConfig) for stage in self.pipeline):
                    if not (
                        self.checkpointing.checkpoint_on_episode_end
                        and self.checkpointing.keep_episode_checkpoints
                    ):
                        raise ValueError(
                            "TIMARL protocol training must preserve every episode-end checkpoint"
                        )
                    for stage in self.pipeline:
                        if not isinstance(stage, TIMARLStageConfig):
                            continue
                        behavior_cloning = stage.hyperparameters.behavior_cloning
                        if (
                            behavior_cloning is not None
                            and behavior_cloning.enabled
                            and (
                                self.simulator.episodes
                                - int(self.simulator.deterministic_finish)
                            )
                            <= behavior_cloning.demonstration_episodes
                        ):
                            raise ValueError(
                                "TIMARL protocol training requires at least one "
                                "post-BC learning episode; simulator.episodes must "
                                "exceed behavior_cloning.demonstration_episodes"
                            )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary using original key names (aliases)."""
        payload = self.model_dump(by_alias=True)
        for stage in payload.get("pipeline", []) or []:
            if not isinstance(stage, dict):
                continue
            networks = stage.get("networks")
            if not isinstance(networks, dict):
                continue
            for network in networks.values():
                if not isinstance(network, dict):
                    continue
                for key in ("state_layers", "action_layers", "joint_layers", "head_layers"):
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
    project = ProjectConfig.model_validate(raw_config)

    for stage in project.pipeline:
        if isinstance(
            stage, (TransformerPPOStageConfig, TransformerMATD3StageConfig)
        ):
            from utils.entity_tokenizer_schema import (
                _load_default_sample,
                load_entity_tokenizer_config,
                validate_against_payload,
            )
            tokenizer_cfg = load_entity_tokenizer_config(stage.tokenizer_config_path)
            sample = _load_default_sample()
            action_names_per_building = [
                [ca.action_field for ca in tokenizer_cfg.ca_types.values()]
            ]
            validate_against_payload(tokenizer_cfg, sample, action_names_per_building)

    return project
