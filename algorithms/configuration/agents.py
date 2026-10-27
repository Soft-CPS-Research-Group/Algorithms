"""Validated agent-owned parameters; no numerical implementation is imported."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

from algorithms.configuration.model import StrictModel as BaseModel


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
        if self.behavior_action_stratified_sampling and self.behavior_action_priority_scope != "ev":
            raise ValueError(
                "behavior_action_stratified_sampling requires behavior_action_priority_scope='ev'"
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
    ev_v2g_buffer_soc: float = Field(default=0.0, ge=0.0)
    ev_subhour_service_guard_steps: float = Field(default=8.0, ge=0.0)
    ev_subhour_deadline_buffer_steps: float = Field(default=2.0, ge=0.0)
    ev_subhour_v2g_guard_steps: float = Field(default=8.0, ge=0.0)
    ev_subhour_min_service_margin_rate: float = Field(default=0.10, ge=0.0)
    ev_subhour_v2g_reserve_soc: float = Field(default=0.05, ge=0.0)
    ev_subhour_tight_feasibility_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    ev_service_soc_tolerance: float = Field(default=0.04, ge=0.0)
    deferrable_import_block_threshold_kw: Optional[float] = Field(default=None, ge=0.0)
    deferrable_community_import_block_threshold_kw: Optional[float] = Field(default=None, ge=0.0)
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
    local_action_safety_ev_minimum_mode: Literal["average", "deadline_feasible"] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = True
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0)


class ExperimentalPPOHyperparameters(BaseModel):
    """Shared schema for experimental hierarchical PPO agents.

    Unknown parameters are rejected; each specialised agent declares its fields.
    """

    model_config = ConfigDict(extra="forbid")

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
    use_env_reward: bool = False

    output_mode: Literal["actions", "signal"] = "actions"
    c_dim: int = Field(default=12, gt=0)
    b_dim: int = Field(default=7, gt=0)
    num_buildings: int = Field(default=17, gt=0)
    cc_action_interval: int = Field(default=1, gt=0)
    net_weight: float = Field(default=0.01, ge=0)


class CCLevel1Hyperparameters(ExperimentalPPOHyperparameters):
    # Phase-1 market maker: emits a global price multiplier.
    bc_collect_steps: int = 336
    bc_dt_hours: float = 0.25
    bc_lr: float = 0.001
    bc_mult_scale: float = 1.0
    bc_pretrain_enabled: bool = False
    bc_price_p20: float = 0.1
    bc_price_p80: float = 0.21
    bc_reference_export: Optional[float] = None
    bc_reference_headroom: float = 2.0
    bc_reference_peak: Optional[float] = None
    bc_reference_price: Optional[float] = None
    bc_reference_ramping: float = 1.878
    bc_target_import: Optional[float] = None
    bc_train_steps: int = 2000
    bc_w_cost: float = 1.0
    bc_w_export: float = 0.05
    bc_w_headroom: float = 1.0
    bc_w_peak: float = 0.6
    bc_w_ramp: float = 0.4
    bc_w_violation: float = 2.0
    w_factor: float = 0.3
    w_smoothness: float = 0.02

    c_dim: int = Field(default=17, gt=0)  # cc_level1 encoding width
    cc_action_interval: int = Field(default=4, gt=0)  # 4 × 15min = hourly
    price_min: float = Field(default=0.5, gt=0)  # min price multiplier
    price_max: float = Field(default=1.5, gt=0)  # max price multiplier
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

    bc_dt_hours: float = 0.25
    bc_mult_scale: float = 1.0
    bc_reference_export: Optional[float] = None
    bc_reference_peak: Optional[float] = None
    bc_target_import: Optional[float] = None
    bc_urgency_horizon: float = 4.0
    bc_w_cost: float = 1.0
    bc_w_ev: float = 0.5
    bc_w_export: float = 0.1
    bc_w_net: float = 0.1
    bc_w_peak: float = 0.3
    bc_w_soc: float = 0.2
    include_community_history: bool = False

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
    ] = "continuous_score"
    bc_collection_policy: Literal["teacher_rollout", "neutral_label_only"] = "teacher_rollout"
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
                raise ValueError("CCLevel2 reference_multipliers length must equal num_buildings")
            if any(value < self.price_min or value > self.price_max for value in values):
                raise ValueError(
                    "CCLevel2 reference_multipliers must lie within the configured price range"
                )
        if self.bc_discount_multiplier < self.price_min:
            raise ValueError("CCLevel2 bc_discount_multiplier must not be below price_min")
        if (
            self.policy_parameterization != "sparse_centered_residual"
            and self.policy_deadband > 0.0
        ):
            raise ValueError(
                "CCLevel2 policy_deadband is only supported by sparse_centered_residual"
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
                "CCLevel2 neutral-label BC collection requires at least one neutral warm-up episode"
            )
        if self.bc_teacher_mode == "oracle_storage_schedule":
            if not self.bc_pretrain_enabled:
                raise ValueError("CCLevel2 oracle_storage_schedule requires bc_pretrain_enabled")
            if not str(self.bc_oracle_schedule_path or "").strip():
                raise ValueError(
                    "CCLevel2 oracle_storage_schedule requires bc_oracle_schedule_path"
                )
        if self.bc_anchor_min_weight > self.bc_anchor_weight:
            raise ValueError("CCLevel2 bc_anchor_min_weight must not exceed bc_anchor_weight")
        if self.neutral_warmup_episodes > 0 and not self.neutral_baseline_enabled:
            raise ValueError("CCLevel2 neutral warm-up episodes require neutral_baseline_enabled")
        if self.training_episodes_per_validation > 0 and not self.neutral_baseline_enabled:
            raise ValueError("CCLevel2 validation episodes require neutral_baseline_enabled")
        if (
            self.restore_best_policy_for_deterministic
            and self.training_episodes_per_validation <= 0
        ):
            raise ValueError(
                "CCLevel2 best-policy restore requires deterministic validation episodes"
            )
        if self.rollback_rejected_validation and self.training_episodes_per_validation <= 0:
            raise ValueError(
                "CCLevel2 validation rollback requires deterministic validation episodes"
            )
        if self.policy_parameterization == "causal_active_only":
            if self.price_max > 1.0:
                raise ValueError("CCLevel2 causal_active_only requires price_max <= 1.0")
            if not self.price_min <= self.causal_initial_multiplier <= self.price_max:
                raise ValueError(
                    "CCLevel2 causal_initial_multiplier must lie within the price range"
                )
            causal_values = self.causal_initial_multipliers
            if causal_values is not None:
                if len(causal_values) != self.num_buildings:
                    raise ValueError(
                        "CCLevel2 causal_initial_multipliers length must equal num_buildings"
                    )
                if any(value < self.price_min or value > self.price_max for value in causal_values):
                    raise ValueError(
                        "CCLevel2 causal_initial_multipliers must lie within the price range"
                    )
            if self.bc_pretrain_enabled:
                raise ValueError("CCLevel2 causal_active_only does not support BC pretraining")
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
            raise ValueError("FixedPriceSignal vector schedule multipliers must not be empty")
        if any(value <= 0 for value in values):
            raise ValueError("FixedPriceSignal vector schedule multipliers must all be positive")
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
            value is not None for value in (self.multipliers, self.schedule, self.vector_schedule)
        )
        if configured_modes > 1:
            raise ValueError(
                "FixedPriceSignal multipliers, schedule and vector_schedule are mutually exclusive"
            )
        configured_schedule = self.schedule if self.schedule is not None else self.vector_schedule
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
                raise ValueError("FixedPriceSignal vector_schedule entries must have equal widths")
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
                "CausalPriceSignal vector_max_multiplier must be greater than vector_min_multiplier"
            )
        if self.discount_multipliers is not None:
            if not self.discount_multipliers:
                raise ValueError("CausalPriceSignal discount_multipliers must not be empty")
            if any(
                value < self.vector_min_multiplier or value > self.vector_max_multiplier
                for value in self.discount_multipliers
            ):
                raise ValueError(
                    "CausalPriceSignal discount_multipliers must lie within "
                    "the configured vector multiplier range"
                )
        return self


class BuildingAgentHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    num_epochs: int = Field(default=10, ge=1)
    mini_batch_size: int = Field(default=64, ge=1)
    clip_coef: float = Field(default=0.2, gt=0)
    vf_coef: float = Field(default=0.5, ge=0)
    ent_coef: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    target_kl: Optional[float] = Field(default=0.02, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    obs_dim: int = Field(default=0, ge=0)  # 0 = auto from env
    action_dim: int = Field(default=0, ge=0)  # 0 = auto from env
    num_steps: int = Field(default=2048, ge=1)
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 64])
    building_cost_weight: float = Field(default=1.0, ge=0)
    community_import_weight: float = Field(default=0.3, ge=0)
    constraint_penalty_weight: float = Field(default=0.5, ge=0)


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


class CCLevel2CapacityHyperparameters(ExperimentalPPOHyperparameters):
    bc_collect_steps: int = 336
    bc_dt_hours: float = 0.25
    bc_lr: float = 0.001
    bc_mult_scale: float = 1.0
    bc_pretrain_enabled: bool = False
    bc_reference_export: Optional[float] = None
    bc_reference_peak: Optional[float] = None
    bc_reference_price: Optional[float] = None
    bc_reference_ramping: float = 1.878
    bc_target_import: Optional[float] = None
    bc_train_steps: int = 2000
    bc_urgency_horizon: float = 4.0
    bc_w_cost: float = 1.0
    bc_w_ev: float = 0.5
    bc_w_export: float = 0.05
    bc_w_flex: float = 0.5
    bc_w_peak: float = 0.6
    bc_w_ramp: float = 0.4
    bc_w_violation: float = 2.0
    c_dim: Optional[int] = None
    cc_action_interval: int = 4
    cc_capacity_reference_kwh: float = 10.0
    num_buildings: int = 17
    per_building_credit: bool = False
    price_max: float = 1.5
    price_min: float = 0.5
    w_factor: float = 0.3
    w_smoothness: float = 0.02

    pass


class CCLevel2CapacityAlgorithmConfig(BaseModel):
    """Historical capacity-aware manager, separate from the current CCLevel2."""

    algorithm: Literal["CCLevel2Capacity"]
    count: Literal[1] = 1
    frozen: bool = False
    hyperparameters: CCLevel2CapacityHyperparameters = Field(
        default_factory=CCLevel2CapacityHyperparameters
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
    hyperparameters: BuildingAgentHyperparameters = Field(
        default_factory=BuildingAgentHyperparameters
    )
    networks: None = None
    replay_buffer: None = None
    exploration: None = None


class ActorStageConfig(BaseModel):
    @model_validator(mode="after")
    def validate_extension_parameters(self):
        if self.exploration is not None:
            from algorithms.configuration.extensions import validate_actor_parameters

            self.exploration.params = validate_actor_parameters(
                self.algorithm, self.exploration.params
            )
        return self


class ActorCriticAlgorithmConfig(ActorStageConfig):
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
        "SignalAwareRBCCommunity",
        "SignalDirectedRBC",
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
    policy: Optional[str] = Field(
        default=None, description="Identifier for the policy architecture"
    )
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
    local_action_safety_ev_minimum_mode: Literal["average", "deadline_feasible"] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = False
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0)


class RuleTeacherConfig(BaseModel):
    @field_validator("hyperparameters", check_fields=False)
    @classmethod
    def validate_teacher_parameters(cls, value):
        return RuleBasedHyperparameters.model_validate(value).model_dump(exclude_unset=True)


class TransformerPPOBehaviorCloningTeacherConfig(RuleTeacherConfig):
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
    deterministic_mode_strategy_by_group_type: Dict[str, Literal["argmax", "expected_signed"]] = (
        Field(default_factory=dict)
    )
    deterministic_expected_signed_gain_by_group_type: Dict[str, float] = Field(default_factory=dict)
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
            gain < 0.0 for gain in self.deterministic_expected_signed_gain_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic expected-signed gains must be non-negative"
            )
        if any(
            margin < 0.0
            for margin in self.deterministic_non_idle_logit_margin_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic non-idle logit margins must be non-negative"
            )
        if any(
            deadband < 0.0 or deadband > 1.0
            for deadband in self.deterministic_expected_signed_deadband_by_group_type.values()
        ):
            raise ValueError(
                "TIMARL actor deterministic expected-signed deadbands must be between zero and one"
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
    critic_loss_type: Literal["mse", "huber"] = "mse"
    critic_huber_delta: float = Field(default=1.0, gt=0.0)
    reward_normalization_enabled: bool = False
    reward_normalization_clip: float = Field(default=10.0, gt=0.0)
    target_policy_smoothing: bool = True
    target_policy_noise: float = Field(ge=0.0)
    target_policy_noise_clip: float = Field(ge=0.0)
    actor_update_interval: int = Field(default=2, gt=0)
    minimum_successful_critic_updates_before_actor: int = Field(default=1, ge=0)
    actor_policy_loss_weight: float = Field(default=1.0, ge=0.0)
    actor_projection_consistency_loss_weight: float = Field(default=0.0, ge=0.0)
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

    @field_validator("warm_start_policy_hyperparameters")
    @classmethod
    def validate_warm_start_parameters(cls, value):
        return RuleBasedHyperparameters.model_validate(value).model_dump(exclude_unset=True)

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
    local_action_safety_service_teacher_enabled: bool = False
    local_action_safety_service_teacher_eval_enabled: Optional[bool] = None
    local_action_safety_fail_on_infeasible: bool = False
    local_action_safety_protect_ev_minimum: bool = True
    local_action_safety_ev_minimum_mode: Literal["average", "deadline_feasible"] = "average"
    local_action_safety_protect_ev_service_target: bool = False
    local_action_safety_allow_ev_service_target_to_use_reserved_headroom: bool = False
    local_action_safety_protect_deferrable_must_start: bool = True
    local_action_safety_allow_discretionary_deferrable_start: bool = False
    local_action_safety_headroom_reserve_kw: float = Field(default=0.0, ge=0.0)
    local_action_safety_runtime_only_export: bool = False
    local_price_conditioning_enabled: bool = False
    local_price_forecast_mode: Literal["real_unmodified", "aligned_vector", "persist_current"] = (
        "real_unmodified"
    )
    local_price_conditioning_runtime_only_export: bool = False

    @model_validator(mode="after")
    def validate_matd3_relationships(self) -> "TransformerMATD3Hyperparameters":
        if self.buffer_capacity < self.batch_size:
            raise ValueError("buffer_capacity must be greater than or equal to batch_size")
        if self.min_sigma > self.sigma:
            raise ValueError("min_sigma must be less than or equal to sigma")
        if self.residual_policy_enabled and not str(self.warm_start_policy_name or "").strip():
            raise ValueError("residual_policy_enabled=true requires warm_start_policy_name")
        if (
            self.local_action_safety_service_teacher_enabled
            or self.local_action_safety_service_teacher_eval_enabled is True
        ) and not str(self.warm_start_policy_name or "").strip():
            raise ValueError(
                "local_action_safety_service_teacher_enabled=true requires warm_start_policy_name"
            )
        if self.local_action_safety_service_teacher_eval_enabled is None:
            self.local_action_safety_service_teacher_eval_enabled = (
                self.local_action_safety_service_teacher_enabled
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
        if self.enabled and self.teacher == "external":
            raise ValueError(
                "replay_based.teacher='external' is not supported by AgentTransformerMATD3"
            )
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
    ev_service_jit_buffer_seconds: float = Field(default=0.0, ge=0.0)
    ev_service_jit_minimum_average_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    protect_ev_service_target: bool = False
    enforce_ev_discharge_reserve: bool = True
    ev_v2g_reserve_margin_ratio: float = Field(default=0.0, ge=0.0, le=0.2)
    enforce_ev_economic_guard: bool = True
    ev_v2g_avoided_import_value_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    ev_v2g_minimum_profit_margin_eur_per_kwh: float = Field(
        default=0.01,
        ge=0.0,
    )
    ev_v2g_degradation_cost_eur_per_kwh: float = Field(
        default=0.0,
        ge=0.0,
    )
    ev_v2g_require_local_demand: bool = True
    headroom_reserve_kw: float = Field(default=0.0, ge=0.0)
    deferrable_service_margin_seconds: float = Field(default=0.0, ge=0.0)


class TIMARLTraceConfig(BaseModel):
    enabled: bool = True
    chunk_size: int = Field(default=256, ge=1)
    snapshot_interval: int = Field(default=256, ge=1)


class TIMARLBehaviorCloningTeacherConfig(RuleTeacherConfig):
    policy: Literal["RBCSmartPolicy"] = "RBCSmartPolicy"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TransformerMATD3BehaviorCloningTeacherConfig(RuleTeacherConfig):
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
    v2g_avoided_import_value_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    v2g_minimum_profit_margin_eur_per_kwh: float = Field(
        default=0.01,
        ge=0.0,
    )
    v2g_degradation_cost_eur_per_kwh: float = Field(
        default=0.0,
        ge=0.0,
    )
    opportunity_value_kind: Literal[
        "tariff",
        "community_marginal_import",
    ] = "tariff"


class TIMARLStoragePlanningConfig(BaseModel):
    """Causal auxiliary target for stationary-storage scheduling."""

    auxiliary_coeff: float = Field(default=0.0, ge=0.0)
    balance_targets: bool = True
    fraction_coeff: float = Field(default=0.25, ge=0.0)
    replay_capacity_per_reason: int = Field(default=16, ge=0)
    replay_samples_per_reason: int = Field(default=8, ge=0)
    charge_fraction: float = Field(default=0.55, gt=0.0, lt=1.0)
    discharge_fraction: float = Field(default=0.45, gt=0.0, lt=1.0)
    minimum_soc_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    maximum_soc_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    price_tie_tolerance: float = Field(default=1.0e-6, ge=0.0)
    minimum_price_spread: float = Field(default=0.01, ge=0.0)
    pv_surplus_threshold_kw: float = Field(default=0.25, ge=0.0)
    import_threshold_kw: float = Field(default=0.25, ge=0.0)
    price_regime_kind: Literal[
        "strict_extrema",
        "relative_forecast",
    ] = "strict_extrema"
    forecast_mean_margin_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    forecast_edge_margin_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    forecast_spread_floor_ratio: float = Field(default=0.05, ge=0.0)
    scale_price_fraction_by_opportunity: bool = False
    minimum_price_fraction_scale: float = Field(default=0.50, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_soc_band(self) -> "TIMARLStoragePlanningConfig":
        if self.minimum_soc_ratio >= self.maximum_soc_ratio:
            raise ValueError("storage_planning minimum_soc_ratio must be below maximum_soc_ratio")
        return self


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
    policy_credit_assignment: Literal["joint_agent", "typed_group"] = "joint_agent"
    ppo_policy_group_types: Optional[List[str]] = None
    actor_update_scope: Literal["all", "selected_group_heads"] = "all"
    actor_update_group_types: Optional[List[str]] = None
    policy_anchor_coeff: float = Field(default=0.0, ge=0)
    policy_anchor_coeff_by_group_type: Dict[str, float] = Field(default_factory=dict)
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
    storage_planning: TIMARLStoragePlanningConfig = TIMARLStoragePlanningConfig()

    @model_validator(mode="after")
    def validate_learning_architecture(self) -> "TIMARLHyperparameters":
        expected = {"ppo": "local", "mappo": "set"}[self.backbone.name]
        if self.critic.kind != expected:
            raise ValueError(
                f"TIMARL backbone.name={self.backbone.name!r} requires critic.kind={expected!r}"
            )
        if any(value < 0.0 for value in self.entropy_coeff_by_group_type.values()):
            raise ValueError("TIMARL entropy_coeff_by_group_type values must be non-negative")
        if any(value < 0.0 for value in self.policy_anchor_coeff_by_group_type.values()):
            raise ValueError("TIMARL policy_anchor_coeff_by_group_type values must be non-negative")
        if (
            self.policy_anchor_coeff_by_group_type
            and self.policy_credit_assignment != "typed_group"
        ):
            raise ValueError(
                "TIMARL policy_anchor_coeff_by_group_type requires "
                "policy_credit_assignment='typed_group'"
            )
        if self.ppo_policy_group_types is not None:
            normalized_group_types = [
                str(group_type).strip() for group_type in self.ppo_policy_group_types
            ]
            if not normalized_group_types or any(
                not group_type for group_type in normalized_group_types
            ):
                raise ValueError(
                    "TIMARL ppo_policy_group_types must contain at least one "
                    "non-empty action-group type"
                )
            if len(normalized_group_types) != len(set(normalized_group_types)):
                raise ValueError("TIMARL ppo_policy_group_types must not contain duplicates")
            if self.policy_credit_assignment != "typed_group":
                raise ValueError(
                    "TIMARL ppo_policy_group_types requires policy_credit_assignment='typed_group'"
                )
            self.ppo_policy_group_types = normalized_group_types
        if self.actor_update_group_types is not None:
            normalized_update_group_types = [
                str(group_type).strip() for group_type in self.actor_update_group_types
            ]
            if not normalized_update_group_types or any(
                not group_type for group_type in normalized_update_group_types
            ):
                raise ValueError(
                    "TIMARL actor_update_group_types must contain at least one "
                    "non-empty action-group type"
                )
            if len(normalized_update_group_types) != len(set(normalized_update_group_types)):
                raise ValueError("TIMARL actor_update_group_types must not contain duplicates")
            self.actor_update_group_types = normalized_update_group_types
        if self.actor_update_scope == "selected_group_heads":
            if self.actor_update_group_types is None:
                raise ValueError(
                    "TIMARL actor_update_scope='selected_group_heads' requires "
                    "actor_update_group_types"
                )
            if self.policy_credit_assignment != "typed_group":
                raise ValueError(
                    "TIMARL selected group-head updates require "
                    "policy_credit_assignment='typed_group'"
                )
        elif self.actor_update_group_types is not None:
            raise ValueError(
                "TIMARL actor_update_group_types requires actor_update_scope='selected_group_heads'"
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


class TransformerNetworkConfig(BaseModel):
    """Configuration for Transformer network hyperparameters."""

    d_model: int = Field(default=64, ge=8, description="Embedding dimension")
    nhead: int = Field(default=4, ge=1, description="Number of attention heads")
    num_layers: int = Field(default=2, ge=1, description="Number of encoder layers")
    dim_feedforward: int = Field(default=128, ge=8, description="FFN hidden size")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate")
    max_tokens: int = Field(default=128, ge=4, description="Maximum total tokens")

    @field_validator("d_model")
    @classmethod
    def validate_d_model_divisible_by_nhead(cls, v: int, info) -> int:
        # Note: nhead validation happens via model_validator since we need both fields
        return v

    @model_validator(mode="after")
    def validate_divisibility(self) -> "TransformerNetworkConfig":
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})")
        return self


class TransformerTokenizerConfig(BaseModel):
    """Configuration for feature-to-token mapping."""

    ca_feature_patterns: List[str] = Field(
        default_factory=lambda: [
            "electric_vehicle_soc",
            "connected_state",
            "departure_time",
            "arrival_time",
        ],
        description="Patterns identifying CA features",
    )
    sro_feature_patterns: List[str] = Field(
        default_factory=lambda: [
            "outdoor_dry_bulb",
            "carbon_intensity",
            "electricity_pricing",
        ],
        description="Patterns identifying SRO features",
    )
    nfc_feature_patterns: List[str] = Field(
        default_factory=lambda: ["non_shiftable_load"],
        description="Patterns identifying NFC features",
    )


class TransformerNetworksConfig(BaseModel):
    """Networks configuration for TransformerMADDPG."""

    transformer: TransformerNetworkConfig = TransformerNetworkConfig()
    lr_actor: float = Field(default=1e-4, gt=0, description="Actor learning rate")
    lr_critic: float = Field(default=1e-3, gt=0, description="Critic learning rate")


class TransformerReplayBufferConfig(BaseModel):
    """Replay buffer configuration for TransformerMADDPG."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(default="TransformerReplayBuffer", alias="class")
    capacity: int = Field(default=100000, ge=1, description="Buffer capacity")
    batch_size: int = Field(default=256, ge=1, description="Training batch size")
    max_ca: Optional[int] = Field(default=None, ge=1, description="Max CA tokens (auto if None)")
    max_sro: Optional[int] = Field(default=None, ge=1, description="Max SRO tokens (auto if None)")


class TransformerMADDPGAlgorithmConfig(ActorStageConfig):
    """Configuration for TransformerMADDPG algorithm."""

    algorithm: Literal["TransformerMADDPG"]
    count: Literal[1] = 1
    frozen: bool = False
    hyperparameters: AlgorithmHyperparameters = AlgorithmHyperparameters(gamma=0.99)
    networks: TransformerNetworksConfig = TransformerNetworksConfig()
    tokenizer: TransformerTokenizerConfig = TransformerTokenizerConfig()
    replay_buffer: TransformerReplayBufferConfig = TransformerReplayBufferConfig()
    exploration: Optional[ExplorationParams] = None


class OfflineTrainingGroup(BaseModel):
    obs_dim: int = Field(gt=0)
    action_dim: int = Field(gt=0)


class OfflineTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_path: str = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    groups: Optional[List[OfflineTrainingGroup]] = None
    val_seeds: Optional[List[int]] = None


class OfflineAgentHyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    model_dir: Optional[str] = Field(default=None, min_length=1)
    offline_training: Optional[OfflineTrainingConfig] = None
    device: str = "cpu"

    @model_validator(mode="after")
    def require_one_policy_source(self):
        if bool(self.model_dir) == (self.offline_training is not None):
            raise ValueError("Choose exactly one of model_dir or offline_training")
        return self


class OfflineAgentStageConfig(BaseModel):
    """Offline optimization followed by the shared simulation/evaluation lifecycle."""

    algorithm: Literal["BCAgent", "IQLAgent", "IQLEntityAgent", "CQLEntityAgent"]
    count: Literal[1] = 1
    frozen: Literal[True] = True
    hyperparameters: OfflineAgentHyperparameters

    @model_validator(mode="after")
    def validate_training_parameters(self):
        training = self.hyperparameters.offline_training
        if training is not None:
            from algorithms.configuration.offline import validate_offline_parameters

            training.parameters = validate_offline_parameters(self.algorithm, training.parameters)
        return self


PipelineStageConfig = Union[
    TransformerMADDPGAlgorithmConfig,
    OfflineAgentStageConfig,
    TIMARLStageConfig,
    BuildingAgentStageConfig,
    CCLevel1AlgorithmConfig,
    CausalPriceSignalAlgorithmConfig,
    FixedPriceSignalAlgorithmConfig,
    CCLevel2AlgorithmConfig,
    CCLevel2CapacityAlgorithmConfig,
    CommunityCoordinatorAlgorithmConfig,
    ActorCriticAlgorithmConfig,
    RuleBasedAlgorithmConfig,
    SingleAgentRLStageConfig,
    TransformerPPOStageConfig,
    TransformerMATD3StageConfig,
]
