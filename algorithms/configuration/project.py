"""Canonical project configuration and cross-service compatibility checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, SerializeAsAny, field_validator, model_validator
from typing_extensions import Literal

from algorithms.configuration.agents import (
    SingleAgentRLStageConfig,
    TIMARLStageConfig,
    TransformerMATD3StageConfig,
    TransformerPPOStageConfig,
)
from algorithms.configuration.common import (
    BundleConfig,
    CheckpointingConfig,
    ExecutionConfig,
    ExperimentProtocolConfig,
    JobConfig,
    MetadataConfig,
    OutputConfig,
    RuntimeConfig,
    SimulatorConfig,
    TopologyConfig,
    TrackingConfig,
    TrainingConfig,
)
from algorithms.configuration.migration import SCHEMA_VERSION, normalize_config
from algorithms.configuration.model import StrictModel as BaseModel
from algorithms.configuration.profiles import resolve_profile, validate_capabilities
from algorithms.registry import agent_spec, import_symbol


class ProjectConfig(BaseModel):
    schema_version: Literal["algorithms.config.v1"] = SCHEMA_VERSION
    job: JobConfig = Field(default_factory=JobConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    metadata: MetadataConfig
    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    simulator: SimulatorConfig
    training: TrainingConfig = TrainingConfig()
    topology: TopologyConfig = TopologyConfig()
    pipeline: List[SerializeAsAny[BaseModel]] = Field(
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

    @field_validator("pipeline", mode="before")
    @classmethod
    def validate_agent_parameters(cls, value):
        if not isinstance(value, list):
            raise ValueError("pipeline must be a list of agent stages")
        stages = []
        for index, stage in enumerate(value):
            if hasattr(stage, "model_dump"):
                stage = stage.model_dump(by_alias=True)
            if not isinstance(stage, dict):
                raise ValueError(f"pipeline[{index}] must be an agent stage mapping")
            name = stage.get("algorithm")
            if name == "SingleAgentRL":
                stages.append(SingleAgentRLStageConfig.model_validate(stage))
                continue
            spec = agent_spec(name)
            if "tokenizer_config_path" in stage and isinstance(stage["tokenizer_config_path"], str):
                from algorithms.configuration.assets import configuration_asset

                stage = dict(stage)
                stage["tokenizer_config_path"] = str(
                    configuration_asset(stage["tokenizer_config_path"])
                )
            if spec.capabilities.single_controller and stage.get("count", 1) != 1:
                raise ValueError(f"{name} pipeline stages require count=1")
            stages.append(import_symbol(spec.configuration).model_validate(stage))
        return stages

    @model_validator(mode="after")
    def validate_cross_constraints(self) -> "ProjectConfig":
        for index, stage in enumerate(self.pipeline):
            if (
                isinstance(stage, (TransformerPPOStageConfig, TransformerMATD3StageConfig))
                and index != len(self.pipeline) - 1
            ):
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
                    raise ValueError("AgentTransformerMATD3 requires simulator.interface='entity'.")
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
            for stage in self.pipeline:
                capabilities = agent_spec(stage.algorithm).capabilities
                if not capabilities.dynamic_topology:
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
                    raise ValueError("development/confirmation must be one deterministic episode")
                export = self.simulator.export
                if not export.export_kpis_on_episode_end or not export.final_episode_only:
                    raise ValueError("development/confirmation must export final-episode KPIs")
                if (
                    self.checkpointing.checkpoint_interval is not None
                    or self.checkpointing.checkpoint_on_episode_end
                ):
                    raise ValueError("evaluation runs must not write training checkpoints")
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
                            and (self.simulator.episodes - int(self.simulator.deterministic_finish))
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
            hyperparameters = stage.get("hyperparameters")
            if isinstance(hyperparameters, dict) and agent_spec(
                stage["algorithm"]
            ).configuration.endswith(":RuleBasedAlgorithmConfig"):
                # Omitted limits retain the agent's unbounded/default behaviour;
                # JSON/YAML do not need Infinity and null must not reach float().
                for key in (
                    "deferrable_import_block_threshold_kw",
                    "deferrable_community_import_block_threshold_kw",
                ):
                    if hyperparameters.get(key, "missing") is None:
                        hyperparameters.pop(key)
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
            '    - algorithm: "<name>"\n'
            "      count: 1\n"
            "      hyperparameters: { ... }\n"
            "      networks: { ... }   # if applicable\n"
            "      replay_buffer: { ... }   # if applicable\n"
            "      exploration: { ... }   # if applicable\n"
        )
    project = ProjectConfig.model_validate(resolve_profile(normalize_config(raw_config)))
    validate_capabilities(project)

    for stage in project.pipeline:
        if isinstance(stage, (TransformerPPOStageConfig, TransformerMATD3StageConfig)):
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
