"""The single simulation/training/evaluation loop, shared by every agent."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Mapping

import numpy as np
import psutil
import torch
from loguru import logger

from algorithms.runtime.randomness import call_hook
from algorithms.runtime.tracking import metrics_active, mlflow

if TYPE_CHECKING:
    from algorithms.runtime.controller import SimulationController


def run_episodes(
    self: "SimulationController", episodes=None, deterministic=None, deterministic_finish=None
):
    """
    Train agent with MLflow logging for rewards (per step and per agent), PyTorch GPU memory usage, and system usage.
    """
    if self.model is None:
        logger.error("Wrapper invoked without a model; aborting training.")
        raise ValueError("Model is not set. Use `set_model` to provide a model.")

    episodes = episodes or self.default_episodes
    deterministic_finish = deterministic_finish if deterministic_finish is not None else False
    deterministic = deterministic if deterministic is not None else False
    if not getattr(self, "_resume_episode", 0):
        self._episode_execution_history = []
    self.learning_step = getattr(self, "learning_step", 0)

    total_rewards_across_episodes = []  # To track overall reward trends

    for episode in range(getattr(self, "_resume_episode", 0), episodes):
        self._at_episode_boundary = False
        start_episode_time = time.time()
        episode_deterministic = bool(deterministic) or (
            deterministic_finish and episode >= episodes - 1
        )
        export_this_episode = self._configure_episode_exports(
            episode, episodes, deterministic=episode_deterministic
        )

        self._entity_model_observations_direct = self._can_use_direct_entity_model_observations()
        self._write_phase_progress(
            phase="episode_reset_start",
            episode=episode,
            step=0,
            episode_total=episodes,
            step_total=None,
            global_step_total=None,
        )
        self._prepare_episode_reset()
        raw_observations, reset_info = self.simulator.reset()
        if self._entity_interface_mode:
            self._entity_current_payload = raw_observations
            self._entity_current_info = dict(reset_info or {})
            self._entity_next_payload = None
            self._entity_step_info = {}
            observations = self._apply_entity_layout(
                raw_observations,
                force_attach=True,
                model_observations=self._entity_model_observations_direct,
            )
        else:
            observations = raw_observations
        on_episode_start = getattr(self.model, "on_episode_start", None)
        if callable(on_episode_start):
            call_hook(
                self.model,
                "on_episode_start",
                stream="evaluation" if episode_deterministic else "learning",
                episode=episode,
                training=not episode_deterministic,
            )
        if episode_deterministic:
            from algorithms.export.references import begin_capture
            from algorithms.runtime.context import OutputPlan
            from algorithms.runtime.evaluation import policy_digest
            from algorithms.runtime.resume import leaves

            selections = []
            for path, unit in leaves(self.model):
                selector = getattr(unit, "select_evaluation_policy", None)
                selected = (
                    call_hook(unit, "select_evaluation_policy", stream="evaluation")
                    if callable(selector)
                    else {"selection": "final"}
                )
                selections.append({"path": path, **selected})
            final_policy = all(item["selection"] == "final" for item in selections)
            if OutputPlan.from_config(self.config).bundle:
                begin_capture(self.model, self.config)
            self.evaluated_policy = {
                "selection": "final" if final_policy else "validated_selection",
                "stages": selections,
                "sha256": policy_digest(self.model),
                "checkpoint": getattr(self.checkpoint_manager, "latest_entry", None)
                if final_policy
                else None,
                "seed": self.config.get("simulator", {}).get("random_seed"),
                "window": [
                    getattr(self.episode_tracker, "episode_start_time_step", None),
                    getattr(self.episode_tracker, "episode_end_time_step", None),
                ],
                "topology": self.describe_environment(),
            }

        self.episode_time_steps = self.episode_tracker.episode_time_steps
        self._record_episode_execution(
            episode=episode,
            deterministic=episode_deterministic,
            training=not episode_deterministic,
            export_enabled=export_this_episode,
        )
        episode_step_total, global_step_total = self._resolve_progress_totals(episodes)
        self._write_phase_progress(
            phase="episode_reset_end",
            episode=episode,
            step=0,
            episode_total=episodes,
            step_total=episode_step_total,
            global_step_total=global_step_total,
        )
        terminated = False
        truncated = False
        time_step = 0
        rewards_list = []  # Stores rewards per step

        while not (terminated or truncated):
            step_start_time = time.time()
            self.global_step += 1
            if not episode_deterministic:
                self.learning_step += 1
            should_profile_step = (
                self.runtime_profiling_enabled
                and self.global_step % self.runtime_profiling_interval == 0
            )
            step_profile_start_time = time.perf_counter() if should_profile_step else 0.0
            runtime_profile_metrics: Dict[str, float] = {}
            model_update_duration = 0.0
            logger.debug(
                "Global step {} (episode {}, timestep {})",
                self.global_step,
                episode,
                time_step,
            )
            self._write_phase_progress(
                phase="step_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            self._enforce_resource_guards(
                phase="step_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )

            step_observations = [np.asarray(obs, dtype=np.float64) for obs in observations]
            self._write_phase_progress(
                phase="predict_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            phase_start_time = time.perf_counter() if should_profile_step else 0.0
            actions = self.predict(
                observations,
                deterministic=episode_deterministic,
                episode_step=time_step,
            )
            if should_profile_step:
                runtime_profile_metrics["Runtime/predict_seconds"] = (
                    time.perf_counter() - phase_start_time
                )
            self._write_phase_progress(
                phase="predict_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            self._write_phase_progress(
                phase="action_prepare_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            phase_start_time = time.perf_counter() if should_profile_step else 0.0
            self._last_proposed_actions = [np.asarray(row).copy() for row in actions]
            actions = self._clip_actions(actions)
            if episode_deterministic:
                from algorithms.export.references import capture_submission

                capture_submission(self.model, actions, time_step)
            if not self._entity_interface_mode:
                self.actions = actions
            logger.debug("Predicted actions: {}", actions)

            # Apply actions to CityLearn environment
            env_actions = self._to_env_actions(actions)
            if should_profile_step:
                runtime_profile_metrics["Runtime/action_prepare_seconds"] = (
                    time.perf_counter() - phase_start_time
                )
            self._write_phase_progress(
                phase="action_prepare_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            self._write_phase_progress(
                phase="env_step_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            phase_start_time = time.perf_counter() if should_profile_step else 0.0
            simulator_step = self.simulator.step(env_actions)
            next_observations_raw, rewards, terminated, truncated, step_info = (
                simulator_step.as_tuple()
            )
            self._last_step_duration_seconds = simulator_step.duration_seconds
            if self._entity_interface_mode:
                self._entity_next_payload = next_observations_raw
                self._entity_step_info = dict(step_info or {})
            if should_profile_step:
                runtime_profile_metrics["Runtime/env_step_seconds"] = (
                    time.perf_counter() - phase_start_time
                )
            self._write_phase_progress(
                phase="env_step_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            self._write_phase_progress(
                phase="entity_layout_start"
                if self._entity_interface_mode
                else "observation_prepare_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            phase_start_time = time.perf_counter() if should_profile_step else 0.0
            entity_layout_snapshot = None
            model_topology_snapshot = None
            restore_model_topology = None
            if self._entity_interface_mode:
                if self._entity_adapter is not None and self._entity_adapter.needs_rebuild(
                    next_observations_raw
                ):
                    entity_layout_snapshot = self._snapshot_entity_layout_state()
                next_observations = self._apply_entity_layout(
                    next_observations_raw,
                    force_attach=False,
                    model_observations=self._entity_model_observations_direct,
                    attach_model=False,
                )
            else:
                next_observations = next_observations_raw
            if should_profile_step:
                entity_layout_seconds = time.perf_counter() - phase_start_time
                runtime_profile_metrics["Runtime/observation_encoding_seconds"] = (
                    entity_layout_seconds
                )
                runtime_profile_metrics["Runtime/entity_layout_seconds"] = entity_layout_seconds
            self._write_phase_progress(
                phase="entity_layout_end"
                if self._entity_interface_mode
                else "observation_prepare_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            self._write_phase_progress(
                phase="reward_shaping_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            phase_start_time = time.perf_counter() if should_profile_step else 0.0
            rewards = self._shape_rewards(rewards, next_observations)
            if self._entity_interface_mode:
                self._entity_step_info = dict(self._entity_step_info)
                self._entity_step_info.setdefault(
                    "reward_components",
                    self._get_reward_components_payload(),
                )
            if should_profile_step:
                runtime_profile_metrics["Runtime/reward_shaping_seconds"] = (
                    time.perf_counter() - phase_start_time
                )
            self._write_phase_progress(
                phase="reward_shaping_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            rewards_list.append(rewards)

            topology_transition_recorded = False
            if self._topology_changed_during_step:
                handles_cross_topology = bool(
                    getattr(self.model, "handles_cross_topology_transitions", False)
                )
                snapshot_model_topology = getattr(self.model, "snapshot_topology_state", None)
                restore_model_topology = getattr(self.model, "restore_topology_state", None)
                try:
                    if callable(snapshot_model_topology) != callable(restore_model_topology):
                        raise RuntimeError(
                            "Model topology rollback requires both snapshot_topology_state() "
                            "and restore_topology_state()."
                        )
                    if callable(snapshot_model_topology):
                        model_topology_snapshot = snapshot_model_topology()
                    if not episode_deterministic and not handles_cross_topology:
                        transition_hook = getattr(self.model, "record_topology_transition", None)
                        if callable(transition_hook):
                            transition_observations = (
                                [np.asarray(obs, dtype=np.float64) for obs in observations]
                                if self._entity_model_observations_direct
                                else self._encode_observations_for_model(observations)
                            )
                            transition_hook(
                                observations=transition_observations,
                                actions=actions,
                                rewards=rewards,
                                terminated=terminated,
                                truncated=truncated,
                                global_learning_step=self.global_step,
                            )
                            topology_transition_recorded = True
                    if episode_deterministic:
                        self._attach_model_environment_metadata()
                    else:
                        self._pending_model_environment_attach = True
                except Exception:
                    if model_topology_snapshot is not None and callable(restore_model_topology):
                        restore_model_topology(model_topology_snapshot)
                    if entity_layout_snapshot is not None:
                        self._restore_entity_layout_state(entity_layout_snapshot)
                    raise

            # Deterministic policies do not learn, but agents with an
            # explicit observation hook may still persist execution
            # evidence and advance their typed transition state.
            if (
                episode_deterministic
                and bool(
                    getattr(
                        self.model,
                        "requires_deterministic_transition_observation",
                        False,
                    )
                )
                and callable(getattr(self.model, "observe_transition", None))
            ):
                phase_start_time = time.perf_counter()
                self.update(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    terminated=terminated,
                    truncated=truncated,
                    learn=False,
                )
                model_update_duration = time.perf_counter() - phase_start_time
                self._last_model_update_seconds = model_update_duration

            # Update model if not in deterministic mode
            if not episode_deterministic:
                if (
                    self._topology_changed_during_step
                    and not self._pending_model_environment_attach
                ):
                    logger.debug(
                        "{} model.update at global step {} due to mid-step topology change.",
                        "Recorded old-layout transition before"
                        if topology_transition_recorded
                        else "Skipping",
                        self.global_step,
                    )
                    self._topology_changed_during_step = False
                else:
                    self._write_phase_progress(
                        phase="model_update_start",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                    self._synchronize_model_cuda_for_timing()
                    phase_start_time = time.perf_counter()
                    set_training_progress = getattr(
                        self.model,
                        "set_training_progress_callback",
                        None,
                    )

                    def report_model_update_progress(
                        payload: Mapping[str, Any],
                    ) -> None:
                        details = dict(payload)
                        training_phase = str(details.pop("phase", "model_update_training"))
                        self._write_phase_progress(
                            phase=training_phase,
                            episode=episode,
                            step=time_step,
                            episode_total=episodes,
                            step_total=episode_step_total,
                            global_step_total=global_step_total,
                            rewards=rewards,
                            force=True,
                            extra=details,
                        )

                    if callable(set_training_progress):
                        set_training_progress(report_model_update_progress)
                    try:
                        if self._pending_model_environment_attach:
                            self._pending_model_environment_attach = False
                            self._refresh_update_schedule()
                            self._attach_model_environment_metadata()
                            if bool(
                                getattr(
                                    self.model,
                                    "handles_cross_topology_transitions",
                                    False,
                                )
                            ):
                                self.update(
                                    observations,
                                    actions,
                                    rewards,
                                    next_observations,
                                    terminated=terminated,
                                    truncated=truncated,
                                )
                        else:
                            self.update(
                                observations,
                                actions,
                                rewards,
                                next_observations,
                                terminated=terminated,
                                truncated=truncated,
                            )
                    except Exception:
                        if model_topology_snapshot is not None and callable(restore_model_topology):
                            restore_model_topology(model_topology_snapshot)
                        if entity_layout_snapshot is not None:
                            self._restore_entity_layout_state(entity_layout_snapshot)
                        raise
                    finally:
                        if callable(set_training_progress):
                            set_training_progress(None)
                    self._synchronize_model_cuda_for_timing()
                    model_update_duration = time.perf_counter() - phase_start_time
                    self._last_model_update_seconds = model_update_duration
                    self._enforce_update_duration_guard(
                        update_duration=model_update_duration,
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                    if should_profile_step:
                        runtime_profile_metrics["Runtime/agent_update_seconds"] = (
                            model_update_duration
                        )
                        runtime_profile_metrics["Runtime/model_observation_encoding_seconds"] = (
                            float(
                                getattr(self, "_last_model_observation_encoding_seconds", 0.0)
                                or 0.0
                            )
                        )
                        runtime_profile_metrics["Runtime/model_update_seconds"] = float(
                            getattr(self, "_last_model_update_seconds", 0.0) or 0.0
                        )
                    logger.debug("Model update executed at global step {}", self.global_step)
                    self._write_phase_progress(
                        phase="model_update_end",
                        episode=episode,
                        step=time_step,
                        episode_total=episodes,
                        step_total=episode_step_total,
                        global_step_total=global_step_total,
                        rewards=rewards,
                    )
                    self._topology_changed_during_step = False

                self._write_phase_progress(
                    phase="checkpoint_start",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
                phase_start_time = time.perf_counter() if should_profile_step else 0.0
                self.checkpoint_manager.maybe_save(
                    agent=self.model,
                    step=self.global_step,
                    initial_exploration_done=self.initial_exploration_done,
                    update_step=self.update_step,
                )
                if should_profile_step:
                    runtime_profile_metrics["Runtime/checkpoint_seconds"] = (
                        time.perf_counter() - phase_start_time
                    )
                self._write_phase_progress(
                    phase="checkpoint_end",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )
            elif should_profile_step:
                runtime_profile_metrics["Runtime/agent_update_seconds"] = 0.0
                runtime_profile_metrics["Runtime/model_observation_encoding_seconds"] = 0.0
                runtime_profile_metrics["Runtime/model_update_seconds"] = 0.0
                runtime_profile_metrics["Runtime/checkpoint_seconds"] = 0.0
            elif self._progress_phase_in_window():
                self._write_phase_progress(
                    phase="model_update_skipped",
                    episode=episode,
                    step=time_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=rewards,
                )

            observations = [o for o in next_observations]
            if self._entity_interface_mode:
                self._entity_current_payload = self._entity_next_payload
                self._entity_current_info = self._entity_step_info
                self._entity_next_payload = None
                self._entity_step_info = {}

            # Reduce system monitoring frequency
            cpu_usage = None
            ram_usage = None
            if self.system_metrics_enabled and (
                self.global_step % self.system_metrics_interval == 0
            ):
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent

            # PyTorch-specific GPU memory tracking (kept only PyTorch measurement)
            if (
                self.system_metrics_enabled
                and (self.global_step % self.system_metrics_interval == 0)
                and torch.cuda.is_available()
            ):
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2  # Reserved for caching
            else:
                gpu_mem_allocated = None
                gpu_mem_reserved = None

            # Step duration calculation
            step_duration = time.time() - step_start_time
            normal_step_duration = max(0.0, step_duration - model_update_duration)
            self._enforce_resource_guards(
                phase="step_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
            )
            self._enforce_step_duration_guard(
                step_duration=normal_step_duration,
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
            )
            self._write_phase_progress(
                phase="step_finalize_start",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
                extra={
                    "step_duration_seconds": round(float(step_duration), 6),
                    "normal_step_duration_seconds": round(float(normal_step_duration), 6),
                    "update_duration_seconds": round(float(model_update_duration), 6),
                },
            )
            if should_profile_step:
                runtime_profile_metrics["Runtime/step_seconds"] = step_duration
                runtime_profile_metrics["Runtime/normal_step_seconds"] = normal_step_duration
                runtime_profile_metrics["Runtime/update_seconds"] = model_update_duration
                runtime_profile_metrics["Runtime/step_perf_seconds"] = (
                    time.perf_counter() - step_profile_start_time
                )
            should_log_step = self._should_log_step(self.global_step) or should_profile_step
            if should_log_step:
                diagnostics_start_time = time.perf_counter()
                metrics = {f"Agent_{i}_Reward": reward for i, reward in enumerate(rewards)}
                if cpu_usage is not None:
                    metrics["CPU_Usage"] = cpu_usage
                    metrics["RAM_Usage"] = ram_usage
                if gpu_mem_allocated is not None:
                    metrics["GPU_PyTorch_Allocated_MB"] = gpu_mem_allocated
                    metrics["GPU_PyTorch_Reserved_MB"] = gpu_mem_reserved
                metrics["Step_Duration"] = step_duration
                if should_profile_step:
                    metrics.update(runtime_profile_metrics)
                metrics.update(self._collect_model_status_metrics())
                metrics.update(self._build_action_diagnostic_metrics(actions, step_observations))
                metrics.update(self._build_reward_component_metrics())
                metrics.update(self._consume_model_training_metrics())
                if should_profile_step:
                    metrics["Runtime/diagnostics_build_seconds"] = (
                        time.perf_counter() - diagnostics_start_time
                    )

                if metrics_active(mlflow):
                    mlflow.log_metrics(metrics, step=self.global_step)
                elif self.local_metrics_logger:
                    self.local_metrics_logger.log(metrics, self.global_step)

                logger.info(
                    "Time step: {}/{}, Episode: {}/{}, Actions: {}, Rewards: {}, CPU: {}%, RAM: {}%, "
                    "GPU Allocated: {} MB, GPU Reserved: {} MB Step Duration: {}",
                    time_step + 1,
                    self.episode_time_steps,
                    episode + 1,
                    episodes,
                    actions,
                    rewards,
                    cpu_usage,
                    ram_usage,
                    gpu_mem_allocated,
                    gpu_mem_reserved,
                    step_duration,
                )

            self._write_phase_progress(
                phase="step_end",
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
                extra={
                    "step_duration_seconds": round(float(step_duration), 6),
                    "normal_step_duration_seconds": round(float(normal_step_duration), 6),
                    "update_duration_seconds": round(float(model_update_duration), 6),
                },
            )
            self._write_step_progress(
                episode=episode,
                step=time_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=rewards,
                step_duration=step_duration,
                normal_step_duration=normal_step_duration,
                update_duration=model_update_duration,
            )

            time_step += 1

        on_episode_end = getattr(self.model, "on_episode_end", None)
        if callable(on_episode_end):
            episode_boundary_step = max(time_step - 1, 0)
            last_rewards = rewards_list[-1] if rewards_list else None
            self._write_phase_progress(
                phase="model_episode_end_start",
                episode=episode,
                step=episode_boundary_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=last_rewards,
                force=True,
            )
            set_training_progress = getattr(
                self.model,
                "set_training_progress_callback",
                None,
            )

            def report_boundary_training_progress(
                payload: Mapping[str, Any],
            ) -> None:
                details = dict(payload)
                training_phase = str(details.pop("phase", "model_episode_end_training"))
                self._write_phase_progress(
                    phase=training_phase,
                    episode=episode,
                    step=episode_boundary_step,
                    episode_total=episodes,
                    step_total=episode_step_total,
                    global_step_total=global_step_total,
                    rewards=last_rewards,
                    force=True,
                    extra=details,
                )

            if callable(set_training_progress):
                set_training_progress(report_boundary_training_progress)
            try:
                call_hook(
                    self.model,
                    "on_episode_end",
                    stream="evaluation" if episode_deterministic else "learning",
                    episode=episode,
                    training=not episode_deterministic,
                )
            finally:
                if callable(set_training_progress):
                    set_training_progress(None)
            self._write_phase_progress(
                phase="model_episode_end_complete",
                episode=episode,
                step=episode_boundary_step,
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                rewards=last_rewards,
                force=True,
            )
        boundary_training_metrics = self._consume_model_training_metrics()
        if boundary_training_metrics:
            if metrics_active(mlflow):
                mlflow.log_metrics(boundary_training_metrics, step=self.global_step)
            elif self.local_metrics_logger:
                self.local_metrics_logger.log(
                    boundary_training_metrics,
                    self.global_step,
                )

        last_rewards = rewards_list[-1] if rewards_list else None
        self._write_phase_progress(
            phase="episode_export_start",
            episode=episode,
            step=max(time_step - 1, 0),
            episode_total=episodes,
            step_total=episode_step_total,
            global_step_total=global_step_total,
            rewards=last_rewards,
        )
        self._export_episode_kpis_if_needed(
            export_this_episode,
            episode=episode,
            is_final_episode=episode + 1 >= episodes,
        )
        self._at_episode_boundary = True
        self._next_episode = episode + 1
        if not episode_deterministic and episode + 1 == episodes - int(bool(deterministic_finish)):
            self._final_training_checkpoint = self.checkpoint_manager.save_final(
                self.model, self.global_step
            )
        self.checkpoint_manager.save_episode_end(
            self.model,
            step=self.global_step,
            episode=episode,
            deterministic=episode_deterministic,
        )
        self._write_phase_progress(
            phase="episode_export_end",
            episode=episode,
            step=max(time_step - 1, 0),
            episode_total=episodes,
            step_total=episode_step_total,
            global_step_total=global_step_total,
            rewards=last_rewards,
        )

        if self.progress_updates_enabled and time_step > 0:
            self.progress_tracker.update(
                episode=episode,
                step=time_step - 1,
                global_step=self.global_step,
                rewards=rewards_list[-1],
                episode_total=episodes,
                step_total=episode_step_total,
                global_step_total=global_step_total,
                status="finalizing" if episode + 1 >= episodes else "running",
                extra={
                    "phase": "episode_end",
                    **self._runtime_resource_snapshot(),
                },
            )

        # Compute rewards statistics for this episode
        reward_vectors = [
            np.asarray(step_rewards, dtype=np.float64).reshape(-1) for step_rewards in rewards_list
        ]
        if len(reward_vectors) == 0:
            rewards_array = np.zeros((0, 0), dtype=np.float64)
        else:
            max_agents = max(vector.shape[0] for vector in reward_vectors)
            rewards_array = np.full((len(reward_vectors), max_agents), np.nan, dtype=np.float64)
            for row, vector in enumerate(reward_vectors):
                rewards_array[row, : vector.shape[0]] = vector

        if rewards_array.size == 0:
            rewards_summary = {
                "sum": np.array([], dtype=np.float64),
                "mean": np.array([], dtype=np.float64),
                "min": np.array([], dtype=np.float64),
                "max": np.array([], dtype=np.float64),
            }
        else:
            valid_mask = ~np.isnan(rewards_array)
            valid_counts = valid_mask.sum(axis=0)
            sums = np.nansum(rewards_array, axis=0)
            means = np.divide(
                sums, np.maximum(valid_counts, 1), where=np.maximum(valid_counts, 1) > 0
            )
            mins = np.array(
                [
                    np.nanmin(rewards_array[:, i]) if valid_counts[i] > 0 else 0.0
                    for i in range(rewards_array.shape[1])
                ],
                dtype=np.float64,
            )
            maxs = np.array(
                [
                    np.nanmax(rewards_array[:, i]) if valid_counts[i] > 0 else 0.0
                    for i in range(rewards_array.shape[1])
                ],
                dtype=np.float64,
            )
            rewards_summary = {
                "sum": sums,
                "mean": means,
                "min": mins,
                "max": maxs,
            }

        # Store rewards for global tracking
        total_rewards_across_episodes.append(rewards_summary["sum"])

        # Log episode statistics
        episode_metrics = {}

        for i in range(len(rewards_summary["sum"])):
            episode_metrics[f"Agent_{i}_Episode_Reward_Sum"] = rewards_summary["sum"][i]
            episode_metrics[f"Agent_{i}_Episode_Reward_Mean"] = rewards_summary["mean"][i]
            episode_metrics[f"Agent_{i}_Episode_Reward_Min"] = rewards_summary["min"][i]
            episode_metrics[f"Agent_{i}_Episode_Reward_Max"] = rewards_summary["max"][i]

        episode_duration = time.time() - start_episode_time
        episode_metrics["Episode_Duration"] = episode_duration
        if metrics_active(mlflow):
            mlflow.log_metrics(episode_metrics, step=episode)
        elif self.local_metrics_logger:
            self.local_metrics_logger.log(episode_metrics, episode)

        logger.info(
            "Completed episode {}/{}, reward summary: {}, duration: {:.2f}s",
            episode + 1,
            episodes,
            rewards_summary,
            episode_duration,
        )

    # Aggregate rewards across episodes
    if len(total_rewards_across_episodes) == 0:
        total_rewards_matrix = np.zeros((0, 0), dtype=np.float64)
    else:
        max_agents = max(
            np.asarray(values).reshape(-1).shape[0] for values in total_rewards_across_episodes
        )
        total_rewards_matrix = np.full(
            (len(total_rewards_across_episodes), max_agents), np.nan, dtype=np.float64
        )
        for row, values in enumerate(total_rewards_across_episodes):
            vector = np.asarray(values, dtype=np.float64).reshape(-1)
            total_rewards_matrix[row, : vector.shape[0]] = vector

    # Compute overall statistics across episodes
    if total_rewards_matrix.size == 0:
        overall_rewards_summary = {
            "sum": np.array([], dtype=np.float64),
            "mean": np.array([], dtype=np.float64),
            "min": np.array([], dtype=np.float64),
            "max": np.array([], dtype=np.float64),
        }
    else:
        valid_mask = ~np.isnan(total_rewards_matrix)
        valid_counts = valid_mask.sum(axis=0)
        sums = np.nansum(total_rewards_matrix, axis=0)
        means = np.divide(sums, np.maximum(valid_counts, 1), where=np.maximum(valid_counts, 1) > 0)
        mins = np.array(
            [
                np.nanmin(total_rewards_matrix[:, i]) if valid_counts[i] > 0 else 0.0
                for i in range(total_rewards_matrix.shape[1])
            ],
            dtype=np.float64,
        )
        maxs = np.array(
            [
                np.nanmax(total_rewards_matrix[:, i]) if valid_counts[i] > 0 else 0.0
                for i in range(total_rewards_matrix.shape[1])
            ],
            dtype=np.float64,
        )
        overall_rewards_summary = {
            "sum": sums,
            "mean": means,
            "min": mins,
            "max": maxs,
        }

    # Log overall statistics
    overall_metrics = {}

    for i in range(len(overall_rewards_summary["sum"])):
        overall_metrics[f"Agent_{i}_Overall_Reward_Sum"] = overall_rewards_summary["sum"][i]
        overall_metrics[f"Agent_{i}_Overall_Reward_Mean"] = overall_rewards_summary["mean"][i]
        overall_metrics[f"Agent_{i}_Overall_Reward_Min"] = overall_rewards_summary["min"][i]
        overall_metrics[f"Agent_{i}_Overall_Reward_Max"] = overall_rewards_summary["max"][i]

    if metrics_active(mlflow):
        mlflow.log_metrics(overall_metrics)
    elif self.local_metrics_logger:
        # Use -1 to denote aggregate metrics when logging locally.
        self.local_metrics_logger.log(overall_metrics, -1)
    self._cancel_stall_watchdog()
    self._close_stall_watchdog_file()
