"""Shared resources service for the simulation controller; no agent-specific training."""

from __future__ import annotations

import faulthandler
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import psutil
import torch
from loguru import logger


class ResourceLifecycle:
    @staticmethod
    def _coerce_positive_int(value) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _coerce_non_negative_int(value) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _coerce_positive_float(value) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(parsed) or np.isinf(parsed) or parsed <= 0.0:
            return None
        return parsed

    @staticmethod
    def _optional_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        parsed = str(value).strip()
        return parsed or None

    def _resolve_progress_totals(self, episodes: int) -> tuple[Optional[int], Optional[int]]:
        state_points = self._coerce_positive_int(self.episode_time_steps)
        step_total = None if state_points is None else state_points - 1
        if step_total is None:
            return None, None
        return step_total, episodes * step_total

    def _progress_phase_in_window(self) -> bool:
        if not self.progress_updates_enabled or not self.progress_phase_updates_enabled:
            return False
        if self.global_step % self.progress_update_interval != 0:
            return False
        if (
            self.progress_phase_start_step is not None
            and self.global_step < self.progress_phase_start_step
        ):
            return False
        if (
            self.progress_phase_end_step is not None
            and self.global_step > self.progress_phase_end_step
        ):
            return False
        return True

    def _runtime_resource_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        try:
            rss_mb = self._process.memory_info().rss / (1024**2)
            snapshot["process_rss_mb"] = round(float(rss_mb), 3)
        except (OSError, psutil.Error, AttributeError):
            pass

        try:
            virtual_memory = psutil.virtual_memory()
            snapshot["system_available_ram_mb"] = round(
                float(virtual_memory.available / (1024**2)),
                3,
            )
            snapshot["system_ram_percent"] = round(float(virtual_memory.percent), 3)
        except (OSError, psutil.Error, AttributeError):
            pass

        if torch.cuda.is_available():
            try:
                snapshot["gpu_allocated_mb"] = round(
                    float(torch.cuda.memory_allocated() / (1024**2)),
                    3,
                )
                snapshot["gpu_reserved_mb"] = round(
                    float(torch.cuda.memory_reserved() / (1024**2)),
                    3,
                )
            except RuntimeError:
                pass

        return snapshot

    def _write_phase_progress(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Optional[List[float]] = None,
        status: str = "running",
        force: bool = False,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # A watchdog scope covers an episode boundary or a continuous window
        # of environment steps.  Individual predict/update/checkpoint phases
        # do not need their own timer: the traceback already identifies the
        # exact blocking call, and repeatedly rebuilding a native watchdog for
        # every subphase creates millions of needless operations on long runs.
        completes_watchdog_scope = phase in {
            "episode_reset_end",
            "episode_export_end",
            "episode_end",
        }
        self._update_stall_watchdog_for_phase(
            phase=phase,
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status=status,
            rewards=rewards,
        )

        try:
            if not force and not self._progress_phase_in_window():
                return

            payload_extra: Dict[str, Any] = {
                "phase": phase,
                "entity_topology_version": self._entity_topology_version,
                "entity_model_observations_direct": bool(
                    getattr(self, "_entity_model_observations_direct", False)
                ),
            }
            payload_extra.update(self._runtime_resource_snapshot())
            if extra:
                payload_extra.update(dict(extra))

            self.progress_tracker.update(
                episode=episode,
                step=step,
                global_step=self.global_step,
                rewards=rewards,
                episode_total=episode_total,
                step_total=step_total,
                global_step_total=global_step_total,
                status=status,
                extra=payload_extra,
            )
        finally:
            # Keep the watchdog armed until the completion progress write has
            # returned.  The previous ordering cancelled it before file I/O,
            # leaving a blocked progress write invisible to the watchdog.
            if completes_watchdog_scope and self.stall_watchdog_enabled:
                self._cancel_stall_watchdog()

    def _write_step_progress(
        self,
        *,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Any,
        step_duration: float,
        normal_step_duration: float,
        update_duration: float,
    ) -> None:
        """Write one lightweight progress heartbeat at the configured interval.

        Phase progress is optional and deliberately more detailed. Ordinary
        progress must remain observable when phase heartbeats are disabled,
        without restoring the former per-phase write amplification.
        """
        if not self.progress_updates_enabled:
            return
        if self.global_step % self.progress_update_interval != 0:
            return
        if self.progress_phase_updates_enabled:
            # ``step_end`` already emitted the same heartbeat through the
            # phase-progress path at this interval.
            return

        self.progress_tracker.update(
            episode=episode,
            step=step,
            global_step=self.global_step,
            rewards=rewards,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status="running",
            extra={
                "phase": "step_end",
                "step_duration_seconds": round(float(step_duration), 6),
                "normal_step_duration_seconds": round(float(normal_step_duration), 6),
                "update_duration_seconds": round(float(update_duration), 6),
                **self._runtime_resource_snapshot(),
            },
        )

    def _update_stall_watchdog_for_phase(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        status: str,
        rewards: Optional[List[float]],
    ) -> None:
        if not self.stall_watchdog_enabled:
            return

        force_refresh = phase in {"episode_reset_start", "episode_export_start"}
        step_refresh = phase == "step_start" and (
            self._stall_watchdog_armed_phase is None
            or self._stall_watchdog_context_is_due(self.global_step)
        )
        if force_refresh or step_refresh:
            self._arm_stall_watchdog(
                phase=phase,
                episode=episode,
                step=step,
                episode_total=episode_total,
                step_total=step_total,
                global_step_total=global_step_total,
                status=status,
                rewards=rewards,
            )
            return

    def _stall_watchdog_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        if (
            self._stall_watchdog_traceback_path is not None
            or self._stall_watchdog_context_path is not None
        ):
            return self._stall_watchdog_traceback_path, self._stall_watchdog_context_path

        raw_path = self.stall_watchdog_traceback_file
        base_dir: Optional[Path] = None
        if self.log_dir:
            base_dir = Path(self.log_dir)
        elif self.progress_tracker.progress_path is not None:
            base_dir = self.progress_tracker.progress_path.parent
        else:
            runtime_job_dir = (
                self.config.get("runtime", {}).get("job_dir")
                if isinstance(self.config, dict)
                else None
            )
            if runtime_job_dir:
                base_dir = Path(runtime_job_dir) / "logs"

        if raw_path:
            traceback_path = Path(raw_path)
            if not traceback_path.is_absolute():
                root = base_dir or Path.cwd()
                traceback_path = root / traceback_path
        elif base_dir is not None:
            safe_job_id = (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.job_id or "job")).strip("_") or "job"
            )
            traceback_path = base_dir / f"{safe_job_id}_stall_watchdog.log"
        else:
            traceback_path = None

        context_path = None
        if traceback_path is not None:
            suffix = traceback_path.suffix
            context_path = traceback_path.with_suffix(
                f"{suffix}.context.json" if suffix else ".context.json"
            )

        self._stall_watchdog_traceback_path = traceback_path
        self._stall_watchdog_context_path = context_path
        return traceback_path, context_path

    def _stall_watchdog_output_file(self):
        # stderr is captured by every supported executor and does not share
        # the progress/artifact filesystem that the watchdog is intended to
        # diagnose.  An explicit path remains supported for local workflows.
        if self.stall_watchdog_traceback_file is None:
            return sys.stderr

        traceback_path, _ = self._stall_watchdog_paths()
        if traceback_path is None:
            return sys.stderr

        if self._stall_watchdog_file_handle is None or self._stall_watchdog_file_handle.closed:
            try:
                traceback_path.parent.mkdir(parents=True, exist_ok=True)
                self._stall_watchdog_file_handle = traceback_path.open(
                    "a", encoding="utf-8", buffering=1
                )
            except OSError as exc:
                logger.warning("Failed to open stall watchdog log {}: {}", traceback_path, exc)
                return sys.stderr

        return self._stall_watchdog_file_handle

    def _stall_watchdog_context_is_due(self, global_step: int, *, force: bool = False) -> bool:
        if force:
            return True
        interval = max(int(self.stall_watchdog_context_interval_steps or 1), 1)
        last_global_step = self._stall_watchdog_last_context_global_step
        return last_global_step is None or (global_step - last_global_step) >= interval

    def _write_stall_watchdog_context(self, context: Mapping[str, Any]) -> None:
        _, context_path = self._stall_watchdog_paths()
        if context_path is None:
            return

        try:
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(
                json.dumps(dict(context), indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Failed to write stall watchdog context {}: {}", context_path, exc)
        else:
            self._stall_watchdog_last_context_global_step = (
                self._coerce_non_negative_int(context.get("global_step")) or 0
            )

    def _arm_stall_watchdog(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        status: str,
        rewards: Optional[List[float]],
    ) -> None:
        timeout = self.stall_watchdog_timeout_seconds
        if timeout is None or timeout <= 0:
            return

        context: Dict[str, Any] = {
            "job_id": self.job_id,
            "phase": phase,
            "status": status,
            "episode": episode,
            "episode_current": max(0, episode) + 1,
            "step": step,
            "step_current": max(0, step) + 1,
            "global_step": self.global_step,
            "episode_total": episode_total,
            "step_total": step_total,
            "global_step_total": global_step_total,
            "timeout_seconds": float(timeout),
            "exit_on_timeout": self.stall_watchdog_exit_on_timeout,
            "repeat": self.stall_watchdog_repeat,
            "entity_topology_version": self._entity_topology_version,
            "entity_model_observations_direct": bool(
                getattr(self, "_entity_model_observations_direct", False)
            ),
        }
        if rewards is not None:
            context["rewards"] = list(rewards)

        # Arm first.  If context/progress I/O itself blocks, the watchdog must
        # already be capable of dumping to the executor's stderr and exiting.
        try:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                float(timeout),
                repeat=self.stall_watchdog_repeat,
                file=self._stall_watchdog_output_file(),
                exit=self.stall_watchdog_exit_on_timeout,
            )
            self._stall_watchdog_armed_phase = phase
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("Failed to arm stall watchdog for phase {}: {}", phase, exc)

        force_context = phase in {"episode_reset_start", "episode_export_start"}
        if self._stall_watchdog_context_is_due(self.global_step, force=force_context):
            context.update(self._runtime_resource_snapshot())
            self._write_stall_watchdog_context(context)

    def _cancel_stall_watchdog(self) -> None:
        if not self.stall_watchdog_enabled:
            return

        try:
            faulthandler.cancel_dump_traceback_later()
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to cancel stall watchdog: {}", exc)
        self._stall_watchdog_armed_phase = None

    def _close_stall_watchdog_file(self) -> None:
        handle = self._stall_watchdog_file_handle
        if handle is None or handle is sys.stderr:
            return

        try:
            if not handle.closed:
                handle.close()
        except OSError as exc:
            logger.warning("Failed to close stall watchdog log handle: {}", exc)
        finally:
            self._stall_watchdog_file_handle = None

    def _enforce_resource_guards(
        self,
        *,
        phase: str,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
    ) -> None:
        if not self.resource_guard_enabled:
            return

        snapshot = self._runtime_resource_snapshot()
        failures: List[str] = []
        rss_mb = snapshot.get("process_rss_mb")
        available_mb = snapshot.get("system_available_ram_mb")
        if (
            self.max_process_rss_mb is not None
            and rss_mb is not None
            and rss_mb > self.max_process_rss_mb
        ):
            failures.append(
                f"process RSS {rss_mb:.1f} MB exceeded limit {self.max_process_rss_mb:.1f} MB"
            )
        if (
            self.min_available_ram_mb is not None
            and available_mb is not None
            and available_mb < self.min_available_ram_mb
        ):
            failures.append(
                f"available RAM {available_mb:.1f} MB below limit {self.min_available_ram_mb:.1f} MB"
            )
        if not failures:
            return

        message = "; ".join(failures)
        self._write_phase_progress(
            phase=phase,
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            status="failed",
            force=True,
            extra={
                "error_type": "ResourceGuardError",
                "error_message": message,
                **snapshot,
            },
        )
        self._cancel_stall_watchdog()
        raise MemoryError(message)

    def _enforce_step_duration_guard(
        self,
        *,
        step_duration: float,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Optional[List[float]],
    ) -> None:
        if self.max_step_seconds is None or step_duration <= self.max_step_seconds:
            return

        message = (
            f"Step duration {step_duration:.3f}s exceeded configured limit "
            f"{self.max_step_seconds:.3f}s at global step {self.global_step}."
        )
        self._write_phase_progress(
            phase="step_duration_guard",
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            rewards=rewards,
            status="failed",
            force=True,
            extra={
                "error_type": "StepDurationGuardError",
                "error_message": message,
                "step_duration_seconds": round(float(step_duration), 6),
                "max_step_seconds": round(float(self.max_step_seconds), 6),
            },
        )
        self._cancel_stall_watchdog()
        raise TimeoutError(message)

    def _enforce_update_duration_guard(
        self,
        *,
        update_duration: float,
        episode: int,
        step: int,
        episode_total: Optional[int],
        step_total: Optional[int],
        global_step_total: Optional[int],
        rewards: Optional[List[float]],
    ) -> None:
        if self.max_update_seconds is None or update_duration <= self.max_update_seconds:
            return

        message = (
            f"Update duration {update_duration:.3f}s exceeded configured limit "
            f"{self.max_update_seconds:.3f}s at global step {self.global_step}."
        )
        self._write_phase_progress(
            phase="update_duration_guard",
            episode=episode,
            step=step,
            episode_total=episode_total,
            step_total=step_total,
            global_step_total=global_step_total,
            rewards=rewards,
            status="failed",
            force=True,
            extra={
                "error_type": "UpdateDurationGuardError",
                "error_message": message,
                "update_duration_seconds": round(float(update_duration), 6),
                "max_update_seconds": round(float(self.max_update_seconds), 6),
            },
        )
        self._cancel_stall_watchdog()
        raise TimeoutError(message)

    def _synchronize_model_cuda_for_timing(self) -> None:
        if getattr(getattr(self.model, "device", None), "type", None) == "cuda":
            torch.cuda.synchronize()
