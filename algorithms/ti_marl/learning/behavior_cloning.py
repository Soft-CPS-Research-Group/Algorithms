"""Typed SMART demonstrations used only to warm-start the TI-MARL actor."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import math
import pickle
from random import Random
import time
from typing import Any, Callable, Mapping, Sequence, Tuple
import zlib

import torch
from loguru import logger
from torch import nn

from algorithms.ti_marl.contracts.models import InterfaceSnapshot, LocalActionBundle
from algorithms.ti_marl.policy.networks import TypedActor


@dataclass(frozen=True)
class TypedDemonstration:
    snapshot: InterfaceSnapshot
    bundles: Tuple[LocalActionBundle, ...]


@dataclass(frozen=True)
class _PreparedDemonstration:
    demonstration: TypedDemonstration
    actions: Mapping[str, Mapping[str, Any]]


class TypedBehaviorCloningWarmStart:
    """Reservoir-store typed demonstrations and pretrain the shared actor.

    Demonstration transitions are intentionally absent: this component never
    writes to the PPO rollout and never trains the centralized critic.
    """

    def __init__(
        self,
        *,
        demonstration_episodes: int,
        max_samples: int,
        pretraining_epochs: int,
        batch_size: int,
        learning_rate: float,
        balance_action_modes: bool,
        mode_balance_exponent: float,
        max_mode_weight: float,
        seed: int,
        balanced_loss_kind: str = "weighted",
        calibration_epochs: int = 0,
        calibration_learning_rate: float | None = None,
    ) -> None:
        self.demonstration_episodes = int(demonstration_episodes)
        self.max_samples = int(max_samples)
        self.pretraining_epochs = int(pretraining_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.balance_action_modes = bool(balance_action_modes)
        self.mode_balance_exponent = float(mode_balance_exponent)
        self.max_mode_weight = float(max_mode_weight)
        if balanced_loss_kind not in {"weighted", "hierarchical_mode_mean"}:
            raise ValueError(
                "balanced_loss_kind must be 'weighted' or "
                "'hierarchical_mode_mean'"
            )
        self.balanced_loss_kind = str(balanced_loss_kind)
        self.calibration_epochs = int(calibration_epochs)
        self.calibration_learning_rate = (
            self.learning_rate
            if calibration_learning_rate is None
            else float(calibration_learning_rate)
        )
        self._rng = Random(int(seed))
        self._demonstrations: list[TypedDemonstration] = []
        self.seen_samples = 0
        self.training_samples = 0
        self.mode_counts: Counter[tuple[str, str]] = Counter()
        self.mode_weights: dict[tuple[str, str], float] = {}
        self.pretraining_complete = False
        self.latest_loss = 0.0
        self.latest_batches = 0
        self.latest_balanced_loss = 0.0
        self.latest_balanced_batches = 0
        self.latest_calibration_loss = 0.0
        self.latest_calibration_batches = 0
        self.mode_diagnostic_metrics: dict[str, float] = {}

    def record(
        self,
        snapshot: InterfaceSnapshot,
        bundles: Sequence[LocalActionBundle],
    ) -> None:
        if self.pretraining_complete:
            raise RuntimeError("Cannot append demonstrations after TI-MARL BC pretraining")
        demonstration = TypedDemonstration(snapshot, tuple(bundles))
        demonstration_counts = self._demonstration_mode_counts(demonstration)
        self.seen_samples += 1
        if len(self._demonstrations) < self.max_samples:
            self._demonstrations.append(demonstration)
            self.mode_counts.update(demonstration_counts)
            return
        replacement = self._rng.randrange(self.seen_samples)
        if replacement < self.max_samples:
            evicted_counts = self._demonstration_mode_counts(
                self._demonstrations[replacement]
            )
            self.mode_counts.subtract(evicted_counts)
            self.mode_counts = Counter(
                {
                    key: count
                    for key, count in self.mode_counts.items()
                    if count > 0
                }
            )
            self._demonstrations[replacement] = demonstration
            self.mode_counts.update(demonstration_counts)

    @staticmethod
    def _demonstration_mode_counts(
        demonstration: TypedDemonstration,
    ) -> Counter[tuple[str, str]]:
        group_types = {
            (group.owner_agent_id, group.group_id): group.group_type
            for group in demonstration.snapshot.action_groups
        }
        counts: Counter[tuple[str, str]] = Counter()
        for bundle in demonstration.bundles:
            for decision in bundle.decisions:
                group_type = group_types.get((bundle.agent_id, decision.group_id))
                if group_type is None:
                    raise ValueError(
                        "Typed demonstration references an unknown action group: "
                        f"{bundle.agent_id}/{decision.group_id}"
                    )
                counts[(group_type, decision.mode)] += 1
        return counts

    def pretrain(
        self,
        actor: TypedActor,
        *,
        max_grad_norm: float,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> Mapping[str, float]:
        if self.pretraining_complete:
            return self.metrics()
        if not self._demonstrations:
            raise RuntimeError("TI-MARL behavior cloning has zero typed demonstrations")
        optimizer = torch.optim.Adam(actor.parameters(), lr=self.learning_rate)
        self.mode_weights = self._build_mode_weights()
        training_batches = self._prepare_training_batches()
        total_epochs = self.pretraining_epochs + self.calibration_epochs
        actor.train()
        actor.encoder.begin_replay_preparation_cache()
        self._report_progress(
            progress_callback,
            phase="behavior_cloning_prepare",
            epoch_current=0,
            epoch_total=total_epochs,
            batches_complete=0,
            batches_total=len(training_batches) * total_epochs,
            loss=None,
        )
        try:
            balanced_losses, balanced_batches = self._train_epochs(
                actor,
                optimizer,
                training_batches=training_batches,
                epochs=self.pretraining_epochs,
                epoch_offset=0,
                total_epochs=total_epochs,
                phase="behavior_cloning_balanced",
                mode_weights=self.mode_weights,
                loss_kind=self.balanced_loss_kind,
                max_grad_norm=max_grad_norm,
                progress_callback=progress_callback,
            )
            calibration_losses: list[float] = []
            calibration_batches = 0
            if self.calibration_epochs > 0:
                for group in optimizer.param_groups:
                    group["lr"] = self.calibration_learning_rate
                calibration_losses, calibration_batches = self._train_epochs(
                    actor,
                    optimizer,
                    training_batches=training_batches,
                    epochs=self.calibration_epochs,
                    epoch_offset=self.pretraining_epochs,
                    total_epochs=total_epochs,
                    phase="behavior_cloning_calibration",
                    mode_weights={key: 1.0 for key in self.mode_counts},
                    loss_kind="weighted",
                    max_grad_norm=max_grad_norm,
                    progress_callback=progress_callback,
                )
            losses = balanced_losses + calibration_losses
            batches = balanced_batches + calibration_batches
            if batches == 0:
                raise RuntimeError(
                    "TI-MARL behavior cloning produced zero trainable batches"
                )
            self.latest_loss = float(sum(losses) / len(losses))
            self.latest_batches = int(batches)
            self.latest_balanced_loss = float(
                sum(balanced_losses) / len(balanced_losses)
            )
            self.latest_balanced_batches = int(balanced_batches)
            self.latest_calibration_loss = (
                0.0
                if not calibration_losses
                else float(sum(calibration_losses) / len(calibration_losses))
            )
            self.latest_calibration_batches = int(calibration_batches)
            self.training_samples = len(self._demonstrations)
            self.mode_diagnostic_metrics = self._evaluate_mode_diagnostics(
                actor,
                training_batches=training_batches,
            )
            self.pretraining_complete = True
            self._report_progress(
                progress_callback,
                phase="behavior_cloning_complete",
                epoch_current=total_epochs,
                epoch_total=total_epochs,
                batches_complete=batches,
                batches_total=len(training_batches) * total_epochs,
                loss=self.latest_loss,
            )
        finally:
            actor.encoder.end_replay_preparation_cache()

        # The demonstrations have already been distilled into the actor. PPO
        # never revisits them, so retaining typed snapshots would inflate every
        # subsequent checkpoint without changing resume semantics.
        self._demonstrations.clear()
        return self.metrics()

    def _prepare_training_batches(
        self,
    ) -> tuple[tuple[_PreparedDemonstration, ...], ...]:
        indices = list(range(len(self._demonstrations)))
        self._rng.shuffle(indices)
        prepared: list[_PreparedDemonstration] = []
        for index in indices:
            demonstration = self._demonstrations[index]
            actions = {
                bundle.agent_id: {
                    decision.group_id: decision
                    for decision in bundle.decisions
                }
                for bundle in demonstration.bundles
            }
            prepared.append(_PreparedDemonstration(demonstration, actions))
        return tuple(
            tuple(prepared[start : start + self.batch_size])
            for start in range(0, len(prepared), self.batch_size)
        )

    def _train_epochs(
        self,
        actor: TypedActor,
        optimizer: torch.optim.Optimizer,
        *,
        training_batches: Sequence[Sequence[_PreparedDemonstration]],
        epochs: int,
        epoch_offset: int,
        total_epochs: int,
        phase: str,
        mode_weights: Mapping[tuple[str, str], float],
        loss_kind: str,
        max_grad_norm: float,
        progress_callback: Callable[[Mapping[str, Any]], None] | None,
    ) -> tuple[list[float], int]:
        losses: list[float] = []
        batches = 0
        total_batch_count = len(training_batches) * int(total_epochs)
        for epoch in range(int(epochs)):
            epoch_started = time.monotonic()
            batch_indices = list(range(len(training_batches)))
            self._rng.shuffle(batch_indices)
            epoch_losses: list[float] = []
            for batch_index in batch_indices:
                prepared_batch = training_batches[batch_index]
                demos = [item.demonstration for item in prepared_batch]
                evaluation = actor.evaluate_actions_many(
                    tuple(
                        (item.demonstration.snapshot, item.actions)
                        for item in prepared_batch
                    )
                )
                sample_losses: list[tuple[str, str, torch.Tensor]] = []
                for demo, group_log_probs in zip(
                    demos, evaluation.log_prob_by_group_step
                ):
                    groups = {
                        (group.owner_agent_id, group.group_id): group
                        for group in demo.snapshot.action_groups
                    }
                    for bundle in demo.bundles:
                        for decision in bundle.decisions:
                            group = groups[(bundle.agent_id, decision.group_id)]
                            log_prob = group_log_probs[bundle.agent_id][
                                decision.group_id
                            ]
                            weight = mode_weights.get(
                                (group.group_type, decision.mode), 1.0
                            )
                            sample_losses.append(
                                (
                                    group.group_type,
                                    decision.mode,
                                    -log_prob * float(weight),
                                )
                            )
                if not sample_losses:
                    continue
                loss = self._reduce_losses(sample_losses, loss_kind=loss_kind)
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Non-finite TI-MARL behavior-cloning loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(actor.parameters(), float(max_grad_norm))
                if not bool(torch.isfinite(torch.as_tensor(gradient))):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        "Non-finite TI-MARL behavior-cloning gradient"
                    )
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                epoch_losses.append(loss_value)
                batches += 1
            if not epoch_losses:
                raise RuntimeError(
                    "TI-MARL behavior cloning produced no trainable batch in "
                    f"epoch {int(epoch_offset) + epoch + 1}"
                )
            absolute_epoch = int(epoch_offset) + epoch + 1
            epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
            epoch_duration = time.monotonic() - epoch_started
            logger.info(
                "event=ti_marl_bc_epoch phase={} epoch={}/{} batches={} "
                "loss={:.8f} duration_seconds={:.3f}",
                phase,
                absolute_epoch,
                total_epochs,
                len(epoch_losses),
                epoch_loss,
                epoch_duration,
            )
            self._report_progress(
                progress_callback,
                phase=phase,
                epoch_current=absolute_epoch,
                epoch_total=total_epochs,
                batches_complete=(int(epoch_offset) + epoch + 1)
                * len(training_batches),
                batches_total=total_batch_count,
                loss=epoch_loss,
                epoch_duration_seconds=epoch_duration,
            )
        return losses, batches

    @staticmethod
    def _report_progress(
        callback: Callable[[Mapping[str, Any]], None] | None,
        **payload: Any,
    ) -> None:
        if callback is None:
            return
        try:
            callback(payload)
        except Exception as exc:  # progress is best-effort telemetry
            logger.warning("Unable to report TI-MARL BC progress: {}", exc)

    @staticmethod
    def _reduce_losses(
        losses: Sequence[tuple[str, str, torch.Tensor]],
        *,
        loss_kind: str,
    ) -> torch.Tensor:
        """Reduce typed BC losses without letting IDLE frequency dominate.

        ``hierarchical_mode_mean`` first averages examples of each action mode,
        then modes within one group type, and finally group types. A rare
        deferrable START therefore receives a useful gradient without making a
        group type with more modes dominate the entire actor update.
        """

        if loss_kind == "weighted":
            return torch.stack([loss for _group, _mode, loss in losses]).mean()
        if loss_kind != "hierarchical_mode_mean":
            raise ValueError(f"Unsupported BC loss kind: {loss_kind!r}")
        by_group: dict[str, dict[str, list[torch.Tensor]]] = {}
        for group_type, mode, loss in losses:
            by_group.setdefault(group_type, {}).setdefault(mode, []).append(loss)
        group_losses = []
        for modes in by_group.values():
            mode_losses = [torch.stack(items).mean() for items in modes.values()]
            group_losses.append(torch.stack(mode_losses).mean())
        return torch.stack(group_losses).mean()

    def in_demonstration_phase(self, *, episode: int, training: bool) -> bool:
        return bool(
            training
            and not self.pretraining_complete
            and int(episode) < self.demonstration_episodes
        )

    def metrics(self) -> Mapping[str, float]:
        metrics = {
            "bc_demonstration_samples": float(
                self.training_samples
                if self.pretraining_complete
                else len(self._demonstrations)
            ),
            "bc_seen_samples": float(self.seen_samples),
            "bc_pretraining_complete": float(self.pretraining_complete),
            "bc_pretraining_loss": float(self.latest_loss),
            "bc_pretraining_batches": float(self.latest_batches),
            "bc_balanced_loss": float(self.latest_balanced_loss),
            "bc_balanced_batches": float(self.latest_balanced_batches),
            "bc_calibration_loss": float(self.latest_calibration_loss),
            "bc_calibration_batches": float(self.latest_calibration_batches),
            (
                "bc_balanced_loss_kind_"
                f"{self.balanced_loss_kind}"
            ): 1.0,
        }
        for (group_type, mode), count in sorted(self.mode_counts.items()):
            key = f"bc_mode_count_{group_type.lower()}_{mode.lower()}"
            metrics[key] = float(count)
            if (group_type, mode) in self.mode_weights:
                metrics[key.replace("count", "weight")] = float(
                    self.mode_weights[(group_type, mode)]
                )
        metrics.update(self.mode_diagnostic_metrics)
        return metrics

    def _evaluate_mode_diagnostics(
        self,
        actor: TypedActor,
        *,
        training_batches: Sequence[Sequence[_PreparedDemonstration]],
    ) -> dict[str, float]:
        totals: Counter[tuple[str, str]] = Counter()
        correct: Counter[tuple[str, str]] = Counter()
        predicted: Counter[tuple[str, str]] = Counter()
        target_probability: Counter[tuple[str, str]] = Counter()
        fraction_absolute_error: Counter[tuple[str, str]] = Counter()
        fraction_signed_error: Counter[tuple[str, str]] = Counter()
        fraction_count: Counter[tuple[str, str]] = Counter()
        actor.eval()
        with torch.no_grad():
            for prepared_batch in training_batches:
                demos = [item.demonstration for item in prepared_batch]
                evaluation = actor.evaluate_actions_many(
                    tuple(
                        (item.demonstration.snapshot, item.actions)
                        for item in prepared_batch
                    )
                )
                for step_index, demo in enumerate(demos):
                    groups = {
                        (group.owner_agent_id, group.group_id): group
                        for group in demo.snapshot.action_groups
                    }
                    for bundle in demo.bundles:
                        for decision in bundle.decisions:
                            group = groups[(bundle.agent_id, decision.group_id)]
                            key = (group.group_type, decision.mode)
                            predicted_index = int(
                                evaluation.predicted_mode_by_group_step[step_index][
                                    bundle.agent_id
                                ][decision.group_id].item()
                            )
                            predicted_mode = actor.group_modes[group.group_type][
                                predicted_index
                            ]
                            totals[key] += 1
                            correct[key] += int(predicted_mode == decision.mode)
                            predicted[(group.group_type, predicted_mode)] += 1
                            target_probability[key] += float(
                                evaluation.mode_log_prob_by_group_step[step_index][
                                    bundle.agent_id
                                ][decision.group_id].exp().cpu()
                            )
                            if decision.mode.startswith(
                                ("CHARGE_", "DISCHARGE_")
                            ):
                                predicted_fraction = float(
                                    evaluation.predicted_fraction_by_group_step[
                                        step_index
                                    ][bundle.agent_id][decision.group_id].cpu()
                                )
                                error = predicted_fraction - float(
                                    decision.fraction
                                )
                                fraction_absolute_error[key] += abs(error)
                                fraction_signed_error[key] += error
                                fraction_count[key] += 1
        actor.train()
        metrics: dict[str, float] = {}
        for key in sorted(set(totals) | set(predicted)):
            count = totals[key]
            group_type, mode = key
            prefix = f"bc_mode_{group_type.lower()}_{mode.lower()}"
            metrics[f"{prefix}_predicted_count"] = float(predicted[key])
            if count:
                metrics[f"{prefix}_recall"] = float(correct[key] / count)
                metrics[f"{prefix}_target_probability"] = float(
                    target_probability[key] / count
                )
            if fraction_count[key]:
                metrics[f"{prefix}_fraction_mae"] = float(
                    fraction_absolute_error[key] / fraction_count[key]
                )
                metrics[f"{prefix}_fraction_bias"] = float(
                    fraction_signed_error[key] / fraction_count[key]
                )
        return metrics

    def _build_mode_weights(self) -> dict[tuple[str, str], float]:
        if self.balanced_loss_kind == "hierarchical_mode_mean":
            return {key: 1.0 for key in self.mode_counts}
        if not self.balance_action_modes:
            return {key: 1.0 for key in self.mode_counts}
        result: dict[tuple[str, str], float] = {}
        group_types = sorted({key[0] for key in self.mode_counts})
        for group_type in group_types:
            counts = {
                mode: count
                for (candidate_type, mode), count in self.mode_counts.items()
                if candidate_type == group_type and count > 0
            }
            if not counts:
                continue
            raw = {
                mode: float(count) ** (-self.mode_balance_exponent)
                for mode, count in counts.items()
            }
            normalizer = sum(counts.values()) / sum(
                counts[mode] * raw[mode] for mode in counts
            )
            for mode, value in raw.items():
                result[(group_type, mode)] = min(
                    value * normalizer,
                    self.max_mode_weight,
                )
        if not all(math.isfinite(value) and value > 0.0 for value in result.values()):
            raise FloatingPointError("Invalid TI-MARL behavior-cloning mode weights")
        return result

    def state_dict(self) -> Mapping[str, Any]:
        serialized_demonstrations = pickle.dumps(
            tuple(self._demonstrations),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        compressed_demonstrations = zlib.compress(
            serialized_demonstrations,
            level=1,
        )
        return {
            "format": "ti_marl_behavior_cloning_v2",
            "demonstrations_codec": "pickle_zlib_v1",
            "demonstrations_zlib": compressed_demonstrations,
            "demonstrations_sha256": hashlib.sha256(
                compressed_demonstrations
            ).hexdigest(),
            "seen_samples": self.seen_samples,
            "training_samples": self.training_samples,
            "mode_counts": dict(self.mode_counts),
            "mode_weights": dict(self.mode_weights),
            "rng_state": self._rng.getstate(),
            "pretraining_complete": self.pretraining_complete,
            "latest_loss": self.latest_loss,
            "latest_batches": self.latest_batches,
            "latest_balanced_loss": self.latest_balanced_loss,
            "latest_balanced_batches": self.latest_balanced_batches,
            "latest_calibration_loss": self.latest_calibration_loss,
            "latest_calibration_batches": self.latest_calibration_batches,
            "balanced_loss_kind": self.balanced_loss_kind,
            "mode_diagnostic_metrics": dict(self.mode_diagnostic_metrics),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        checkpoint_format = payload.get("format")
        if checkpoint_format not in {
            "ti_marl_behavior_cloning_v1",
            "ti_marl_behavior_cloning_v2",
        }:
            raise ValueError("Unsupported TI-MARL behavior-cloning checkpoint format")
        checkpoint_loss_kind = str(
            payload.get("balanced_loss_kind", "weighted")
        )
        if checkpoint_loss_kind != self.balanced_loss_kind:
            raise ValueError(
                "TI-MARL BC balanced_loss_kind differs from the checkpoint"
            )
        if checkpoint_format == "ti_marl_behavior_cloning_v2":
            if payload.get("demonstrations_codec") != "pickle_zlib_v1":
                raise ValueError(
                    "Unsupported TI-MARL behavior-cloning demonstration codec"
                )
            compressed = payload.get("demonstrations_zlib", b"")
            if not isinstance(compressed, bytes):
                raise ValueError("Invalid compressed TI-MARL demonstrations")
            expected_hash = str(payload.get("demonstrations_sha256", ""))
            if hashlib.sha256(compressed).hexdigest() != expected_hash:
                raise ValueError("Corrupt compressed TI-MARL demonstrations")
            try:
                demonstrations = list(
                    pickle.loads(zlib.decompress(compressed))
                )
            except (
                EOFError,
                pickle.UnpicklingError,
                TypeError,
                ValueError,
                zlib.error,
            ) as exc:
                raise ValueError(
                    "Unable to decode compressed TI-MARL demonstrations"
                ) from exc
        else:
            demonstrations = list(payload.get("demonstrations", ()))
        if len(demonstrations) > self.max_samples:
            raise ValueError("TI-MARL BC checkpoint exceeds configured sample capacity")
        self._demonstrations = demonstrations
        self.seen_samples = int(payload.get("seen_samples", len(demonstrations)))
        self.training_samples = int(
            payload.get(
                "training_samples",
                self.seen_samples if payload.get("pretraining_complete", False) else 0,
            )
        )
        self.mode_counts = Counter(payload.get("mode_counts", {}))
        self.mode_weights = {
            tuple(key): float(value)
            for key, value in dict(payload.get("mode_weights", {})).items()
        }
        self._rng.setstate(payload["rng_state"])
        self.pretraining_complete = bool(payload.get("pretraining_complete", False))
        self.latest_loss = float(payload.get("latest_loss", 0.0))
        self.latest_batches = int(payload.get("latest_batches", 0))
        self.latest_balanced_loss = float(
            payload.get("latest_balanced_loss", self.latest_loss)
        )
        self.latest_balanced_batches = int(
            payload.get("latest_balanced_batches", self.latest_batches)
        )
        self.latest_calibration_loss = float(
            payload.get("latest_calibration_loss", 0.0)
        )
        self.latest_calibration_batches = int(
            payload.get("latest_calibration_batches", 0)
        )
        self.mode_diagnostic_metrics = {
            str(key): float(value)
            for key, value in dict(
                payload.get("mode_diagnostic_metrics", {})
            ).items()
        }
        if self.pretraining_complete:
            self._demonstrations.clear()
