from __future__ import annotations

import random
from collections import deque
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from algorithms.transformer_matd3.types import (
    ArrayTuple,
    LayoutSignature,
    ReplayBatch,
    ReplayTransition,
)


class SignatureBucketedReplayBuffer:
    STATE_FORMAT = "signature_bucketed_v1"

    def __init__(self, capacity: int, num_agents: int, batch_size: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if num_agents <= 0:
            raise ValueError("replay num_agents must be positive")
        if batch_size <= 0:
            raise ValueError("replay batch_size must be positive")
        if batch_size > capacity:
            raise ValueError("replay batch_size cannot exceed capacity")
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.batch_size = int(batch_size)
        self._buckets: dict[LayoutSignature, deque[ReplayTransition]] = {}
        self._global_fifo: deque[tuple[int, LayoutSignature]] = deque()
        self._next_sequence_id = 0
        self._total_size = 0
        self._rng = random.Random()

    def push(
        self,
        *,
        encoded_obs: Sequence[Any],
        next_encoded_obs: Sequence[Any],
        actions: Optional[Sequence[Any]] = None,
        reward: Any,
        terminated: Any,
        truncated: Any,
        layout_signature: LayoutSignature,
        proposed_actions: Optional[Sequence[Any]] = None,
        executed_actions: Optional[Sequence[Any]] = None,
        base_actions: Optional[Sequence[Any]] = None,
        behavior_actions: Optional[Sequence[Any]] = None,
        next_behavior_actions: Optional[Sequence[Any]] = None,
        cloning_actions: Optional[Sequence[Any]] = None,
    ) -> None:
        self._validate_signature(layout_signature)
        observations = self._coerce_agent_vectors(encoded_obs, "encoded_obs")
        next_observations = self._coerce_agent_vectors(
            next_encoded_obs,
            "next_encoded_obs",
        )
        legacy_actions = actions
        if legacy_actions is None:
            legacy_actions = executed_actions or proposed_actions
        if legacy_actions is None:
            raise ValueError(
                "replay push requires actions or an explicit proposed/executed action"
            )
        action_vectors = self._coerce_agent_vectors(legacy_actions, "action")
        proposed_vectors = (
            action_vectors
            if proposed_actions is None
            else self._coerce_agent_vectors(proposed_actions, "proposed_actions")
        )
        executed_vectors = (
            action_vectors
            if executed_actions is None
            else self._coerce_agent_vectors(executed_actions, "executed_actions")
        )
        base_vectors = self._coerce_optional_actions(base_actions, "base_actions")
        self._validate_transition_shapes(
            layout_signature,
            observations,
            next_observations,
            action_vectors,
            proposed_vectors,
            executed_vectors,
        )
        optional_vectors = (
            self._coerce_optional_actions(behavior_actions, "behavior_actions"),
            self._coerce_optional_actions(
                next_behavior_actions,
                "next_behavior_actions",
            ),
            self._coerce_optional_actions(cloning_actions, "cloning_actions"),
        )
        self._validate_optional_presence(
            layout_signature,
            (*optional_vectors, base_vectors, proposed_vectors, executed_vectors),
        )
        transition = ReplayTransition(
            sequence_id=self._next_sequence_id,
            signature=layout_signature,
            observations=observations,
            next_observations=next_observations,
            actions=action_vectors,
            rewards=self._coerce_agent_values(reward, np.float32, "reward"),
            terminated=self._coerce_agent_values(
                terminated,
                np.bool_,
                "terminated",
                broadcast_scalar=True,
            ),
            truncated=self._coerce_agent_values(
                truncated,
                np.bool_,
                "truncated",
                broadcast_scalar=True,
            ),
            proposed_actions=proposed_vectors,
            executed_actions=executed_vectors,
            base_actions=base_vectors,
            behavior_actions=optional_vectors[0],
            next_behavior_actions=optional_vectors[1],
            cloning_actions=optional_vectors[2],
        )
        self._evict_if_full()
        self._append_transition(transition)
        self._next_sequence_id += 1

    def sample(self, signature: LayoutSignature, k: int) -> ReplayBatch:
        self._validate_signature(signature)
        if k <= 0:
            raise ValueError("sample size must be positive")
        bucket = self._buckets.get(signature, ())
        if len(bucket) < k:
            raise ValueError(
                f"signature bucket contains {len(bucket)} transitions; "
                f"cannot sample {k}"
            )
        transitions = self._rng.sample(list(bucket), k)
        return self._build_batch(signature, transitions)

    def signatures(self) -> Iterable[LayoutSignature]:
        return tuple(self._buckets.keys())

    def bucket_size(self, signature: LayoutSignature) -> int:
        self._validate_signature(signature)
        return len(self._buckets.get(signature, ()))

    def total_size(self) -> int:
        return self._total_size

    def get_state(self) -> dict[str, Any]:
        transitions_by_id = {
            transition.sequence_id: transition
            for bucket in self._buckets.values()
            for transition in bucket
        }
        transitions = [
            self._copy_transition(transitions_by_id[sequence_id])
            for sequence_id, _ in self._global_fifo
        ]
        buckets = {
            signature: tuple(
                self._copy_transition(transition) for transition in bucket
            )
            for signature, bucket in self._buckets.items()
        }
        return {
            "format": self.STATE_FORMAT,
            "capacity": self.capacity,
            "num_agents": self.num_agents,
            "batch_size": self.batch_size,
            "next_sequence_id": self._next_sequence_id,
            "total_size": self._total_size,
            "transitions": transitions,
            "buckets": buckets,
            "global_fifo": tuple(self._global_fifo),
            "rng_state": self._rng.getstate(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("replay state must be a mapping")
        state_format = state.get("format", self.STATE_FORMAT)
        if state_format != self.STATE_FORMAT:
            raise ValueError(f"unsupported replay state format: {state_format!r}")
        for field in ("capacity", "num_agents", "batch_size"):
            value = state.get(field)
            expected = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected
            ):
                raise ValueError(
                    f"replay state {field}={value!r} does not match "
                    f"buffer value {expected!r}"
                )
        transitions = state.get("transitions")
        if not isinstance(transitions, list) or len(transitions) > self.capacity:
            raise ValueError("replay state transitions are invalid")
        try:
            restored = [self._copy_transition(item) for item in transitions]
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("replay state contains an invalid transition") from exc
        for transition in restored:
            self._validate_sequence_id(transition.sequence_id)
        sequence_ids = [item.sequence_id for item in restored]
        if sequence_ids != sorted(set(sequence_ids)):
            raise ValueError(
                "replay transition sequence IDs must be unique and ordered"
            )
        next_sequence_id = state.get("next_sequence_id")
        if (
            isinstance(next_sequence_id, bool)
            or not isinstance(next_sequence_id, int)
            or next_sequence_id < 0
        ):
            raise ValueError("replay next_sequence_id must be non-negative")
        if sequence_ids and next_sequence_id <= sequence_ids[-1]:
            raise ValueError("replay next_sequence_id must exceed stored sequence IDs")
        restored_rng = random.Random()
        try:
            restored_rng.setstate(state["rng_state"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("replay rng_state is invalid") from exc
        candidate = SignatureBucketedReplayBuffer(
            capacity=self.capacity,
            num_agents=self.num_agents,
            batch_size=self.batch_size,
        )
        for transition in restored:
            candidate._validate_restored_transition(transition)
            candidate._append_transition(transition)
        if "total_size" in state:
            total_size = state["total_size"]
            if (
                isinstance(total_size, bool)
                or not isinstance(total_size, int)
                or total_size != candidate._total_size
            ):
                raise ValueError("replay state total_size does not match transitions")
        candidate._validate_bucket_snapshot(state.get("buckets"), restored)
        candidate._validate_fifo_snapshot(state.get("global_fifo"), restored)
        candidate._next_sequence_id = next_sequence_id
        candidate._rng.setstate(restored_rng.getstate())
        self._buckets = candidate._buckets
        self._global_fifo = candidate._global_fifo
        self._total_size = candidate._total_size
        self._next_sequence_id = candidate._next_sequence_id
        self._rng = candidate._rng

    def _validate_bucket_snapshot(
        self,
        buckets: Any,
        transitions: Sequence[ReplayTransition],
    ) -> None:
        if buckets is None:
            return
        if not isinstance(buckets, Mapping):
            raise ValueError("replay state buckets are invalid")
        expected = self._buckets
        if tuple(buckets) != tuple(expected):
            raise ValueError("replay state buckets do not match transitions")
        transitions_by_id = {
            transition.sequence_id: transition for transition in transitions
        }
        for signature, bucket in buckets.items():
            if not isinstance(bucket, (list, tuple)):
                raise ValueError("replay state bucket entries are invalid")
            expected_bucket = expected[signature]
            if len(bucket) != len(expected_bucket):
                raise ValueError("replay state bucket size does not match transitions")
            for restored, expected_transition in zip(bucket, expected_bucket):
                if not isinstance(restored, ReplayTransition):
                    raise ValueError(
                        "replay state bucket contains an invalid transition"
                    )
                canonical = transitions_by_id.get(restored.sequence_id)
                if canonical is None or not self._transitions_equal(
                    restored, canonical
                ) or not self._transitions_equal(restored, expected_transition):
                    raise ValueError(
                        "replay state bucket order does not match transitions"
                    )

    def _validate_fifo_snapshot(
        self,
        global_fifo: Any,
        transitions: Sequence[ReplayTransition],
    ) -> None:
        if global_fifo is None:
            return
        if not isinstance(global_fifo, (list, tuple)):
            raise ValueError("replay state global_fifo is invalid")
        expected = tuple(
            (transition.sequence_id, transition.signature) for transition in transitions
        )
        try:
            restored = tuple(global_fifo)
        except TypeError as exc:
            raise ValueError("replay state global_fifo is invalid") from exc
        for item in restored:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("replay state global_fifo entries are invalid")
            self._validate_sequence_id(item[0])
            self._validate_signature(item[1])
        if restored != expected:
            raise ValueError("replay state global_fifo does not match transitions")

    @staticmethod
    def _transitions_equal(left: ReplayTransition, right: ReplayTransition) -> bool:
        if (
            left.sequence_id != right.sequence_id
            or left.signature != right.signature
            or not np.array_equal(left.rewards, right.rewards)
            or not np.array_equal(left.terminated, right.terminated)
            or not np.array_equal(left.truncated, right.truncated)
        ):
            return False
        for field in (
            "observations",
            "next_observations",
            "actions",
            "proposed_actions",
            "executed_actions",
            "base_actions",
            "behavior_actions",
            "next_behavior_actions",
            "cloning_actions",
        ):
            left_values = getattr(left, field, None)
            right_values = getattr(right, field, None)
            if (left_values is None) != (right_values is None):
                return False
            if left_values is not None and (
                len(left_values) != len(right_values)
                or any(
                    not np.array_equal(a, b)
                    for a, b in zip(left_values, right_values)
                )
            ):
                return False
        return True

    def _validate_signature(self, signature: LayoutSignature) -> None:
        if not isinstance(signature, tuple) or len(signature) != self.num_agents:
            raise ValueError(
                "layout_signature building count must match num_agents "
                f"({self.num_agents})"
            )
        for building_index, building in enumerate(signature):
            if not isinstance(building, tuple) or len(building) != 6:
                raise ValueError(
                    f"layout_signature building {building_index} must contain "
                    "six fields"
                )
            n_sro, n_ca, action_names, segments, excluded_names, type_widths = (
                building
            )
            self._validate_non_negative_count(n_sro, building_index, "n_sro")
            self._validate_non_negative_count(n_ca, building_index, "n_ca")
            self._validate_string_tuple(
                action_names,
                building_index,
                "ca_action_names",
            )
            if len(action_names) != n_ca:
                raise ValueError(
                    f"layout_signature building {building_index} has {n_ca} CAs "
                    f"but {len(action_names)} action names"
                )
            self._validate_segments(segments, building_index, n_sro, n_ca)
            self._validate_string_tuple(
                excluded_names,
                building_index,
                "excluded_feature_names",
            )
            self._validate_type_widths(
                type_widths,
                building_index,
                segments,
            )

    @staticmethod
    def _validate_non_negative_count(
        value: Any,
        building_index: int,
        label: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"layout_signature building {building_index} {label} must be "
                "a non-negative integer"
            )

    @staticmethod
    def _validate_string_tuple(
        values: Any,
        building_index: int,
        label: str,
    ) -> None:
        if not isinstance(values, tuple) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(
                f"layout_signature building {building_index} {label} must be "
                "a tuple of strings"
            )

    def _validate_segments(
        self,
        segments: Any,
        building_index: int,
        n_sro: int,
        n_ca: int,
    ) -> None:
        if not isinstance(segments, tuple):
            raise ValueError(
                f"layout_signature building {building_index} segments must be a tuple"
            )
        family_counts = {"sro": 0, "nfc": 0, "ca": 0}
        for segment_index, segment in enumerate(segments):
            if not isinstance(segment, tuple) or len(segment) != 5:
                raise ValueError(
                    "layout_signature segment must contain family, type, instance, "
                    "ordered feature names, and NFC expression; "
                    f"instance ID; building {building_index}, segment {segment_index}"
                )
            family, type_name, instance_id, feature_names, expression = segment
            if not isinstance(family, str):
                raise ValueError(
                    "layout_signature segment family must be a string"
                )
            if family not in family_counts:
                raise ValueError(
                    f"layout_signature segment family {family!r} is invalid"
                )
            if not isinstance(type_name, str) or not type_name:
                raise ValueError(
                    "layout_signature segment type must be a non-empty string"
                )
            if instance_id is not None and not isinstance(instance_id, str):
                raise ValueError(
                    "layout_signature segment instance ID must be a string or None"
                )
            self._validate_string_tuple(
                feature_names,
                building_index,
                "segment_feature_names",
            )
            if not feature_names:
                raise ValueError(
                    "layout_signature segment feature names must not be empty"
                )
            if family == "nfc":
                if (
                    not isinstance(expression, tuple)
                    or len(expression) != 3
                    or not all(isinstance(value, str) and value for value in expression)
                ):
                    raise ValueError(
                        "layout_signature NFC segment expression must contain "
                        "operation, left feature, and right feature"
                    )
            elif expression is not None:
                raise ValueError(
                    "layout_signature non-NFC segment expression must be None"
                )
            family_counts[family] += 1
        if family_counts != {"sro": n_sro, "nfc": 1, "ca": n_ca}:
            raise ValueError(
                f"layout_signature building {building_index} segment counts do not "
                "match n_sro and n_ca"
            )

    @staticmethod
    def _validate_type_widths(
        type_widths: Any,
        building_index: int,
        segments: tuple[Any, ...],
    ) -> None:
        if not isinstance(type_widths, tuple):
            raise ValueError(
                f"layout_signature building {building_index} type widths must "
                "be a tuple"
            )
        widths_by_type: dict[str, int] = {}
        for item in type_widths:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "layout_signature type width must contain a type name and width"
                )
            type_name, width = item
            if (
                not isinstance(type_name, str)
                or not type_name
                or isinstance(width, bool)
                or not isinstance(width, int)
                or width <= 0
            ):
                raise ValueError(
                    "layout_signature type widths must be positive integers"
                )
            if type_name in widths_by_type:
                raise ValueError(
                    f"layout_signature type width {type_name!r} is duplicated"
                )
            widths_by_type[type_name] = width
        segment_types = {segment[1] for segment in segments}
        if set(widths_by_type) != segment_types:
            raise ValueError(
                f"layout_signature building {building_index} type widths must "
                "match segment types"
            )

    def _validate_restored_transition(self, transition: ReplayTransition) -> None:
        self._validate_sequence_id(transition.sequence_id)
        self._validate_signature(transition.signature)

        required_fields = (
            ("observations", transition.observations),
            ("next_observations", transition.next_observations),
            ("actions", transition.actions),
        )
        for label, vectors in required_fields:
            self._validate_stored_vector_tuple(vectors, label, np.float32)
        self._validate_transition_shapes(
            transition.signature,
            transition.observations,
            transition.next_observations,
            transition.actions,
            getattr(transition, "proposed_actions", None) or transition.actions,
            getattr(transition, "executed_actions", None) or transition.actions,
        )
        optional_values = (
            transition.behavior_actions,
            transition.next_behavior_actions,
            transition.cloning_actions,
            getattr(transition, "base_actions", None),
            getattr(transition, "proposed_actions", None),
            getattr(transition, "executed_actions", None),
        )
        optional_labels = (
            "behavior_actions",
            "next_behavior_actions",
            "cloning_actions",
            "base_actions",
            "proposed_actions",
            "executed_actions",
        )
        for label, vectors in zip(optional_labels, optional_values):
            if vectors is not None:
                self._validate_stored_vector_tuple(vectors, label, np.float32)
        self._validate_optional_presence(transition.signature, optional_values)
        self._validate_stored_agent_values(transition.rewards, "rewards", np.float32)
        self._validate_stored_agent_values(
            transition.terminated,
            "terminated",
            np.bool_,
        )
        self._validate_stored_agent_values(transition.truncated, "truncated", np.bool_)

    @staticmethod
    def _validate_sequence_id(sequence_id: Any) -> None:
        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
        ):
            raise ValueError("replay transition sequence_id must be non-negative")

    def _validate_stored_vector_tuple(
        self,
        vectors: Any,
        label: str,
        dtype: Any,
    ) -> None:
        if not isinstance(vectors, tuple) or len(vectors) != self.num_agents:
            raise ValueError(f"replay {label} must contain one vector per agent")
        for vector in vectors:
            if not isinstance(vector, np.ndarray) or vector.ndim != 1:
                raise ValueError(
                    f"replay {label} entries must be one-dimensional arrays"
                )
            if vector.dtype != np.dtype(dtype):
                raise ValueError(
                    f"replay {label} entries must have dtype {np.dtype(dtype)}"
                )
            if vector.dtype.kind == "f" and not np.isfinite(vector).all():
                raise ValueError(f"replay {label} entries must contain finite values")

    def _validate_stored_agent_values(
        self,
        values: Any,
        label: str,
        dtype: Any,
    ) -> None:
        if (
            not isinstance(values, np.ndarray)
            or values.ndim != 1
            or values.shape[0] != self.num_agents
            or values.dtype != np.dtype(dtype)
        ):
            raise ValueError(
                f"replay {label} must be a {np.dtype(dtype)} vector with "
                f"width {self.num_agents}"
            )
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"replay {label} must contain finite values")

    def _coerce_agent_vectors(self, values: Sequence[Any], label: str) -> ArrayTuple:
        try:
            value_count = len(values)
        except TypeError as exc:
            raise ValueError(f"{label} must contain one vector per agent") from exc
        if value_count != self.num_agents:
            raise ValueError(f"{label} count must match num_agents ({self.num_agents})")
        return tuple(
            self._immutable_vector(value, label, dtype=np.float32)
            for value in values
        )

    def _coerce_optional_actions(
        self,
        values: Optional[Sequence[Any]],
        label: str,
    ) -> Optional[ArrayTuple]:
        if values is None:
            return None
        vectors = self._coerce_agent_vectors(values, label)
        for index, vector in enumerate(vectors):
            self._validate_action_width(vector, index, label)
        return vectors

    def _validate_transition_shapes(
        self,
        signature: LayoutSignature,
        observations: ArrayTuple,
        next_observations: ArrayTuple,
        actions: ArrayTuple,
        proposed_actions: ArrayTuple,
        executed_actions: ArrayTuple,
    ) -> None:
        for index in range(self.num_agents):
            if observations[index].shape != next_observations[index].shape:
                raise ValueError(
                    f"next_encoded_obs[{index}] shape must match encoded_obs[{index}]"
            )
            self._validate_action_width(actions[index], index, "action", signature)
            self._validate_action_width(
                proposed_actions[index], index, "proposed_actions", signature
            )
            self._validate_action_width(
                executed_actions[index], index, "executed_actions", signature
            )
        bucket = self._buckets.get(signature)
        if not bucket:
            return
        expected = bucket[0]
        for index in range(self.num_agents):
            if observations[index].shape != expected.observations[index].shape:
                raise ValueError(
                    f"encoded_obs[{index}] shape changed within signature bucket"
                )

    def _validate_action_width(
        self,
        vector: np.ndarray,
        index: int,
        label: str,
        signature: Optional[LayoutSignature] = None,
    ) -> None:
        active_signature = signature
        if active_signature is None:
            return
        expected = active_signature[index][1]
        if vector.shape[0] != expected:
            raise ValueError(
                f"{label}[{index}] width is {vector.shape[0]}, expected {expected}"
            )

    def _validate_optional_presence(
        self,
        signature: LayoutSignature,
        values: tuple[Optional[ArrayTuple], ...],
    ) -> None:
        labels = (
            "behavior_actions",
            "next_behavior_actions",
            "cloning_actions",
            "base_actions",
            "proposed_actions",
            "executed_actions",
        )
        for label, vectors in zip(labels, values):
            if vectors is not None:
                for index, vector in enumerate(vectors):
                    self._validate_action_width(vector, index, label, signature)
        bucket = self._buckets.get(signature)
        if not bucket:
            return
        expected = bucket[0]
        for label, vectors in zip(labels, values):
            if (getattr(expected, label, None) is None) != (vectors is None):
                raise ValueError(f"{label} presence must remain stable within a bucket")

    def _coerce_agent_values(
        self,
        value: Any,
        dtype: Any,
        label: str,
        *,
        broadcast_scalar: bool = False,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=dtype).reshape(-1)
        if broadcast_scalar and array.shape[0] == 1:
            array = np.repeat(array, self.num_agents)
        if array.shape[0] != self.num_agents:
            raise ValueError(f"{label} width must match num_agents ({self.num_agents})")
        return self._immutable_vector(array, label, dtype=dtype)

    @staticmethod
    def _immutable_vector(value: Any, label: str, *, dtype: Any) -> np.ndarray:
        array = np.asarray(value, dtype=dtype).reshape(-1).copy()
        if array.dtype.kind == "f":
            array = array.astype(np.float32, copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f"{label} must contain only finite values")
        array.setflags(write=False)
        return array

    def _evict_if_full(self) -> None:
        if self._total_size < self.capacity:
            return
        sequence_id, signature = self._global_fifo.popleft()
        bucket = self._buckets[signature]
        transition = bucket.popleft()
        if transition.sequence_id != sequence_id:
            raise RuntimeError("replay FIFO and signature bucket are inconsistent")
        if not bucket:
            del self._buckets[signature]
        self._total_size -= 1

    def _append_transition(self, transition: ReplayTransition) -> None:
        bucket = self._buckets.setdefault(transition.signature, deque())
        bucket.append(transition)
        self._global_fifo.append((transition.sequence_id, transition.signature))
        self._total_size += 1

    @staticmethod
    def _copy_transition(transition: ReplayTransition) -> ReplayTransition:
        if not isinstance(transition, ReplayTransition):
            raise ValueError("replay state contains an invalid transition")

        def copied(values: Optional[ArrayTuple]) -> Optional[ArrayTuple]:
            if values is None:
                return None
            result = tuple(np.array(value, copy=True) for value in values)
            for value in result:
                value.setflags(write=False)
            return result

        rewards = np.array(transition.rewards, copy=True)
        terminated = np.array(transition.terminated, copy=True)
        truncated = np.array(transition.truncated, copy=True)
        rewards.setflags(write=False)
        terminated.setflags(write=False)
        truncated.setflags(write=False)
        return ReplayTransition(
            sequence_id=transition.sequence_id,
            signature=transition.signature,
            observations=copied(transition.observations) or (),
            next_observations=copied(transition.next_observations) or (),
            actions=copied(transition.actions) or (),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            proposed_actions=copied(getattr(transition, "proposed_actions", None)),
            executed_actions=copied(getattr(transition, "executed_actions", None)),
            base_actions=copied(getattr(transition, "base_actions", None)),
            behavior_actions=copied(transition.behavior_actions),
            next_behavior_actions=copied(transition.next_behavior_actions),
            cloning_actions=copied(transition.cloning_actions),
        )

    @staticmethod
    def _build_batch(
        signature: LayoutSignature,
        transitions: Sequence[ReplayTransition],
    ) -> ReplayBatch:
        def stacked(field: str) -> ArrayTuple:
            values = getattr(transitions[0], field)
            return tuple(
                np.stack([getattr(item, field)[index] for item in transitions])
                for index in range(len(values))
            )

        def optional_stacked(field: str) -> Optional[ArrayTuple]:
            if getattr(transitions[0], field, None) is None:
                return None
            return stacked(field)

        def action_stacked(field: str) -> ArrayTuple:
            if getattr(transitions[0], field, None) is None:
                return stacked("actions")
            return stacked(field)

        terminated = np.stack([item.terminated for item in transitions])
        truncated = np.stack([item.truncated for item in transitions])
        return ReplayBatch(
            signature=signature,
            observations=stacked("observations"),
            next_observations=stacked("next_observations"),
            actions=stacked("actions"),
            rewards=np.stack([item.rewards for item in transitions]),
            terminated=terminated,
            truncated=truncated,
            done=np.logical_or(terminated, truncated),
            proposed_actions=action_stacked("proposed_actions"),
            executed_actions=action_stacked("executed_actions"),
            base_actions=optional_stacked("base_actions"),
            behavior_actions=optional_stacked("behavior_actions"),
            next_behavior_actions=optional_stacked("next_behavior_actions"),
            cloning_actions=optional_stacked("cloning_actions"),
        )
