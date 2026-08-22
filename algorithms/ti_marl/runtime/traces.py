"""Buffered, deduplicated TI-MARL trace persistence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from algorithms.ti_marl.contracts.models import (
    InterfaceSnapshot,
    TypedTransition,
    canonical_value,
)


class BufferedTraceWriter:
    """Write chunks instead of per-field/per-step watchdog-style files."""

    def __init__(
        self,
        output_dir: str | Path | None,
        *,
        chunk_size: int = 256,
        snapshot_interval: int = 256,
        enabled: bool = True,
    ) -> None:
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.chunk_size = max(int(chunk_size), 1)
        self.snapshot_interval = max(int(snapshot_interval), 1)
        self.enabled = bool(enabled and self.output_dir is not None)
        self._snapshots: Dict[str, Mapping[str, Any]] = {}
        self._transitions: list[Mapping[str, Any]] = []
        self._known_snapshot_hashes: set[str] = set()
        self._chunk_index = 0
        self.transition_count = 0
        self.snapshot_count = 0
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        current: InterfaceSnapshot,
        following: InterfaceSnapshot,
        transition: TypedTransition,
    ) -> None:
        if not self.enabled:
            return
        force_snapshot = (
            current.topology_version != following.topology_version
            or self._event_signature(current) != self._event_signature(following)
            or self.transition_count % self.snapshot_interval == 0
        )
        self._stage_snapshot(current, force=force_snapshot)
        self._stage_snapshot(following, force=force_snapshot)
        self._transitions.append(canonical_value(transition))
        self.transition_count += 1
        if len(self._transitions) >= self.chunk_size:
            self.flush()

    @staticmethod
    def _event_signature(snapshot: InterfaceSnapshot) -> tuple:
        """Return event state without per-step ages or active durations."""

        return tuple(
            sorted(
                (
                    item.event_domain.value,
                    item.fault_mode,
                    item.target_type,
                    item.target_id,
                    item.target_feature,
                    item.availability.value,
                    item.connection.value,
                    item.quality.value,
                    item.event_ids,
                )
                for item in snapshot.fault_evidence
            )
        )

    def _stage_snapshot(self, snapshot: InterfaceSnapshot, *, force: bool) -> None:
        # Transitions are always persisted, but complete snapshots are sparse:
        # first/periodic intervals and health/topology events only. Persisting
        # every value-bearing snapshot defeats buffering because its hash
        # changes on practically every simulator step.
        if not force:
            return
        digest = snapshot.snapshot_hash
        if digest in self._known_snapshot_hashes or digest in self._snapshots:
            return
        payload = canonical_value(snapshot)
        payload["capture_reason"] = "event_or_interval"
        self._snapshots[digest] = payload

    def flush(self) -> None:
        if not self.enabled or (not self._snapshots and not self._transitions):
            return
        path = self.output_dir / f"trace-{self._chunk_index:06d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for digest, snapshot in sorted(self._snapshots.items()):
                handle.write(json.dumps({"kind": "snapshot", "hash": digest, "payload": snapshot}, sort_keys=True))
                handle.write("\n")
                self._known_snapshot_hashes.add(digest)
                self.snapshot_count += 1
            for transition in self._transitions:
                handle.write(json.dumps({"kind": "transition", "payload": transition}, sort_keys=True))
                handle.write("\n")
        self._snapshots.clear()
        self._transitions.clear()
        self._chunk_index += 1

    def close(self) -> None:
        self.flush()

    def manifest(self) -> Mapping[str, Any]:
        return {
            "format": "ti_marl_trace_v1",
            "enabled": self.enabled,
            "path": None if self.output_dir is None else str(self.output_dir),
            "chunks": self._chunk_index,
            "transitions": self.transition_count,
            "snapshots": self.snapshot_count,
            "chunk_size": self.chunk_size,
            "snapshot_interval": self.snapshot_interval,
        }
