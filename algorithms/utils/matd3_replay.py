"""Topology-partitioned replay buffer for AgentTransformerMATD3."""
from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt


def compute_topology_signature(
    building_ids: List[str],
    observation_names: List[List[str]],
    action_names: List[List[str]],
    ca_action_names: List[List[str]],
    per_type_feature_dims: Dict[str, int],
) -> str:
    """Compute a stable hash of the topology configuration."""
    payload = {
        "building_ids": building_ids,
        "observation_names": observation_names,
        "action_names": action_names,
        "ca_action_names": ca_action_names,
        "per_type_feature_dims": dict(sorted(per_type_feature_dims.items())),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class LayoutSummary:
    """Lightweight per-building layout metadata stored in replay."""
    building_id: str
    n_ca: int
    n_sro: int
    obs_dim: int
    action_dim: int


@dataclass
class TransitionData:
    """Single multi-agent transition to store in replay."""
    observations: List[npt.NDArray[np.float32]]
    next_observations: List[npt.NDArray[np.float32]]
    actions: List[npt.NDArray[np.float32]]
    base_actions: List[npt.NDArray[np.float32]]
    next_base_actions: List[npt.NDArray[np.float32]]
    rewards: List[float]
    done: bool
    topology_signature: str
    layout_summaries: List[LayoutSummary]


@dataclass
class SampledBatch:
    """Batch sampled from a single topology partition."""
    observations: List[npt.NDArray[np.float32]]
    next_observations: List[npt.NDArray[np.float32]]
    actions: List[npt.NDArray[np.float32]]
    base_actions: List[npt.NDArray[np.float32]]
    next_base_actions: List[npt.NDArray[np.float32]]
    rewards: List[npt.NDArray[np.float32]]
    done: npt.NDArray[np.float32]
    topology_signature: str
    layout_summaries: List[LayoutSummary]


class _Partition:
    """Append-only partition with oldest-entry eviction."""

    def __init__(self) -> None:
        self.transitions: List[TransitionData] = []
        self.position: int = 0

    @property
    def size(self) -> int:
        return len(self.transitions)

    def push(self, transition: TransitionData) -> None:
        self.transitions.append(transition)

    def evict_oldest(self) -> bool:
        if not self.transitions:
            return False
        self.transitions.pop(0)
        self.position = 0
        return True

    def sample_indices(self, batch_size: int) -> List[int]:
        return random.sample(range(self.size), batch_size)


class TopologyPartitionedReplay:
    """Global-capacity replay buffer partitioned by topology signature."""

    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self._active_signature: Optional[str] = None
        self._partitions: OrderedDict[str, _Partition] = OrderedDict()

    @property
    def active_signature(self) -> Optional[str]:
        return self._active_signature

    @property
    def active_partition_size(self) -> int:
        if self._active_signature is None:
            return 0
        partition = self._partitions.get(self._active_signature)
        return partition.size if partition else 0

    @property
    def total_size(self) -> int:
        return sum(p.size for p in self._partitions.values())

    @property
    def partition_count(self) -> int:
        return len(self._partitions)

    def partition_size(self, signature: str) -> int:
        partition = self._partitions.get(signature)
        return partition.size if partition else 0

    def set_active_signature(self, signature: str) -> None:
        """Switch the active topology signature."""
        self._active_signature = signature
        if signature not in self._partitions:
            self._partitions[signature] = _Partition()

    def push(self, transition: TransitionData) -> None:
        """Store a transition in the appropriate partition."""
        sig = transition.topology_signature
        if sig not in self._partitions:
            self._partitions[sig] = _Partition()
        while self.total_size >= self.capacity:
            if not self._evict_one():
                break
        self._partitions[sig].push(transition)

    def sample(self) -> Optional[SampledBatch]:
        """Sample a batch from the active partition."""
        if self._active_signature is None:
            return None
        partition = self._partitions.get(self._active_signature)
        if partition is None or partition.size < self.batch_size:
            return None
        transitions = [partition.transitions[i] for i in partition.sample_indices(self.batch_size)]
        n_buildings = len(transitions[0].observations)
        observations = [np.stack([t.observations[b] for t in transitions]) for b in range(n_buildings)]
        next_observations = [
            np.stack([t.next_observations[b] for t in transitions]) for b in range(n_buildings)
        ]
        actions = [np.stack([t.actions[b] for t in transitions]) for b in range(n_buildings)]
        base_actions = [np.stack([t.base_actions[b] for t in transitions]) for b in range(n_buildings)]
        next_base_actions = [
            np.stack([t.next_base_actions[b] for t in transitions]) for b in range(n_buildings)
        ]
        rewards = [
            np.array([t.rewards[b] for t in transitions], dtype=np.float32)
            for b in range(n_buildings)
        ]
        done = np.array([float(t.done) for t in transitions], dtype=np.float32)
        return SampledBatch(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            base_actions=base_actions,
            next_base_actions=next_base_actions,
            rewards=rewards,
            done=done,
            topology_signature=self._active_signature,
            layout_summaries=transitions[0].layout_summaries,
        )

    def _evict_one(self) -> bool:
        """Evict one transition, preferring oldest non-active partition."""
        for sig in list(self._partitions.keys()):
            if sig == self._active_signature:
                continue
            partition = self._partitions[sig]
            if partition.size > 0:
                partition.evict_oldest()
                if partition.size == 0:
                    del self._partitions[sig]
                return True
        if self._active_signature and self._active_signature in self._partitions:
            return self._partitions[self._active_signature].evict_oldest()
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay state for checkpointing."""
        partitions_state = {}
        for sig, partition in self._partitions.items():
            partitions_state[sig] = {
                "transitions": [
                    {
                        "observations": [obs.tolist() for obs in t.observations],
                        "next_observations": [obs.tolist() for obs in t.next_observations],
                        "actions": [a.tolist() for a in t.actions],
                        "base_actions": [a.tolist() for a in t.base_actions],
                        "next_base_actions": [a.tolist() for a in t.next_base_actions],
                        "rewards": t.rewards,
                        "done": t.done,
                        "topology_signature": t.topology_signature,
                        "layout_summaries": [ls.__dict__ for ls in t.layout_summaries],
                    }
                    for t in partition.transitions
                ],
                "position": partition.position,
            }
        return {"active_signature": self._active_signature, "partitions": partitions_state}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay state from checkpoint."""
        self._active_signature = state["active_signature"]
        self._partitions = OrderedDict()
        for sig, pstate in state["partitions"].items():
            partition = _Partition()
            partition.position = pstate["position"]
            for tdata in pstate["transitions"]:
                partition.transitions.append(
                    TransitionData(
                        observations=[np.array(o, dtype=np.float32) for o in tdata["observations"]],
                        next_observations=[
                            np.array(o, dtype=np.float32) for o in tdata["next_observations"]
                        ],
                        actions=[np.array(a, dtype=np.float32) for a in tdata["actions"]],
                        base_actions=[np.array(a, dtype=np.float32) for a in tdata["base_actions"]],
                        next_base_actions=[
                            np.array(a, dtype=np.float32) for a in tdata["next_base_actions"]
                        ],
                        rewards=tdata["rewards"],
                        done=tdata["done"],
                        topology_signature=tdata["topology_signature"],
                        layout_summaries=[LayoutSummary(**ls) for ls in tdata["layout_summaries"]],
                    )
                )
            self._partitions[sig] = partition
