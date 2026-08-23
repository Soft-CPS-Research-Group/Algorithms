from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, TypeAlias

import numpy as np


NfcExpressionSignature: TypeAlias = Tuple[str, str, str]
SegmentSignature: TypeAlias = Tuple[
    str,
    str,
    Optional[str],
    Tuple[str, ...],
    Optional[NfcExpressionSignature],
]
TypeWidth: TypeAlias = Tuple[str, int]
BuildingLayoutSignature: TypeAlias = Tuple[
    int,
    int,
    Tuple[str, ...],
    Tuple[SegmentSignature, ...],
    Tuple[str, ...],
    Tuple[TypeWidth, ...],
]
LayoutSignature: TypeAlias = Tuple[BuildingLayoutSignature, ...]
ArrayTuple: TypeAlias = Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ReplayTransition:
    sequence_id: int
    signature: LayoutSignature
    observations: ArrayTuple
    next_observations: ArrayTuple
    actions: ArrayTuple
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    behavior_actions: Optional[ArrayTuple] = None
    next_behavior_actions: Optional[ArrayTuple] = None
    cloning_actions: Optional[ArrayTuple] = None


@dataclass(frozen=True)
class ReplayBatch:
    signature: LayoutSignature
    observations: ArrayTuple
    next_observations: ArrayTuple
    actions: ArrayTuple
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    behavior_actions: Optional[ArrayTuple] = None
    next_behavior_actions: Optional[ArrayTuple] = None
    cloning_actions: Optional[ArrayTuple] = None
