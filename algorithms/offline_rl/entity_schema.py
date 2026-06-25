"""Schema conventions for the entity-interface offline-RL dataset.

The dataset produced by ``scripts/collect_rbcsmart_dataset.py`` stores one row
per (agent, timestep). Observations, actions, and next-observations each live in
wide sparse columns named with a fixed prefix plus the entity-scoped field name.

This module is intentionally schema-agnostic: it does not hard-code observation
or action names — those are read from the parquet file at load time.  It only
defines:

* Column prefix constants.
* Helper functions to extract obs/action column names from a dataframe.
* A dataclass describing the shape of one agent group.

Agent groups
------------
The dataset covers four heterogeneous agent groups identified by
``(obs_dim, action_dim)``:

==========  ==========  =====  ==========================
obs_dim     action_dim  Count  Equipment
==========  ==========  =====  ==========================
627         1           10     electrical storage only
706         2           5      storage + EV (1 charger)
749         3           1      storage + EV + deferrable
785         3           1      storage + 2 EV chargers
==========  ==========  =====  ==========================

Downstream IQL/CQL training trains *one policy per group*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Column prefix constants
# ---------------------------------------------------------------------------

OBS_PREFIX: str = "obs__"
NEXT_OBS_PREFIX: str = "next_obs__"
ACTION_PREFIX: str = "action__"

# Scalar meta-columns present on every row
META_COLUMNS: Tuple[str, ...] = (
    "seed",
    "episode",
    "timestep",
    "agent_idx",
    "obs_dim",
    "action_dim",
    "reward",
    "terminated",
    "truncated",
)

# Known heterogeneous agent groups: (obs_dim, action_dim)
AGENT_GROUPS: Tuple[Tuple[int, int], ...] = (
    (627, 1),
    (706, 2),
    (749, 3),
    (785, 3),
)


# ---------------------------------------------------------------------------
# Group descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentGroupSpec:
    """Describes one (obs_dim, action_dim) agent group within the dataset."""

    obs_dim: int
    action_dim: int
    obs_names: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)
    # Note: List[str] is mutable despite frozen=True; callers should not mutate this field.
    buildings: List[str] = field(default_factory=list)

    @property
    def group_key(self) -> str:
        return f"obs{self.obs_dim}_act{self.action_dim}"


# ---------------------------------------------------------------------------
# Column helpers (operate on a pandas DataFrame)
# ---------------------------------------------------------------------------


def obs_columns(df_columns) -> List[str]:
    """Return all obs__ columns from a column index (sorted for stability)."""
    return sorted(c for c in df_columns if c.startswith(OBS_PREFIX))


def next_obs_columns(df_columns) -> List[str]:
    """Return all next_obs__ columns from a column index."""
    return sorted(c for c in df_columns if c.startswith(NEXT_OBS_PREFIX))


def action_columns(df_columns) -> List[str]:
    """Return all action__ columns from a column index."""
    return sorted(c for c in df_columns if c.startswith(ACTION_PREFIX))


def obs_name_from_col(col: str) -> str:
    """Strip obs__ prefix from a column name."""
    assert col.startswith(OBS_PREFIX), col
    return col[len(OBS_PREFIX):]


def action_name_from_col(col: str) -> str:
    """Strip action__ prefix from a column name."""
    assert col.startswith(ACTION_PREFIX), col
    return col[len(ACTION_PREFIX):]


def next_obs_col(obs_col: str) -> str:
    """Given an obs__ column name, return the matching next_obs__ column name."""
    assert obs_col.startswith(OBS_PREFIX), obs_col
    return NEXT_OBS_PREFIX + obs_col[len(OBS_PREFIX):]


def obs_cols_for_group(all_obs_cols: List[str], obs_dim: int) -> List[str]:
    """Return the first ``obs_dim`` obs columns (stable order = file order)."""
    # In the wide format, each agent group uses a disjoint prefix subset of
    # obs columns.  The simplest way to recover the group's columns is to
    # filter rows for that group and take the non-NaN obs columns; that is done
    # in EntityDataset.  This helper exists for tests / introspection.
    return all_obs_cols[:obs_dim]


# ---------------------------------------------------------------------------
# Schema utility functions
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path
from typing import Union as _Union


def episode_steps_for_schema(schema_path: _Union[str, _Path]) -> int:
    """Return the number of environment steps in one calendar day.

    Reads ``seconds_per_time_step`` from the CityLearn schema JSON.
    Falls back to 3600 (hourly) if the key is absent.
    """
    raw = _json.loads(_Path(schema_path).read_text())
    sps = int(raw.get("seconds_per_time_step", 3600))
    return 86400 // sps


def probe_agent_groups(schema_path: _Union[str, _Path]) -> List["AgentGroupSpec"]:
    """Instantiate the CityLearn env once to discover (obs_dim, action_dim) per building.

    Groups buildings that share the same (obs_dim, action_dim) into one
    ``AgentGroupSpec``.  Works for any dataset without requiring a
    pre-collected parquet file.

    Note: creates and immediately discards a CityLearnEnv (takes a few seconds).
    """
    from collections import defaultdict as _dd
    from citylearn.citylearn import CityLearnEnv as _Env
    from utils.entity_adapter import EntityContractAdapter as _Adapter

    env = _Env(
        schema=str(schema_path),
        central_agent=False,
        interface="entity",
        topology_mode="static",
        episode_time_steps=1,  # minimal episode to make reset() cheap; we only need obs/action dims
    )
    try:
        # CityLearn (gymnasium-style) returns (payload, info) — we only need the payload.
        payload, _info = env.reset()
        adapter = _Adapter(env, normalization_enabled=False, clip=False)
        # to_agent_encoded_observations returns (obs_arrays, obs_names, obs_spaces).
        obs_arrays, _, _ = adapter.to_agent_encoded_observations(payload)
        # env.action_names is a per-building list-of-lists (one inner list per agent).
        # In the entity interface env.action_space is a nested Dict over tables/entities,
        # NOT a per-building sequence; iterating it yields keys (strings), which previously
        # caused `AttributeError: 'str' object has no attribute 'low'`.
        action_names_per_building = list(env.action_names)
        building_names = [b.name for b in env.buildings]
        groups: dict = _dd(list)
        for i, (obs, anames) in enumerate(zip(obs_arrays, action_names_per_building)):
            key = (len(obs), len(anames))
            groups[key].append(building_names[i])
    finally:
        env.close()
    return [
        AgentGroupSpec(obs_dim=k[0], action_dim=k[1], buildings=sorted(v))
        for k, v in sorted(groups.items())
    ]


def buildings_to_group_keys(
    buildings: List[str],
    groups: List["AgentGroupSpec"],
) -> List[str]:
    """Return group_key strings for groups that contain any of ``buildings``."""
    requested = set(buildings)
    return [g.group_key for g in groups if any(b in requested for b in g.buildings)]
