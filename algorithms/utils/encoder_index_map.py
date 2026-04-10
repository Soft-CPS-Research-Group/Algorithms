"""Utility to map raw observation feature names to post-encoding slice indices.

This is the foundational piece for the tokenizer: given a building's raw
observation names and the encoder configuration, it computes which indices
in the flat post-encoded vector correspond to each raw feature.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple


class EncoderSlice(NamedTuple):
    """Post-encoding position of a single raw observation feature.

    Attributes:
        start_idx: Inclusive start index in the post-encoded flat vector.
        end_idx: Exclusive end index in the post-encoded flat vector.
        n_dims: Number of post-encoding dimensions (end_idx - start_idx).
    """

    start_idx: int
    end_idx: int
    n_dims: int


# ---------------------------------------------------------------------------
# Rule matching (mirrors utils/wrapper_citylearn._matches_rule)
# ---------------------------------------------------------------------------


def _matches_rule(name: str, match_spec: Dict[str, Any]) -> bool:
    """Check whether *name* satisfies the match specification of an encoder rule.

    Supports the same match types as ``wrapper_citylearn._matches_rule``:
    ``equals``, ``contains``, ``prefixes``, ``suffixes``, and ``default``.
    """

    equals = match_spec.get("equals")
    if equals is not None and name in equals:
        return True

    contains = match_spec.get("contains")
    if contains is not None and any(token in name for token in contains):
        return True

    prefixes = match_spec.get("prefixes")
    if prefixes is not None and any(name.startswith(p) for p in prefixes):
        return True

    suffixes = match_spec.get("suffixes")
    if suffixes is not None and any(name.endswith(s) for s in suffixes):
        return True

    return bool(match_spec.get("default", False))


# ---------------------------------------------------------------------------
# Dimension computation
# ---------------------------------------------------------------------------


def _compute_encoded_dims(encoder_spec: Dict[str, Any]) -> int:
    """Return the number of post-encoding dimensions for an encoder spec.

    Rules
    -----
    * ``PeriodicNormalization`` → **2** (sin, cos)
    * ``OnehotEncoding``        → **len(classes)**
    * ``RemoveFeature``         → **0**
    * Everything else (``NoNormalization``, ``Normalize``,
      ``NormalizeWithMissing``) → **1**
    """

    encoder_type = encoder_spec.get("type")

    if encoder_type == "RemoveFeature":
        return 0

    if encoder_type == "PeriodicNormalization":
        return 2

    if encoder_type == "OnehotEncoding":
        classes = encoder_spec.get("params", {}).get("classes", [])
        return len(classes)

    # NoNormalization, Normalize, NormalizeWithMissing → 1 dim
    return 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_encoder_index_map(
    observation_names: List[str],
    encoder_config: Dict[str, Any],
) -> OrderedDict[str, EncoderSlice]:
    """Build an ordered mapping from raw feature names to post-encoding slices.

    Parameters
    ----------
    observation_names:
        Raw feature names for **one** building, as provided by
        ``attach_environment(observation_names=...)``.
    encoder_config:
        The full encoder configuration dict (loaded from
        ``configs/encoders/default.json``).  Must contain a ``"rules"`` key.

    Returns
    -------
    OrderedDict[str, EncoderSlice]
        Maps each raw name to ``(start_idx, end_idx, n_dims)`` describing
        its slice in the post-encoded flat vector.  Features encoded with
        ``RemoveFeature`` have ``n_dims=0`` and ``start_idx == end_idx``.

    Raises
    ------
    ValueError
        If the encoder config contains no rules, or if no rule matches
        a given observation name.
    """

    rules = encoder_config.get("rules", [])
    if not rules:
        raise ValueError("Encoder config must contain at least one rule")

    index_map: OrderedDict[str, EncoderSlice] = OrderedDict()
    current_idx = 0

    for name in observation_names:
        rule = next(
            (r for r in rules if _matches_rule(name, r.get("match", {}))),
            None,
        )
        if rule is None:
            raise ValueError(f"No encoder rule matches observation: {name!r}")

        encoder_spec = rule.get("encoder", {})
        n_dims = _compute_encoded_dims(encoder_spec)

        start_idx = current_idx
        end_idx = current_idx + n_dims
        index_map[name] = EncoderSlice(
            start_idx=start_idx,
            end_idx=end_idx,
            n_dims=n_dims,
        )
        current_idx = end_idx

    return index_map
