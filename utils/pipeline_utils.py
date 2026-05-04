"""Helpers for inspecting the resolved ``pipeline`` config block."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def pipeline_algorithm_names(config: Dict[str, Any]) -> List[str]:
    """Return the ordered list of algorithm names declared in the pipeline.

    Empty entries (missing/blank ``algorithm`` field) are filtered out.
    """
    pipeline_cfg = config.get("pipeline") or []
    names: List[str] = []
    for stage in pipeline_cfg:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("algorithm") or "").strip()
        if name:
            names.append(name)
    return names


def summarise_pipeline_algorithms(config: Dict[str, Any], default: Optional[str] = "unknown_algorithm") -> Optional[str]:
    """Summarise pipeline algorithms for compact tagging / reporting.

    * Empty pipeline → ``default`` (configurable to ``None`` for callers
      that prefer to skip the field entirely).
    * Single-stage pipeline → the lone algorithm name (matches the
      historical ``opeva.algorithm`` MLflow tag for backwards
      compatibility).
    * Multi-stage pipeline → algorithms joined by ``+`` from top to
      bottom of the pipeline.
    """
    names = pipeline_algorithm_names(config)
    if not names:
        return default
    if len(names) == 1:
        return names[0]
    return "+".join(names)
