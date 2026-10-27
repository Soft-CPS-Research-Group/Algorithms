"""CQL inference agent for the entity-interface environment.

Functionally identical to :class:`algorithms.offline_rl.iql_entity_agent.IQLEntityAgent`
— the inference contract is the same (load a ``GaussianPolicy`` per group,
dispatch by ``obs_dim``, apply the stored standardiser).  The only difference
is that the underlying policies were trained with a CQL conservative penalty.

See :mod:`algorithms.offline_rl.iql_entity_agent` for full documentation.

Configuration
-------------
Same as ``IQLEntityAgent``:

``model_dir`` *(required)*
    Path to the CQL output root written by ``train_cql_entity.py``.

``device`` *(optional, default ``"cpu"``)*

Loading examples
----------------
Via config dict (runner path)::

    agent = CQLEntityAgent(config={
        "algorithm": {
            "hyperparameters": {
                "model_dir": "runs/offline_cql_entity/run-001",
            }
        }
    })

Direct instantiation::

    agent = CQLEntityAgent.from_model_dir(
        Path("runs/offline_cql_entity/run-001"),
        device="cpu",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch

from algorithms.agents.base_agent import BaseAgent
from algorithms.offline_rl.iql_entity_agent import IQLEntityAgent, _pick_best_seed_dir


class CQLEntityAgent(IQLEntityAgent):
    """Offline CQL inference agent for the entity-interface environment.

    Inherits all inference logic from :class:`IQLEntityAgent`.  The only
    difference is the ``artifact_type`` tag in ``export_artifacts`` and the
    class name visible in registry/config.
    """

    def export_artifacts(  # type: ignore[override]
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manifest = super().export_artifacts(output_dir, context)
        manifest["artifact_type"] = "cql_entity_agent"
        # Rewrite manifest json with corrected type
        out_root = Path(output_dir)
        (out_root / "iql_entity_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return manifest

    @classmethod
    def from_model_dir(  # type: ignore[override]
        cls,
        model_dir: Path,
        *,
        device: str = "cpu",
    ) -> "CQLEntityAgent":
        return cls(
            config={
                "algorithm": {
                    "hyperparameters": {
                        "model_dir": str(model_dir),
                        "device": device,
                    }
                }
            }
        )
