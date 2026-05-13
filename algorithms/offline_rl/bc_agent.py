"""BC inference agent.

CityLearn ``BaseAgent`` that:

  * Controls **Building 5** with the trained BC policy.
  * Controls the other 16 buildings with ``OfflineRBC`` — the same
    behaviour policy used during data collection. This keeps the rest of the
    district fixed across the RBC-vs-BC comparison so any KPI delta is
    attributable to BC's actions on Building 5.

Loading
-------
    agent = BCAgent.from_seed_dir(
        Path("runs/offline_bc/<run>/seed_<N>"),
        rbc_config=None,  # default-empty hyperparameters
    )

The seed dir must contain ``policy.pt``, ``obs_standardiser.npz``, and
``architecture.json`` (all written by ``bc_trainer.train_single_seed``).

Notes
-----
* ``self.use_raw_observations = True`` — we read CityLearn's raw 35-D obs
  for Building 5, standardise it ourselves with the bundled standardiser,
  then run the policy. We do **not** rely on a wrapper to encode obs.
* ``predict()`` is deterministic (the policy is in eval mode, no dropout).
* ``update()`` and ``export_artifacts()`` are no-ops / passthroughs — this
  is an inference-only agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch

from algorithms.agents.base_agent import BaseAgent
from algorithms.offline_rl import schema as S
from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.bc_policy import BCPolicy
from algorithms.offline_rl.rbc import OfflineRBC


# Default CityLearn schema path used when callers don't supply rbc_config.
# Same value the collector and benchmark harness use, so the off-target
# buildings get the same RBC config as during data collection.
_DEFAULT_DATASET_SCHEMA: str = (
    "./datasets/citylearn_three_phase_dynamic_topology_demo_v1/schema.json"
)


def _default_rbc_config() -> Dict[str, Any]:
    return {
        "algorithm": {"hyperparameters": {}},
        "simulator": {"dataset_path": _DEFAULT_DATASET_SCHEMA},
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BCAgent(BaseAgent):
    """BC for Building 5, RBC for the other buildings."""

    def __init__(
        self,
        policy: BCPolicy,
        standardiser: ObservationStandardiser,
        *,
        target_building_index: int = S.TARGET_BUILDING_INDEX,
        rbc_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.use_raw_observations = True

        self._policy = policy.eval()
        self._standardiser = standardiser
        self._target_idx = int(target_building_index)
        self._device = torch.device(device)
        self._policy = self._policy.to(self._device)

        # The companion RBC for the other 16 buildings. Same config the
        # collector uses.
        self._rbc = OfflineRBC(
            config=rbc_config if rbc_config is not None else _default_rbc_config()
        )
        self._attached: bool = False

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def attach_environment(  # type: ignore[override]
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate the target building's observation order against the schema
        # the policy was trained on. The schema is checked by the collector
        # too, but we re-check at inference time so a stale checkpoint can't
        # silently produce garbage actions.
        S.validate_observation_names(observation_names[self._target_idx])
        S.validate_action_names(action_names[self._target_idx])

        # Forward to the RBC so its obs/action index maps are populated for
        # all 17 buildings (including B5 — RBC will compute B5's action too,
        # but we'll overwrite it).
        self._rbc.attach_environment(
            observation_names=observation_names,
            action_names=action_names,
            action_space=action_space,
            observation_space=observation_space,
            metadata=metadata,
        )
        self._attached = True

    def predict(  # type: ignore[override]
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: Optional[bool] = None,
    ) -> List[List[float]]:
        if not self._attached:
            raise RuntimeError(
                "BCAgent.predict called before attach_environment"
            )
        # 1. Let the RBC fill in actions for every building.
        rbc_actions = self._rbc.predict(observations, deterministic=deterministic)

        # 2. Overwrite Building 5's action with the BC policy.
        b5_obs_raw = np.asarray(
            observations[self._target_idx], dtype=np.float32
        ).reshape(-1)
        if b5_obs_raw.shape[0] != self._standardiser.mean.shape[0]:
            raise ValueError(
                f"B5 obs has {b5_obs_raw.shape[0]} features, "
                f"standardiser was fit on {self._standardiser.mean.shape[0]}"
            )
        b5_obs_std = self._standardiser.transform(b5_obs_raw)
        with torch.no_grad():
            obs_t = torch.from_numpy(b5_obs_std).to(self._device).unsqueeze(0)
            action_t = self._policy(obs_t).squeeze(0)
        bc_action = action_t.cpu().numpy().astype(np.float64).tolist()

        # Action vector length must match the env's expectation; CityLearn
        # will clip to bounds afterwards (via _benchmark_common.clip_actions
        # in the rollout helper).
        rbc_actions[self._target_idx] = list(bc_action)
        return rbc_actions

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # Inference-only.
        return None

    def export_artifacts(  # type: ignore[override]
        self, output_dir: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), out_dir / "policy.pt")
        self._standardiser.save(out_dir / "obs_standardiser.npz")
        return {
            "artifact_type": "bc_agent",
            "policy_path": str(out_dir / "policy.pt"),
            "standardiser_path": str(out_dir / "obs_standardiser.npz"),
            "target_building_index": self._target_idx,
            "architecture": self._policy.architecture_summary(),
        }

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_seed_dir(
        cls,
        seed_dir: Path,
        *,
        rbc_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> "BCAgent":
        seed_dir = Path(seed_dir)
        arch_path = seed_dir / "architecture.json"
        policy_path = seed_dir / "policy.pt"
        std_path = seed_dir / "obs_standardiser.npz"
        for p in (arch_path, policy_path, std_path):
            if not p.exists():
                raise FileNotFoundError(f"missing {p}")

        arch = json.loads(arch_path.read_text())
        policy = BCPolicy(
            obs_dim=int(arch["obs_dim"]),
            action_dim=int(arch["action_dim"]),
            hidden_layers=list(arch["hidden_layers"]),
            dropout=float(arch.get("dropout", 0.0)),
        )
        # weights_only=True for safety on torch>=2.6 default change.
        try:
            state = torch.load(policy_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(policy_path, map_location="cpu")
        policy.load_state_dict(state)

        standardiser = ObservationStandardiser.load(std_path)
        return cls(policy, standardiser, rbc_config=rbc_config, device=device)
