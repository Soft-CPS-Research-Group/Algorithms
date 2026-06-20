"""IQL inference agent for the entity-interface environment.

Loads one :class:`~algorithms.offline_rl.iql_networks.GaussianPolicy` per
agent group (identified by ``obs_dim × action_dim``) from a trained model
directory produced by :mod:`algorithms.offline_rl.iql_entity_trainer`.

At predict time the agent:

1. Receives a list of per-agent observation vectors (one per building).
2. Identifies each agent's group by ``obs_dim`` (unique within the four
   groups: 627/706/749/785).
3. Applies the group-specific ``ObservationStandardiser``.
4. Runs the trained policy's deterministic head
   (``tanh(mean)``) and returns the clamped action.

``update()`` is a no-op — this is a pure inference wrapper.

Configuration
-------------
The agent reads from ``config["algorithm"]["hyperparameters"]``:

``model_dir`` *(required)*
    Path to the output root written by ``train_iql_entity.py``.  Must
    contain one subdirectory per group (``obs<D>_act<A>/``) each with a
    ``multi_seed_summary.json`` file; the best seed (lowest
    ``best_val_policy_mse``) is loaded automatically.

    Alternatively, ``model_dir`` may point directly to a single group's
    *seed* directory (containing ``architecture.json``, ``policy.pt``,
    ``obs_standardiser.npz``).  In that case the agent loads exactly one
    group.

``device`` *(optional, default ``"cpu"``)*
    PyTorch device string.

Loading examples
----------------
Via config dict (runner path)::

    agent = IQLEntityAgent(config={
        "algorithm": {
            "hyperparameters": {
                "model_dir": "runs/offline_iql_entity/run-001",
                "device": "cpu",
            }
        }
    })

Direct instantiation::

    agent = IQLEntityAgent.from_model_dir(
        Path("runs/offline_iql_entity/run-001"),
        device="cpu",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from algorithms.agents.base_agent import BaseAgent
from algorithms.offline_rl.bc_dataset import ObservationStandardiser
from algorithms.offline_rl.iql_networks import GaussianPolicy


class IQLEntityAgent(BaseAgent):
    """Offline IQL inference agent for the entity-interface environment.

    Dispatches each building's observation to the policy trained for that
    agent's ``(obs_dim, action_dim)`` group.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        super().__init__()
        hyp: Dict[str, Any] = (
            config.get("algorithm", {}).get("hyperparameters", {})
        )
        model_dir = Path(hyp["model_dir"])
        device: str = str(hyp.get("device", "cpu"))

        self._device = torch.device(device)
        # Keyed by obs_dim (sufficient for dispatch — unique in our 4 groups)
        self._policy_map: Dict[int, GaussianPolicy] = {}
        self._standardiser_map: Dict[int, ObservationStandardiser] = {}
        # Full group key for metadata
        self._group_keys: Dict[int, str] = {}
        self._action_dims: Dict[int, int] = {}

        self._load_model_dir(model_dir, device)

        # Populated by attach_environment if called
        self._obs_dims_per_agent: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_dir(self, model_dir: Path, device: str) -> None:
        """Auto-detect layout (multi-group root or single seed dir) and load."""
        if not model_dir.exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        # Single-seed directory: has policy.pt or architecture.json directly.
        # Check policy.pt as well so we raise FileNotFoundError if architecture
        # is missing (rather than falling through to the multi-group search).
        if (model_dir / "policy.pt").exists() or (model_dir / "architecture.json").exists():
            self._load_seed_dir(model_dir, device)
            return

        # Multi-group root: each subdir is a group (obs<D>_act<A>/)
        loaded = 0
        for group_dir in sorted(model_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            multi_path = group_dir / "multi_seed_summary.json"
            if multi_path.exists():
                best_seed_dir = _pick_best_seed_dir(group_dir, multi_path)
                self._load_seed_dir(best_seed_dir, device)
                loaded += 1

        if loaded == 0:
            raise ValueError(
                f"No group directories with multi_seed_summary.json found "
                f"in {model_dir}. Expected layout: "
                "<model_dir>/<group_key>/multi_seed_summary.json"
            )

    def _load_seed_dir(self, seed_dir: Path, device: str) -> None:
        arch_path = seed_dir / "architecture.json"
        policy_path = seed_dir / "policy.pt"
        std_path = seed_dir / "obs_standardiser.npz"
        for p in (arch_path, policy_path, std_path):
            if not p.exists():
                raise FileNotFoundError(f"missing required file: {p}")

        arch = json.loads(arch_path.read_text())
        obs_dim = int(arch["obs_dim"])
        action_dim = int(arch["action_dim"])

        policy = GaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=list(arch["hidden_layers"]),
            dropout=float(arch.get("dropout", 0.0)),
            log_std_init=float(arch.get("log_std_init", 0.0)),
        )
        try:
            state = torch.load(policy_path, map_location="cpu", weights_only=True)
        except TypeError:  # older PyTorch
            state = torch.load(policy_path, map_location="cpu")
        policy.load_state_dict(state)
        policy.eval().to(torch.device(device))

        standardiser = ObservationStandardiser.load(std_path)

        self._policy_map[obs_dim] = policy
        self._standardiser_map[obs_dim] = standardiser
        self._action_dims[obs_dim] = action_dim
        self._group_keys[obs_dim] = f"obs{obs_dim}_act{action_dim}"

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
        """Cache per-agent obs_dim for fast dispatch at predict time."""
        self._obs_dims_per_agent = [len(names) for names in observation_names]
        # Validate that every agent has a loaded policy
        for i, obs_dim in enumerate(self._obs_dims_per_agent):
            if obs_dim not in self._policy_map:
                available = sorted(self._policy_map.keys())
                raise ValueError(
                    f"Agent {i} has obs_dim={obs_dim} but no policy was "
                    f"loaded for that group.  Loaded obs_dims: {available}"
                )

    def predict(  # type: ignore[override]
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: Optional[bool] = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        """Return deterministic IQL actions for all agents."""
        del context  # offline inference agent ignores pipeline context
        actions: List[List[float]] = []
        for i, obs_i in enumerate(observations):
            obs_arr = np.asarray(obs_i, dtype=np.float32).reshape(-1)
            obs_dim = obs_arr.shape[0]

            if obs_dim not in self._policy_map:
                raise ValueError(
                    f"Agent {i}: obs_dim={obs_dim} not in loaded policies "
                    f"{sorted(self._policy_map.keys())}."
                )

            # Fill NaN → 0 (consistent with EntityOfflineDataset loader)
            obs_arr = np.nan_to_num(obs_arr, nan=0.0)

            obs_std = self._standardiser_map[obs_dim].transform(obs_arr)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_std).to(self._device).unsqueeze(0)
                act_t = self._policy_map[obs_dim].predict_deterministic(obs_t)
                act_t = act_t.squeeze(0)
            actions.append(act_t.cpu().numpy().astype(np.float64).tolist())

        return actions

    def update(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """No-op: offline agent does not learn online."""
        return None

    def export_artifacts(  # type: ignore[override]
        self,
        output_dir: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save policies and standardisers; return manifest metadata."""
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        groups_meta: Dict[str, Any] = {}
        for obs_dim, policy in self._policy_map.items():
            group_key = self._group_keys[obs_dim]
            group_dir = out_root / group_key
            group_dir.mkdir(exist_ok=True)

            policy_path = group_dir / "policy.pt"
            std_path = group_dir / "obs_standardiser.npz"
            torch.save(policy.state_dict(), policy_path)
            self._standardiser_map[obs_dim].save(std_path)

            groups_meta[group_key] = {
                "obs_dim": obs_dim,
                "action_dim": self._action_dims[obs_dim],
                "policy_path": str(policy_path),
                "standardiser_path": str(std_path),
                "architecture": policy.architecture_summary(),
            }

        manifest = {
            "artifact_type": "iql_entity_agent",
            "n_groups": len(groups_meta),
            "groups": groups_meta,
        }
        (out_root / "iql_entity_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        return manifest

    # ------------------------------------------------------------------
    # Convenience classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        *,
        device: str = "cpu",
    ) -> "IQLEntityAgent":
        """Load from a trained model directory without a full config dict."""
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pick_best_seed_dir(group_dir: Path, multi_path: Path) -> Path:
    """Return the seed dir with the lowest ``best_val_policy_mse``."""
    multi = json.loads(multi_path.read_text())
    per_seed = multi.get("per_seed", [])
    if not per_seed:
        raise ValueError(
            f"multi_seed_summary.json has no 'per_seed' entries in {group_dir}"
        )
    best_entry = min(per_seed, key=lambda s: float(s.get("best_val_policy_mse", float("inf"))))
    seed_dir = Path(best_entry["output_dir"])
    if not seed_dir.exists():
        # Fall back to relative path from group_dir
        seed_dir = group_dir / f"seed_{best_entry['seed']}"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Best seed dir not found: {seed_dir} (from {group_dir})"
        )
    return seed_dir
