# AgentTransformerMATD3 — Plan A: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the skeleton `AgentTransformerMATD3` agent with config schema, registry, per-building actor stack, `attach_environment`, `predict`, and `export_artifacts` — enough to run a no-op forward pass and export ONNX actors.

**Architecture:** Per-building Transformer actor (tokenizer → backbone → deterministic head), registered via `TransformerMATD3StageConfig`. Reuses shared entity-transformer modules. No critic, no replay, no training — those come in Plan B/C/D.

**Tech Stack:** Python 3.10+, PyTorch, Pydantic (config schema), ONNX export, pytest.

**Spec:** `docs/transformer_matd3_spec.md`

**Depends on:** Nothing (first plan in the series).

**Produces:** A registered agent that passes `predict` and `export_artifacts` unit tests and can be wired into the wrapper for smoke testing (but does not train yet).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `utils/config_schema.py` (modify) | Add `TransformerMATD3StageConfig` + sub-models |
| `algorithms/registry.py` (modify) | Import + register `AgentTransformerMATD3` |
| `algorithms/agents/agent_transformer_matd3.py` (create) | Main agent class skeleton |
| `algorithms/utils/matd3_actor_head.py` (create) | Deterministic actor head (MLP per CA embedding → tanh scalar) |
| `tests/test_agent_transformer_matd3_foundation.py` (create) | Unit tests for Plan A scope |
| `configs/templates/rl/transformer_matd3_local.yaml` (create) | Minimal local config template |

---

## Task 1: Config Schema — TransformerMATD3StageConfig

**Files:**
- Modify: `utils/config_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_transformer_matd3_foundation.py
"""Plan A tests for AgentTransformerMATD3 foundation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from utils.config_schema import TransformerMATD3StageConfig


class TestTransformerMATD3StageConfig:
    def test_valid_minimal_config(self):
        cfg = TransformerMATD3StageConfig(
            algorithm="AgentTransformerMATD3",
            tokenizer_config_path="configs/tokenizers/entity_default.json",
            transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
            hyperparameters={
                "gamma": 0.99,
                "tau": 0.005,
                "batch_size": 256,
                "replay_capacity": 100000,
                "actor_lr": 1e-4,
                "critic_lr": 3e-4,
                "target_policy_noise": 0.2,
                "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2,
            },
        )
        assert cfg.algorithm == "AgentTransformerMATD3"
        assert cfg.transformer_actor.d_model == 64
        assert cfg.transformer_critic.d_model == 64

    def test_rejects_wrong_algorithm_name(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="MATD3",
                tokenizer_config_path="configs/tokenizers/entity_default.json",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )

    def test_rejects_missing_tokenizer_path(self):
        with pytest.raises(ValidationError):
            TransformerMATD3StageConfig(
                algorithm="AgentTransformerMATD3",
                tokenizer_config_path="",
                transformer_actor={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                transformer_critic={"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128},
                hyperparameters={"gamma": 0.99, "tau": 0.005, "batch_size": 256, "replay_capacity": 100000, "actor_lr": 1e-4, "critic_lr": 3e-4, "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5, "actor_update_interval": 2},
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestTransformerMATD3StageConfig -v`
Expected: ImportError — `TransformerMATD3StageConfig` does not exist yet.

- [ ] **Step 3: Implement the schema models**

Add to `utils/config_schema.py` (after `TransformerPPOStageConfig`, before `PipelineStageConfig`):

```python
class TransformerMATD3TransformerConfig(BaseModel):
    """Transformer architecture config (reused for both actor and critic)."""
    d_model: int = Field(ge=1)
    nhead: int = Field(ge=1)
    num_layers: int = Field(ge=1)
    dim_feedforward: int = Field(ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)


class TransformerMATD3Hyperparameters(BaseModel):
    gamma: float = Field(gt=0, le=1.0)
    tau: float = Field(gt=0, le=1.0)
    batch_size: int = Field(ge=1)
    replay_capacity: int = Field(ge=1)
    actor_lr: float = Field(gt=0)
    critic_lr: float = Field(gt=0)
    target_policy_noise: float = Field(ge=0)
    target_policy_noise_clip: float = Field(ge=0)
    actor_update_interval: int = Field(ge=1)
    critic_action_input_mode: Literal["final", "final_base_delta", "final_base_delta_normalized"] = "final"
    reward_normalization: bool = False
    reward_normalization_clip: float = Field(default=10.0, gt=0)
    reward_normalization_epsilon: float = Field(default=1e-8, gt=0)


class TransformerMATD3StageConfig(BaseModel):
    algorithm: Literal["AgentTransformerMATD3"]
    count: int = Field(default=1, ge=1)
    frozen: bool = False
    tokenizer_config_path: str = Field(min_length=1)
    transformer_actor: TransformerMATD3TransformerConfig
    transformer_critic: TransformerMATD3TransformerConfig
    hyperparameters: TransformerMATD3Hyperparameters
    exploration: Optional[Any] = None
    residual: Optional[Any] = None
    behavior_cloning: Optional[Any] = None
    diagnostics: Optional[Any] = None
```

Update `PipelineStageConfig` union to include `TransformerMATD3StageConfig`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestTransformerMATD3StageConfig -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/config_schema.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "feat(matd3-t): add TransformerMATD3StageConfig schema"
```

---

## Task 2: Deterministic Actor Head

**Files:**
- Create: `algorithms/utils/matd3_actor_head.py`
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
import torch

from algorithms.utils.matd3_actor_head import DeterministicActorHead


class TestDeterministicActorHead:
    def test_output_shape(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(2, 3, 16)  # [batch=2, n_ca=3, d_model=16]
        actions = head(ca_emb)
        assert actions.shape == (2, 3, 1)

    def test_output_range_tanh(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(4, 5, 16) * 10.0  # large inputs
        actions = head(ca_emb)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_deterministic_same_output(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        head.eval()
        ca_emb = torch.randn(1, 2, 16)
        a1 = head(ca_emb)
        a2 = head(ca_emb)
        assert torch.allclose(a1, a2)

    def test_pre_tanh_accessor(self):
        head = DeterministicActorHead(d_model=16, hidden_dim=32)
        ca_emb = torch.randn(1, 2, 16)
        actions, pre_tanh = head.forward_with_pre_tanh(ca_emb)
        assert torch.allclose(actions, torch.tanh(pre_tanh))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestDeterministicActorHead -v`
Expected: ImportError — `matd3_actor_head` module does not exist.

- [ ] **Step 3: Implement the actor head**

Create `algorithms/utils/matd3_actor_head.py`:

```python
"""Deterministic actor head for Transformer-MATD3.

Applies an MLP to each CA token embedding independently, producing one
scalar action per CA token. Output is tanh-squashed to [-1, 1].
Unlike PPO's stochastic ActorHead, this is purely deterministic —
exploration noise is added externally.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DeterministicActorHead(nn.Module):
    """MLP per CA embedding → tanh-squashed scalar action."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ca_embeddings: torch.Tensor) -> torch.Tensor:
        """Return tanh-squashed actions [B, N_ca, 1]."""
        return torch.tanh(self.mlp(ca_embeddings))

    def forward_with_pre_tanh(
        self, ca_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh_actions, pre_tanh_means) for target smoothing."""
        pre_tanh = self.mlp(ca_embeddings)
        return torch.tanh(pre_tanh), pre_tanh
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestDeterministicActorHead -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_actor_head.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "feat(matd3-t): add DeterministicActorHead"
```

---

## Task 3: Agent Skeleton + Registry

**Files:**
- Create: `algorithms/agents/agent_transformer_matd3.py`
- Modify: `algorithms/registry.py`
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
from algorithms.registry import ALGORITHM_REGISTRY


class TestRegistry:
    def test_agent_registered(self):
        assert "AgentTransformerMATD3" in ALGORITHM_REGISTRY

    def test_supports_dynamic_topology(self):
        cls = ALGORITHM_REGISTRY["AgentTransformerMATD3"]
        assert cls.supports_dynamic_topology is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestRegistry -v`
Expected: KeyError or ImportError.

- [ ] **Step 3: Create the agent skeleton**

Create `algorithms/agents/agent_transformer_matd3.py`:

```python
"""AgentTransformerMATD3 — per-building Transformer actor + centralized twin critics.

Plan A scope: actor stack only (predict + export). Critics, replay, and
training are added in Plans B/C/D.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn

from algorithms.agents.base_agent import BaseAgent
from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    EntityTokenLayoutBuilder,
)
from algorithms.utils.matd3_actor_head import DeterministicActorHead
from algorithms.utils.transformer_backbone import TransformerBackbone
from utils.entity_tokenizer_schema import (
    load_entity_tokenizer_config,
    EntityTokenizerConfig,
)


@dataclass
class _ActorState:
    """Per-building actor stack (deployable)."""
    building_id: str
    tokenizer: EntityObservationTokenizer
    backbone: TransformerBackbone
    actor: DeterministicActorHead
    target_actor: DeterministicActorHead
    optimizer: torch.optim.Optimizer
    layout: BuildingTokenLayout
    obs_names_tuple: Tuple[str, ...]
    action_names_tuple: Tuple[str, ...]
    topology_version: int = 0


class AgentTransformerMATD3(BaseAgent):
    """Per-building Transformer actor + centralized twin TD3 critics.

    Plan A implements actor stack only. Training (critics, replay, residual,
    BC) will be wired in Plans B/C/D.
    """

    supports_dynamic_topology: ClassVar[bool] = True

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        algo = config["algorithm"]

        self._tokenizer_config_path: str = str(algo["tokenizer_config_path"])
        self._tokenizer_config: EntityTokenizerConfig = load_entity_tokenizer_config(
            self._tokenizer_config_path
        )

        actor_cfg = dict(algo["transformer_actor"])
        self._actor_d_model = int(actor_cfg["d_model"])
        self._actor_nhead = int(actor_cfg["nhead"])
        self._actor_num_layers = int(actor_cfg["num_layers"])
        self._actor_dim_feedforward = int(actor_cfg.get("dim_feedforward", 256))
        self._actor_dropout = float(actor_cfg.get("dropout", 0.1))

        h = dict(algo["hyperparameters"])
        self._actor_lr = float(h["actor_lr"])
        self._actor_hidden_dim = int(h.get("actor_hidden_dim", max(32, self._actor_d_model * 2)))

        self._layout_builder = EntityTokenLayoutBuilder(self._tokenizer_config)
        self._actors: List[_ActorState] = []
        self._first_attach_done = False

    # ==========================================================================
    # BaseAgent contract
    # ==========================================================================

    def attach_environment(
        self,
        *,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        action_space: List[Any],
        observation_space: List[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._first_attach_done:
            self._build_all_actor_states(observation_names, action_names, metadata)
            self._first_attach_done = True
            return

        if len(self._actors) != len(observation_names):
            raise ValueError(
                f"AgentTransformerMATD3: building count changed from "
                f"{len(self._actors)} to {len(observation_names)}. "
                "Building-count changes are not supported."
            )

        for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names)):
            new_obs = tuple(obs_n)
            new_act = tuple(act_n)
            state = self._actors[b]
            if state.obs_names_tuple == new_obs and state.action_names_tuple == new_act:
                continue
            state.obs_names_tuple = new_obs
            state.action_names_tuple = new_act
            self._handle_topology_change(b)

    def predict(
        self,
        observations: List[npt.NDArray[np.float64]],
        deterministic: bool | None = None,
        *,
        context: Any = None,
    ) -> List[List[float]]:
        out: List[List[float]] = []
        for state, obs in zip(self._actors, observations):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                tokenized = state.tokenizer(obs_t, state.layout)
                ca_emb, _ = state.backbone(
                    tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
                )
                actions = state.actor(ca_emb)
            out.append(actions.squeeze(0).squeeze(-1).clamp(-1.0, 1.0).tolist())
        return out

    def update(
        self,
        observations: List[npt.NDArray[np.float64]],
        actions: List[npt.NDArray[np.float64]],
        rewards: List[float],
        next_observations: List[npt.NDArray[np.float64]],
        terminated: bool,
        truncated: bool,
        *,
        update_target_step: bool,
        global_learning_step: int,
        update_step: bool,
        initial_exploration_done: bool,
    ) -> None:
        # Plan A: no-op. Training wired in Plan B/C/D.
        pass

    def export_artifacts(
        self, output_dir: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        models_dir = out / "onnx_models"
        models_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[Dict[str, Any]] = []
        for b, state in enumerate(self._actors):
            obs_dim = self._infer_obs_dim(state.layout)
            relpath = f"onnx_models/agent_{b}__topology_v{state.topology_version}.onnx"
            self._export_actor_onnx(state, out / relpath, obs_dim)
            cfg = {
                "building_id": state.building_id,
                "topology_version": state.topology_version,
                "obs_dim": obs_dim,
                "n_sro": state.layout.n_sro,
                "n_ca": state.layout.n_ca,
                "sro_types": [s.type_name for s in state.layout.segments if s.family == "sro"],
                "ca_types": [s.type_name for s in state.layout.segments if s.family == "ca"],
                "ca_action_names": list(state.layout.ca_action_names),
                "action_low": [-1.0] * state.layout.n_ca,
                "action_high": [1.0] * state.layout.n_ca,
            }
            artifacts.append({
                "agent_index": b,
                "path": relpath,
                "format": "onnx",
                "config": cfg,
            })
        return {
            "format": "onnx",
            "artifacts": artifacts,
            "tokenizer_config_path": self._tokenizer_config_path,
            "supports_dynamic_topology": True,
        }

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        # Plan A: minimal checkpoint.
        out = Path(output_dir) / "checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"transformer_matd3_step{step}.pt"
        payload = {
            "step": step,
            "actors": [
                {
                    "building_id": s.building_id,
                    "tokenizer_state": s.tokenizer.state_dict(),
                    "backbone_state": s.backbone.state_dict(),
                    "actor_state": s.actor.state_dict(),
                    "target_actor_state": s.target_actor.state_dict(),
                    "optimizer_state": s.optimizer.state_dict(),
                    "obs_names": list(s.obs_names_tuple),
                    "action_names": list(s.action_names_tuple),
                    "topology_version": s.topology_version,
                }
                for s in self._actors
            ],
        }
        torch.save(payload, path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        actors_data = payload["actors"]
        if len(actors_data) != len(self._actors):
            raise ValueError(
                f"Checkpoint has {len(actors_data)} buildings; current agent "
                f"has {len(self._actors)}. Building-count mismatch."
            )
        for state, saved in zip(self._actors, actors_data):
            state.tokenizer.load_state_dict(saved["tokenizer_state"])
            state.backbone.load_state_dict(saved["backbone_state"])
            state.actor.load_state_dict(saved["actor_state"])
            state.target_actor.load_state_dict(saved["target_actor_state"])
            state.optimizer.load_state_dict(saved["optimizer_state"])

    # ==========================================================================
    # Internal
    # ==========================================================================

    def _build_all_actor_states(
        self,
        observation_names: List[List[str]],
        action_names: List[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        building_names = (metadata or {}).get("building_names")
        for b, (obs_n, act_n) in enumerate(zip(observation_names, action_names)):
            building_id = (
                building_names[b]
                if building_names and b < len(building_names)
                else f"building_{b}"
            )
            state = self._build_one_actor_state(building_id, list(obs_n), list(act_n))
            self._actors.append(state)

    def _build_one_actor_state(
        self, building_id: str, observation_names: List[str], action_names: List[str]
    ) -> _ActorState:
        layout = self._layout_builder.build(building_id, observation_names, action_names)
        # Validate CA order (prefix tolerance like TransformerPPO)
        for af, an in zip(layout.ca_action_names, action_names):
            if an == af or an.startswith(af + "_"):
                continue
            raise ValueError(
                f"BuildingTokenLayout.ca_action_names {layout.ca_action_names!r} "
                f"does not match action_names {tuple(action_names)!r} "
                f"for building {building_id!r}."
            )
        type_input_dims = self._compute_type_input_dims(layout)
        tokenizer = EntityObservationTokenizer(
            tokenizer_config=self._tokenizer_config,
            d_model=self._actor_d_model,
            type_input_dims=type_input_dims,
        )
        backbone = TransformerBackbone(
            d_model=self._actor_d_model,
            nhead=self._actor_nhead,
            num_layers=self._actor_num_layers,
            dim_feedforward=self._actor_dim_feedforward,
            dropout=self._actor_dropout,
        )
        actor = DeterministicActorHead(
            d_model=self._actor_d_model, hidden_dim=self._actor_hidden_dim
        )
        target_actor = DeterministicActorHead(
            d_model=self._actor_d_model, hidden_dim=self._actor_hidden_dim
        )
        target_actor.load_state_dict(actor.state_dict())
        params = list(tokenizer.parameters()) + list(backbone.parameters()) + list(actor.parameters())
        optimizer = torch.optim.Adam(params, lr=self._actor_lr)
        return _ActorState(
            building_id=building_id,
            tokenizer=tokenizer,
            backbone=backbone,
            actor=actor,
            target_actor=target_actor,
            optimizer=optimizer,
            layout=layout,
            obs_names_tuple=tuple(observation_names),
            action_names_tuple=tuple(action_names),
        )

    def _handle_topology_change(self, building_idx: int) -> None:
        state = self._actors[building_idx]
        new_layout = self._layout_builder.build(
            state.building_id,
            list(state.obs_names_tuple),
            list(state.action_names_tuple),
        )
        new_dims = self._compute_type_input_dims(new_layout)
        for tname, dim in new_dims.items():
            if tname not in state.tokenizer.projections:
                raise ValueError(
                    f"Topology change for {state.building_id!r}: new type "
                    f"{tname!r} has no projection. Restart required."
                )
            if int(state.tokenizer.projections[tname].in_features) != int(dim):
                raise ValueError(
                    f"Topology change for {state.building_id!r}: feature count "
                    f"for type {tname!r} changed. Weights cannot be preserved."
                )
        state.layout = new_layout
        state.topology_version += 1
        logger.info(
            "Topology change: {} v{} — n_ca={}",
            state.building_id, state.topology_version, new_layout.n_ca,
        )

    def _compute_type_input_dims(self, layout: BuildingTokenLayout) -> Dict[str, int]:
        nfc_name = self._tokenizer_config.nfc.type_name
        dims: Dict[str, int] = {nfc_name: 1}
        for seg in layout.segments:
            if seg.family == "nfc":
                continue
            existing = dims.get(seg.type_name)
            new = len(seg.feature_indices)
            if existing is not None and existing != new:
                raise ValueError(
                    f"Inconsistent input dim for type {seg.type_name!r}: "
                    f"{existing} vs {new}"
                )
            dims[seg.type_name] = new
        for tname, ca_cfg in self._tokenizer_config.ca_types.items():
            dims.setdefault(tname, int(ca_cfg.input_dim_fallback))
        for tname, sro_cfg in self._tokenizer_config.sro_types.items():
            dims.setdefault(tname, int(sro_cfg.input_dim_fallback))
        return dims

    @staticmethod
    def _infer_obs_dim(layout: BuildingTokenLayout) -> int:
        return max(max(seg.feature_indices) for seg in layout.segments) + 1

    def _export_actor_onnx(self, state: _ActorState, path: Path, obs_dim: int) -> None:
        layout = state.layout
        sros_idx = [
            torch.tensor(list(s.feature_indices), dtype=torch.long)
            for s in layout.segments if s.family == "sro"
        ]
        ca_idx = [
            torch.tensor(list(s.feature_indices), dtype=torch.long)
            for s in layout.segments if s.family == "ca"
        ]
        nfc_seg = next(s for s in layout.segments if s.family == "nfc")
        nfc_idx = torch.tensor(list(nfc_seg.feature_indices), dtype=torch.long)
        nfc_l = nfc_seg.derived.left_index_in_segment
        nfc_r = nfc_seg.derived.right_index_in_segment
        sro_types = [s.type_name for s in layout.segments if s.family == "sro"]
        ca_types = [s.type_name for s in layout.segments if s.family == "ca"]
        tokenizer = state.tokenizer
        backbone = state.backbone
        actor = state.actor

        class _Wrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.tokenizer = tokenizer
                self_inner.backbone = backbone
                self_inner.actor = actor

            def forward(self_inner, encoded_obs: torch.Tensor) -> torch.Tensor:
                sro_list = []
                for i, idx in enumerate(sros_idx):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    sro_list.append(self_inner.tokenizer.projections[sro_types[i]](g).unsqueeze(1))
                ca_list = []
                for i, idx in enumerate(ca_idx):
                    g = encoded_obs.index_select(dim=1, index=idx)
                    ca_list.append(self_inner.tokenizer.projections[ca_types[i]](g).unsqueeze(1))
                nfc_grp = encoded_obs.index_select(dim=1, index=nfc_idx)
                scalar = (nfc_grp[:, nfc_l] - nfc_grp[:, nfc_r]).unsqueeze(1)
                nfc_tok = self_inner.tokenizer.projections["building_nfc"](scalar).unsqueeze(1)
                sros = torch.cat(sro_list, dim=1) if sro_list else encoded_obs.new_zeros(encoded_obs.shape[0], 0, backbone.d_model)
                cas = torch.cat(ca_list, dim=1) if ca_list else encoded_obs.new_zeros(encoded_obs.shape[0], 0, backbone.d_model)
                ca_emb, _ = self_inner.backbone(sros, nfc_tok, cas)
                return self_inner.actor(ca_emb).squeeze(-1)

        wrapper = _Wrapper().eval()
        dummy = torch.zeros(1, obs_dim)
        with torch.no_grad():
            try:
                from torch.backends.mha import set_fastpath_enabled
                set_fastpath_enabled(False)
                _restore = True
            except ImportError:
                _restore = False
            try:
                torch.onnx.export(
                    wrapper, (dummy,), str(path),
                    input_names=["encoded_obs"], output_names=["actions"],
                    dynamic_axes={"encoded_obs": {0: "batch"}, "actions": {0: "batch"}},
                    opset_version=17,
                )
            finally:
                if _restore:
                    from torch.backends.mha import set_fastpath_enabled
                    set_fastpath_enabled(True)
```

Register in `algorithms/registry.py`:

```python
# Add import after AgentTransformerPPO import
from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3

# Add to ALGORITHM_REGISTRY dict
"AgentTransformerMATD3": AgentTransformerMATD3,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestRegistry -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py algorithms/registry.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "feat(matd3-t): add AgentTransformerMATD3 skeleton + registry"
```

---

## Task 4: Attach Environment + Predict

**Files:**
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3
from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"
_DEFAULT_ACTIONS = ["electrical_storage", "electric_vehicle_storage"]


def _matd3_config() -> dict:
    return {
        "algorithm": {
            "name": "AgentTransformerMATD3",
            "tokenizer_config_path": _TOKENIZER_CFG,
            "transformer_actor": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "transformer_critic": {
                "d_model": 16, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 32, "dropout": 0.0,
            },
            "hyperparameters": {
                "gamma": 0.99, "tau": 0.005, "batch_size": 4,
                "replay_capacity": 100, "actor_lr": 1e-3, "critic_lr": 3e-4,
                "target_policy_noise": 0.2, "target_policy_noise_clip": 0.5,
                "actor_update_interval": 2, "actor_hidden_dim": 32,
            },
        },
    }


def _make_matd3(n_buildings: int = 1):
    obs_names = load_sample_observation_names_for_first_building()
    obs_per = [list(obs_names) for _ in range(n_buildings)]
    act_per = [list(_DEFAULT_ACTIONS) for _ in range(n_buildings)]
    agent = AgentTransformerMATD3(_matd3_config())
    agent.attach_environment(
        observation_names=obs_per,
        action_names=act_per,
        action_space=[None] * n_buildings,
        observation_space=[None] * n_buildings,
        metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
    )
    obs_dim = len(obs_names)
    return agent, obs_per, act_per, obs_dim


class TestAttachAndPredict:
    def test_attach_builds_actors(self):
        agent, _, _, _ = _make_matd3(n_buildings=2)
        assert len(agent._actors) == 2
        for s in agent._actors:
            assert s.layout.n_ca == 2

    def test_attach_noop_on_same_names(self):
        agent, obs_per, act_per, _ = _make_matd3()
        layout_before = agent._actors[0].layout
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[None],
            observation_space=[None],
        )
        assert agent._actors[0].layout is layout_before

    def test_attach_rejects_building_count_change(self):
        agent, obs_per, act_per, _ = _make_matd3(n_buildings=1)
        with pytest.raises(ValueError, match="building count changed"):
            agent.attach_environment(
                observation_names=obs_per + [obs_per[0]],
                action_names=act_per + [act_per[0]],
                action_space=[None, None],
                observation_space=[None, None],
            )

    def test_predict_returns_correct_shape(self):
        agent, _, act_per, obs_dim = _make_matd3(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        actions = agent.predict(obs, deterministic=True)
        assert len(actions) == 2
        for a, expected_names in zip(actions, act_per):
            assert len(a) == len(expected_names)

    def test_predict_actions_in_range(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        actions = agent.predict(obs, deterministic=True)
        for val in actions[0]:
            assert -1.0 <= val <= 1.0

    def test_predict_deterministic_reproducible(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        a1 = agent.predict(obs, deterministic=True)
        a2 = agent.predict(obs, deterministic=True)
        assert a1 == a2
```

- [ ] **Step 2: Run test to verify it passes** (should pass since agent skeleton was written in Task 3)

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestAttachAndPredict -v`
Expected: 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent_transformer_matd3_foundation.py
git commit -m "test(matd3-t): attach_environment and predict tests"
```

---

## Task 5: Topology Change + Export

**Files:**
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
import tempfile


class TestTopologyChange:
    def test_topology_change_rebuilds_layout(self):
        agent, obs_per, act_per, _ = _make_matd3()
        # Add a fake charger to observation names
        new_obs = list(obs_per[0]) + ["charger::Building_0/charger_new::connected_state"]
        new_act = list(act_per[0]) + ["electric_vehicle_storage_charger_new"]
        agent.attach_environment(
            observation_names=[new_obs],
            action_names=[new_act],
            action_space=[None],
            observation_space=[None],
        )
        assert agent._actors[0].layout.n_ca == 3
        assert agent._actors[0].topology_version == 1

    def test_topology_feature_count_drift_fails(self):
        agent, obs_per, act_per, _ = _make_matd3()
        # Remove features from a storage segment to trigger feature-count drift
        new_obs = [n for n in obs_per[0] if "storage::" not in n]
        # This should fail because storage type feature count changes
        with pytest.raises(ValueError, match="feature count"):
            agent.attach_environment(
                observation_names=[new_obs],
                action_names=[["electrical_storage"]],
                action_space=[None],
                observation_space=[None],
            )


class TestExport:
    def test_export_creates_onnx_files(self):
        agent, _, _, _ = _make_matd3(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            assert manifest["format"] == "onnx"
            assert len(manifest["artifacts"]) == 2
            for art in manifest["artifacts"]:
                onnx_path = Path(tmpdir) / art["path"]
                assert onnx_path.exists()
                assert art["config"]["n_ca"] == 2
                assert "action_low" in art["config"]
                assert "action_high" in art["config"]

    def test_export_manifest_has_no_critic(self):
        agent, _, _, _ = _make_matd3()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = agent.export_artifacts(tmpdir)
            # No critic keys in manifest
            assert "critic" not in str(manifest).lower() or "critic" not in manifest

    def test_checkpoint_round_trip(self):
        agent, _, _, obs_dim = _make_matd3()
        obs = [np.random.randn(obs_dim).astype(np.float64)]
        a_before = agent.predict(obs, deterministic=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent.save_checkpoint(tmpdir, step=100)
            # Create a fresh agent and load
            agent2, _, _, _ = _make_matd3()
            agent2.load_checkpoint(ckpt_path)
            a_after = agent2.predict(obs, deterministic=True)
        assert a_before == a_after

    def test_checkpoint_rejects_building_count_mismatch(self):
        agent1, _, _, _ = _make_matd3(n_buildings=1)
        agent2, _, _, _ = _make_matd3(n_buildings=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent1.save_checkpoint(tmpdir, step=1)
            with pytest.raises(ValueError, match="Building-count mismatch"):
                agent2.load_checkpoint(ckpt_path)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestTopologyChange tests/test_agent_transformer_matd3_foundation.py::TestExport -v`
Expected: 5 tests PASS (topology change tests may need minor fixture adjustments for the new charger feature names — fix iteratively).

- [ ] **Step 3: Fix any failures iteratively, then commit**

```bash
git add tests/test_agent_transformer_matd3_foundation.py
git commit -m "test(matd3-t): topology change + export + checkpoint tests"
```

---

## Task 6: Config Template YAML

**Files:**
- Create: `configs/templates/rl/transformer_matd3_local.yaml`

- [ ] **Step 1: Create the template**

```yaml
# Local AgentTransformerMATD3 template (Plan A: predict-only, no training).
metadata:
  experiment_name: "transformer_matd3_local_template"
  run_name: "Transformer MATD3 Local"
  community_name: "default_community"
  description: "Per-building Transformer MATD3 over entity interface"

runtime:
  log_dir: null
  job_dir: null
  mlflow_uri: null
  job_id: null
  run_id: null
  run_name: null
  tracking_uri: null
  experiment_id: null
  mlflow_run_url: null

tracking:
  mlflow_enabled: false
  log_level: "INFO"
  log_frequency: 1
  mlflow_step_sample_interval: 10
  mlflow_artifacts_profile: minimal
  progress_updates_enabled: true
  progress_update_interval: 5
  system_metrics_enabled: false
  system_metrics_interval: 10

checkpointing:
  resume_training: false
  checkpoint_run_id: null
  checkpoint_artifact: "transformer_matd3_checkpoint.pt"
  use_best_checkpoint_artifact: false
  reset_replay_buffer: false
  freeze_pretrained_layers: false
  fine_tune: false
  checkpoint_interval: 1
  require_update_step: true
  require_initial_exploration_done: true

bundle:
  bundle_version: null
  description: null
  alias_mapping_path: null
  require_observations_envelope: false
  artifact_config: {}
  per_agent_artifact_config: {}

simulator:
  dataset_name: citylearn_three_phase_dynamic_assets_only_demo_15s_parquet
  dataset_path: ./datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet/schema.json
  central_agent: false
  interface: entity
  topology_mode: dynamic
  entity_encoding:
    enabled: true
    normalization: minmax_space
    clip: true
  reward_function: CostHardConstraintReward
  reward_function_kwargs: {}
  episodes: 1
  simulation_start_time_step: 0
  simulation_end_time_step: 500
  episode_time_steps: 501
  export:
    mode: end
    export_kpis_on_episode_end: true
    session_name: null
  wrapper_reward:
    enabled: false
    profile: cost_limits_v1
    clip_enabled: true
    clip_min: -10.0
    clip_max: 10.0
    squash: none

training:
  seed: 42
  steps_between_training_updates: 4
  target_update_interval: 2

topology:
  num_agents: null
  observation_dimensions: null
  action_dimensions: null
  action_space: null

pipeline:
  - algorithm: "AgentTransformerMATD3"
    count: 1
    tokenizer_config_path: "configs/tokenizers/entity_default.json"
    transformer_actor:
      d_model: 64
      nhead: 4
      num_layers: 2
      dim_feedforward: 128
      dropout: 0.1
    transformer_critic:
      d_model: 64
      nhead: 4
      num_layers: 2
      dim_feedforward: 128
      dropout: 0.1
    hyperparameters:
      gamma: 0.99
      tau: 0.005
      batch_size: 256
      replay_capacity: 100000
      actor_lr: 1.0e-4
      critic_lr: 3.0e-4
      target_policy_noise: 0.2
      target_policy_noise_clip: 0.5
      actor_update_interval: 2

execution: null
```

- [ ] **Step 2: Commit**

```bash
git add configs/templates/rl/transformer_matd3_local.yaml
git commit -m "feat(matd3-t): add local config template"
```

---

## Task 7: Config Schema Guardrail

**Files:**
- Modify: `utils/config_schema.py`
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the test**

Append to test file:

```python
class TestDynamicTopologyGuardrail:
    def test_transformer_matd3_allows_dynamic_topology(self):
        """AgentTransformerMATD3 should not trigger the dynamic-topology error."""
        from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3
        assert AgentTransformerMATD3.supports_dynamic_topology is True

    def test_legacy_matd3_still_rejects_dynamic(self):
        """Legacy MATD3 error message must remain unchanged."""
        from algorithms.agents.matd3_agent import MATD3
        assert not getattr(MATD3, "supports_dynamic_topology", False)
```

- [ ] **Step 2: Run test — should pass immediately**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestDynamicTopologyGuardrail -v`
Expected: PASS (the class var is already set in the skeleton).

- [ ] **Step 3: Verify schema accepts the new stage in config validation**

Ensure `TransformerMATD3StageConfig` is in the `PipelineStageConfig` Union and that config validation against the dynamic guardrail at line 794-808 of `utils/config_schema.py` passes for `AgentTransformerMATD3`. If needed, add the new stage type to the dynamic-mode validation loop.

- [ ] **Step 4: Commit**

```bash
git add utils/config_schema.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "test(matd3-t): dynamic topology guardrail tests"
```

---

## Task 8: Final Verification

- [ ] **Step 1: Run full test suite for Plan A**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run existing tests to confirm no regressions**

Run: `pytest tests/ -x --timeout=120`
Expected: No new failures.

- [ ] **Step 3: Verify import**

```python
python -c "from algorithms.registry import ALGORITHM_REGISTRY; print('AgentTransformerMATD3' in ALGORITHM_REGISTRY)"
```
Expected: `True`

---

## Plan A Complete

After Task 8, the agent is registered, can predict actions, exports ONNX artifacts, handles topology changes at the actor level, and has a working config template. It does not train yet — that's Plan B (twin critics + replay) and Plan C (residual/BC/warm-start).
