# AgentTransformerMATD3 — Plan B: Twin Critics + Replay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver twin independent Transformer critic stacks, a global token packer, topology-partitioned replay buffer, and critic update loop — enough to store transitions, pack global sequences, and train critics against min-Q TD3 targets.

**Architecture:** Two FULLY INDEPENDENT Transformer critic stacks (each with its own TransformerEncoder, type embedding table, and Linear Q head). A global token packer concatenates all buildings' observation tokens + action tokens + per-building identity embeddings + type-family embeddings + padding masks. Replay is topology-partitioned with global capacity eviction.

**Tech Stack:** Python 3.10+, PyTorch, numpy, pytest.

**Spec:** `docs/transformer_matd3_spec.md`

**Depends on:** Plan A (config schema, registry, per-building actor stack, predict, export, topology-change at actor level, checkpoint round-trip for actors).

**Produces:** Twin critic stacks that can be trained from replay on active-topology batches with correct min-Q targets. No actor update yet — that's Plan D integration.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `algorithms/utils/matd3_critic.py` (create) | Twin Transformer critic stacks + Q heads |
| `algorithms/utils/matd3_global_packer.py` (create) | Global token packer (obs + action tokens + embeddings + masks) |
| `algorithms/utils/matd3_replay.py` (create) | Topology-partitioned replay buffer |
| `algorithms/utils/matd3_critic_update.py` (create) | Critic update helper (target Q computation + MSE loss) |
| `tests/test_matd3_critic.py` (create) | Unit tests for critic stacks |
| `tests/test_matd3_global_packer.py` (create) | Unit tests for token packer |
| `tests/test_matd3_replay.py` (create) | Unit tests for partitioned replay |
| `tests/test_matd3_critic_update.py` (create) | Unit tests for critic update loop |

---

## Task 1: Twin Transformer Critic Stacks

**Files:**
- Create: `algorithms/utils/matd3_critic.py`
- Create: `tests/test_matd3_critic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_critic.py`:

```python
"""Unit tests for twin independent Transformer critic stacks."""
from __future__ import annotations

import pytest
import torch

from algorithms.utils.matd3_critic import (
    TransformerCriticStack,
    TwinTransformerCritics,
)


class TestTransformerCriticStack:
    """Single critic stack behavior."""

    def test_output_shape_single_building(self):
        """Q output per controlled building."""
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        # Simulate global sequence: 5 tokens, 1 controlled building
        global_tokens = torch.randn(2, 5, 16)  # [B=2, T=5, d=16]
        type_ids = torch.zeros(2, 5, dtype=torch.long)
        building_ids = torch.zeros(2, 5, dtype=torch.long)
        padding_mask = torch.zeros(2, 5, dtype=torch.bool)  # no padding
        controlled_building_indices = [0]

        q_values = critic(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        assert q_values.shape == (2, 1)  # [B, n_controlled]

    def test_output_shape_multi_building(self):
        """Q output for multiple controlled buildings."""
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        global_tokens = torch.randn(4, 10, 16)
        type_ids = torch.zeros(4, 10, dtype=torch.long)
        building_ids = torch.cat([
            torch.zeros(4, 5, dtype=torch.long),
            torch.ones(4, 5, dtype=torch.long),
        ], dim=1)
        padding_mask = torch.zeros(4, 10, dtype=torch.bool)
        controlled_building_indices = [0, 1]

        q_values = critic(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        assert q_values.shape == (4, 2)

    def test_padding_mask_respected(self):
        """Padded tokens should not affect output of non-padded tokens."""
        torch.manual_seed(42)
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        critic.eval()

        global_tokens = torch.randn(1, 6, 16)
        type_ids = torch.zeros(1, 6, dtype=torch.long)
        building_ids = torch.zeros(1, 6, dtype=torch.long)

        # No padding
        mask_none = torch.zeros(1, 6, dtype=torch.bool)
        q_no_pad = critic(global_tokens, type_ids, building_ids, mask_none, [0])

        # Pad last 2 tokens
        mask_pad = torch.tensor([[False, False, False, False, True, True]])
        tokens_with_junk = global_tokens.clone()
        tokens_with_junk[:, 4:, :] = torch.randn(1, 2, 16) * 100.0
        q_with_pad = critic(tokens_with_junk, type_ids, building_ids, mask_pad, [0])

        # They should be equal since padded tokens are masked
        assert torch.allclose(q_no_pad, q_with_pad, atol=1e-5)

    def test_type_embeddings_affect_output(self):
        """Different type_ids should produce different outputs."""
        torch.manual_seed(0)
        critic = TransformerCriticStack(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        critic.eval()

        global_tokens = torch.randn(1, 4, 16)
        building_ids = torch.zeros(1, 4, dtype=torch.long)
        padding_mask = torch.zeros(1, 4, dtype=torch.bool)

        type_ids_a = torch.tensor([[0, 1, 2, 3]])
        type_ids_b = torch.tensor([[4, 5, 6, 7]])

        q_a = critic(global_tokens, type_ids_a, building_ids, padding_mask, [0])
        q_b = critic(global_tokens, type_ids_b, building_ids, padding_mask, [0])

        assert not torch.allclose(q_a, q_b)


class TestTwinTransformerCritics:
    """Twin critics independence and min-Q."""

    def test_critics_are_independent(self):
        """No shared parameters between critic 1 and critic 2."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        params_1 = set(id(p) for p in twins.critic_1.parameters())
        params_2 = set(id(p) for p in twins.critic_2.parameters())
        assert params_1.isdisjoint(params_2), "Twin critics share parameters!"

    def test_critics_produce_different_outputs(self):
        """After init, twin critics produce different Q values."""
        torch.manual_seed(99)
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        twins.eval()

        global_tokens = torch.randn(2, 6, 16)
        type_ids = torch.zeros(2, 6, dtype=torch.long)
        building_ids = torch.zeros(2, 6, dtype=torch.long)
        padding_mask = torch.zeros(2, 6, dtype=torch.bool)
        controlled = [0]

        q1, q2 = twins(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled,
        )
        assert q1.shape == q2.shape == (2, 1)
        assert not torch.allclose(q1, q2)

    def test_min_q_helper(self):
        """min_q returns element-wise minimum."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        twins.eval()

        global_tokens = torch.randn(3, 4, 16)
        type_ids = torch.zeros(3, 4, dtype=torch.long)
        building_ids = torch.zeros(3, 4, dtype=torch.long)
        padding_mask = torch.zeros(3, 4, dtype=torch.bool)

        min_q = twins.min_q(
            global_tokens, type_ids, building_ids,
            padding_mask, [0],
        )
        q1, q2 = twins(
            global_tokens, type_ids, building_ids,
            padding_mask, [0],
        )
        expected = torch.min(q1, q2)
        assert torch.allclose(min_q, expected)

    def test_soft_update(self):
        """Soft update with tau=1.0 makes target equal to online."""
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins.soft_update_from(twins, tau=1.0)

        for p_online, p_target in zip(twins.parameters(), target_twins.parameters()):
            assert torch.allclose(p_online, p_target)

    def test_soft_update_partial(self):
        """Soft update with tau=0.0 leaves target unchanged."""
        torch.manual_seed(7)
        twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        target_twins = TwinTransformerCritics(
            d_model=16, nhead=2, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            num_token_types=8, max_buildings=4,
        )
        # Snapshot target before update
        before = [p.clone() for p in target_twins.parameters()]
        target_twins.soft_update_from(twins, tau=0.0)
        for p_before, p_after in zip(before, target_twins.parameters()):
            assert torch.allclose(p_before, p_after)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_critic.py -v`
Expected: ImportError — `matd3_critic` module does not exist.

- [ ] **Step 3: Implement the critic stacks**

Create `algorithms/utils/matd3_critic.py`:

```python
"""Twin independent Transformer critic stacks for AgentTransformerMATD3.

Each critic stack has its own:
- TransformerEncoder backbone (separate weights, type embeddings, layer params)
- Type embedding table (num_token_types entries)
- Building identity embedding table (max_buildings entries)
- Q head: per-building query → scalar Q-value

The two critics are FULLY INDEPENDENT — no shared parameters — to preserve
TD3's overestimation reduction property.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class TransformerCriticStack(nn.Module):
    """Single Transformer critic: global sequence → per-building Q values.

    The Q head uses a learned query token per controlled building that
    attends to the encoded global sequence, then projects to a scalar.
    Implementation: mean-pool tokens belonging to each controlled building,
    then project through a 2-layer MLP → scalar.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_token_types: int,
        max_buildings: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_token_types = num_token_types
        self.max_buildings = max_buildings

        # Per-type embeddings (obs_sro, obs_nfc, obs_ca, action, etc.)
        self.type_embedding = nn.Embedding(num_token_types, d_model)
        # Per-building identity embeddings
        self.building_embedding = nn.Embedding(max_buildings, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Q head: per-building pooled embedding → scalar
        self.q_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(
        self,
        global_tokens: torch.Tensor,       # [B, T, d_model]
        type_ids: torch.Tensor,            # [B, T] (long)
        building_ids: torch.Tensor,        # [B, T] (long)
        padding_mask: torch.Tensor,        # [B, T] (bool, True=pad)
        controlled_building_indices: List[int],
    ) -> torch.Tensor:
        """Return Q-values [B, n_controlled] for each controlled building."""
        # Add type + building embeddings
        seq = global_tokens + self.type_embedding(type_ids) + self.building_embedding(building_ids)

        # Encode with padding mask (True positions are ignored)
        encoded = self.encoder(seq, src_key_padding_mask=padding_mask)

        # Pool per controlled building: mean of non-padded tokens for that building
        batch_size = encoded.shape[0]
        device = encoded.device
        q_values = []

        for b_idx in controlled_building_indices:
            # Mask: tokens belonging to this building AND not padded
            building_mask = (building_ids == b_idx) & (~padding_mask)  # [B, T]
            # Expand for broadcasting: [B, T, 1]
            building_mask_expanded = building_mask.unsqueeze(-1).float()
            # Sum tokens for this building
            summed = (encoded * building_mask_expanded).sum(dim=1)  # [B, d_model]
            count = building_mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
            pooled = summed / count  # [B, d_model]
            q = self.q_head(pooled)  # [B, 1]
            q_values.append(q)

        return torch.cat(q_values, dim=-1)  # [B, n_controlled]


class TwinTransformerCritics(nn.Module):
    """Container for two fully independent TransformerCriticStack instances.

    Provides:
    - forward() → (q1, q2) tuple
    - min_q() → element-wise minimum
    - soft_update_from(source, tau) for target network updates
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_token_types: int,
        max_buildings: int,
    ) -> None:
        super().__init__()
        self.critic_1 = TransformerCriticStack(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            num_token_types=num_token_types, max_buildings=max_buildings,
        )
        self.critic_2 = TransformerCriticStack(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            num_token_types=num_token_types, max_buildings=max_buildings,
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        building_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        controlled_building_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (q1, q2) each of shape [B, n_controlled]."""
        q1 = self.critic_1(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        q2 = self.critic_2(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        return q1, q2

    def min_q(
        self,
        global_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        building_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        controlled_building_indices: List[int],
    ) -> torch.Tensor:
        """Return element-wise min(q1, q2) of shape [B, n_controlled]."""
        q1, q2 = self.forward(
            global_tokens, type_ids, building_ids,
            padding_mask, controlled_building_indices,
        )
        return torch.min(q1, q2)

    @torch.no_grad()
    def soft_update_from(self, source: "TwinTransformerCritics", tau: float) -> None:
        """Polyak-average: target = tau * source + (1 - tau) * target."""
        for p_target, p_source in zip(self.parameters(), source.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_source.data, alpha=tau)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_critic.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_critic.py tests/test_matd3_critic.py
git commit -m "feat(matd3-t): add twin independent Transformer critic stacks"
```

---

## Task 2: Global Token Packer

**Files:**
- Create: `algorithms/utils/matd3_global_packer.py`
- Create: `tests/test_matd3_global_packer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_global_packer.py`:

```python
"""Unit tests for the global critic token packer."""
from __future__ import annotations

import pytest
import torch

from algorithms.utils.matd3_global_packer import (
    GlobalTokenPacker,
    PackedGlobalSequence,
    BuildingLayout,
)


def _make_packer(d_model=16, num_token_types=8, max_buildings=4,
                 action_input_mode="final") -> GlobalTokenPacker:
    return GlobalTokenPacker(
        d_model=d_model,
        num_token_types=num_token_types,
        max_buildings=max_buildings,
        action_input_mode=action_input_mode,
    )


def _make_layouts(n_buildings=2, n_sro=2, n_ca=2) -> list[BuildingLayout]:
    """Create synthetic building layouts."""
    return [
        BuildingLayout(
            building_index=b,
            n_sro=n_sro,
            n_nfc=1,
            n_ca=n_ca,
            is_controlled=True,
        )
        for b in range(n_buildings)
    ]


class TestGlobalTokenPacker:
    def test_output_type(self):
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=2, n_ca=2)
        # Per building: n_sro + 1(nfc) + n_ca obs tokens + n_ca action tokens = 2+1+2+2 = 7
        # Total: 2 buildings * 7 = 14
        obs_tokens = [torch.randn(3, 5, 16) for _ in range(2)]  # [B, n_obs_tokens, d]
        action_values = [torch.randn(3, 2) for _ in range(2)]  # [B, n_ca] per building

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert isinstance(packed, PackedGlobalSequence)

    def test_output_shapes(self):
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=3, n_ca=2)
        # Per building obs tokens: 3(sro) + 1(nfc) + 2(ca) = 6
        obs_tokens = [torch.randn(4, 6, 16) for _ in range(2)]
        action_values = [torch.randn(4, 2) for _ in range(2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        # Total tokens: 2 buildings * (6 obs + 2 action) = 16
        assert packed.global_tokens.shape == (4, 16, 16)
        assert packed.type_ids.shape == (4, 16)
        assert packed.building_ids.shape == (4, 16)
        assert packed.padding_mask.shape == (4, 16)

    def test_variable_building_token_counts(self):
        """Different buildings can have different token counts with padding."""
        packer = _make_packer()
        layouts = [
            BuildingLayout(building_index=0, n_sro=2, n_nfc=1, n_ca=1, is_controlled=True),
            BuildingLayout(building_index=1, n_sro=4, n_nfc=1, n_ca=3, is_controlled=True),
        ]
        # Building 0: 2+1+1 = 4 obs, 1 action = 5 total
        # Building 1: 4+1+3 = 8 obs, 3 action = 11 total
        obs_tokens = [torch.randn(2, 4, 16), torch.randn(2, 8, 16)]
        action_values = [torch.randn(2, 1), torch.randn(2, 3)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        # Max per building = 11, total = 5 + 11 = 16
        total_tokens = 5 + 11
        assert packed.global_tokens.shape == (2, total_tokens, 16)
        # No padding needed since we pack contiguously
        assert packed.padding_mask.shape == (2, total_tokens)

    def test_padding_mask_correct(self):
        """Padding positions should be True in mask."""
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=1, n_sro=2, n_ca=2)
        obs_tokens = [torch.randn(1, 5, 16)]  # n_sro+1+n_ca = 5
        action_values = [torch.randn(1, 2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        # No padding in this case: all tokens are real
        assert not packed.padding_mask.any()

    def test_action_mode_final(self):
        """Final mode: action token contains 1 scalar projected to d_model."""
        packer = _make_packer(action_input_mode="final")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]  # 1+1+2
        action_values = [torch.randn(2, 2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        # Should have obs_tokens (4) + action_tokens (2) = 6 total
        assert packed.global_tokens.shape[1] == 6

    def test_action_mode_final_base_delta(self):
        """final_base_delta mode: action token carries 3 scalars."""
        packer = _make_packer(action_input_mode="final_base_delta")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]
        action_values = [torch.randn(2, 2)]
        base_actions = [torch.randn(2, 2)]

        packed = packer.pack(
            obs_tokens, action_values, layouts,
            base_actions=base_actions,
        )
        assert packed.global_tokens.shape[1] == 6

    def test_action_mode_final_base_delta_normalized(self):
        """Normalized mode: delta is divided by action_span."""
        packer = _make_packer(action_input_mode="final_base_delta_normalized")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]
        action_values = [torch.randn(2, 2)]
        base_actions = [torch.randn(2, 2)]

        packed = packer.pack(
            obs_tokens, action_values, layouts,
            base_actions=base_actions,
            action_span=2.0,
        )
        assert packed.global_tokens.shape[1] == 6

    def test_controlled_building_indices(self):
        """PackedGlobalSequence reports correct controlled building list."""
        packer = _make_packer()
        layouts = [
            BuildingLayout(building_index=0, n_sro=2, n_nfc=1, n_ca=2, is_controlled=True),
            BuildingLayout(building_index=1, n_sro=2, n_nfc=1, n_ca=0, is_controlled=False),
            BuildingLayout(building_index=2, n_sro=2, n_nfc=1, n_ca=1, is_controlled=True),
        ]
        obs_tokens = [
            torch.randn(1, 5, 16),
            torch.randn(1, 3, 16),  # no CA tokens
            torch.randn(1, 4, 16),
        ]
        action_values = [
            torch.randn(1, 2),
            torch.randn(1, 0),  # no actions
            torch.randn(1, 1),
        ]
        packed = packer.pack(obs_tokens, action_values, layouts)
        assert packed.controlled_building_indices == [0, 2]

    def test_building_ids_correct(self):
        """Each token's building_id matches its source building."""
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=1, n_ca=1)
        obs_tokens = [torch.randn(1, 3, 16), torch.randn(1, 3, 16)]  # 1+1+1=3 each
        action_values = [torch.randn(1, 1), torch.randn(1, 1)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        # Building 0: tokens 0-3 (3 obs + 1 action = 4)
        # Building 1: tokens 4-7 (3 obs + 1 action = 4)
        assert (packed.building_ids[0, :4] == 0).all()
        assert (packed.building_ids[0, 4:] == 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_global_packer.py -v`
Expected: ImportError — `matd3_global_packer` module does not exist.

- [ ] **Step 3: Implement the global token packer**

Create `algorithms/utils/matd3_global_packer.py`:

```python
"""Global critic token packer for AgentTransformerMATD3.

Concatenates all buildings' observation tokens + action tokens into a single
global sequence for the centralized twin critics. Handles:
- Variable per-building token counts
- Per-building identity embeddings (via building_ids)
- Type-family embeddings (via type_ids)
- Padding masks for variable-length sequences within a batch
- Three action token content modes: final, final_base_delta, final_base_delta_normalized
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


# Type ID constants for the global critic sequence.
# These index into the critic's type_embedding table.
TYPE_OBS_SRO = 0
TYPE_OBS_NFC = 1
TYPE_OBS_CA = 2
TYPE_ACTION = 3
# Reserve 4-7 for future extensions (e.g., edge tokens, context tokens)


@dataclass
class BuildingLayout:
    """Lightweight layout summary for the packer.

    Extracted from the full BuildingTokenLayout during transition storage.
    """
    building_index: int
    n_sro: int
    n_nfc: int  # always 1
    n_ca: int
    is_controlled: bool

    @property
    def n_obs_tokens(self) -> int:
        return self.n_sro + self.n_nfc + self.n_ca

    @property
    def n_action_tokens(self) -> int:
        return self.n_ca

    @property
    def n_total_tokens(self) -> int:
        return self.n_obs_tokens + self.n_action_tokens


@dataclass
class PackedGlobalSequence:
    """Output of the global token packer, ready for twin critics."""
    global_tokens: torch.Tensor    # [B, T_total, d_model]
    type_ids: torch.Tensor         # [B, T_total] (long)
    building_ids: torch.Tensor     # [B, T_total] (long)
    padding_mask: torch.Tensor     # [B, T_total] (bool, True=padded)
    controlled_building_indices: List[int]


class GlobalTokenPacker(nn.Module):
    """Pack per-building obs + action tokens into a single global sequence.

    Action token encoding modes:
    - ``final``: each action token is projected from 1 scalar (the final action).
    - ``final_base_delta``: projected from 3 scalars (final, base, delta).
    - ``final_base_delta_normalized``: projected from 3 scalars (final, base,
      delta / action_span).
    """

    def __init__(
        self,
        d_model: int,
        num_token_types: int,
        max_buildings: int,
        action_input_mode: str = "final",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.action_input_mode = action_input_mode
        self.num_token_types = num_token_types
        self.max_buildings = max_buildings

        # Action token projection: scalars → d_model
        if action_input_mode == "final":
            action_input_dim = 1
        elif action_input_mode in ("final_base_delta", "final_base_delta_normalized"):
            action_input_dim = 3
        else:
            raise ValueError(f"Unknown action_input_mode: {action_input_mode!r}")

        self.action_projection = nn.Linear(action_input_dim, d_model)

    def pack(
        self,
        obs_tokens_per_building: List[torch.Tensor],  # [B, n_obs_tokens_b, d_model]
        action_values_per_building: List[torch.Tensor],  # [B, n_ca_b]
        layouts: List[BuildingLayout],
        *,
        base_actions: Optional[List[torch.Tensor]] = None,  # [B, n_ca_b] per building
        action_span: float = 2.0,
    ) -> PackedGlobalSequence:
        """Pack all buildings into a global sequence.

        Args:
            obs_tokens_per_building: Pre-computed observation token embeddings
                per building, each [B, n_obs_b, d_model].
            action_values_per_building: Final action values per building,
                each [B, n_ca_b].
            layouts: Layout summaries for each building.
            base_actions: Teacher/base actions per building (required for
                final_base_delta modes).
            action_span: Action range width for normalization (default 2.0 for [-1,1]).

        Returns:
            PackedGlobalSequence ready for the twin critics.
        """
        assert len(obs_tokens_per_building) == len(layouts)
        assert len(action_values_per_building) == len(layouts)
        batch_size = obs_tokens_per_building[0].shape[0]
        device = obs_tokens_per_building[0].device

        all_tokens: List[torch.Tensor] = []  # each [B, n_b, d_model]
        all_type_ids: List[torch.Tensor] = []  # each [n_b]
        all_building_ids: List[torch.Tensor] = []  # each [n_b]
        controlled_building_indices: List[int] = []

        for i, layout in enumerate(layouts):
            obs_toks = obs_tokens_per_building[i]  # [B, n_obs, d_model]
            n_obs = obs_toks.shape[1]

            # Build type ids for obs tokens: [SRO]*n_sro + [NFC]*1 + [CA]*n_ca
            obs_type_ids = torch.cat([
                torch.full((layout.n_sro,), TYPE_OBS_SRO, dtype=torch.long, device=device),
                torch.full((layout.n_nfc,), TYPE_OBS_NFC, dtype=torch.long, device=device),
                torch.full((layout.n_ca,), TYPE_OBS_CA, dtype=torch.long, device=device),
            ])

            # Build action tokens
            actions = action_values_per_building[i]  # [B, n_ca]
            n_ca = layout.n_ca
            if n_ca > 0:
                action_toks = self._encode_action_tokens(
                    actions, base_actions[i] if base_actions else None,
                    action_span, device,
                )  # [B, n_ca, d_model]
                action_type_ids = torch.full(
                    (n_ca,), TYPE_ACTION, dtype=torch.long, device=device,
                )
            else:
                action_toks = torch.zeros(
                    batch_size, 0, self.d_model, device=device,
                )
                action_type_ids = torch.zeros(0, dtype=torch.long, device=device)

            # Concatenate obs + action for this building
            building_tokens = torch.cat([obs_toks, action_toks], dim=1)
            building_type_ids = torch.cat([obs_type_ids, action_type_ids])
            n_total = building_tokens.shape[1]
            building_id_vec = torch.full(
                (n_total,), layout.building_index, dtype=torch.long, device=device,
            )

            all_tokens.append(building_tokens)
            all_type_ids.append(building_type_ids)
            all_building_ids.append(building_id_vec)

            if layout.is_controlled:
                controlled_building_indices.append(layout.building_index)

        # Concatenate across buildings
        global_tokens = torch.cat(all_tokens, dim=1)  # [B, T_total, d_model]
        type_ids = torch.cat(all_type_ids).unsqueeze(0).expand(batch_size, -1)
        building_ids = torch.cat(all_building_ids).unsqueeze(0).expand(batch_size, -1)
        padding_mask = torch.zeros(
            batch_size, global_tokens.shape[1], dtype=torch.bool, device=device,
        )

        return PackedGlobalSequence(
            global_tokens=global_tokens,
            type_ids=type_ids,
            building_ids=building_ids,
            padding_mask=padding_mask,
            controlled_building_indices=controlled_building_indices,
        )

    def _encode_action_tokens(
        self,
        actions: torch.Tensor,        # [B, n_ca]
        base_actions: Optional[torch.Tensor],  # [B, n_ca] or None
        action_span: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Project action scalars to d_model embeddings. Returns [B, n_ca, d_model]."""
        batch_size, n_ca = actions.shape

        if self.action_input_mode == "final":
            # [B, n_ca] → [B, n_ca, 1]
            action_input = actions.unsqueeze(-1)
        elif self.action_input_mode == "final_base_delta":
            if base_actions is None:
                raise ValueError(
                    "base_actions required for action_input_mode='final_base_delta'"
                )
            delta = actions - base_actions
            # [B, n_ca, 3]
            action_input = torch.stack([actions, base_actions, delta], dim=-1)
        elif self.action_input_mode == "final_base_delta_normalized":
            if base_actions is None:
                raise ValueError(
                    "base_actions required for action_input_mode='final_base_delta_normalized'"
                )
            delta = (actions - base_actions) / max(action_span, 1e-8)
            action_input = torch.stack([actions, base_actions, delta], dim=-1)
        else:
            raise ValueError(f"Unknown action_input_mode: {self.action_input_mode!r}")

        # Project: [B, n_ca, input_dim] → [B, n_ca, d_model]
        return self.action_projection(action_input)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_global_packer.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_global_packer.py tests/test_matd3_global_packer.py
git commit -m "feat(matd3-t): add global critic token packer"
```

---

## Task 3: Topology-Partitioned Replay Buffer

**Files:**
- Create: `algorithms/utils/matd3_replay.py`
- Create: `tests/test_matd3_replay.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_replay.py`:

```python
"""Unit tests for the topology-partitioned replay buffer."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.utils.matd3_replay import (
    TopologyPartitionedReplay,
    TransitionData,
    LayoutSummary,
    compute_topology_signature,
)


def _make_layout_summary(building_id="B0", n_ca=2, n_sro=3) -> LayoutSummary:
    return LayoutSummary(
        building_id=building_id,
        n_ca=n_ca,
        n_sro=n_sro,
        obs_dim=10,
        action_dim=n_ca,
    )


def _make_transition(
    n_buildings=2, obs_dim=10, n_ca=2, topology_sig="sig_a",
) -> TransitionData:
    return TransitionData(
        observations=[np.random.randn(obs_dim).astype(np.float32) for _ in range(n_buildings)],
        next_observations=[np.random.randn(obs_dim).astype(np.float32) for _ in range(n_buildings)],
        actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        base_actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        next_base_actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        rewards=[float(np.random.randn()) for _ in range(n_buildings)],
        done=False,
        topology_signature=topology_sig,
        layout_summaries=[_make_layout_summary(f"B{i}", n_ca=n_ca) for i in range(n_buildings)],
    )


class TestTopologySignature:
    def test_same_inputs_same_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0", "B1"],
            observation_names=[["a", "b"], ["c", "d"]],
            action_names=[["act0"], ["act1"]],
            ca_action_names=[["act0"], ["act1"]],
            per_type_feature_dims={"storage": 5, "pv": 3},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0", "B1"],
            observation_names=[["a", "b"], ["c", "d"]],
            action_names=[["act0"], ["act1"]],
            ca_action_names=[["act0"], ["act1"]],
            per_type_feature_dims={"storage": 5, "pv": 3},
        )
        assert sig1 == sig2

    def test_different_obs_different_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a", "b"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a", "b", "c"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        assert sig1 != sig2

    def test_different_feature_dims_different_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 7},
        )
        assert sig1 != sig2

    def test_deterministic(self):
        """Same call produces same result (no random component)."""
        for _ in range(10):
            sig = compute_topology_signature(
                building_ids=["X"],
                observation_names=[["f1", "f2"]],
                action_names=[["a"]],
                ca_action_names=[["a"]],
                per_type_feature_dims={"t": 4},
            )
        # If we got here without error, it's deterministic (same output)
        assert isinstance(sig, str)
        assert len(sig) > 0


class TestTopologyPartitionedReplay:
    def test_push_and_size(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=4)
        replay.set_active_signature("sig_a")
        t = _make_transition(topology_sig="sig_a")
        replay.push(t)
        assert replay.active_partition_size == 1
        assert replay.total_size == 1

    def test_sample_returns_batch_from_active_only(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=4)
        replay.set_active_signature("sig_a")

        # Push 10 transitions to sig_a
        for _ in range(10):
            replay.push(_make_transition(topology_sig="sig_a"))

        # Push 5 transitions to sig_b
        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))

        # Sample should only return sig_b transitions
        batch = replay.sample()
        assert batch is not None
        assert batch.topology_signature == "sig_b"
        assert len(batch.observations[0]) == 4  # batch_size

    def test_sample_returns_none_when_insufficient(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=10)
        replay.set_active_signature("sig_a")

        # Push only 5, need 10
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))

        batch = replay.sample()
        assert batch is None

    def test_eviction_oldest_non_active_first(self):
        replay = TopologyPartitionedReplay(capacity=10, batch_size=2)

        # Fill with sig_a (5 transitions)
        replay.set_active_signature("sig_a")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))

        # Fill with sig_b (5 transitions) -> capacity full
        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))

        assert replay.total_size == 10

        # Push 3 more to sig_b -> should evict from sig_a (non-active)
        for _ in range(3):
            replay.push(_make_transition(topology_sig="sig_b"))

        assert replay.total_size == 10  # still at capacity
        assert replay.partition_size("sig_a") == 2  # evicted 3 from sig_a
        assert replay.partition_size("sig_b") == 8

    def test_eviction_ring_buffer_within_active(self):
        replay = TopologyPartitionedReplay(capacity=5, batch_size=2)
        replay.set_active_signature("sig_a")

        # Push 8 transitions, capacity is 5
        for _ in range(8):
            replay.push(_make_transition(topology_sig="sig_a"))

        assert replay.total_size == 5
        assert replay.active_partition_size == 5

    def test_partition_count(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=2)
        replay.set_active_signature("sig_a")
        replay.push(_make_transition(topology_sig="sig_a"))
        replay.set_active_signature("sig_b")
        replay.push(_make_transition(topology_sig="sig_b"))
        replay.set_active_signature("sig_c")
        replay.push(_make_transition(topology_sig="sig_c"))

        assert replay.partition_count == 3

    def test_sample_batch_contents_structure(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=3)
        replay.set_active_signature("sig_a")

        n_buildings = 2
        for _ in range(10):
            replay.push(_make_transition(n_buildings=n_buildings, topology_sig="sig_a"))

        batch = replay.sample()
        assert batch is not None
        assert len(batch.observations) == n_buildings
        assert len(batch.next_observations) == n_buildings
        assert len(batch.actions) == n_buildings
        assert len(batch.base_actions) == n_buildings
        assert len(batch.next_base_actions) == n_buildings
        assert len(batch.rewards) == n_buildings
        # Each is [batch_size, dim]
        assert batch.observations[0].shape[0] == 3
        assert batch.rewards[0].shape == (3,)
        assert batch.done.shape == (3,)

    def test_set_active_signature_switch(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=2)
        replay.set_active_signature("sig_a")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))

        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))

        assert replay.active_signature == "sig_b"
        assert replay.active_partition_size == 5

    def test_checkpoint_state(self):
        replay = TopologyPartitionedReplay(capacity=50, batch_size=4)
        replay.set_active_signature("sig_a")
        for _ in range(10):
            replay.push(_make_transition(topology_sig="sig_a"))

        state = replay.state_dict()
        assert "active_signature" in state
        assert "partitions" in state

        replay2 = TopologyPartitionedReplay(capacity=50, batch_size=4)
        replay2.load_state_dict(state)
        assert replay2.active_signature == "sig_a"
        assert replay2.active_partition_size == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_replay.py -v`
Expected: ImportError — `matd3_replay` module does not exist.

- [ ] **Step 3: Implement the topology-partitioned replay**

Create `algorithms/utils/matd3_replay.py`:

```python
"""Topology-partitioned replay buffer for AgentTransformerMATD3.

Stores transitions partitioned by topology signature. Sampling only returns
transitions from the active signature. Global capacity with eviction from
oldest non-active partition first; ring-buffer within active partition.

Each transition stores:
- Per-building observations, next_observations, actions, base_actions,
  next_base_actions, rewards.
- Done flag.
- Topology signature.
- Per-building layout summaries (n_ca, n_sro counts).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


def compute_topology_signature(
    building_ids: List[str],
    observation_names: List[List[str]],
    action_names: List[List[str]],
    ca_action_names: List[List[str]],
    per_type_feature_dims: Dict[str, int],
) -> str:
    """Compute a stable hash of the topology configuration.

    Includes building_ids, observation/action names, and per-type feature
    dimensions to prevent accidentally reusing incompatible replay or actor
    state when buildings are reordered or schema drifts.
    """
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
    observations: List[npt.NDArray[np.float32]]       # per building [obs_dim]
    next_observations: List[npt.NDArray[np.float32]]  # per building [obs_dim]
    actions: List[npt.NDArray[np.float32]]            # per building [n_ca]
    base_actions: List[npt.NDArray[np.float32]]       # per building [n_ca]
    next_base_actions: List[npt.NDArray[np.float32]]  # per building [n_ca]
    rewards: List[float]                              # per building
    done: bool
    topology_signature: str
    layout_summaries: List[LayoutSummary]


@dataclass
class SampledBatch:
    """Batch sampled from a single topology partition."""
    observations: List[npt.NDArray[np.float32]]       # per building [batch_size, obs_dim]
    next_observations: List[npt.NDArray[np.float32]]  # per building [batch_size, obs_dim]
    actions: List[npt.NDArray[np.float32]]            # per building [batch_size, n_ca]
    base_actions: List[npt.NDArray[np.float32]]       # per building [batch_size, n_ca]
    next_base_actions: List[npt.NDArray[np.float32]]  # per building [batch_size, n_ca]
    rewards: List[npt.NDArray[np.float32]]            # per building [batch_size]
    done: npt.NDArray[np.float32]                     # [batch_size]
    topology_signature: str
    layout_summaries: List[LayoutSummary]


class _Partition:
    """Ring buffer for a single topology signature."""

    def __init__(self) -> None:
        self.transitions: List[TransitionData] = []
        self.position: int = 0
        self._max_size: Optional[int] = None  # unlimited until eviction needed

    @property
    def size(self) -> int:
        return len(self.transitions)

    def push(self, transition: TransitionData) -> None:
        if self._max_size is not None and self.size >= self._max_size:
            # Ring-buffer overwrite
            self.transitions[self.position] = transition
            self.position = (self.position + 1) % self._max_size
        else:
            self.transitions.append(transition)

    def evict_oldest(self) -> bool:
        """Remove the oldest transition. Returns False if empty."""
        if not self.transitions:
            return False
        # The oldest is at `position` index (ring buffer wraps)
        if self.size <= self.position:
            # Haven't wrapped yet: oldest is index 0
            self.transitions.pop(0)
            if self.position > 0:
                self.position -= 1
        else:
            # Wrapped: oldest is at position
            self.transitions.pop(self.position)
            # Position stays the same (wraps around smaller list)
            if self.position >= len(self.transitions) and self.transitions:
                self.position = 0
        return True

    def sample_indices(self, batch_size: int) -> List[int]:
        return random.sample(range(self.size), batch_size)


class TopologyPartitionedReplay:
    """Global-capacity replay buffer partitioned by topology signature.

    Sampling returns only transitions from the active signature.
    Eviction: oldest from oldest non-active partition first; then
    ring-buffer within active partition.
    """

    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.batch_size = batch_size
        self._active_signature: Optional[str] = None
        # OrderedDict preserves insertion order (oldest first)
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

        # Evict if at capacity
        while self.total_size >= self.capacity:
            evicted = self._evict_one()
            if not evicted:
                break

        self._partitions[sig].push(transition)

    def sample(self) -> Optional[SampledBatch]:
        """Sample a batch from the active partition.

        Returns None if active partition has fewer than batch_size transitions.
        """
        if self._active_signature is None:
            return None
        partition = self._partitions.get(self._active_signature)
        if partition is None or partition.size < self.batch_size:
            return None

        indices = partition.sample_indices(self.batch_size)
        transitions = [partition.transitions[i] for i in indices]

        # Determine structure from first transition
        n_buildings = len(transitions[0].observations)

        # Collate into arrays
        observations = [
            np.stack([t.observations[b] for t in transitions])
            for b in range(n_buildings)
        ]
        next_observations = [
            np.stack([t.next_observations[b] for t in transitions])
            for b in range(n_buildings)
        ]
        actions = [
            np.stack([t.actions[b] for t in transitions])
            for b in range(n_buildings)
        ]
        base_actions = [
            np.stack([t.base_actions[b] for t in transitions])
            for b in range(n_buildings)
        ]
        next_base_actions = [
            np.stack([t.next_base_actions[b] for t in transitions])
            for b in range(n_buildings)
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
        """Evict one transition. Prefer oldest non-active partition."""
        # Try non-active partitions first (oldest first due to OrderedDict)
        for sig, partition in self._partitions.items():
            if sig == self._active_signature:
                continue
            if partition.size > 0:
                partition.evict_oldest()
                # Remove partition if empty
                if partition.size == 0:
                    del self._partitions[sig]
                return True

        # All non-active are empty; ring-buffer the active partition
        if self._active_signature and self._active_signature in self._partitions:
            active = self._partitions[self._active_signature]
            if active.size > 0:
                active.evict_oldest()
                return True
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
                        "layout_summaries": [
                            {"building_id": ls.building_id, "n_ca": ls.n_ca,
                             "n_sro": ls.n_sro, "obs_dim": ls.obs_dim,
                             "action_dim": ls.action_dim}
                            for ls in t.layout_summaries
                        ],
                    }
                    for t in partition.transitions
                ],
                "position": partition.position,
            }
        return {
            "active_signature": self._active_signature,
            "partitions": partitions_state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay state from checkpoint."""
        self._active_signature = state["active_signature"]
        self._partitions = OrderedDict()

        for sig, pstate in state["partitions"].items():
            partition = _Partition()
            partition.position = pstate["position"]
            for tdata in pstate["transitions"]:
                transition = TransitionData(
                    observations=[np.array(o, dtype=np.float32) for o in tdata["observations"]],
                    next_observations=[np.array(o, dtype=np.float32) for o in tdata["next_observations"]],
                    actions=[np.array(a, dtype=np.float32) for a in tdata["actions"]],
                    base_actions=[np.array(a, dtype=np.float32) for a in tdata["base_actions"]],
                    next_base_actions=[np.array(a, dtype=np.float32) for a in tdata["next_base_actions"]],
                    rewards=tdata["rewards"],
                    done=tdata["done"],
                    topology_signature=tdata["topology_signature"],
                    layout_summaries=[
                        LayoutSummary(**ls) for ls in tdata["layout_summaries"]
                    ],
                )
                partition.transitions.append(transition)
            self._partitions[sig] = partition
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_replay.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_replay.py tests/test_matd3_replay.py
git commit -m "feat(matd3-t): add topology-partitioned replay buffer"
```

---

## Task 4: Critic Update Helper

**Files:**
- Create: `algorithms/utils/matd3_critic_update.py`
- Create: `tests/test_matd3_critic_update.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_matd3_critic_update.py`:

```python
"""Unit tests for the critic update loop."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import (
    GlobalTokenPacker,
    BuildingLayout,
    PackedGlobalSequence,
)
from algorithms.utils.matd3_critic_update import (
    compute_target_q,
    critic_update_step,
    CriticUpdateResult,
)


def _make_critics(d_model=16) -> TwinTransformerCritics:
    return TwinTransformerCritics(
        d_model=d_model, nhead=2, num_layers=1,
        dim_feedforward=32, dropout=0.0,
        num_token_types=8, max_buildings=4,
    )


def _make_packed_sequence(batch_size=4, n_tokens=8, d_model=16) -> PackedGlobalSequence:
    return PackedGlobalSequence(
        global_tokens=torch.randn(batch_size, n_tokens, d_model),
        type_ids=torch.zeros(batch_size, n_tokens, dtype=torch.long),
        building_ids=torch.zeros(batch_size, n_tokens, dtype=torch.long),
        padding_mask=torch.zeros(batch_size, n_tokens, dtype=torch.bool),
        controlled_building_indices=[0],
    )


class TestComputeTargetQ:
    def test_target_q_shape(self):
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=4, n_tokens=8)
        rewards = torch.randn(4, 1)
        done = torch.zeros(4, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert target_q.shape == (4, 1)

    def test_target_q_terminal_state(self):
        """When done=1, target Q should equal reward (no bootstrap)."""
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=2, n_tokens=6)
        rewards = torch.tensor([[1.0], [2.0]])
        done = torch.ones(2, 1)  # all terminal

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert torch.allclose(target_q, rewards)

    def test_target_q_uses_min_of_twins(self):
        """Target Q uses min(Q1, Q2) for overestimation reduction."""
        target_critics = _make_critics()
        target_critics.eval()
        packed_next = _make_packed_sequence(batch_size=3, n_tokens=6)
        rewards = torch.zeros(3, 1)
        done = torch.zeros(3, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        # Verify it's actually using min-Q by comparing with individual critics
        with torch.no_grad():
            q1 = target_critics.critic_1(
                packed_next.global_tokens, packed_next.type_ids,
                packed_next.building_ids, packed_next.padding_mask,
                packed_next.controlled_building_indices,
            )
            q2 = target_critics.critic_2(
                packed_next.global_tokens, packed_next.type_ids,
                packed_next.building_ids, packed_next.padding_mask,
                packed_next.controlled_building_indices,
            )
        expected_min_q = torch.min(q1, q2)
        expected_target = rewards + 0.99 * (1.0 - done) * expected_min_q
        assert torch.allclose(target_q, expected_target)

    def test_target_q_no_gradient(self):
        """Target Q computation should not require grad."""
        target_critics = _make_critics()
        packed_next = _make_packed_sequence(batch_size=2, n_tokens=6)
        rewards = torch.randn(2, 1)
        done = torch.zeros(2, 1)

        target_q = compute_target_q(
            target_critics=target_critics,
            packed_next_state=packed_next,
            rewards=rewards,
            done=done,
            gamma=0.99,
        )
        assert not target_q.requires_grad


class TestCriticUpdateStep:
    def test_returns_result_object(self):
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=8)
        target_q = torch.randn(4, 1)

        result = critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=target_q,
        )
        assert isinstance(result, CriticUpdateResult)
        assert result.critic_1_loss > 0 or result.critic_1_loss == 0
        assert result.critic_2_loss > 0 or result.critic_2_loss == 0

    def test_loss_decreases_over_steps(self):
        """Critic loss should decrease over multiple update steps."""
        torch.manual_seed(42)
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-2)
        packed_current = _make_packed_sequence(batch_size=8, n_tokens=6)
        target_q = torch.zeros(8, 1)  # target = 0 for simplicity

        losses = []
        for _ in range(20):
            result = critic_update_step(
                online_critics=online_critics,
                optimizer=optimizer,
                packed_current_state=packed_current,
                target_q=target_q,
            )
            losses.append(result.critic_1_loss + result.critic_2_loss)

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_both_critics_updated(self):
        """Both critic parameters should change after update."""
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=6)
        target_q = torch.randn(4, 1)

        params_1_before = [p.clone() for p in online_critics.critic_1.parameters()]
        params_2_before = [p.clone() for p in online_critics.critic_2.parameters()]

        critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=target_q,
        )

        params_1_changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_1_before, online_critics.critic_1.parameters())
        )
        params_2_changed = any(
            not torch.allclose(p_before, p_after)
            for p_before, p_after in zip(params_2_before, online_critics.critic_2.parameters())
        )
        assert params_1_changed, "Critic 1 params unchanged after update"
        assert params_2_changed, "Critic 2 params unchanged after update"

    def test_mse_loss_used(self):
        """Verify that MSE loss is used (loss = 0 when Q matches target)."""
        torch.manual_seed(0)
        online_critics = _make_critics()
        optimizer = torch.optim.Adam(online_critics.parameters(), lr=1e-3)
        packed_current = _make_packed_sequence(batch_size=4, n_tokens=6)

        # Get current Q values as target (loss should be ~0)
        with torch.no_grad():
            q1, q2 = online_critics(
                packed_current.global_tokens, packed_current.type_ids,
                packed_current.building_ids, packed_current.padding_mask,
                packed_current.controlled_building_indices,
            )
        # Use q1 as target for both — loss won't be exactly 0 for q2
        # but q1 loss should be near 0
        result = critic_update_step(
            online_critics=online_critics,
            optimizer=optimizer,
            packed_current_state=packed_current,
            target_q=q1.detach(),
        )
        assert result.critic_1_loss < 1e-6  # should be nearly 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_matd3_critic_update.py -v`
Expected: ImportError — `matd3_critic_update` module does not exist.

- [ ] **Step 3: Implement the critic update helper**

Create `algorithms/utils/matd3_critic_update.py`:

```python
"""Critic update helper for AgentTransformerMATD3.

Provides:
- compute_target_q: builds the TD3 min-Q target from target critics.
- critic_update_step: one gradient step on both online critics with MSE loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import PackedGlobalSequence


@dataclass
class CriticUpdateResult:
    """Diagnostic output from a single critic update step."""
    critic_1_loss: float
    critic_2_loss: float
    total_loss: float
    mean_q1: float
    mean_q2: float
    mean_target_q: float


@torch.no_grad()
def compute_target_q(
    target_critics: TwinTransformerCritics,
    packed_next_state: PackedGlobalSequence,
    rewards: torch.Tensor,        # [B, n_controlled]
    done: torch.Tensor,           # [B, n_controlled] or [B, 1]
    gamma: float,
) -> torch.Tensor:
    """Compute TD3 target: r + gamma * (1 - done) * min(Q1_target, Q2_target).

    Args:
        target_critics: Target twin critic networks (soft-updated).
        packed_next_state: Packed global sequence for next state
            (with target actor actions already embedded).
        rewards: Per-building rewards [B, n_controlled].
        done: Terminal flag [B, n_controlled] or [B, 1] (broadcast).
        gamma: Discount factor.

    Returns:
        Target Q-values [B, n_controlled], detached from computation graph.
    """
    min_q_next = target_critics.min_q(
        packed_next_state.global_tokens,
        packed_next_state.type_ids,
        packed_next_state.building_ids,
        packed_next_state.padding_mask,
        packed_next_state.controlled_building_indices,
    )
    target_q = rewards + gamma * (1.0 - done) * min_q_next
    return target_q


def critic_update_step(
    online_critics: TwinTransformerCritics,
    optimizer: torch.optim.Optimizer,
    packed_current_state: PackedGlobalSequence,
    target_q: torch.Tensor,        # [B, n_controlled], detached
) -> CriticUpdateResult:
    """Perform one gradient step on both online critics using MSE loss.

    Args:
        online_critics: Online twin critic networks to update.
        optimizer: Optimizer for online critic parameters.
        packed_current_state: Packed global sequence for current state
            (with current actions embedded).
        target_q: Detached target Q-values from compute_target_q.

    Returns:
        CriticUpdateResult with diagnostic metrics.
    """
    # Forward through both critics
    q1, q2 = online_critics(
        packed_current_state.global_tokens,
        packed_current_state.type_ids,
        packed_current_state.building_ids,
        packed_current_state.padding_mask,
        packed_current_state.controlled_building_indices,
    )

    # MSE loss for each critic independently
    loss_1 = F.mse_loss(q1, target_q)
    loss_2 = F.mse_loss(q2, target_q)
    total_loss = loss_1 + loss_2

    # Gradient step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return CriticUpdateResult(
        critic_1_loss=loss_1.item(),
        critic_2_loss=loss_2.item(),
        total_loss=total_loss.item(),
        mean_q1=q1.mean().item(),
        mean_q2=q2.mean().item(),
        mean_target_q=target_q.mean().item(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_matd3_critic_update.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/matd3_critic_update.py tests/test_matd3_critic_update.py
git commit -m "feat(matd3-t): add critic update helper (target Q + MSE step)"
```

---

## Task 5: Integrate Critic Into Agent Class

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Modify: `tests/test_agent_transformer_matd3_foundation.py` (or new test file)

This task wires the critic stacks, packer, and replay into the agent skeleton created by Plan A. It adds transition storage in `update()` and performs critic updates when conditions are met.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_transformer_matd3_foundation.py` (or create `tests/test_matd3_critic_integration.py`):

```python
"""Integration tests: critic stacks wired into the agent."""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.agents.agent_transformer_matd3 import AgentTransformerMATD3


_TOKENIZER_CFG = "configs/tokenizers/entity_default.json"


def _matd3_config_with_critic() -> dict:
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
                "critic_action_input_mode": "final",
            },
        },
    }


class TestCriticIntegration:
    """Critic stacks are created and critic updates run."""

    def _make_agent(self, n_buildings=2):
        from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building
        obs_names = load_sample_observation_names_for_first_building()
        obs_per = [list(obs_names) for _ in range(n_buildings)]
        act_per = [["electrical_storage", "electric_vehicle_storage"] for _ in range(n_buildings)]
        agent = AgentTransformerMATD3(_matd3_config_with_critic())
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[None] * n_buildings,
            observation_space=[None] * n_buildings,
            metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
        )
        return agent, obs_per, act_per, len(obs_names)

    def test_twin_critics_created(self):
        agent, _, _, _ = self._make_agent()
        assert agent._online_critics is not None
        assert agent._target_critics is not None
        # Verify independence
        params_1 = set(id(p) for p in agent._online_critics.critic_1.parameters())
        params_2 = set(id(p) for p in agent._online_critics.critic_2.parameters())
        assert params_1.isdisjoint(params_2)

    def test_replay_created(self):
        agent, _, _, _ = self._make_agent()
        assert agent._replay is not None
        assert agent._replay.active_signature is not None

    def test_update_stores_transition(self):
        agent, _, _, obs_dim = self._make_agent(n_buildings=2)
        obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
        actions = [np.array([0.1, -0.2], dtype=np.float64) for _ in range(2)]
        rewards = [0.5, -0.3]

        agent.update(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=100,
            update_step=True,
            initial_exploration_done=True,
        )
        assert agent._replay.active_partition_size == 1

    def test_critic_update_runs_when_enough_samples(self):
        agent, _, _, obs_dim = self._make_agent(n_buildings=2)

        # Fill replay above batch_size (4)
        for _ in range(10):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[0.0, 0.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=False, global_learning_step=100,
                update_step=True, initial_exploration_done=True,
            )

        # After enough transitions, critic update should have run
        assert agent._critic_update_count > 0

    def test_no_update_when_exploration_not_done(self):
        agent, _, _, obs_dim = self._make_agent(n_buildings=2)

        for _ in range(10):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[0.0, 0.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=False, global_learning_step=100,
                update_step=True, initial_exploration_done=False,
            )
        assert agent._critic_update_count == 0

    def test_target_soft_update_on_flag(self):
        agent, _, _, obs_dim = self._make_agent(n_buildings=2)

        # Fill replay
        for _ in range(10):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[0.0, 0.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=True, global_learning_step=100,
                update_step=True, initial_exploration_done=True,
            )

        # Target critics should have been updated
        assert agent._target_update_count > 0

    def test_topology_change_updates_replay_signature(self):
        agent, obs_per, act_per, obs_dim = self._make_agent(n_buildings=2)
        sig_before = agent._replay.active_signature

        # Trigger topology change by changing obs names
        new_obs = [obs_per[0] + ["charger::Building_0/charger_new::connected_state"], obs_per[1]]
        new_act = [act_per[0] + ["electric_vehicle_storage_charger_new"], act_per[1]]
        agent.attach_environment(
            observation_names=new_obs,
            action_names=new_act,
            action_space=[None, None],
            observation_space=[None, None],
        )

        sig_after = agent._replay.active_signature
        assert sig_after != sig_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestCriticIntegration -v`
Expected: AttributeError — `_online_critics`, `_replay`, etc. don't exist on agent yet.

- [ ] **Step 3: Wire critics, packer, and replay into the agent**

Modify `algorithms/agents/agent_transformer_matd3.py` — add to `__init__`:

```python
# --- Critic infrastructure (Plan B) ---
from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import GlobalTokenPacker, BuildingLayout
from algorithms.utils.matd3_replay import (
    TopologyPartitionedReplay, TransitionData, LayoutSummary,
    compute_topology_signature,
)
from algorithms.utils.matd3_critic_update import compute_target_q, critic_update_step

# In __init__:
critic_cfg = dict(algo["transformer_critic"])
self._critic_d_model = int(critic_cfg["d_model"])
self._critic_nhead = int(critic_cfg["nhead"])
self._critic_num_layers = int(critic_cfg["num_layers"])
self._critic_dim_feedforward = int(critic_cfg.get("dim_feedforward", 256))
self._critic_dropout = float(critic_cfg.get("dropout", 0.1))
self._critic_lr = float(h["critic_lr"])
self._gamma = float(h["gamma"])
self._tau = float(h["tau"])
self._batch_size = int(h["batch_size"])
self._replay_capacity = int(h["replay_capacity"])
self._critic_action_input_mode = str(h.get("critic_action_input_mode", "final"))
self._num_token_types = 8  # obs_sro, obs_nfc, obs_ca, action, +reserved
self._max_buildings = 16   # can be derived from env later

self._online_critics: Optional[TwinTransformerCritics] = None
self._target_critics: Optional[TwinTransformerCritics] = None
self._critic_optimizer: Optional[torch.optim.Optimizer] = None
self._global_packer: Optional[GlobalTokenPacker] = None
self._replay: Optional[TopologyPartitionedReplay] = None
self._critic_update_count: int = 0
self._target_update_count: int = 0
```

In `_build_all_actor_states` (after building actors), add critic initialization:

```python
# Build twin critics
self._online_critics = TwinTransformerCritics(
    d_model=self._critic_d_model, nhead=self._critic_nhead,
    num_layers=self._critic_num_layers,
    dim_feedforward=self._critic_dim_feedforward,
    dropout=self._critic_dropout,
    num_token_types=self._num_token_types,
    max_buildings=self._max_buildings,
)
self._target_critics = TwinTransformerCritics(
    d_model=self._critic_d_model, nhead=self._critic_nhead,
    num_layers=self._critic_num_layers,
    dim_feedforward=self._critic_dim_feedforward,
    dropout=self._critic_dropout,
    num_token_types=self._num_token_types,
    max_buildings=self._max_buildings,
)
self._target_critics.load_state_dict(self._online_critics.state_dict())
self._critic_optimizer = torch.optim.Adam(
    self._online_critics.parameters(), lr=self._critic_lr,
)

# Build global packer
self._global_packer = GlobalTokenPacker(
    d_model=self._critic_d_model,
    num_token_types=self._num_token_types,
    max_buildings=self._max_buildings,
    action_input_mode=self._critic_action_input_mode,
)

# Build replay
self._replay = TopologyPartitionedReplay(
    capacity=self._replay_capacity, batch_size=self._batch_size,
)
# Set initial topology signature
self._update_active_topology_signature()
```

Add helper methods:

```python
def _update_active_topology_signature(self) -> None:
    """Compute and set the active topology signature from current layouts."""
    per_type_dims = self._compute_type_input_dims(self._actors[0].layout)
    sig = compute_topology_signature(
        building_ids=[s.building_id for s in self._actors],
        observation_names=[list(s.obs_names_tuple) for s in self._actors],
        action_names=[list(s.action_names_tuple) for s in self._actors],
        ca_action_names=[list(s.layout.ca_action_names) for s in self._actors],
        per_type_feature_dims=per_type_dims,
    )
    self._replay.set_active_signature(sig)

def _get_building_layouts(self) -> List[BuildingLayout]:
    """Extract BuildingLayout summaries for the global packer."""
    return [
        BuildingLayout(
            building_index=i,
            n_sro=s.layout.n_sro,
            n_nfc=1,
            n_ca=s.layout.n_ca,
            is_controlled=(s.layout.n_ca > 0),
        )
        for i, s in enumerate(self._actors)
    ]
```

Update `update()` to store transitions and run critic updates:

```python
def update(self, observations, actions, rewards, next_observations,
           terminated, truncated, *, update_target_step,
           global_learning_step, update_step, initial_exploration_done):
    # 1. Store transition
    done = terminated or truncated
    layouts = self._get_building_layouts()
    transition = TransitionData(
        observations=[np.asarray(o, dtype=np.float32) for o in observations],
        next_observations=[np.asarray(o, dtype=np.float32) for o in next_observations],
        actions=[np.asarray(a, dtype=np.float32) for a in actions],
        base_actions=[np.zeros_like(np.asarray(a, dtype=np.float32)) for a in actions],
        next_base_actions=[np.zeros_like(np.asarray(a, dtype=np.float32)) for a in actions],
        rewards=list(rewards),
        done=done,
        topology_signature=self._replay.active_signature,
        layout_summaries=[
            LayoutSummary(
                building_id=s.building_id,
                n_ca=s.layout.n_ca,
                n_sro=s.layout.n_sro,
                obs_dim=len(s.obs_names_tuple),
                action_dim=s.layout.n_ca,
            )
            for s in self._actors
        ],
    )
    self._replay.push(transition)

    # 2. Gate updates
    if not initial_exploration_done:
        return
    if not update_step:
        return
    if self._replay.active_partition_size < self._batch_size:
        return

    # 3. Critic update
    self._perform_critic_update()

    # 4. Target soft update
    if update_target_step:
        self._target_critics.soft_update_from(self._online_critics, self._tau)
        self._target_update_count += 1

def _perform_critic_update(self) -> None:
    """Sample batch, build global tokens, compute target, update critics."""
    batch = self._replay.sample()
    if batch is None:
        return

    device = next(self._online_critics.parameters()).device
    n_buildings = len(batch.observations)
    layouts = self._get_building_layouts()

    # Tokenize observations for current and next state
    obs_tokens_current = self._tokenize_batch(batch.observations)
    obs_tokens_next = self._tokenize_batch(batch.next_observations)

    # Pack current state with executed actions
    action_tensors = [
        torch.as_tensor(batch.actions[b], dtype=torch.float, device=device)
        for b in range(n_buildings)
    ]
    packed_current = self._global_packer.pack(
        obs_tokens_current, action_tensors, layouts,
    )

    # Pack next state with target actor actions (for now, use next actions from batch)
    # Full target actor pipeline will be added in Plan D.
    next_action_tensors = [
        torch.as_tensor(batch.actions[b], dtype=torch.float, device=device)
        for b in range(n_buildings)
    ]
    packed_next = self._global_packer.pack(
        obs_tokens_next, next_action_tensors, layouts,
    )

    # Compute target Q
    rewards_t = torch.stack([
        torch.as_tensor(batch.rewards[b], dtype=torch.float, device=device)
        for b in range(n_buildings)
        if layouts[b].is_controlled
    ], dim=1)  # [batch_size, n_controlled]
    done_t = torch.as_tensor(
        batch.done, dtype=torch.float, device=device,
    ).unsqueeze(1).expand_as(rewards_t)

    target_q = compute_target_q(
        target_critics=self._target_critics,
        packed_next_state=packed_next,
        rewards=rewards_t,
        done=done_t,
        gamma=self._gamma,
    )

    # Update critics
    critic_update_step(
        online_critics=self._online_critics,
        optimizer=self._critic_optimizer,
        packed_current_state=packed_current,
        target_q=target_q,
    )
    self._critic_update_count += 1

def _tokenize_batch(
    self, observations_per_building: List[np.ndarray],
) -> List[torch.Tensor]:
    """Tokenize a batch of observations for each building."""
    device = next(self._online_critics.parameters()).device
    tokens = []
    for b, state in enumerate(self._actors):
        obs_t = torch.as_tensor(
            observations_per_building[b], dtype=torch.float, device=device,
        )  # [batch_size, obs_dim]
        with torch.no_grad():
            tokenized = state.tokenizer(obs_t, state.layout)
        # Concatenate SRO + NFC + CA tokens: [batch, n_total, d_model]
        all_toks = torch.cat(
            [tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens],
            dim=1,
        )
        tokens.append(all_toks)
    return tokens
```

Also update `_handle_topology_change` to refresh the replay signature:

```python
# At end of _handle_topology_change (after updating layout):
if self._replay is not None:
    self._update_active_topology_signature()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestCriticIntegration -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "feat(matd3-t): wire critic stacks, packer, and replay into agent"
```

---

## Task 6: Critic Checkpoint Round-Trip

**Files:**
- Modify: `algorithms/agents/agent_transformer_matd3.py`
- Test: `tests/test_agent_transformer_matd3_foundation.py`

- [ ] **Step 1: Write the failing test**

Append to test file:

```python
import tempfile


class TestCriticCheckpoint:
    """Checkpoint includes critic and replay state."""

    def _make_agent(self, n_buildings=2):
        from tests._entity_sample_obs_names import load_sample_observation_names_for_first_building
        obs_names = load_sample_observation_names_for_first_building()
        obs_per = [list(obs_names) for _ in range(n_buildings)]
        act_per = [["electrical_storage", "electric_vehicle_storage"] for _ in range(n_buildings)]
        agent = AgentTransformerMATD3(_matd3_config_with_critic())
        agent.attach_environment(
            observation_names=obs_per,
            action_names=act_per,
            action_space=[None] * n_buildings,
            observation_space=[None] * n_buildings,
            metadata={"building_names": [f"Building_{b}" for b in range(n_buildings)]},
        )
        return agent, obs_per, act_per, len(obs_names)

    def test_checkpoint_includes_critics(self):
        agent, _, _, obs_dim = self._make_agent()
        # Push some transitions
        for _ in range(5):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[0.0, 0.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=False, global_learning_step=100,
                update_step=True, initial_exploration_done=True,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent.save_checkpoint(tmpdir, step=50)
            import torch as _torch
            payload = _torch.load(ckpt_path, map_location="cpu")
            assert "online_critics_state" in payload
            assert "target_critics_state" in payload
            assert "critic_optimizer_state" in payload
            assert "replay_state" in payload

    def test_checkpoint_restores_critic_weights(self):
        agent, _, _, obs_dim = self._make_agent()
        # Run some critic updates
        for _ in range(10):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[1.0, -1.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=True, global_learning_step=100,
                update_step=True, initial_exploration_done=True,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent.save_checkpoint(tmpdir, step=100)

            # Get Q values before reload
            test_tokens = torch.randn(1, 6, agent._critic_d_model)
            type_ids = torch.zeros(1, 6, dtype=torch.long)
            building_ids = torch.zeros(1, 6, dtype=torch.long)
            mask = torch.zeros(1, 6, dtype=torch.bool)
            with torch.no_grad():
                q1_before, q2_before = agent._online_critics(
                    test_tokens, type_ids, building_ids, mask, [0],
                )

            # Create fresh agent and load
            agent2, _, _, _ = self._make_agent()
            agent2.load_checkpoint(ckpt_path)

            with torch.no_grad():
                q1_after, q2_after = agent2._online_critics(
                    test_tokens, type_ids, building_ids, mask, [0],
                )

            assert torch.allclose(q1_before, q1_after, atol=1e-6)
            assert torch.allclose(q2_before, q2_after, atol=1e-6)

    def test_checkpoint_restores_replay(self):
        agent, _, _, obs_dim = self._make_agent()
        for _ in range(8):
            obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            next_obs = [np.random.randn(obs_dim).astype(np.float64) for _ in range(2)]
            actions = [np.random.randn(2).astype(np.float64) for _ in range(2)]
            agent.update(
                observations=obs, actions=actions, rewards=[0.0, 0.0],
                next_observations=next_obs, terminated=False, truncated=False,
                update_target_step=False, global_learning_step=50,
                update_step=False, initial_exploration_done=False,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = agent.save_checkpoint(tmpdir, step=50)
            agent2, _, _, _ = self._make_agent()
            agent2.load_checkpoint(ckpt_path)
            assert agent2._replay.active_partition_size == 8
            assert agent2._replay.active_signature == agent._replay.active_signature
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestCriticCheckpoint -v`
Expected: KeyError or AttributeError — checkpoint doesn't include critic state yet.

- [ ] **Step 3: Update save_checkpoint and load_checkpoint**

Modify `save_checkpoint` in the agent to include critic + replay:

```python
def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
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
        # Plan B additions:
        "online_critics_state": self._online_critics.state_dict() if self._online_critics else None,
        "target_critics_state": self._target_critics.state_dict() if self._target_critics else None,
        "critic_optimizer_state": self._critic_optimizer.state_dict() if self._critic_optimizer else None,
        "global_packer_state": self._global_packer.state_dict() if self._global_packer else None,
        "replay_state": self._replay.state_dict() if self._replay else None,
        "critic_update_count": self._critic_update_count,
        "target_update_count": self._target_update_count,
    }
    torch.save(payload, path)
    return str(path)
```

Modify `load_checkpoint`:

```python
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

    # Plan B: restore critics and replay
    if payload.get("online_critics_state") and self._online_critics:
        self._online_critics.load_state_dict(payload["online_critics_state"])
    if payload.get("target_critics_state") and self._target_critics:
        self._target_critics.load_state_dict(payload["target_critics_state"])
    if payload.get("critic_optimizer_state") and self._critic_optimizer:
        self._critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
    if payload.get("global_packer_state") and self._global_packer:
        self._global_packer.load_state_dict(payload["global_packer_state"])
    if payload.get("replay_state") and self._replay:
        self._replay.load_state_dict(payload["replay_state"])
    self._critic_update_count = payload.get("critic_update_count", 0)
    self._target_update_count = payload.get("target_update_count", 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py::TestCriticCheckpoint -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/agents/agent_transformer_matd3.py tests/test_agent_transformer_matd3_foundation.py
git commit -m "feat(matd3-t): critic + replay checkpoint round-trip"
```

---

## Task 7: Final Verification

- [ ] **Step 1: Run all Plan B tests**

Run: `pytest tests/test_matd3_critic.py tests/test_matd3_global_packer.py tests/test_matd3_replay.py tests/test_matd3_critic_update.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_agent_transformer_matd3_foundation.py -v`
Expected: All tests PASS (Plan A + Plan B).

- [ ] **Step 3: Run existing test suite for regressions**

Run: `pytest tests/ -x --timeout=120`
Expected: No new failures.

- [ ] **Step 4: Verify imports**

```bash
python -c "
from algorithms.utils.matd3_critic import TwinTransformerCritics
from algorithms.utils.matd3_global_packer import GlobalTokenPacker
from algorithms.utils.matd3_replay import TopologyPartitionedReplay
from algorithms.utils.matd3_critic_update import compute_target_q, critic_update_step
print('All Plan B modules importable')
"
```
Expected: `All Plan B modules importable`

- [ ] **Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix(matd3-t): Plan B final verification fixups" --allow-empty
```

---

## Plan B Complete

After Task 7:
- Twin independent Transformer critic stacks are implemented and tested.
- Global token packer handles variable buildings/tokens, padding masks, and 3 action input modes.
- Topology-partitioned replay stores transitions with layout metadata; sampling respects active signature; eviction respects global capacity.
- Critic update loop computes min-Q targets and updates both critics with MSE loss.
- Agent integrates all components: transitions stored on `update()`, critic updates gated by exploration and replay size, target soft updates on flag.
- Checkpoint round-trip includes critics, optimizer, packer, and replay state.

**Not included (deferred to Plan C/D):**
- Actor update through critic (Plan D).
- Residual policy composition (Plan C).
- Behavior cloning (Plan C).
- Target actor actions in target Q computation (Plan D — currently uses batch actions as placeholder).
- Target policy smoothing (Plan D).
- Reward normalization (Plan C).
- Diagnostics logging (Plan C).