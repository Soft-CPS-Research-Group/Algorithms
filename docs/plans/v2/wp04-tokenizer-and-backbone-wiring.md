# WP04 — `EntityObservationTokenizer` + Backbone Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **For all v2 WPs:** every production-code task MUST follow `superpowers:test-driven-development`. The WP MUST end with `superpowers:requesting-code-review`.

**Goal:** Implement `EntityObservationTokenizer` (`algorithms/utils/entity_observation_tokenizer.py`) per spec §8, and update the ported `TransformerBackbone` (`algorithms/utils/transformer_backbone.py`) to (a) use a 3-entry type embedding table with semantics `SRO=0, NFC=1, CA=2`, and (b) accept `forward(sros, nfc, cas)` returning the correct CA slice. Also add a small integration test that runs the tokenizer → backbone → ported PPO components end-to-end on a single-batch tensor (no agent yet).

**Architecture:** Per-building tokenizer is a single `nn.Module` with a `nn.ModuleDict` of `nn.Linear` projections keyed by **type name** (`storage`, `charger`, `pv`, `ev_connected`, `ev_incoming`, `district_*`, `building_*`, plus `building_nfc`). One projection per type, regardless of how many instances are present in the layout (zero new parameters when topology grows — spec §8.4). Slicing uses `index_select` against pre-registered `LongTensor` buffers per segment, so non-contiguous feature indices (interleaved district forecasts) work correctly. NFC pre-projection scalar reduction is the only non-`Linear`-only special case.

**Tech Stack:** Python 3.11, PyTorch (already in repo), pytest. Consumes `BuildingTokenLayout` from WP03 and the parsed `EntityTokenizerConfig` from WP02.

**Branch:** `gj/wp04-tokenizer-backbone`
**Base branch:** `gj/wp03-layout-builder`

---

## Scope

**Files created:**

- `algorithms/utils/entity_observation_tokenizer.py` — `TokenizedObservation` dataclass + `EntityObservationTokenizer` (`nn.Module`).
- `tests/test_entity_observation_tokenizer.py` — covers spec §16.2.

**Files modified:**

- `algorithms/utils/transformer_backbone.py` — change type-embedding semantics and `forward` signature per spec §9.
- `tests/test_transformer_backbone.py` — replace v1 marker-era tests with the §16.3 test rows. **The v1 tests ported in WP01 will partially break here; that is the intended TDD red signal.**

**Out of scope:**

- The agent (WP05).
- The wrapper hook (WP05).
- Adapter changes.

---

## File Structure

```
algorithms/utils/
  entity_observation_tokenizer.py    # NEW (~200 lines)
  transformer_backbone.py             # MODIFIED (forward signature + type embedding semantics)
tests/
  test_entity_observation_tokenizer.py  # NEW (covers §16.2, ~8 tests)
  test_transformer_backbone.py          # REWRITTEN (covers §16.3, 6 tests)
```

---

## Key Design Decisions

- **Per-type Linear**: `projections = nn.ModuleDict({type_name: nn.Linear(input_dim, d_model)})`. Keys exactly equal the union of `cfg.ca_types.keys() | cfg.sro_types.keys() | {cfg.nfc.type_name}`.
- **`type_input_dims` is supplied by the agent** (WP05), not inferred by the tokenizer. The tokenizer is dumb about where dims come from; this keeps it testable in isolation.
- **`type_input_dims` validation**: at construction, every key declared in the tokenizer config (every CA, SRO, and the NFC type) must appear in `type_input_dims`. Missing → `ValueError`. The NFC entry's value MUST equal `1` (the spec is unambiguous — §8.5).
- **Index buffers**: registered with `register_buffer(name, tensor, persistent=False)` so the buffer follows the module to GPU but is not saved in checkpoints (the layout is rebuilt from config + observation_names, not deserialized).
- **Index buffers are layout-dependent, not module-state**: each `forward()` rebuilds them from the supplied `layout` (cheaply — `torch.tensor(seg.feature_indices, …, device=encoded_obs.device)`). This is fine because:
  - the tokenizer is constructed once per building per topology-version;
  - layouts are cached in WP03;
  - `forward()` is on the hot path, but `torch.tensor(small_list, device=…)` is microseconds and amortised by batched PPO updates.

  An optimization (pre-register buffers indexed by `(type_name, instance_id)`) is reserved for a future tuning WP if profiling shows hot.
- **NFC pre-projection**: the tokenizer reads `seg.derived` and computes `lhs - rhs`, unsqueezes to `[batch, 1]`, projects via `projections["building_nfc"]` → `[batch, d_model]`, then unsqueezes again to `[batch, 1, d_model]`.
- **Backbone changes**: minimal — modify only what spec §9 requires:
  1. `nn.Embedding(3, d_model)` with documented constants `_TYPE_SRO=0, _TYPE_NFC=1, _TYPE_CA=2`.
  2. `forward(sros: Tensor, nfc: Tensor, cas: Tensor) -> Tuple[ca_embeddings, pooled]`.
  3. Concatenation order `[sros, nfc, cas]`; CA slice = `concat[:, N_sro+1 : N_sro+1+N_ca, :]`; mean pool across all tokens for critic.

---

## Tasks

### Task 1: Branch + verify prerequisites

- [ ] **Step 1: Create branch from WP03**

```bash
git checkout gj/wp03-layout-builder
git checkout -b gj/wp04-tokenizer-backbone
```

- [ ] **Step 2: Verify WP03 deliverables**

```bash
test -f algorithms/utils/entity_token_layout.py && python -c "from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder, BuildingTokenLayout, TokenSegment, NfcExpression; print('OK')"
```

- [ ] **Step 3: Verify ports from WP01 still present**

```bash
test -f algorithms/utils/transformer_backbone.py && test -f algorithms/utils/ppo_components.py && echo "OK"
```

---

### Task 2: Update `TransformerBackbone` — type embedding table size + semantics

This task modifies the ported v1 backbone. The v1 backbone's tests will break first (RED), then we re-author the backbone (GREEN), then re-author the tests to match §16.3 (replacing v1 marker-era assertions).

**Files:**
- Modify: `algorithms/utils/transformer_backbone.py`
- Rewrite: `tests/test_transformer_backbone.py`

- [ ] **Step 1: Inspect the ported backbone**

```bash
wc -l algorithms/utils/transformer_backbone.py
grep -n "nn.Embedding\|def forward\|TYPE_\|num_embeddings" algorithms/utils/transformer_backbone.py
```

Identify:
  - The line where `nn.Embedding(<n>, d_model)` is constructed.
  - The `forward(...)` signature.
  - Any v1 type constants (e.g. `TYPE_CTX`, `TYPE_FA`).

- [ ] **Step 2: Replace `tests/test_transformer_backbone.py` with the §16.3 catalog**

The v1 test file may have completely different assertions. We discard it and write a fresh test module aligned to §16.3:

```python
# tests/test_transformer_backbone.py
"""Tests for TransformerBackbone (entity-mode contract). Covers spec §16.3."""
from __future__ import annotations

import pytest
import torch

D_MODEL = 16


@pytest.fixture
def backbone():
    from algorithms.utils.transformer_backbone import TransformerBackbone
    return TransformerBackbone(
        d_model=D_MODEL, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0,
    )


def test_type_embedding_table_size_3(backbone):
    """Spec §9: nn.Embedding(3, d_model) for {SRO=0, NFC=1, CA=2}."""
    # Find the embedding submodule. The exact attribute name comes from the
    # ported file; assert by walking modules.
    embeddings = [
        m for m in backbone.modules() if isinstance(m, torch.nn.Embedding)
    ]
    assert any(e.num_embeddings == 3 and e.embedding_dim == D_MODEL for e in embeddings), (
        f"Expected an nn.Embedding(3, {D_MODEL}); got "
        f"{[(e.num_embeddings, e.embedding_dim) for e in embeddings]}"
    )


def test_concat_order_sros_nfc_cas(backbone):
    """The internal concat order is sros → nfc → cas. We assert it indirectly:
    the returned ca_embeddings must come from the LAST N_ca rows of the
    concatenated sequence (verified by the slice test below)."""
    n_sro, n_ca = 3, 2
    sros = torch.randn(1, n_sro, D_MODEL)
    nfc = torch.randn(1, 1, D_MODEL)
    cas = torch.randn(1, n_ca, D_MODEL)
    ca_emb, _ = backbone(sros, nfc, cas)
    assert ca_emb.shape == (1, n_ca, D_MODEL)


def test_ca_embeddings_sliced_at_correct_offset(backbone):
    """Replace cas with sentinel tensor; the returned ca_embeddings, while
    transformed by self-attention, must depend on cas through gradients only —
    the slice positions are [N_sro + 1 : N_sro + 1 + N_ca]. We verify the
    slice positions by zeroing out the relevant rows AFTER the embedding
    addition (impractical), so instead use a gradient probe:
    backward through ca_emb[0,0] should accumulate gradient on cas[0,0] but
    not on cas[0,1] specifically — but with self-attention all positions
    interact. Use a different approach: monkey-patch the backbone's
    transformer_encoder to identity, then verify slice positions numerically."""
    n_sro, n_ca = 4, 3
    sros = torch.zeros(1, n_sro, D_MODEL)
    nfc = torch.zeros(1, 1, D_MODEL)
    cas = torch.arange(n_ca * D_MODEL, dtype=torch.float).view(1, n_ca, D_MODEL)
    # Replace transformer encoder with identity to bypass attention mixing.
    import torch.nn as nn
    backbone.transformer_encoder = nn.Identity()
    # Disable type embedding addition by zeroing the embedding weight (or
    # patch). Simplest: zero out the embedding weight in-place.
    for m in backbone.modules():
        if isinstance(m, nn.Embedding) and m.num_embeddings == 3:
            with torch.no_grad():
                m.weight.zero_()
    ca_emb, _ = backbone(sros, nfc, cas)
    # With identity attention and zero type embeddings, ca_emb == cas.
    assert torch.allclose(ca_emb, cas), (
        f"CA slice misaligned. Expected {cas}, got {ca_emb}."
    )


def test_pooled_includes_sro_and_nfc_tokens(backbone):
    """Mean pool spans all tokens. Replace encoder with identity, zero type
    embeddings, then verify pooled == mean of [sros, nfc, cas]."""
    import torch.nn as nn
    n_sro, n_ca = 2, 2
    sros = torch.ones(1, n_sro, D_MODEL) * 1.0
    nfc = torch.ones(1, 1, D_MODEL) * 5.0
    cas = torch.ones(1, n_ca, D_MODEL) * 2.0
    backbone.transformer_encoder = nn.Identity()
    for m in backbone.modules():
        if isinstance(m, nn.Embedding) and m.num_embeddings == 3:
            with torch.no_grad():
                m.weight.zero_()
    _, pooled = backbone(sros, nfc, cas)
    expected_mean = (n_sro * 1.0 + 1 * 5.0 + n_ca * 2.0) / (n_sro + 1 + n_ca)
    assert torch.allclose(pooled, torch.full_like(pooled, expected_mean))


def test_gradient_flow_through_sro_tokens(backbone):
    n_sro, n_ca = 2, 1
    sros = torch.randn(1, n_sro, D_MODEL, requires_grad=True)
    nfc = torch.randn(1, 1, D_MODEL, requires_grad=True)
    cas = torch.randn(1, n_ca, D_MODEL, requires_grad=True)
    ca_emb, pooled = backbone(sros, nfc, cas)
    (ca_emb.sum() + pooled.sum()).backward()
    assert sros.grad is not None and sros.grad.abs().sum() > 0


def test_gradient_flow_through_nfc_token(backbone):
    n_sro, n_ca = 2, 1
    sros = torch.randn(1, n_sro, D_MODEL, requires_grad=True)
    nfc = torch.randn(1, 1, D_MODEL, requires_grad=True)
    cas = torch.randn(1, n_ca, D_MODEL, requires_grad=True)
    ca_emb, pooled = backbone(sros, nfc, cas)
    (ca_emb.sum() + pooled.sum()).backward()
    assert nfc.grad is not None and nfc.grad.abs().sum() > 0
```

- [ ] **Step 3: Run new tests — expect all to FAIL** (signature mismatch, embedding-size mismatch, etc.)

```bash
pytest tests/test_transformer_backbone.py -v
```
Expected: every test FAIL with errors like `forward() takes 2 positional arguments but 4 were given`, `expected 3 num_embeddings, got <other>`, etc.

- [ ] **Step 4: Edit the backbone**

Open `algorithms/utils/transformer_backbone.py`. Make the minimal changes:

  1. **Type embedding**: ensure the embedding is `nn.Embedding(3, d_model)`. If the v1 file uses a different constant (e.g. 4), change it to 3. Add module-level constants:
  ```python
  _TYPE_SRO = 0
  _TYPE_NFC = 1
  _TYPE_CA = 2
  ```

  2. **`forward` signature**: replace the v1 signature with:
  ```python
  def forward(
      self,
      sros: torch.Tensor,    # [B, N_sro, d_model]
      nfc: torch.Tensor,     # [B, 1,     d_model]
      cas: torch.Tensor,     # [B, N_ca,  d_model]
  ) -> Tuple[torch.Tensor, torch.Tensor]:
      """Returns (ca_embeddings: [B, N_ca, d_model], pooled: [B, d_model])."""
      n_sro = sros.shape[1]
      n_ca = cas.shape[1]
      # Build type-id sequence: [SRO]*n_sro + [NFC] + [CA]*n_ca
      type_ids = torch.cat([
          torch.full((n_sro,), _TYPE_SRO, dtype=torch.long, device=sros.device),
          torch.full((1,),     _TYPE_NFC, dtype=torch.long, device=sros.device),
          torch.full((n_ca,),  _TYPE_CA,  dtype=torch.long, device=sros.device),
      ])
      type_emb = self.type_embedding(type_ids)             # [N_total, d_model]
      seq = torch.cat([sros, nfc, cas], dim=1)             # [B, N_total, d_model]
      seq = seq + type_emb.unsqueeze(0)                     # broadcast over batch
      out = self.transformer_encoder(seq)                   # [B, N_total, d_model]
      ca_embeddings = out[:, n_sro + 1 : n_sro + 1 + n_ca, :]
      pooled = out.mean(dim=1)                              # mean over all tokens
      return ca_embeddings, pooled
  ```
  Adapt attribute names (`self.type_embedding`, `self.transformer_encoder`) to whatever the ported file uses.

  3. Remove any v1 marker-handling code (token type for "marker", fused MLP for marker tokens, etc.). If unsure whether code is marker-related, search for "marker" / "FA" / "CTX" and inspect.

- [ ] **Step 5: Re-run tests until all 6 PASS**

```bash
pytest tests/test_transformer_backbone.py -v
```
Iterate. Note: `test_ca_embeddings_sliced_at_correct_offset` and `test_pooled_includes_sro_and_nfc_tokens` rely on the test setting `transformer_encoder = nn.Identity()` and zeroing the type embedding weight. If the backbone has a layer-norm INSIDE `forward()` outside the encoder (e.g. final pre-LN), those tests may fail with a tiny numerical offset — adjust the assertion to `torch.allclose(..., atol=1e-5)` if needed, or remove the LN outside the encoder if the spec doesn't require it.

- [ ] **Step 6: Commit**

```bash
git add algorithms/utils/transformer_backbone.py tests/test_transformer_backbone.py
git commit -m "feat(wp04): adapt TransformerBackbone to entity-mode forward(sros, nfc, cas) per spec §9"
```

---

### Task 3: `EntityObservationTokenizer` — first failing test

**Files:**
- Create: `algorithms/utils/entity_observation_tokenizer.py`
- Create: `tests/test_entity_observation_tokenizer.py`

- [ ] **Step 1: Write the failing test (basic forward shapes)**

```python
# tests/test_entity_observation_tokenizer.py
"""Tests for EntityObservationTokenizer. Covers spec §16.2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


D_MODEL = 8


@pytest.fixture
def cfg_and_layout():
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_adapter import EntityContractAdapter

    cfg = load_entity_tokenizer_config("configs/tokenizers/entity_default.json")
    payload = json.loads(
        Path("datasets/tmp_entity_obs_full_step2200_named.json").read_text()
    )
    adapter = EntityContractAdapter()
    obs_per_b, names_per_b = adapter.to_agent_observations(payload)
    builder = EntityTokenLayoutBuilder(cfg)
    layout = builder.build(
        "Building_1", names_per_b[0],
        ["electrical_storage", "electric_vehicle_storage"],
    )
    obs_tensor = torch.as_tensor(obs_per_b[0], dtype=torch.float).unsqueeze(0)
    return cfg, layout, obs_tensor


def _type_input_dims_for_layout(cfg, layout):
    """Compute per-type raw input dims from the layout itself (test stub).
    Mirrors what the agent will compute from entity_specs."""
    dims = {}
    for seg in layout.segments:
        if seg.family == "nfc":
            dims[cfg.nfc.type_name] = 1
        else:
            # Each segment of the same type_name has the same feature count
            # (singleton SROs always; per-asset SROs per spec §8.5).
            dims[seg.type_name] = len(seg.feature_indices)
    return dims


def test_forward_shapes_baseline(cfg_and_layout):
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer

    cfg, layout, obs = cfg_and_layout
    tok = EntityObservationTokenizer(
        tokenizer_config=cfg, d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    out = tok(obs, layout)
    assert out.sro_tokens.shape == (1, layout.n_sro, D_MODEL)
    assert out.nfc_token.shape == (1, 1, D_MODEL)
    assert out.ca_tokens.shape == (1, layout.n_ca, D_MODEL)
    assert len(out.sro_types) == layout.n_sro
    assert len(out.ca_types) == layout.n_ca
    assert out.n_sro == layout.n_sro
    assert out.n_ca == layout.n_ca
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_entity_observation_tokenizer.py::test_forward_shapes_baseline -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement minimal `EntityObservationTokenizer` to pass this test**

Create `algorithms/utils/entity_observation_tokenizer.py`:

```python
"""EntityObservationTokenizer — slice + project encoded per-building observation.

See docs/specv2.md §8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

import torch
from torch import nn

# Local imports — both stdlib-only modules, no risk of cycles.
from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout, NfcExpression, TokenSegment,
)


@dataclass
class TokenizedObservation:
    sro_tokens: torch.Tensor    # [B, N_sro, d_model]
    nfc_token: torch.Tensor     # [B, 1,     d_model]
    ca_tokens: torch.Tensor     # [B, N_ca,  d_model]
    sro_types: List[str]
    ca_types: List[str]
    n_sro: int
    n_ca: int


class EntityObservationTokenizer(nn.Module):
    def __init__(
        self,
        tokenizer_config,
        d_model: int,
        type_input_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        self._cfg = tokenizer_config
        self._d_model = d_model
        # Validate type_input_dims covers every declared type.
        required = set(tokenizer_config.ca_types.keys()) \
            | set(tokenizer_config.sro_types.keys()) \
            | {tokenizer_config.nfc.type_name}
        missing = required - set(type_input_dims)
        if missing:
            raise ValueError(
                f"type_input_dims is missing entries: {sorted(missing)}"
            )
        nfc_name = tokenizer_config.nfc.type_name
        if type_input_dims[nfc_name] != 1:
            raise ValueError(
                f"NFC type {nfc_name!r} input dim must be 1 "
                f"(scalar from NfcExpression), got {type_input_dims[nfc_name]}."
            )
        # Build one Linear per declared type. Per-asset/SRO/CA share weights
        # across instances by construction (one projection key per type_name).
        self.projections = nn.ModuleDict({
            type_name: nn.Linear(int(in_dim), d_model)
            for type_name, in_dim in type_input_dims.items()
        })

    def forward(
        self,
        encoded_obs: torch.Tensor,    # [B, obs_dim]
        layout: BuildingTokenLayout,
    ) -> TokenizedObservation:
        if encoded_obs.dim() != 2:
            raise ValueError(
                f"encoded_obs must be 2-D [B, obs_dim], got shape "
                f"{tuple(encoded_obs.shape)}"
            )
        device = encoded_obs.device
        sro_tokens: List[torch.Tensor] = []
        ca_tokens: List[torch.Tensor] = []
        sro_types: List[str] = []
        ca_types: List[str] = []
        nfc_token: torch.Tensor | None = None

        for seg in layout.segments:
            idx = torch.tensor(
                list(seg.feature_indices), dtype=torch.long, device=device,
            )
            group = encoded_obs.index_select(dim=1, index=idx)  # [B, k]
            if seg.family == "nfc":
                expr: NfcExpression = seg.derived  # type: ignore[assignment]
                if expr.op != "subtract":
                    raise ValueError(f"unsupported NFC op: {expr.op!r}")
                lhs = group[:, expr.left_index_in_segment]
                rhs = group[:, expr.right_index_in_segment]
                scalar = (lhs - rhs).unsqueeze(1)             # [B, 1]
                projected = self.projections[seg.type_name](scalar)  # [B, d_model]
                nfc_token = projected.unsqueeze(1)            # [B, 1, d_model]
            elif seg.family == "sro":
                projected = self.projections[seg.type_name](group)  # [B, d_model]
                sro_tokens.append(projected.unsqueeze(1))           # [B, 1, d_model]
                sro_types.append(seg.type_name)
            elif seg.family == "ca":
                projected = self.projections[seg.type_name](group)
                ca_tokens.append(projected.unsqueeze(1))
                ca_types.append(seg.type_name)
            else:
                raise ValueError(f"unknown segment family: {seg.family!r}")

        assert nfc_token is not None, "layout missing NFC segment"
        sro_stack = (
            torch.cat(sro_tokens, dim=1) if sro_tokens
            else torch.zeros(encoded_obs.shape[0], 0, self._d_model, device=device)
        )
        ca_stack = (
            torch.cat(ca_tokens, dim=1) if ca_tokens
            else torch.zeros(encoded_obs.shape[0], 0, self._d_model, device=device)
        )
        return TokenizedObservation(
            sro_tokens=sro_stack,
            nfc_token=nfc_token,
            ca_tokens=ca_stack,
            sro_types=sro_types,
            ca_types=ca_types,
            n_sro=len(sro_tokens),
            n_ca=len(ca_tokens),
        )
```

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/test_entity_observation_tokenizer.py::test_forward_shapes_baseline -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algorithms/utils/entity_observation_tokenizer.py tests/test_entity_observation_tokenizer.py
git commit -m "feat(wp04): add EntityObservationTokenizer (forward shapes baseline)"
```

---

### Task 4: NFC scalar value test (§16.2 row 5–6)

```python
def test_nfc_token_value_equals_subtract_op(cfg_and_layout):
    """Verify the pre-projection scalar = lhs - rhs."""
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer

    cfg, layout, _ = cfg_and_layout
    tok = EntityObservationTokenizer(
        tokenizer_config=cfg, d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    # Build a synthetic obs vector where non_shiftable_load=5, solar_generation=2
    # and everything else is 0.
    obs_dim = max(
        max(seg.feature_indices) for seg in layout.segments
    ) + 1
    obs = torch.zeros(1, obs_dim)
    nfc_seg = next(s for s in layout.segments if s.family == "nfc")
    obs[0, nfc_seg.feature_indices[0]] = 5.0  # non_shiftable_load
    obs[0, nfc_seg.feature_indices[1]] = 2.0  # solar_generation

    # Replace the NFC projection with identity so we can see the raw scalar
    # propagate to nfc_token (which is [B, 1, d_model]).
    nfc_name = cfg.nfc.type_name
    with torch.no_grad():
        # Set weight = ones, bias = 0; output = scalar * sum(weight) = scalar*d_model
        tok.projections[nfc_name].weight.fill_(1.0)
        tok.projections[nfc_name].bias.fill_(0.0)

    out = tok(obs, layout)
    expected_scalar = 5.0 - 2.0
    # nfc_token shape = [1,1,d_model]; with weight=1, output = scalar * 1 per dim.
    assert torch.allclose(out.nfc_token, torch.full_like(out.nfc_token, expected_scalar))


def test_nfc_input_dim_is_one(cfg_and_layout):
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
    cfg, layout, _ = cfg_and_layout
    tok = EntityObservationTokenizer(
        tokenizer_config=cfg, d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    nfc_name = cfg.nfc.type_name
    assert tok.projections[nfc_name].in_features == 1
```

- [ ] **Step 1: Run, expect PASS**

```bash
pytest tests/test_entity_observation_tokenizer.py -k "nfc_token_value or nfc_input_dim" -v
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_observation_tokenizer.py
git commit -m "test(wp04): cover NFC scalar reduction and projection input dim"
```

---

### Task 5: Variable cardinality + zero-new-params test (§16.2 row 3)

```python
def test_projection_is_per_type_no_new_params_on_topology_grow(cfg_and_layout):
    """Adding a second charger / pv must NOT add new projection parameters."""
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
    from algorithms.utils.entity_token_layout import (
        EntityTokenLayoutBuilder, BuildingTokenLayout, TokenSegment,
    )

    cfg, layout_one_charger, _ = cfg_and_layout
    tok = EntityObservationTokenizer(
        tokenizer_config=cfg, d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout_one_charger),
    )
    params_before = sum(p.numel() for p in tok.parameters())

    # Construct a synthetic layout that adds one extra charger CA segment by
    # cloning the existing charger segment with a different instance_id.
    charger_segs = [s for s in layout_one_charger.segments if s.family == "ca" and s.type_name == "charger"]
    if not charger_segs:
        pytest.skip("Sample layout has no charger CA segment")
    base_charger = charger_segs[0]
    extra = TokenSegment(
        family="ca", type_name="charger",
        instance_id=(base_charger.instance_id or "x") + "_dup",
        feature_indices=base_charger.feature_indices,
        feature_names=base_charger.feature_names,
    )
    # Insert after existing CAs, update n_ca and ca_action_names accordingly.
    new_segments = list(layout_one_charger.segments) + [extra]
    new_layout = BuildingTokenLayout(
        building_id=layout_one_charger.building_id,
        segments=tuple(new_segments),
        n_sro=layout_one_charger.n_sro,
        n_ca=layout_one_charger.n_ca + 1,
        ca_action_names=layout_one_charger.ca_action_names + ("electric_vehicle_storage",),
        excluded_feature_names=layout_one_charger.excluded_feature_names,
    )
    obs_dim = max(max(s.feature_indices) for s in new_layout.segments) + 1
    obs = torch.zeros(1, obs_dim)
    out = tok(obs, new_layout)
    assert out.ca_tokens.shape == (1, layout_one_charger.n_ca + 1, D_MODEL)
    params_after = sum(p.numel() for p in tok.parameters())
    assert params_before == params_after, (
        f"Parameter count changed on topology grow: {params_before} -> {params_after}"
    )
```

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_observation_tokenizer.py::test_projection_is_per_type_no_new_params_on_topology_grow -v
```
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_observation_tokenizer.py
git commit -m "test(wp04): cover variable cardinality with zero-new-parameter guarantee"
```

---

### Task 6: Non-contiguous slicing test (§16.2 row 4)

```python
def test_index_select_handles_non_contiguous_sro_segment(cfg_and_layout):
    """Using a sentinel obs vector where each position holds its own index,
    confirm that the tokenizer reads exactly the indices listed in the segment
    (not a contiguous slice)."""
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer

    cfg, layout, obs = cfg_and_layout
    tok = EntityObservationTokenizer(
        tokenizer_config=cfg, d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    # Find the district_weather_forecast segment, which is non-contiguous in
    # the bundled sample (forecasts are interleaved per horizon).
    fwd = next(
        (s for s in layout.segments if s.type_name == "district_weather_forecast"),
        None,
    )
    if fwd is None or len(fwd.feature_indices) < 2:
        pytest.skip("sample has no non-contiguous forecast segment")
    # Build sentinel: obs[i] = i.0 for every i.
    obs_dim = max(max(s.feature_indices) for s in layout.segments) + 1
    sentinel = torch.arange(obs_dim, dtype=torch.float).unsqueeze(0)
    # Set the projection for "district_weather_forecast" to identity so the
    # output recovers the gathered values.
    proj = tok.projections["district_weather_forecast"]
    in_dim = proj.in_features
    with torch.no_grad():
        proj.weight.zero_()
        # First d_model columns of the input → first d_model output dims via
        # diagonal pattern; for our purposes it's enough to ensure SUM of input
        # equals dot product. Use sum: weight = ones[1, in_dim] then expand.
        proj.weight.copy_(torch.ones(D_MODEL, in_dim))
        proj.bias.zero_()
    out = tok(sentinel, layout)
    # Find the position of district_weather_forecast in sro_tokens.
    pos = out.sro_types.index("district_weather_forecast")
    summed = out.sro_tokens[0, pos, 0].item()
    expected = float(sum(fwd.feature_indices))
    assert summed == pytest.approx(expected), (
        f"Sliced sum mismatch: got {summed}, expected {expected}. "
        f"Indices were {fwd.feature_indices}"
    )
```

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_observation_tokenizer.py::test_index_select_handles_non_contiguous_sro_segment -v
```
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_observation_tokenizer.py
git commit -m "test(wp04): cover non-contiguous index_select gathering"
```

---

### Task 7: Validation tests (§16.2 row 7) and dtype/device propagation (row 8)

```python
def test_input_dim_mismatch_raises(cfg_and_layout):
    """Constructing the tokenizer with the wrong NFC dim or a missing CA type
    raises a clear ValueError."""
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
    cfg, layout, _ = cfg_and_layout
    dims = _type_input_dims_for_layout(cfg, layout)
    # Case 1: NFC dim wrong.
    bad = dict(dims); bad[cfg.nfc.type_name] = 7
    with pytest.raises(ValueError, match="NFC"):
        EntityObservationTokenizer(cfg, D_MODEL, bad)
    # Case 2: missing type.
    bad = dict(dims); del bad["storage"]
    with pytest.raises(ValueError, match="storage"):
        EntityObservationTokenizer(cfg, D_MODEL, bad)


def test_dtype_and_device_propagation(cfg_and_layout):
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
    cfg, layout, obs = cfg_and_layout
    tok = EntityObservationTokenizer(
        cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout),
    )
    obs32 = obs.float()
    out = tok(obs32, layout)
    assert out.sro_tokens.dtype == torch.float32
    assert out.nfc_token.dtype == torch.float32
    assert out.ca_tokens.dtype == torch.float32
    assert out.sro_tokens.device == obs32.device
    assert out.nfc_token.device == obs32.device
    assert out.ca_tokens.device == obs32.device
    # CUDA branch — only run if available
    if torch.cuda.is_available():
        tok_cuda = tok.cuda()
        obs_cuda = obs32.cuda()
        out_cuda = tok_cuda(obs_cuda, layout)
        assert out_cuda.sro_tokens.is_cuda
        assert out_cuda.nfc_token.is_cuda
        assert out_cuda.ca_tokens.is_cuda
```

- [ ] **Step 1: Run**

```bash
pytest tests/test_entity_observation_tokenizer.py -k "input_dim_mismatch or dtype_and_device" -v
```
Expected: PASS.

- [ ] **Step 2: Commit**

```bash
git add tests/test_entity_observation_tokenizer.py
git commit -m "test(wp04): cover tokenizer construction validation and dtype/device propagation"
```

---

### Task 8: Tokenizer → Backbone → PPO components integration smoke test

This is not a §16 row, but it's the smallest end-to-end test that proves WP01 + WP04 compose correctly before the agent layer (WP05). It's the "would I trust this stack?" check.

**Files:**
- Modify: `tests/test_entity_observation_tokenizer.py`

```python
def test_tokenizer_backbone_ppo_components_integration(cfg_and_layout):
    """Tokenizer → backbone → ActorHead → CriticHead end-to-end on one batch."""
    from algorithms.utils.entity_observation_tokenizer import EntityObservationTokenizer
    from algorithms.utils.transformer_backbone import TransformerBackbone
    from algorithms.utils.ppo_components import ActorHead, CriticHead

    cfg, layout, obs = cfg_and_layout
    tok = EntityObservationTokenizer(cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout))
    backbone = TransformerBackbone(d_model=D_MODEL, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0)
    actor = ActorHead(d_model=D_MODEL)   # signature per ported v1 API; adjust if ActorHead requires more
    critic = CriticHead(d_model=D_MODEL)

    tokenized = tok(obs.float(), layout)
    ca_emb, pooled = backbone(tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens)
    means, log_std = actor(ca_emb)       # API may differ; if so, document in self-review
    value = critic(pooled)
    assert means.shape == (1, layout.n_ca)
    assert value.shape == (1, 1) or value.shape == (1,)
```

- [ ] **Step 1: Run; iterate on `ActorHead`/`CriticHead` API mismatches**

Inspect the ported v1 components if signatures differ:
```bash
grep -n "class ActorHead\|class CriticHead\|def forward" algorithms/utils/ppo_components.py
```
Adapt the test to call them with their actual signatures. Do **not** modify the ported components in this WP — they belong to WP01 and WP05 will adapt them at the agent layer if needed.

- [ ] **Step 2: Commit when green**

```bash
git add tests/test_entity_observation_tokenizer.py
git commit -m "test(wp04): integration smoke — tokenizer → backbone → PPO heads"
```

---

### Task 9: Full sweep + lint

- [ ] **Step 1: Full test run**

```bash
pytest -x -q
```
Expected: exit 0. The previously ported test `tests/test_transformer_refactor_helpers.py` from WP01 may break if a helper imports the v1 backbone signature. If so, evaluate:
  - If the helper is genuinely v1-only (marker tokenizer driven) and won't be reused in v2, **delete the test file** and document in the commit message.
  - If the helper survives v2 (e.g. `state_helper`, `update_helper`, `export_helper` are reused by WP05), adapt the test to the new backbone signature.

```bash
git add -A
git commit -m "chore(wp04): adapt/remove WP01 helper tests broken by backbone signature change"
```

---

## Self-Review Checklist (run before requesting code review)

- [ ] **Spec §16.2 coverage:** all 8 rows present in `tests/test_entity_observation_tokenizer.py`. `grep -c "^def test_" tests/test_entity_observation_tokenizer.py` ≥ 8 (+ integration smoke).
- [ ] **Spec §16.3 coverage:** all 6 rows present in `tests/test_transformer_backbone.py`. Count: ≥ 6.
- [ ] **`forward(sros, nfc, cas)` signature:** `grep "def forward" algorithms/utils/transformer_backbone.py` shows the new 3-tensor signature. The old single-sequence-tensor v1 signature is gone.
- [ ] **Type embedding table size = 3:** `python -c "from algorithms.utils.transformer_backbone import TransformerBackbone; b = TransformerBackbone(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0); import torch.nn as nn; print([m.num_embeddings for m in b.modules() if isinstance(m, nn.Embedding)])"` → `[3]`.
- [ ] **NFC dim hard-coded to 1:** the tokenizer constructor rejects any other NFC dim with a ValueError mentioning "NFC".
- [ ] **Zero new params on topology grow:** `test_projection_is_per_type_no_new_params_on_topology_grow` PASSes.
- [ ] **Non-contiguous slicing works:** `test_index_select_handles_non_contiguous_sro_segment` PASSes against the bundled sample (which has interleaved district forecasts).
- [ ] **Integration smoke passes:** tokenizer → backbone → actor/critic produces sensible shapes.
- [ ] **No marker references remain:** `grep -RIn "marker\|FA\|CTX" algorithms/utils/transformer_backbone.py algorithms/utils/entity_observation_tokenizer.py` → either no matches, or only docstring mentions of removal.
- [ ] **Full repo test suite passes:** `pytest -x -q` exits 0.

---

## Code Review

After self-review passes, invoke `superpowers:requesting-code-review`. Reviewer should check the diff against this plan and §8, §9, §16.2, §16.3 of `docs/specv2.md`.

---

## PR Description

```markdown
## Summary
Adapts the v1 `TransformerBackbone` to the v2 entity-mode `forward(sros, nfc, cas)` contract (spec §9) and adds the new `EntityObservationTokenizer` (spec §8) that consumes a `BuildingTokenLayout` (from WP03), slices the encoded per-building observation via `index_select`, computes the NFC scalar pre-projection, and projects each token via per-type `nn.Linear` weights (zero new parameters when topology grows).

## Key Changes
- Modify `algorithms/utils/transformer_backbone.py`: `nn.Embedding(3, d_model)` for `{SRO=0, NFC=1, CA=2}`; new `forward(sros, nfc, cas) -> (ca_embeddings, pooled)` with concat order `[sros, nfc, cas]`, CA slice at `[N_sro+1 : N_sro+1+N_ca]`, mean-pool over all tokens.
- Rewrite `tests/test_transformer_backbone.py` to cover all six rows of spec §16.3.
- Add `algorithms/utils/entity_observation_tokenizer.py`: `TokenizedObservation` dataclass + `EntityObservationTokenizer` (`nn.Module`) with per-type `ModuleDict` of `nn.Linear`, NFC scalar reduction via `NfcExpression`, validation that NFC dim == 1.
- Add `tests/test_entity_observation_tokenizer.py` covering all of spec §16.2 plus an integration smoke test (tokenizer → backbone → ported PPO heads).
- If WP01 helper tests required adaptation due to signature changes, those changes are included with a focused commit message.
```
