"""Tests for ``EntityObservationTokenizer``. Covers spec §16.2 + integration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


D_MODEL = 8


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _action_names_for_sample() -> list[str]:
    """The Building_1 sample has 1 storage + 1 charger attached, so the
    action vector has the two CA actuators in the canonical order used by
    the legacy CityLearn flat interface."""
    return ["electrical_storage", "electric_vehicle_storage"]


@pytest.fixture
def cfg():
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    return load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )


@pytest.fixture
def layout(cfg):
    from algorithms.utils.entity_token_layout import (
        EntityTokenLayoutBuilder,
    )

    obs_names = load_sample_observation_names_for_first_building()
    builder = EntityTokenLayoutBuilder(cfg)
    return builder.build("Building_1", obs_names, _action_names_for_sample())


@pytest.fixture
def sentinel_obs(layout) -> torch.Tensor:
    """``obs[0, i] = float(i)`` — lets tests recover which feature indices
    each segment selected by inspecting the projected output."""
    obs_dim = max(max(s.feature_indices) for s in layout.segments) + 1
    return torch.arange(obs_dim, dtype=torch.float).unsqueeze(0)


def _type_input_dims_for_layout(cfg, layout) -> dict[str, int]:
    """Compute per-type raw input dim from the layout. Mirrors what the
    agent will compute from ``entity_specs`` at runtime."""
    dims: dict[str, int] = {}
    for seg in layout.segments:
        if seg.family == "nfc":
            dims[cfg.nfc.type_name] = 1
        else:
            dims[seg.type_name] = len(seg.feature_indices)
    # Some declared types may not be present in this particular sample
    # (e.g. ``ev_connected`` if no EV is plugged in). The constructor still
    # requires entries for them; pick a placeholder dim of 1.
    declared = (
        set(cfg.ca_types.keys())
        | set(cfg.sro_types.keys())
        | {cfg.nfc.type_name}
    )
    for type_name in declared:
        dims.setdefault(type_name, 1)
    return dims


# --------------------------------------------------------------------------
# §16.2 row 1: forward shape baseline
# --------------------------------------------------------------------------


def test_forward_shapes_baseline(cfg, layout, sentinel_obs) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        tokenizer_config=cfg,
        d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    out = tok(sentinel_obs, layout)

    assert out.sro_tokens.shape == (1, layout.n_sro, D_MODEL)
    assert out.nfc_token.shape == (1, 1, D_MODEL)
    assert out.ca_tokens.shape == (1, layout.n_ca, D_MODEL)
    assert len(out.sro_types) == layout.n_sro
    assert len(out.ca_types) == layout.n_ca
    assert out.n_sro == layout.n_sro
    assert out.n_ca == layout.n_ca


# --------------------------------------------------------------------------
# §16.2 row 2: NFC scalar reduction = lhs - rhs
# --------------------------------------------------------------------------


def test_nfc_token_value_equals_subtract_op(cfg, layout) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        tokenizer_config=cfg,
        d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )

    # Synthetic obs: zero everywhere except the two NFC operand positions.
    obs_dim = max(max(s.feature_indices) for s in layout.segments) + 1
    obs = torch.zeros(1, obs_dim)
    nfc_seg = next(s for s in layout.segments if s.family == "nfc")
    obs[0, nfc_seg.feature_indices[0]] = 5.0  # lhs
    obs[0, nfc_seg.feature_indices[1]] = 2.0  # rhs

    # Set NFC projection to ones-weight, zero-bias — output value per dim
    # equals the input scalar.
    nfc_name = cfg.nfc.type_name
    with torch.no_grad():
        tok.projections[nfc_name].weight.fill_(1.0)
        tok.projections[nfc_name].bias.fill_(0.0)

    out = tok(obs, layout)
    expected = 5.0 - 2.0
    assert torch.allclose(
        out.nfc_token, torch.full_like(out.nfc_token, expected)
    )


def test_nfc_projection_input_dim_is_one(cfg, layout) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        tokenizer_config=cfg,
        d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    nfc_name = cfg.nfc.type_name
    assert tok.projections[nfc_name].in_features == 1


# --------------------------------------------------------------------------
# §16.2 row 3: zero new params on topology grow
# --------------------------------------------------------------------------


def test_projection_is_per_type_no_new_params_on_topology_grow(
    cfg, layout
) -> None:
    """Adding a second charger (or any extra CA of an existing type) must
    NOT add new projection parameters — that is the whole point of per-type
    weight sharing in spec §8.4."""
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )
    from algorithms.utils.entity_token_layout import (
        BuildingTokenLayout,
        TokenSegment,
    )

    tok = EntityObservationTokenizer(
        tokenizer_config=cfg,
        d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    params_before = sum(p.numel() for p in tok.parameters())

    charger_segs = [
        s
        for s in layout.segments
        if s.family == "ca" and s.type_name == "charger"
    ]
    if not charger_segs:
        pytest.skip("Sample layout has no charger CA segment")
    base = charger_segs[0]
    extra = TokenSegment(
        family="ca",
        type_name="charger",
        instance_id=(base.instance_id or "x") + "_dup",
        feature_indices=base.feature_indices,
        feature_names=base.feature_names,
    )
    grown = BuildingTokenLayout(
        building_id=layout.building_id,
        segments=tuple(list(layout.segments) + [extra]),
        n_sro=layout.n_sro,
        n_ca=layout.n_ca + 1,
        ca_action_names=layout.ca_action_names
        + ("electric_vehicle_storage",),
        excluded_feature_names=layout.excluded_feature_names,
    )

    obs_dim = max(max(s.feature_indices) for s in grown.segments) + 1
    obs = torch.zeros(1, obs_dim)
    out = tok(obs, grown)

    assert out.ca_tokens.shape == (1, layout.n_ca + 1, D_MODEL)
    params_after = sum(p.numel() for p in tok.parameters())
    assert params_before == params_after, (
        "Parameter count changed on topology grow: "
        f"{params_before} -> {params_after}"
    )


# --------------------------------------------------------------------------
# §16.2 row 4: non-contiguous slicing via index_select
# --------------------------------------------------------------------------


def test_index_select_handles_non_contiguous_sro_segment(
    cfg, layout, sentinel_obs
) -> None:
    """Pick any SRO segment with ≥ 2 indices and verify the gathered values
    sum to the sum of those raw indices (sentinel ``obs[i]==i``)."""
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    target = next(
        (
            s
            for s in layout.segments
            if s.family == "sro" and len(s.feature_indices) >= 2
        ),
        None,
    )
    if target is None:
        pytest.skip("sample has no multi-feature SRO segment")

    tok = EntityObservationTokenizer(
        tokenizer_config=cfg,
        d_model=D_MODEL,
        type_input_dims=_type_input_dims_for_layout(cfg, layout),
    )
    proj = tok.projections[target.type_name]
    in_dim = proj.in_features
    with torch.no_grad():
        proj.weight.copy_(torch.ones(D_MODEL, in_dim))
        proj.bias.zero_()

    out = tok(sentinel_obs, layout)
    # Find the position of this segment in sro_tokens.
    pos = next(
        i
        for i, seg in enumerate(
            [s for s in layout.segments if s.family == "sro"]
        )
        if seg is target
    )
    summed = out.sro_tokens[0, pos, 0].item()
    expected = float(sum(target.feature_indices))
    assert summed == pytest.approx(expected), (
        f"Sliced sum mismatch: got {summed}, expected {expected}. "
        f"Indices were {target.feature_indices}"
    )


# --------------------------------------------------------------------------
# §16.2 row 5: construction validation (NFC dim + missing types)
# --------------------------------------------------------------------------


def test_construction_rejects_wrong_nfc_dim(cfg, layout) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    dims = _type_input_dims_for_layout(cfg, layout)
    bad = dict(dims)
    bad[cfg.nfc.type_name] = 7
    with pytest.raises(ValueError, match="NFC"):
        EntityObservationTokenizer(cfg, D_MODEL, bad)


def test_construction_rejects_missing_type(cfg, layout) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    dims = _type_input_dims_for_layout(cfg, layout)
    bad = dict(dims)
    del bad["storage"]
    with pytest.raises(ValueError, match="storage"):
        EntityObservationTokenizer(cfg, D_MODEL, bad)


# --------------------------------------------------------------------------
# §16.2 row 6: dtype + device propagation
# --------------------------------------------------------------------------


def test_dtype_and_device_propagation(cfg, layout, sentinel_obs) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout)
    )
    obs32 = sentinel_obs.float()
    out = tok(obs32, layout)

    assert out.sro_tokens.dtype == torch.float32
    assert out.nfc_token.dtype == torch.float32
    assert out.ca_tokens.dtype == torch.float32
    assert out.sro_tokens.device == obs32.device
    assert out.nfc_token.device == obs32.device
    assert out.ca_tokens.device == obs32.device


# --------------------------------------------------------------------------
# §16.2 row 7: rejects non-2D input
# --------------------------------------------------------------------------


def test_forward_rejects_non_2d_input(cfg, layout) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout)
    )
    with pytest.raises(ValueError, match="2-D"):
        tok(torch.zeros(5), layout)  # 1-D
    with pytest.raises(ValueError, match="2-D"):
        tok(torch.zeros(2, 3, 4), layout)  # 3-D


# --------------------------------------------------------------------------
# §16.2 row 8: gradient flows through projections
# --------------------------------------------------------------------------


def test_gradient_flows_through_projections(
    cfg, layout, sentinel_obs
) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )

    tok = EntityObservationTokenizer(
        cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout)
    )
    out = tok(sentinel_obs, layout)
    loss = out.sro_tokens.sum() + out.nfc_token.sum() + out.ca_tokens.sum()
    loss.backward()

    grad_present = [
        (name, p.grad is not None and p.grad.abs().sum() > 0)
        for name, p in tok.named_parameters()
    ]
    # At minimum every projection touched by the layout must have grad.
    touched_types = {seg.type_name for seg in layout.segments}
    for type_name in touched_types:
        for name, has_grad in grad_present:
            if name.startswith(f"projections.{type_name}.weight"):
                assert has_grad, (
                    f"projection {type_name} has no gradient"
                )


# --------------------------------------------------------------------------
# Integration smoke: tokenizer → backbone → ActorHead/CriticHead
# --------------------------------------------------------------------------


def test_tokenizer_backbone_ppo_components_integration(
    cfg, layout, sentinel_obs
) -> None:
    from algorithms.utils.entity_observation_tokenizer import (
        EntityObservationTokenizer,
    )
    from algorithms.utils.transformer_backbone import TransformerBackbone
    from algorithms.utils.ppo_components import ActorHead, CriticHead

    tok = EntityObservationTokenizer(
        cfg, D_MODEL, _type_input_dims_for_layout(cfg, layout)
    )
    backbone = TransformerBackbone(
        d_model=D_MODEL,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    actor = ActorHead(d_model=D_MODEL, hidden_dim=16)
    critic = CriticHead(d_model=D_MODEL, hidden_dim=16)

    tokenized = tok(sentinel_obs.float(), layout)
    ca_emb, pooled = backbone(
        tokenized.sro_tokens, tokenized.nfc_token, tokenized.ca_tokens
    )
    actions, log_probs, means = actor(ca_emb, deterministic=True)
    value = critic(pooled)

    assert actions.shape == (1, layout.n_ca, 1)
    assert log_probs.shape == (1, layout.n_ca)
    assert means.shape == (1, layout.n_ca, 1)
    # Critic returns either [B, 1] or [B] depending on internal squeeze.
    assert value.shape in {(1, 1), (1,)}
