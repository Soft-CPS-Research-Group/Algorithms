"""EntityObservationTokenizer — slice + project encoded per-building observation.

Spec ``docs/specv2.md`` §8: per-building module that consumes
(a) a 2-D ``encoded_obs`` tensor of shape ``[B, obs_dim]`` (already encoded
by the wrapper's per-feature encoders), and
(b) a ``BuildingTokenLayout`` produced by ``EntityTokenLayoutBuilder``.

It returns three token banks (SRO, NFC, CA) ready to be fed to
``TransformerBackbone.forward(sros, nfc, cas)``.

Design (spec §8.4-§8.5):
- One ``nn.Linear`` per declared **type** (not per instance) so adding a
  second charger / PV / storage adds zero new parameters.
- NFC has a fixed ``in_features = 1`` because the upstream
  ``NfcExpression`` reduces its operands to a scalar before projection.
- Slicing uses ``index_select`` so non-contiguous segments (interleaved
  district forecasts) work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

import torch
from torch import nn

from algorithms.utils.entity_token_layout import (
    BuildingTokenLayout,
    NfcExpression,
)


@dataclass
class TokenizedObservation:
    """Output bank of per-token embeddings for one building.

    Shapes are batched: ``[B, N, d_model]`` where ``N`` varies by family.
    ``sro_types`` and ``ca_types`` parallel the rows of their respective
    token tensors and let downstream code (e.g. action routing) recover
    the type of each token.
    """

    sro_tokens: torch.Tensor  # [B, N_sro, d_model]
    nfc_token: torch.Tensor  # [B, 1,     d_model]
    ca_tokens: torch.Tensor  # [B, N_ca,  d_model]
    sro_types: List[str]
    ca_types: List[str]
    n_sro: int
    n_ca: int


class EntityObservationTokenizer(nn.Module):
    """Slice ``encoded_obs`` per the layout, then project each segment to
    ``d_model`` via the per-type ``nn.Linear`` from ``self.projections``.
    """

    def __init__(
        self,
        tokenizer_config,
        d_model: int,
        type_input_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        self._cfg = tokenizer_config
        self._d_model = int(d_model)

        nfc_name = tokenizer_config.nfc.type_name
        required = (
            set(tokenizer_config.ca_types.keys())
            | set(tokenizer_config.sro_types.keys())
            | {nfc_name}
        )
        missing = required - set(type_input_dims)
        if missing:
            raise ValueError(
                "EntityObservationTokenizer: type_input_dims is missing "
                f"required entries: {sorted(missing)}"
            )
        if int(type_input_dims[nfc_name]) != 1:
            raise ValueError(
                f"EntityObservationTokenizer: NFC type {nfc_name!r} "
                "input dim must be 1 (the NfcExpression reduces operands to "
                f"a scalar); got {type_input_dims[nfc_name]}."
            )

        self.projections = nn.ModuleDict(
            {
                type_name: nn.Linear(int(in_dim), self._d_model)
                for type_name, in_dim in type_input_dims.items()
            }
        )

    def forward(
        self,
        encoded_obs: torch.Tensor,  # [B, obs_dim]
        layout: BuildingTokenLayout,
    ) -> TokenizedObservation:
        if encoded_obs.dim() != 2:
            raise ValueError(
                "encoded_obs must be 2-D [B, obs_dim], got shape "
                f"{tuple(encoded_obs.shape)}"
            )
        device = encoded_obs.device
        batch = encoded_obs.shape[0]

        sro_tokens: List[torch.Tensor] = []
        ca_tokens: List[torch.Tensor] = []
        sro_types: List[str] = []
        ca_types: List[str] = []
        nfc_token: torch.Tensor | None = None

        for seg in layout.segments:
            idx = torch.tensor(
                list(seg.feature_indices), dtype=torch.long, device=device
            )
            group = encoded_obs.index_select(dim=1, index=idx)  # [B, k]

            if seg.family == "nfc":
                expr = seg.derived
                if not isinstance(expr, NfcExpression):
                    raise ValueError(
                        "NFC segment is missing its NfcExpression "
                        f"(got {type(expr).__name__})."
                    )
                if expr.op != "subtract":
                    raise ValueError(
                        f"unsupported NFC op: {expr.op!r}"
                    )
                lhs = group[:, expr.left_index_in_segment]
                rhs = group[:, expr.right_index_in_segment]
                scalar = (lhs - rhs).unsqueeze(1)  # [B, 1]
                projected = self.projections[seg.type_name](scalar)
                nfc_token = projected.unsqueeze(1)  # [B, 1, d_model]

            elif seg.family == "sro":
                projected = self.projections[seg.type_name](group)
                sro_tokens.append(projected.unsqueeze(1))
                sro_types.append(seg.type_name)

            elif seg.family == "ca":
                projected = self.projections[seg.type_name](group)
                ca_tokens.append(projected.unsqueeze(1))
                ca_types.append(seg.type_name)

            else:  # pragma: no cover — guarded by the layout dataclass
                raise ValueError(
                    f"unknown segment family: {seg.family!r}"
                )

        if nfc_token is None:
            raise ValueError("layout is missing the NFC segment")

        sro_stack = (
            torch.cat(sro_tokens, dim=1)
            if sro_tokens
            else torch.zeros(batch, 0, self._d_model, device=device)
        )
        ca_stack = (
            torch.cat(ca_tokens, dim=1)
            if ca_tokens
            else torch.zeros(batch, 0, self._d_model, device=device)
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
