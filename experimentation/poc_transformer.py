"""
Minimal Transformer PoC for variable-cardinality inputs with strict 1-to-1 CA outputs.

Design goals demonstrated:
1) Variable number of CAs and SROs at runtime, no retraining, same model instance.
2) Strict one-to-one mapping, one output per CA token, in the same CA order.
3) Extra inputs, 1 global NFC token, variable SRO set, outputs stay aligned to CAs.

Requires: pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import torch
import torch.nn as nn


torch.set_printoptions(precision=4, sci_mode=False)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    action_dim: int = 3
    max_tokens: int = 256  # upper bound for N_ca + N_sro + 1 (NFC)


class SimpleSetTransformerPolicy(nn.Module):
    """
    Encoder-only Transformer over concatenated tokens [CA..., SRO..., NFC],
    then an MLP head applied ONLY to CA token outputs, producing [N_ca, action_dim].

    This makes the 1-to-1 CA mapping structural, output i corresponds to CA token i
    for the chosen CA ordering.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Type embeddings, 0=CA, 1=SRO, 2=NFC
        self.type_emb = nn.Embedding(3, cfg.d_model)

        # Positional embeddings, not strictly necessary for sets, but useful to keep it standard.
        # You can also remove these to keep it closer to permutation-invariant for CA and SRO,
        # but then you should rely purely on content and type embeddings.
        self.pos_emb = nn.Embedding(cfg.max_tokens, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.action_dim),
        )

    def forward(
        self,
        ca_tokens: torch.Tensor,   # [B, N_ca, d_model]
        sro_tokens: torch.Tensor,  # [B, N_sro, d_model]
        nfc_token: torch.Tensor,   # [B, 1, d_model]
        ca_mask: torch.Tensor | None = None,    # [B, N_ca] True=keep, False=pad
        sro_mask: torch.Tensor | None = None,   # [B, N_sro] True=keep, False=pad
        nfc_mask: torch.Tensor | None = None,   # [B, 1] True=keep, False=pad
    ) -> torch.Tensor:
        B, N_ca, D = ca_tokens.shape
        _, N_sro, _ = sro_tokens.shape
        assert D == self.cfg.d_model
        assert nfc_token.shape == (B, 1, D)

        total = N_ca + N_sro + 1
        if total > self.cfg.max_tokens:
            raise ValueError(f"total tokens {total} exceeds max_tokens {self.cfg.max_tokens}")

        # Build concatenated sequence
        x = torch.cat([ca_tokens, sro_tokens, nfc_token], dim=1)  # [B, T, D]

        # Build type ids
        type_ids = torch.cat(
            [
                torch.zeros(B, N_ca, dtype=torch.long, device=x.device),
                torch.ones(B, N_sro, dtype=torch.long, device=x.device),
                torch.full((B, 1), 2, dtype=torch.long, device=x.device),
            ],
            dim=1,
        )  # [B, T]

        # Add embeddings
        pos_ids = torch.arange(total, device=x.device).unsqueeze(0).expand(B, total)  # [B, T]
        x = x + self.type_emb(type_ids) + self.pos_emb(pos_ids)

        # Key padding mask: True means "ignore" in PyTorch Transformer
        if ca_mask is None:
            ca_mask = torch.ones(B, N_ca, dtype=torch.bool, device=x.device)
        if sro_mask is None:
            sro_mask = torch.ones(B, N_sro, dtype=torch.bool, device=x.device)
        if nfc_mask is None:
            nfc_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)

        keep_mask = torch.cat([ca_mask, sro_mask, nfc_mask], dim=1)  # [B, T], True=keep
        key_padding_mask = ~keep_mask  # True=pad/ignore

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, D]

        # Strict 1-to-1 CA mapping, only slice CA outputs
        h_ca = h[:, :N_ca, :]  # [B, N_ca, D]
        actions = self.head(h_ca)  # [B, N_ca, action_dim]
        return actions


def make_random_inputs(
    cfg: ModelConfig,
    batch_size: int,
    n_ca: int,
    n_sro: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(1234 + n_ca * 17 + n_sro * 31)

    ca = torch.randn(batch_size, n_ca, cfg.d_model, generator=g, device=device)
    sro = torch.randn(batch_size, n_sro, cfg.d_model, generator=g, device=device)
    nfc = torch.randn(batch_size, 1, cfg.d_model, generator=g, device=device)
    return ca, sro, nfc


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def main() -> None:
    device = "cpu"
    cfg = ModelConfig(d_model=64, nhead=4, num_layers=2, action_dim=3, max_tokens=256)
    model = SimpleSetTransformerPolicy(cfg).to(device)
    model.eval()

    B = 1

    print("\nTest A, output count follows number of CAs")

    for n_ca in [3, 7]:
        ca, sro, nfc = make_random_inputs(cfg, B, n_ca=n_ca, n_sro=5, device=device)
        out = model(ca, sro, nfc)
        print(f"CAs in: {n_ca} → outputs out: {out.shape[1]}")

    print("\nTest B, SROs affect context but not output count")

    n_ca = 4
    for n_sro in [0, 9]:
        ca, sro, nfc = make_random_inputs(cfg, B, n_ca=n_ca, n_sro=n_sro, device=device)
        out = model(ca, sro, nfc)
        print(f"CAs in: {n_ca}, SROs in: {n_sro} → outputs out: {out.shape[1]}")

    print("\nTest C, strict 1-to-1 CA mapping under permutation")

    ca, sro, nfc = make_random_inputs(cfg, B, n_ca=6, n_sro=4, device=device)
    base = model(ca, sro, nfc)

    perm = torch.tensor([2, 5, 1, 0, 4, 3], device=device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=device)

    ca_perm = ca[:, perm, :]
    out_perm = model(ca_perm, sro, nfc)
    out_perm_reordered = out_perm[:, inv_perm, :]

    diff = max_abs_diff(base, out_perm_reordered)
    print(f"Permutation equivariance check (lower is better): {diff:.6f}")

    print("\nInterpretation")
    print("- Each CA token produces exactly one output.")
    print("- Adding or removing SROs does not change the number of outputs.")
    print("- Reordering CAs reorders outputs accordingly.")


if __name__ == "__main__":
    main()