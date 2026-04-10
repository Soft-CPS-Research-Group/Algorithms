import torch
import torch.nn as nn


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 2,
        feedforward_dim: int = 128,
        action_dim: int = 3,
    ):
        super().__init__()

        # Token type embeddings: 0 = CA, 1 = SRO, 2 = NFC
        self.token_type_embedding = nn.Embedding(3, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # CA-only output head
        self.ca_output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, action_dim),
        )

    def forward(
        self,
        ca_embeddings: torch.Tensor,    # [num_cas, embedding_dim]
        sro_embeddings: torch.Tensor,   # [num_sros, embedding_dim]
        nfc_embedding: torch.Tensor,    # [embedding_dim]
    ) -> torch.Tensor:

        num_cas, embedding_dim = ca_embeddings.shape
        num_sros, embedding_dim_sro = sro_embeddings.shape

        assert embedding_dim == embedding_dim_sro
        assert nfc_embedding.shape == (embedding_dim,)

        # NFC as a single token
        nfc_token = nfc_embedding.unsqueeze(0)  # [1, embedding_dim]

        # Concatenate all tokens: [total_tokens, embedding_dim]
        all_tokens = torch.cat(
            [ca_embeddings, sro_embeddings, nfc_token],
            dim=0,
        )

        # Token type identifiers
        token_type_ids = torch.cat(
            [
                torch.zeros(num_cas, dtype=torch.long),
                torch.ones(num_sros, dtype=torch.long),
                torch.full((1,), 2, dtype=torch.long),
            ],
            dim=0,
        )

        # Add type embeddings (no positional embeddings, set semantics)
        all_tokens = all_tokens + self.token_type_embedding(token_type_ids)

        # Fake batch dimension for TransformerEncoder
        all_tokens = all_tokens.unsqueeze(0)  # [1, total_tokens, embedding_dim]

        encoded_tokens = self.transformer_encoder(all_tokens)[0]

        # CA tokens are always the first num_cas tokens
        ca_encoded_tokens = encoded_tokens[:num_cas]

        # One action per CA
        ca_actions = self.ca_output_head(ca_encoded_tokens)
        return ca_actions