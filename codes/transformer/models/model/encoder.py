from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding.transformer_embedding import TransformerEmbedding
from models.layers.encoder_layer import EncoderLayer
Tensor = torch.Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        device,
    ) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            model_dim=model_dim,
            max_len=max_len,
            dropout=dropout,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x