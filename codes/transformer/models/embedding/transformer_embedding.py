from typing import *
import torch
import torch.nn as nn
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding
Tensor = torch.Tensor


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        max_len: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            model_dim=model_dim,
        )
        self.positional_encoding = PositionalEncoding(
            model_dim=model_dim,
            max_len=max_len,
            device=device
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        token_embedding = self.token_embedding(x)
        positional_encoding = self.positional_encoding(x)
        return self.dropout(token_embedding + positional_encoding)