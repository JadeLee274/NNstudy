from typing import *
import torch
import torch.nn as nn
from models.layers.attention import MultiHeadAttention
Tensor = torch.Tensor


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads, 
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim, 1e-12)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim, 1e-12)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        _x = x
        x = self.attention(x, x, x, mask)
        x += self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        
        return x 
