from typing import *
import torch
import torch.nn as nn
from models.layers.attention import MultiHeadAttention
Tensor = torch.Tensor


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)

        self.enc_dec_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(
        self,
        dec_input: Tensor,
        enc_output: Tensor,
        target_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        _x = dec_input
        x = self.attention(dec_input, dec_input, dec_input, target_mask)

        x = self.dropout1(x)
        x += _x
        x = self.norm1(x)

        if enc_output is not None:
            _x = x
            x = self.enc_dec_attention(x, enc_output, enc_output, src_mask)
            x = self.dropout2(x)
            x += _x
            x = self.norm2(x)
        
        _x = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x += _x
        x = self.norm3(x)

        return x