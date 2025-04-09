from typing import *
import torch
import torch.nn as nn
Tensor = torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()
        pe = torch.empty(max_len, model_dim) # (max_len, model_dim)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float()
        * (-torch.log(10000.0) / model_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, max_len, model_dim)
        self.register_buffer("pe", pe)
    
    def forward(
        self, 
        x: Tensor,
    ) -> Tensor:
        """
        x     : (batch_size, seq_len, d_modal)
        output: x + positional_encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    