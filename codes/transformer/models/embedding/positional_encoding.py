from typing import *
import math
import torch
import torch.nn as nn
Tensor = torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        model_dim: int,
        max_len: int,
        device,
    ) -> None:
        super().__init__()
        self.encoding = torch.empty(max_len, model_dim, device=device)
        self.encoding.requires_grad = False

        position = torch.arange(0, max_len, device=device,).unsqueeze(1)

        div_term = torch.arange(0, model_dim, 2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(
            position / (10000 * (div_term / model_dim))
        )
        self.encoding[:, 1::2] = torch.cos(
            position / (10000 * (div_term / model_dim))
        )
    
    def forward(
        self, 
        x: Tensor,
    ) -> Tensor:
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]
    