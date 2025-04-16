from typing import *
import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> None:
        super().__init__(vocab_size, model_dim, padding_idx=1)

