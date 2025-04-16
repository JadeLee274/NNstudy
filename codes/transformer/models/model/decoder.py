from typing import *
import torch
import torch.nn as nn
from models.layers.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding
Tensor = torch.Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        model_dim: int,
        feedforward_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()
        self.embeddimg = TransformerEmbedding(
            vocab_size=vocab_size,
            model_dim=model_dim,
            max_len=max_len,
            dropout=dropout,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(
        self,
        target: Tensor,
        encoder_src: Tensor,
        target_mask: Tensor,
        src_mask: Tensor,
    ) -> Tensor:
        target = self.embeddimg(target)

        for layer in self.layers:
            target = layer(
                dec_input=target,
                enc_output=encoder_src,
                target_mask=target_mask,
                src_mask=src_mask,
            )
        
        out = self.linear(target)
        
        return out
    