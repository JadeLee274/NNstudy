from typing import *
import torch
import torch.nn as nn
from models.model.encoder import Encoder
from models.model.decoder import Decoder
Tensor = torch.Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        target_pad_idx: int,
        target_sos_idx: int,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        model_dim: int,
        num_heads: int,
        max_len: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.target_sos_idx = target_sos_idx
        self.device = device
        
        self.encoder = Encoder(
            vocab_size=encoder_vocab_size,
            model_dim=model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
            device=device,
        )

        self.decoder = Decoder(
            vocab_size=decoder_vocab_size,
            max_len=max_len,
            model_dim=model_dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )

    def forward(
        self,
        src: Tensor,
        target: Tensor,
    ) -> Tensor:
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        encoder_src = self.encoder(src, src_mask)
        out = self.decoder(target, encoder_src, target_mask, src_mask)
        return out
    
    def make_src_mask(
        self,
        src: Tensor,
    ) -> Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_target_mask(
        self,
        target: Tensor,
    ) -> Tensor:
        target_pad_mask = (target != self.target_pad_idx).unsqueeze(1).unsqueeze(3)
        target_len = target.shape[1]
        target_sub_mask = torch.tril(
            torch.ones(
                target_len, target_len
            ).type(torch.bool).to(self.device)
        )
        target_mask = target_pad_mask & target_sub_mask
        return target_mask
