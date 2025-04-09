from typing import *
import os
import torch
import torch.nn as nn
Tensor = torch.Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        ) # (B, C, H, W) -> (B, embed_dim, patch_size, patch_size)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        B, C, H, W = x.shape
        assert (H == self.img_size and W == self.img_size,
        "Input image size must match model config."
        )

        x = self.proj(x) # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = x.flatten(2) # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)

        x = x + self.pos_embedding
        return x

