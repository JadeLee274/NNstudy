import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from typing import *
Tensor = torch.Tensor


class Generator(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 128,
        num_doublesize: int = 5,
        n_filters: int = 64,
    ) -> None:
        param_list = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=latent_dim,
                    out_channels=(n_filters * (2 ** (num_doublesize - 2))),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(n_filters * (2 ** (num_doublesize - 2))),
                nn.LeakyReLU(),
            ),
        ]

        for i in range(num_doublesize - 2):
            param_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=(n_filters * (2 ** (num_doublesize - 2 - i))),
                        out_channels=(n_filters * (2 ** (num_doublesize - 1 - i))),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(n_filters * (2 ** (num_doublesize - 1 - i))),
                    nn.LeakyReLU(),
                ),
            )
        
        param_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=n_filters,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            ),
        )

        self.main = nn.Sequential(*param_list)

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_halfsize: int = 5,
        num_filters: int = 64,
    ) -> None:
        param_list = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(),
            ),
        ]

        for i in range(num_halfsize - 2):
            param_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=(num_filters * (2 ** i)),
                        out_channels=(num_filters * (2 ** (i + 1))),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_filters * (2 ** (i + 1))),
                    nn.LeakyReLU(),
                ),
            )
        
        param_list.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=(num_filters * (2 ** (num_halfsize - 2))),
                    out_channels=1,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        )

        self.main = nn.Sequential(*param_list)
        
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.main(x)
        return out
