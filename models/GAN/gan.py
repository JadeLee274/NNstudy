import torch
import torch.nn as nn
from typing import *
Tensor = torch.Tensor


class Generator(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 128,
        num_doublesize: int = 5,
        num_filters: int = 64,
        final_act: str = 'tanh'
    ) -> None:
        super().__init__()

        if final_act == None:
            self.final_act = nn.Identity()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'tanh':
            self.final_act = nn.Tanh()
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'leakyrelu':
            self.final_act = nn.LeakyReLU(
                negative_slope=0.2,
            )

        param_list = [
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=latent_dim,
                    out_channels=(num_filters * (2 ** (num_doublesize - 2))),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(num_filters * (2 ** (num_doublesize - 2))),
                nn.ReLU(inplace=True),
            ),
        ]

        for i in range(num_doublesize - 2):
            param_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=(num_filters * (2 ** (num_doublesize - 2 - i))),
                        out_channels=(num_filters * (2 ** (num_doublesize - 3 - i))),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(num_filters * (2 ** (num_doublesize - 3 - i))),
                    nn.ReLU(inplace=True),
                ),
            )
        
        param_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=num_filters,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                self.final_act,
            ),
        )

        self.main = nn.Sequential(*param_list)

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(
                    tensor=m.weight.data,
                    mean=0.0,
                    std=0.02,
                )
                nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.normal_(
            #         tensor=m.weight.data,
            #         mean=1.0,
            #         std=0.02,
            #     )
            #     nn.init.zeros_(m.bias)

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
        final_act: str = 'sigmoid'
    ) -> None:
        super().__init__()

        if final_act == None:
            self.final_act = nn.Identity()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        elif final_act == 'tanh':
            self.final_act = nn.Tanh()
        elif final_act == 'relu':
            self.final_act = nn.ReLU()
        elif final_act == 'leakyrelu':
            self.final_act = nn.LeakyReLU()

        param_list = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(
                    negative_slope=0.2,
                    inplace=True,
                ),
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
                    nn.LeakyReLU(
                        negative_slope=0.2,
                        inplace=True,
                    ),
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
                self.final_act,
            )
        )

        self.main = nn.Sequential(*param_list)
        
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(
                    tensor=m.weight.data,
                    mean=0.0,
                    std=0.02,
                )
                nn.init.zeros_(m.bias.data)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.normal_(
            #         tensor=m.weight.data,
            #         mean=1.0,
            #         std=0.02,
            #     )
            #     nn.init.zeros_(m.bias.data)

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        out = self.main(x)
        return out
