from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_list: List[int],
        hidden_dim: int,
        input_size: int,
        kld_weight: float = 2e-04,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        self.hidden_dim = hidden_dim
        self.kld_weight = kld_weight

        encoder_list = []
        channel = in_channels

        for out_channels in out_channels_list:
            encoder_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channel,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(
                        num_features=out_channels,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channel = out_channels
        self.encoder = nn.Sequential(*encoder_list)

        self.fc_mu = nn.Linear(
            in_features=(
                out_channels_list[-1] 
                * int((input_size / (2 ** len(out_channels_list))))
                * int((input_size / (2 ** len(out_channels_list))))
            ),
            out_features=hidden_dim,
            )
        
        self.fc_var = nn.Linear(
            in_features=(
                out_channels_list[-1] 
                * int((input_size / (2 ** len(out_channels_list))))
                * int((input_size / (2 ** len(out_channels_list))))
            ),
            out_features=hidden_dim,
            )

        self.fc_dec = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim,
                out_features=(
                    out_channels_list[-1]
                    * int((input_size / (2 ** len(out_channels_list))))
                    * int((input_size / (2 ** len(out_channels_list))))
                ),
            )
        )

        dec_list = []
        for i in range(len(out_channels_list) - 1):
            dec_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=out_channels_list[len(out_channels_list)
                                                     - (i + 1)],
                        out_channels=out_channels_list[len(out_channels_list)
                                                       - (i + 2)],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(
                        num_features=out_channels_list[len(out_channels_list)
                                                       - (i + 2)],
                    ),
                    nn.LeakyReLU(),
                )
            )
        dec_list.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=out_channels_list[0],
                    out_channels=out_channels_list[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(
                    num_features=out_channels_list[0],
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=out_channels_list[0],
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )
        self.decoder = nn.Sequential(*dec_list)

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        out = self.encoder(x)
        out = out.view(out.shape[0], -1)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return mu, log_var
    
    def decode(
        self,
        z: Tensor,
    ) -> Tensor:
        out = self.fc_dec(z)
        out = out.view(
            -1,
            self.out_channels_list[-1],
            int((self.input_size / (2 ** len(self.out_channels_list)))),
            int((self.input_size / (2 ** len(self.out_channels_list))))
        )
        reconstrucsion = self.decoder(out)
        return reconstrucsion
    
    def reparametrization(
        self,
        mu: Tensor,
        log_var: Tensor,
    ) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + (std * eps)
        return z
    
    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparametrization(mu, log_var)
        return self.decode(z), x, mu, log_var
    
    def loss_fn(
        self,
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Dict[str, Tensor]:
        reconstruction_loss = F.mse_loss(x_hat, x)
        kld_loss = torch.mean(
            - 0.5 * torch.sum(
                input=(1 + log_var - mu ** 2 - torch.exp(log_var)),
                dim=1,
            ),
            dim=0,
        )
        loss = reconstruction_loss + self.kld_weight * kld_loss
        return {
            'Loss': loss,
            'Reconstruction Loss': reconstruction_loss.detach(),
            'KLD Loss': (-kld_loss).detach(),
        }
    
    def sample(
        self,
        num_samples: int,
        device: torch.device,
    ) -> Tensor:
        z = torch.randn(num_samples, self.hidden_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def generate(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.forward(x)[0]
