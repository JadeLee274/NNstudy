import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import DataParallel as dp
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from gan import Generator, Discriminator
ROOT = '/data/home/tmdals274/NNstudy/data'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)
parser.add_argument(
    "--latent-dim",
    type=int,
    default=128,
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2e-4,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
)
parser.add_argument(
    "--num-gpu",
    type=int,
    default=4,
)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mnist_set = MNIST(
    root=ROOT,
    train=True,
    transform=T.Compose(
        [
            T.Resize(32),
            T.ToTensor(),
        ]
    )
)

dataloader = DataLoader(
    dataset=mnist_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)

netG = Generator(
    latent_dim=args.latent_dim,
    out_channels=1,
    num_filters=64,
    num_doublesize=5,
).to(device)

netD = Discriminator(
    in_channels=1,
    num_filters=64,
    num_halfsize=5,
).to(device)

if (device.type == 'cuda') and (args.num_gpu > 1):
    netG = dp(
        module=netG,
        device_ids=list(range(args.num_gpu))
    )
    netD = dp(
        module=netD,
        device_ids=list(range(args.num_gpu))
    )

criterion = nn.BCEWithLogitsLoss()

real_label = 1.
fake_label = 0.

optimizer_G = optim.Adam(
    params=netG.parameters(),
    lr=args.learning_rate,
    betas=(0.5, 0.999),
)

optimizer_D = optim.Adam(
    params=netD.parameters(),
    lr=args.learning_rate,
    betas=(0.5, 0.999),
)


def train() -> None:
    for epoch in range(args.epochs):
        for i, (data, _) in enumerate(dataloader):
            optimizer_D.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full(
                size=(batch_size,),
                fill_value=real_label,
                dtype=torch.float,
                device=device,
            )
            output_real = netD(real_data).view(-1)
            D_err_real = criterion(output_real, label)
            D_err_real.backward()
            D_x = output_real.mean().item()

            noise = torch.randn(args.batch_size, args.latent_dim, 1, 1)

            fake_data = netG(noise)
            label.fill_(fake_label)
            output_fake_detach = netD(fake_data.detach()).view(-1)
            D_err_fake = criterion(output_fake_detach, label)
            D_G_z1 = output_fake_detach.mean().item()

            D_err = D_err_real + D_err_fake
            optimizer_D.step()

            optimizer_G.zero_grad()
            label.fill_(real_label)
            output_fake = netD(fake_data).view(-1)
            G_err = criterion(output_fake, label)
            G_err.backward()
            D_G_z2 = output_fake.mean().item()
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1} | Iter {i + 1} : Loss_D = {D_err:.3e}, Loss_G = {G_err:.3e}, D(x) = {D_x:.3e}, D(G(z)) = {D_G_z1:.3e} / {D_G_z2:.3e}")
                fixed_noise = torch.randn(args.batch_size, args.latent_dim, 1, 1)
                save_image(
                    tensor=netG(fixed_noise).detach().cpu(),
                    fp=os.path.join(
                        './output_mnist',
                        f'./mnist_epoch_{(epoch + 1):03d}_iter_{(i + 1):04d}.png'
                    )
                )

if __name__ == "__main__":
    train()