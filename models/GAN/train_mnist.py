import os
import argparse
import torch
import tensorboard
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from gan import *
from ddp_setup import *
ROOT = '/data/home/tmdals274/NNstudy/data'
OUTPUT_DIR = './output_mnist'


parser = argparse.ArgumentParser()

parser.add_argument(
    "--master_addr",
    type=str,
    required=True,
)
parser.add_argument(
    "--master-port",
    type=str,
    required=True,
)
parser.add_argument(
    "--backend",
    type=str,
    default='nccl'
)
parser.add_argument(
    "--latent_dim",
    type=int,
    default=128,
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=2e-3,
)
parser.add_argument(
    "--epochs",
    type=int,
    default=25,
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=10,
)
args = parser.parse_args()


def train(
    rank: int,
    world_size: int,
) -> None:
    setup(rank, world_size)

    sampler = DistributedSampler(
        dataset=mnist_set,
        num_replicas=world_size,
        rank=rank,
    )

    mnist_set = MNIST(
        root=ROOT,
        download=False,
        train=True,
        transform=T.ToTensor(),
    )

    dataloader = DataLoader(
        dataset=mnist_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    Gen = Generator(
        out_channels=1,
        latent_dim=args.latent_dim,
        num_doublesize=5,
        n_filters=64,
    )

    Disc = Discriminator(
        in_channels=1,
        num_halfsize=5,
        num_filters=64,
    )

    Gen = Gen.to(rank)
    Gen = DDP(Gen, device_ids=[rank])
    Disc = Disc.to(rank)
    Disc = DDP(Disc, device_ids=[rank])
    
    criterion = nn.BCELoss().to(rank)

    optimizer_g = optim.Adam(
        params=Gen.parameters(),
        lr=args.learning_rate,
    )
    
    optimizer_d = optim.Adam(
        params=Disc.parameters(),
        lr=args.learning_rate,
    )

    real_label = 1
    fake_label = 0

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, (data, _) in enumerate(dataloader):
            optimizer_d.zero_grad()
            real_data = data.to(rank)
            batch_size = real_data.size(0)
            label = torch.full(
                size=(batch_size,),
                fill_value=real_label,
                device=rank,
            )
            output = Disc(real_data)
            disc_err_real = criterion(output, label)
            disc_err_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=rank)
            fake_data = Gen(noise)
            label.fill_(fake_label)
            output = Disc(fake_data.detach())
            disc_err_fake = criterion(output, label)
            disc_err_fake.backward()
            D_G_Z1 = output.mean().item()

            disc_err = disc_err_real + disc_err_fake
            optimizer_d.step()

            optimizer_g.zero_grad()
            label.fill_(real_label)
            output = Disc(fake_data)
            gen_err = criterion(output, label)
            gen_err.backward()
            D_G_Z2 = output.mean().item()
            optimizer_g.step()

            print(
                f"Epoch {epoch + 1} [{i + 1}/{len(dataloader)}]: " \
                f"D_Loss = {disc_err.item():.4e} | " \
                f"G_Loss = {gen_err.item():.4e} | D(x) = {D_x:.4e} | " \
                f"D(G(z)) = {D_G_Z1:.4e} / {D_G_Z2:.4e}"
            )

            if i % 100 == 0:
                real_save_path = os.path.join(
                    OUTPUT_DIR, f'{epoch + 1:03d}_iter{i + 1}_real.png'
                )
                fake_save_path = os.path.join(
                    OUTPUT_DIR, f'{epoch + 1:03d}_iter{i + 1}_fake.png'
                )
                save_image(
                    tensor=real_data,
                    fp=real_save_path,
                )
                fake = Gen(noise)
                save_image(
                    tensor=fake.detach(),
                    fp=fake_save_path,
                )
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    'gen': Gen.state_dict(),
                    'disc': Disc.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                },
                f'./model_save/mnist_gan/gan_epoch_{epoch + 1}.pt'
            )
    
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        fn=train,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
