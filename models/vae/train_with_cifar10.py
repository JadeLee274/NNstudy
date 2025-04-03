import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
from torch.optim import lr_scheduler as scheduler
import matplotlib.pyplot as plt
from typing import *
from vae import *
from tensordataset import TensorDataset
DATA_ROOT = '/data/home/tmdals274/NNstudy/data'
OUTPUT_DIR = './outputs_cifar10_2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--out-channels-list",
        type=list,
        default=[64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--sched-gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--sched-epoch",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-lr-rate",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--kld-weight",
        type=float,
        default=2e-4
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(DATA_ROOT, 'cifar10'),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=args.in_channels,
        out_channels_list=args.out_channels_list,
        hidden_dim=args.hidden_dim,
        input_size=args.input_size,
        kld_weight=args.kld_weight
    )
    model = model.to(device)

    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
    )

    MIN_LR = args.learning_rate * args.min_lr_rate

    def lr_lambda(epoch: int) -> float:
        lr = args.sched_gamma ** (epoch // args.sched_epoch)
        lr = max(lr, MIN_LR)
        return lr

    sched = scheduler.LambdaLR(
        optimizer=optim,
        lr_lambda=lr_lambda,
    )

    dataset=TensorDataset(
        data_root=args.data_root,
        transform=T.Compose(
            [
                T.Resize(
                    size=args.input_size,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.247, 0.243, 0.261),
                )
            ]
        )
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if args.checkpoint != 0:
        checkpoint = torch.load(
            f"./model_save/cifar10_vae/vae_epoch_{args.checkpoint}.pt"
        )
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        sched.load_state_dict(checkpoint['sched'])
        print(f"Starting Training Loop from epoch {args.checkpoint}...")

    else:
        print("Starting Training Loop...")

    for epoch in range(args.checkpoint, args.epochs):
        model.train()
        avg_train_loss, avg_recon_loss, avg_kld_loss = 0, 0, 0
        for datas in dataloader:
            optim.zero_grad()
            datas = datas.to(device)
            datas_hat, datas, mu, log_var = model(datas)

            loss_dict = model.loss_fn(
                x_hat=datas_hat,
                x=datas,
                mu=mu,
                log_var=log_var,
            )
            loss = loss_dict['Loss']
            recon_loss = loss_dict['Reconstruction Loss']
            kld_loss = loss_dict['KLD Loss']

            loss.backward()
            optim.step()
            avg_train_loss += loss
            avg_recon_loss += recon_loss
            avg_kld_loss += kld_loss

        avg_train_loss /= len(dataloader)
        avg_recon_loss /= len(dataloader)
        avg_kld_loss /= len(dataloader)
        sched.step()

        print(f"Epoch {epoch + 1:03d}: Train = {avg_train_loss:.4e}")
        print(f"Epoch {epoch + 1:03d}: Recon = {avg_recon_loss:.4e}")
        print(f"Epoch {epoch + 1:03d}: KLD   = {avg_kld_loss:.4e}")
        print('=' * 50)

        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                },
                f'./model_save/cifar10_vae/vae_epoch_{epoch + 1}.pt',
            )

            with torch.no_grad():
                model.eval()
                for img in dataloader:
                    img = img.to(device)
                    recon = model(img)[0]
                    sample = model.sample(img.shape[0], device)
                    recon_path = os.path.join(
                        OUTPUT_DIR,
                        f'epoch_{epoch + 1:03d}_recon.png',
                    )
                    sample_path = os.path.join(
                        OUTPUT_DIR,
                        f'epoch_{epoch + 1:03d}_sample.png',
                    )
                    save_image(recon, recon_path)
                    save_image(sample, sample_path)
                    break
            