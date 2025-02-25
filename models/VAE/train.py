import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from vae import VAE
from tensordataset import TensorDataset
from typing import *
DATA_ROOT = "/data/home/tmdals274/NNstudy/data/celeba/img_align_celeba"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--out_channels_list",
        type=list,
        default=[16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=args.in_channels,
        out_channels_list=args.out_channels_list,
        hidden_dim=args.hidden_dim,
        input_size=args.input_size,
    )
    model = model.to(device)

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-04,
    )

    dataset=TensorDataset(
        data_root=args.data_root,
        transform=T.Compose(
            [
                T.Resize(
                    size=args.input_size,
                ),
                T.CenterCrop(
                    size=args.input_size,
                ),
                T.ToTensor()
            ]
        )
    )

    train_set, val_set = random_split(
        dataset=dataset,
        lengths=[0.9, 0.1],
        generator=torch.manual_seed(42),
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=64,
        shuffle=False,
    )

    if args.checkpoint != 0:
        checkpoint = torch.load(
            f"./model_save/vae_epoch_{args.epoch}.pt"
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Starting Training Loop from epoch {args.checkpoint}...")

    else:
        print("Starting Training Loop...")

    for epoch in range(args.checkpoint, args.epochs):
        model.train()
        train_loss, recon_loss, kld_loss = 0., 0., 0.
        for datas in train_loader:
            optimizer.zero_grad()
            datas = datas.to(device)
            datas_hat, datas, mu, log_var = model(datas)

            loss = model.loss_fn(
                x_hat=datas_hat,
                x=datas,
                mu=mu,
                log_var=log_var,
            )['Loss']

            recon = model.loss_fn(
                x_hat=datas_hat,
                x=datas,
                mu=mu,
                log_var=log_var,
            )['Reconstruction Loss']

            kld = model.loss_fn(
                x_hat=datas_hat,
                x=datas,
                mu=mu,
                log_var=log_var,
            )['KLD Loss']

            loss.backward()
            optimizer.step()
            train_loss += loss
            recon_loss += recon
            kld_loss += kld

        print(f"Epoch {epoch + 1}: Sum of Train Loss = {train_loss:.4e}")
        print(f"Epoch {epoch + 1}: Sum of Recon Loss = {recon_loss:.4e}")
        print(f"Epoch {epoch + 1}: Sum of KLD Loss = {kld_loss:.4e}")

        if (epoch + 1) % args.save_interval == 0:
            model.eval()
            val_loss, val_recon_loss, val_kld_loss = 0., 0., 0.
            for datas in val_loader:
                datas = datas.to(device)
                with torch.no_grad():
                    datas_hat, datas, mu, log_var = model(datas)

                    loss = model.loss_fn(
                        x_hat=datas_hat,
                        x=datas,
                        mu=mu,
                        log_var=log_var,
                    )['Loss']

                    recon = model.loss_fn(
                        x_hat=datas_hat,
                        x=datas,
                        mu=mu,
                        log_var=log_var,
                    )['Reconstruction Loss']

                    kld = model.loss_fn(
                        x_hat=datas_hat,
                        x=datas,
                        mu=mu,
                        log_var=log_var,
                    )['KLD Loss']

                    val_loss += loss
                    val_recon_loss += recon
                    val_kld_loss += kld

            print(f"Sum of Val Loss = {val_loss:.4e}")
            print(f"Sum of Val Recon Loss = {val_recon_loss:.4e}")
            print(f"Sum of Val KLD Loss = {val_kld_loss:.4e}")

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                f'./model_save/vae_epoch_{epoch + 1}.pt',
            )
            model.train()
