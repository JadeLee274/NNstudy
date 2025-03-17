import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.linalg import sqrtm
from typing import *
Matrix = np.ndarray
Tensor = torch.Tensor
Module = nn.Module

__all__ = ['calculate_fid', 'calculate_inception']


inception_model = models.inception_v3()
inception_model.fc = nn.Identity()
inception_model.eval()


def get_activations(
    images: Tensor,
    model: Module,
    batch_size: int = 1024,
) -> Matrix:
    num_images = len(images)
    pred_arr = np.zeros((num_images, 2048))

    for i in range(0, num_images, batch_size):
        batch = images[i: i + batch_size]
        batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            pred = model(batch)
            pred_arr[i: i + batch.shape[0]] = pred.cpu().numpy()
    
    return pred_arr


def calculate_fid(
    real_images: Tensor,
    fake_images: Tensor,
    model: Module,
) -> float:
    """
    calculates the FID scores of real image and fake image
    """
    act_real = get_activations(real_images, model)
    act_fake = get_activations(fake_images, model)

    mu_real, sigma_real = act_real.mean(axis=0), np.cov(
        m=act_real,
        rowvar=False,
    )
    mu_fake, sigma_fake = act_fake.mean(axis=0), np.cov(
        m=act_fake,
        rowvar=False,
    )

    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = sqrtm(sigma_real @ sigma_fake)

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)

    return fid


def calculate_inception(
    images: Tensor,
    model: Module,
    batch_size: int = 1024,
    splits: int = 10,
) -> Tuple[float, float]:
    """
    calculates the inception score (IS) of generated images
    """
    images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
    preds = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        with torch.no_grad():
            pred = F.softmax(
                input=model(batch),
                dim=1,
            ).cpu().numpy()
        preds.append(pred)

        split_scores = []
        for k in np.array_split(preds, splits):
            p_y = np.mean(k, axis=0)
            kl_div = k * (np.log(k) - np.log(p_y))
            split_scores.append(
                np.exp(
                    np.mean(
                        np.sum(kl_div, axis=1)
                    )
                )
            )

    return np.mean(split_scores), np.std(split_scores)