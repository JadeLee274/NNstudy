import sys
import os
import cv2
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset
from typing import *

import numpy as np

Tensor = np.ndarray

class NumpyData(Dataset):
    def __init__(self, 
                 root: str, 
                 train: bool,
                 transform) -> None:
        self.dataset = MNIST(root = root, 
                             train = train, 
                             download = False, 
                             transform = transform)
        self.images = self.dataset.data.numpy()
        self.labels = self.dataset.targets.numpy()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, 
                    idx: int) -> Tuple[Tensor, Tensor]:
        image, label = self.images[idx], self.labels[idx]
        return image, label