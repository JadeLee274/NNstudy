import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from typing import *
Tensor = torch.Tensor


class TensorDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            transform: Any = None,
    ) -> None:
        super().__init__()
        self.dataset = [
            os.path.join(data_root, file) for file in os.listdir(data_root)
        ]
        self.transform = transform

    def __len__(
            self,
    ) -> int:
        return len(self.dataset)
    
    def __getitem__(
            self,
            idx: int
    ) -> Tensor:
        data_path = self.dataset[idx]
        data = Image.open(data_path).convert('RGB')
        if self.transform:
            data = self.transform(data)
        return data
