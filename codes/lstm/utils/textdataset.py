import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from typing import *
Tensor = torch.Tensor


class TextDataset(Dataset):
    def __init__(
            self,
            texts: List[int],
            labels: List[int],
    ) -> None:
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(
            self,
    ) -> int:
        return len(self.texts)
    
    def __getitem__(
            self,
            idx: int,
    ) -> Tuple[List[int], List[int]]:
        return self.texts[idx], self.texts[idx]