import h5py
import numpy as np
from torch.utils.data import Dataset
from typing import *

class TrainDataset(Dataset):
    def __init__(self, h5_file: str) -> None:
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __len__(self) -> int:
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(a = f['lr'][idx] / 255., axis = 0), np.expand_dims(a = f['hr'][idx] / 255., axis = 0)


class EvalDataset(Dataset):
    def __init__(self, h5_file: str) -> None:
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
    
    def __len__(self) -> int:
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(a = f['lr'][str(idx)][:, :] / 255., axis = 0), np.expand_dims(a = f['hr'][str(idx)][:, :] / 255., axis = 0)