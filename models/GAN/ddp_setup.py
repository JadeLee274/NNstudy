import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

     
def setup(
    rank: int,
    world_size: int,
) -> None:
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        backend="nccl",
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank,
     )
