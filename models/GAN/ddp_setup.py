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
    master_addr: str,
    master_port: str,
) -> None:
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )


def cleanup() -> None:
    dist.destroy_process_group()
