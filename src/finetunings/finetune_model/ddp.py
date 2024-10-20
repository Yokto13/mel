import os

import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("ncll", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
