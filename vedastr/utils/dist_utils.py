# adapted from mmcv and mmdetection

import functools
import os
import torch
import torch.distributed as dist


def init_dist_pytorch(backend='nccl', **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def reduce_tensor(data, average=True):
    rank, world_size = get_dist_info()
    if world_size < 2:
        return data

    with torch.no_grad():
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).cuda()
        dist.reduce(data, dst=0)
        if rank == 0 and average:
            data /= world_size
    return data


def gather_tensor(data):
    _, world_size = get_dist_info()
    if world_size < 2:
        return data

    with torch.no_grad():
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data).cuda()

        gather_list = [torch.ones_like(data) for _ in range(world_size)]
        dist.all_gather(gather_list, data)
        gather_data = torch.cat(gather_list, 0)

    return gather_data


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def master_only(func):
    """Don't use master_only to decorate function which have random state.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
