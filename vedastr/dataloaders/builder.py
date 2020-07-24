import torch

from .registry import DATALOADERS
from vedastr.utils import build_from_cfg


def build_dataloader(cfg, default_args=None):
    try:
        dataloader = build_from_cfg(cfg, torch.utils.data, default_args, src='module')
    except:
        dataloader = build_from_cfg(cfg, DATALOADERS, default_args)

    return dataloader
