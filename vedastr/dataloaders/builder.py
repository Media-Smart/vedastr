import torch.utils.data as tud

from vedastr.utils import build_from_cfg
from .registry import DATALOADERS


def build_dataloader(cfg, default_args=None):
    try:
        dataloader = build_from_cfg(cfg, tud, default_args, src='module')
    except:
        dataloader = build_from_cfg(cfg, DATALOADERS, default_args)

    return dataloader
