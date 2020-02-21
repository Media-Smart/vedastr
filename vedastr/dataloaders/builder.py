from .registry import DATALOADERS

from vedastr.utils import build_from_cfg


def build_dataloader(cfg, default_args=None):
    dataloader = build_from_cfg(cfg, DATALOADERS, default_args)
    return dataloader
