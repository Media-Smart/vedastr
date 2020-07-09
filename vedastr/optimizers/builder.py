import torch.optim as torch_optim

from vedastr.utils import build_from_cfg


def build_optimizer(cfg, default_args=None):
    optim = build_from_cfg(cfg, torch_optim, default_args, 'module')

    return optim
