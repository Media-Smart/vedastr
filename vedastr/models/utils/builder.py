import torch.nn as nn

from vedastr.utils import build_from_cfg
from .registry import UTILS


def build_module(cfg, default_args=None):
    util = build_from_cfg(cfg, UTILS, default_args)
    return util


def build_torch_nn(cfg, default_args=None):
    module = build_from_cfg(cfg, nn, default_args, 'module')
    return module
