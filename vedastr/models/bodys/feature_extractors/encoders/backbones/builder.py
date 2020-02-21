from vedastr.utils import build_from_cfg
from .registry import BACKBONES


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)

    return backbone
