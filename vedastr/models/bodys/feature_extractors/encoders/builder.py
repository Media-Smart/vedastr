import torch.nn as nn

from .backbones import build_backbone
from .enhance_modules import build_enhance_module


def build_encoder(cfg, default_args=None):
    backbone = build_backbone(cfg['backbone'])

    enhance_cfg = cfg.get('enhance')
    if enhance_cfg:
        enhance_module = build_enhance_module(enhance_cfg)
        encoder = nn.Sequential(backbone, enhance_module)
    else:
        encoder = backbone

    return encoder
