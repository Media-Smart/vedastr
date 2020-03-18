import torch.nn as nn

from .encoders import build_encoder
from .decoders import build_brick, build_decoder


def build_feature_extractor(cfg):
    encoder = build_encoder(cfg.get('encoder'))

    if cfg.get('decoder'):
        middle = build_decoder(cfg.get('decoder'))
        if 'collect' in cfg:
            final = build_brick(cfg.get('collect'))
            feature_extractor = nn.Sequential(encoder, middle, final)
        else:
            feature_extractor = nn.Sequential(encoder, middle)
        # assert 'collect' not in cfg
    else:
        assert 'collect' in cfg
        middle = build_brick(cfg.get('collect'))
        feature_extractor = nn.Sequential(encoder, middle)

    return feature_extractor
