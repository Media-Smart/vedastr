from vedastr.utils import build_from_cfg
from .registry import TRANSFORMER_ENCODER_LAYERS, TRANSFORMER_DECODER_LAYERS


def build_encoder_layer(cfg, default_args=None):
    encoder_layer = build_from_cfg(cfg, TRANSFORMER_ENCODER_LAYERS, default_args)

    return encoder_layer


def build_decoder_layer(cfg, default_args=None):
    decoder_layer = build_from_cfg(cfg, TRANSFORMER_DECODER_LAYERS, default_args)

    return decoder_layer
