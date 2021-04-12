from vedastr.utils import build_from_cfg
from .registry import SEQUENCE_DECODERS, SEQUENCE_ENCODERS


def build_sequence_encoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_ENCODERS, default_args)

    return sequence_encoder


def build_sequence_decoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_DECODERS, default_args)

    return sequence_encoder
