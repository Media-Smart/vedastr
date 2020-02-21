from vedastr.utils import build_from_cfg
from .registry import DECODERS


def build_decoder(cfg, default_args=None):
    decoder = build_from_cfg(cfg, DECODERS, default_args)
    return decoder
