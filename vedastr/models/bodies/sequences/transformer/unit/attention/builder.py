from vedastr.utils import build_from_cfg
from .registry import TRANSFORMER_ATTENTIONS


def build_attention(cfg, default_args=None):
    attention = build_from_cfg(cfg, TRANSFORMER_ATTENTIONS, default_args)

    return attention
