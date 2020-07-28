import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from .attention import build_attention
from .feedforward import build_feedforward
from .registry import TRANSFORMER_DECODER_LAYERS


@TRANSFORMER_DECODER_LAYERS.register_module
class TransformerDecoderLayer1D(nn.Module):
    def __init__(self,
                 self_attention,
                 self_attention_norm,
                 attention,
                 attention_norm,
                 feedforward,
                 feedforward_norm):
        super(TransformerDecoderLayer1D, self).__init__()

        self.self_attention = build_attention(self_attention)
        self.self_attention_norm = build_torch_nn(self_attention_norm)

        self.attention = build_attention(attention)
        self.attention_norm = build_torch_nn(attention_norm)

        self.feedforward = build_feedforward(feedforward)
        self.feedforward_norm = build_torch_nn(feedforward_norm)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        attn1, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        out1 = self.self_attention_norm(tgt + attn1)

        size = src.size()
        if len(size) == 4:
            b, c, h, w = size
            src = src.view(b, c, h * w).transpose(1, 2)
            if src_mask is not None:
                src_mask = src_mask.view(b, 1, h * w)

        attn2, _ = self.attention(out1, src, src, src_mask)
        out2 = self.attention_norm(out1 + attn2)

        ffn_out = self.feedforward(out2)
        out3 = self.feedforward_norm(out2 + ffn_out)

        return out3
