import logging
import math

import torch
import torch.nn as nn

from vedastr.models.bodies import build_sequence_decoder
from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class TransformerHead(nn.Module):
    def __init__(self,
                 decoder,
                 generator,
                 embedding,
                 num_steps,
                 pad_id,
                 src_from,
                 src_mask_from=None,
                 ):
        super(TransformerHead, self).__init__()

        self.decoder = build_sequence_decoder(decoder)
        self.generator = build_torch_nn(generator)
        self.embedding = build_torch_nn(embedding)
        self.num_steps = num_steps
        self.pad_id = pad_id
        self.src_from = src_from
        self.src_mask_from = src_mask_from

        logger.info('TransformerHead init weights')
        init_weights(self.modules())

    def pad_mask(self, text):
        pad_mask = (text == self.pad_id)
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, text):
        t = text.size(1)
        order_mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(text.device)

        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(self, feats, texts):
        src = feats[self.src_from]
        if self.src_mask_from:
            src_mask = feats[self.src_mask_from]
        else:
            src_mask = None

        if self.training:
            tgt = self.text_embedding(texts)
            tgt_mask = (self.pad_mask(texts) | self.order_mask(texts))

            out = self.decoder(tgt, src, tgt_mask, src_mask)
            out = self.generator(out)
        else:
            out = None
            for _ in range(self.num_steps):
                tgt = self.text_embedding(texts)
                tgt_mask = self.order_mask(texts)
                out = self.decoder(tgt, src, tgt_mask, src_mask)
                out = self.generator(out)
                next_text = torch.argmax(out[:, -1:, :], dim=-1)

                texts = torch.cat([texts, next_text], dim=-1)

        return out
