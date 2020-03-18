import logging

import torch
import torch.nn as nn

from vedastr.models.bodies import build_brick, build_sequence_decoder
from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class AttHead(nn.Module):
    def __init__(self,
                 cell,
                 generator,
                 num_steps,
                 num_class,
                 input_attention_block=None,
                 output_attention_block=None,
                 text_transform=None,
                 holistic_input_from=None):
        super(AttHead, self).__init__()
        # from vedastr import utils
        # utils.set_random_seed(1)
        if input_attention_block is not None:
            self.input_attention_block = build_brick(input_attention_block)

        self.cell = build_sequence_decoder(cell)
        self.generator = build_torch_nn(generator)
        self.num_steps = num_steps
        self.num_class = num_class

        if output_attention_block is not None:
            self.output_attention_block = build_brick(output_attention_block)

        if text_transform is not None:
            self.text_transform = build_torch_nn(text_transform)

        if holistic_input_from is not None:
            self.holistic_input_from = holistic_input_from

        self.register_buffer('embeddings', torch.diag(torch.ones(self.num_class)))
        logger.info('AttHead init weights')
        init_weights(self.modules())

    @property
    def with_holistic_input(self):
        return hasattr(self, 'holistic_input_from') and self.holistic_input_from

    @property
    def with_input_attention(self):
        return hasattr(self, 'input_attention_block') and self.input_attention_block is not None

    @property
    def with_output_attention(self):
        return hasattr(self, 'output_attention_block') and self.output_attention_block is not None

    @property
    def with_text_transform(self):
        return hasattr(self, 'text_transform') and self.text_transform

    def forward(self, feats, texts):
        batch_size = texts.size(0)

        hidden = self.cell.init_hidden(batch_size, device=texts.device)
        if self.with_holistic_input:
            holistic_input = feats[self.holistic_input_from][:, :, 0, -1]
            hidden = self.cell(holistic_input, hidden)

        out = []

        if self.training:
            use_gt = True
            assert self.num_steps == texts.size(1)
        else:
            use_gt = False
            assert texts.size(1) == 1

        for i in range(self.num_steps):
            if i == 0:
                indexes = texts[:, i]
            else:
                if use_gt:
                    indexes = texts[:, i]
                else:
                    _, indexes = out[-1].max(1)
            text_feat = self.embeddings.index_select(0, indexes)

            if self.with_text_transform:
                text_feat = self.text_transform(text_feat)

            if self.with_input_attention:
                attention_feat = self.input_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                cell_input = torch.cat([attention_feat, text_feat], dim=1)
            else:
                cell_input = text_feat
            hidden = self.cell(cell_input, hidden)
            out_feat = self.cell.get_output(hidden)

            if self.with_output_attention:
                attention_feat = self.output_attention_block(feats, self.cell.get_output(hidden).unsqueeze(-1).unsqueeze(-1))
                out_feat = torch.cat([self.cell.get_output(hidden), attention_feat], dim=1)

            out.append(self.generator(out_feat))

        out = torch.stack(out, dim=1)

        return out
