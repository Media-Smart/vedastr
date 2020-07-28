import math
import copy

import torch.nn as nn
import torch
from torch.nn import functional as F

from vedastr.models.utils import build_module, ConvModule
from .registry import BRICKS


@BRICKS.register_module
class JunctionBlock(nn.Module):
    """JunctionBlock

    Args:
    """

    def __init__(self, top_down, lateral, post, to_layer, fusion_method=None):
        super(JunctionBlock, self).__init__()

        self.from_layer = {}

        self.to_layer = to_layer
        top_down_ = copy.copy(top_down)
        lateral_ = copy.copy(lateral)
        self.fusion_method = fusion_method

        self.top_down_block = []
        if top_down_:
            self.from_layer['top_down'] = top_down_.pop('from_layer')
            if 'trans' in top_down_:
                self.top_down_block.append(build_module(top_down_['trans']))
            if 'upsample' in top_down_:
                self.top_down_block.append(build_module(top_down_['upsample']))
        self.top_down_block = nn.Sequential(*self.top_down_block)

        if lateral_:
            self.from_layer['lateral'] = lateral_.pop('from_layer')
            if lateral_:
                self.lateral_block = build_module(lateral_)
            else:
                self.lateral_block = nn.Sequential()
        else:
            self.lateral_block = nn.Sequential()

        if post:
            self.post_block = build_module(post)
        else:
            self.post_block = nn.Sequential()

    def forward(self, top_down=None, lateral=None):

        if top_down is not None:
            top_down = self.top_down_block(top_down)
        if lateral is not None:
            lateral = self.lateral_block(lateral)

        if top_down is not None:
            if lateral is not None:
                assert self.fusion_method in ('concat', 'add')
                if self.fusion_method == 'concat':
                    feat = torch.cat([top_down, lateral], 1)
                elif self.fusion_method == 'add':
                    feat = top_down + lateral
            else:
                assert self.fusion_method is None
                feat = top_down
        else:
            assert self.fusion_method is None
            if lateral is not None:
                feat = lateral
            else:
                raise ValueError('There is neither top down feature nor lateral feature')

        feat = self.post_block(feat)

        return feat


@BRICKS.register_module
class FusionBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self,
                 method,
                 from_layers,
                 feat_strides,
                 in_channels_list,
                 out_channels_list,
                 upsample,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 activation='relu',
                 inplace=True,
                 common_stride=4,
                 ):
        super(FusionBlock, self).__init__()

        assert method in ('add', 'concat')
        self.method = method
        self.from_layers = from_layers

        assert len(in_channels_list) == len(out_channels_list)

        self.blocks = nn.ModuleList()
        for idx in range(len(from_layers)):
            in_channels = in_channels_list[idx]
            out_channels = out_channels_list[idx]
            feat_stride = feat_strides[idx]
            ups_num = int(max(1, math.log2(feat_stride) - math.log2(common_stride)))
            head_ops = []
            for idx2 in range(ups_num):
                cur_in_channels = in_channels if idx2 == 0 else out_channels
                conv = ConvModule(
                    cur_in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=activation,
                    inplace=inplace,
                )
                head_ops.append(conv)
                if int(feat_stride) != int(common_stride):
                    head_ops.append(build_module(upsample))
            self.blocks.append(nn.Sequential(*head_ops))

    def forward(self, feats):
        outs = []
        for idx, key in enumerate(self.from_layers):
            block = self.blocks[idx]
            feat = feats[key]
            out = block(feat)
            outs.append(out)
        if self.method == 'add':
            res = torch.stack(outs, 0).sum(0)
        else:
            res = torch.cat(outs, 1)
        return res


@BRICKS.register_module
class CollectBlock(nn.Module):
    """CollectBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super(CollectBlock, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):

        if self.to_layer is None:
            if isinstance(self.from_layer, str):
                return feats[self.from_layer]
            elif isinstance(self.from_layer, list):
                return {f_layer: feats[f_layer] for f_layer in self.from_layer}
        else:
            if isinstance(self.from_layer, str):
                feats[self.to_layer] = feats[self.from_layer]
            elif isinstance(self.from_layer, list):
                feats[self.to_layer] = {f_layer: feats[f_layer] for f_layer in self.from_layer}


@BRICKS.register_module
class CellAttentionBlock(nn.Module):
    def __init__(self, feat, hidden, fusion_method='add', post=None, post_activation='softmax'):
        super(CellAttentionBlock, self).__init__()

        feat_ = feat.copy()
        self.feat_from = feat_.pop('from_layer')
        self.feat_block = build_module(feat_)
        self.hidden_block = build_module(hidden)

        self.fusion_method = fusion_method
        self.activate = post_activation

        if post is not None:
            self.post_block = build_module(post)
        else:
            self.post_block = nn.Sequential()

    def forward(self, feats, hidden):
        feat = feats[self.feat_from]
        b, c = feat.size(0), feat.size(1)
        feat_to_attend = feat.view(b, c, -1)

        x = self.feat_block(feat)
        y = self.hidden_block(hidden)

        assert self.fusion_method in ['add', 'dot']
        if self.fusion_method == 'add':
            attention_map = x + y
        elif self.fusion_method == 'dot':
            attention_map = x * y

        attention_map = self.post_block(attention_map)
        b, c = attention_map.size(0), attention_map.size(1)
        attention_map = attention_map.view(b, c, -1)

        assert self.activate in ['softmax', 'sigmoid']
        if self.activate == 'softmax':
            attention_map = F.softmax(attention_map, dim=2)
        elif self.activate == 'sigmoid':
            attention_map = F.sigmoid(attention_map)

        feat = feat_to_attend * attention_map
        feat = feat.sum(2)

        return feat
