import torch
import torch.nn as nn

from .registry import TRANSFORMER_ATTENTIONS


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        return out, attn


@TRANSFORMER_ATTENTIONS.register_module
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_channels: int,
                 k_channels: int,
                 v_channels: int,
                 n_head: int = 8,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.in_channels = in_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.n_head = n_head

        self.q_linear = nn.Linear(in_channels, n_head * k_channels)
        self.k_linear = nn.Linear(in_channels, n_head * k_channels)
        self.v_linear = nn.Linear(in_channels, n_head * v_channels)
        self.attention = ScaledDotProductAttention(
            temperature=k_channels**0.5, dropout=dropout)
        self.out_linear = nn.Linear(n_head * v_channels, in_channels)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.q_linear(q).view(b, q_len, self.n_head,
                                  self.k_channels).transpose(1, 2)
        k = self.k_linear(k).view(b, k_len, self.n_head,
                                  self.k_channels).transpose(1, 2)
        v = self.v_linear(v).view(b, v_len, self.n_head,
                                  self.v_channels).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1,
                            2).contiguous().view(b, q_len,
                                                 self.n_head * self.v_channels)
        out = self.out_linear(out)
        out = self.dropout(out)

        return out, attn
