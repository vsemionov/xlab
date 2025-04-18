# Copyright 2025 Victor Semionov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# references:
# https://arxiv.org/abs/1706.03762
# https://peterbloem.nl/blog/transformers
# https://github.com/karpathy/minGPT
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
# https://nlp.seas.harvard.edu/annotated-transformer/

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils import checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        assert d_model % 2 == 0
        super().__init__()
        pos = torch.arange(max_len).float()
        i = torch.arange(d_model // 2)
        den = 10_000 ** (2 * i / d_model)
        p_i = pos.unsqueeze(1) / den
        enc = torch.empty(max_len, d_model)
        enc[:, 0::2] = torch.sin(p_i)
        enc[:, 1::2] = torch.cos(p_i)
        self.register_buffer('enc', enc, persistent=False)

    def forward(self, n):
        return self.enc[:n]


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        idx = torch.arange(max_len)
        self.register_buffer('idx', idx, persistent=False)

    def forward(self, n):
        return self.emb(self.idx[:n])


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation, dropout=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.seq(x)


class MultiHeadSelfAttention(nn.Module):
    invert_mask = True

    def __init__(self, d_model, n_heads, dropout=0.1):
        assert d_model % n_heads == 0
        super().__init__()
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def sdpa(self, q, k, v, mask=None, is_causal=False):
        a = q @ k.transpose(2, 3) / math.sqrt(q.size(-1))  # bhnn
        if mask is not None:
            mask = mask.unsqueeze(1)  # b1nn
            a = a.masked_fill_(mask, float('-inf'))
        a = a.softmax(dim=3)  # bhnn
        a = self.dropout(a)

        y = a @ v  # bhns

        return y

    def forward(self, x, mask=None, is_causal=False):
        b, n, d = x.size()
        h = self.n_heads
        s = d // h

        q, k, v = self.qkv_proj(x).split(d, dim=2)  # bnd
        q, k, v = [z.view(b, n, h, s).transpose(1, 2) for z in (q, k, v)]  # bhns

        y = self.sdpa(q, k, v, mask=mask, is_causal=is_causal)  # bhns

        y = y.transpose(1, 2).reshape(b, n, d)  # bnd
        y = self.out_proj(y)  # bnd

        return y


class CheckpointMultiHeadSelfAttention(MultiHeadSelfAttention):
    def sdpa(self, q, k, v, mask=None, is_causal=False):
        return checkpoint.checkpoint(super().sdpa, q, k, v, mask=mask, is_causal=is_causal, use_reentrant=False)


class FlashMultiHeadSelfAttention(MultiHeadSelfAttention):
    invert_mask = False

    def sdpa(self, q, k, v, mask=None, is_causal=False):
        mask = None if is_causal else mask  # flash attention is faster with implicit causal masks
        dropout_p = self.dropout.p if self.training else 0
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, prenorm=False, norm=nn.LayerNorm,
            attention=MultiHeadSelfAttention, activation=nn.ReLU(),
            dropout=0.1, block_drop=True, attn_drop=True, ff_drop=True):
        super().__init__()
        self.prenorm = prenorm
        self.mhsa = attention(d_model, n_heads, dropout=(dropout * attn_drop))
        self.dropout1 = nn.Dropout(dropout * block_drop)
        self.norm1 = norm(d_model)
        self.ff = FeedForward(d_model, d_ff, activation=activation, dropout=(dropout * ff_drop))
        self.dropout2 = nn.Dropout(dropout * block_drop)
        self.norm2 = norm(d_model)

    @property
    def invert_mask(self):
        return self.mhsa.invert_mask

    def forward(self, x, mask=None, is_causal=False):
        if self.prenorm:
            x = x + self.dropout1(self.mhsa(self.norm1(x), mask=mask, is_causal=is_causal))
            x = x + self.dropout2(self.ff(self.norm2(x)))

        else:
            x = self.norm1(x + self.dropout1(self.mhsa(x, mask=mask, is_causal=is_causal)))
            x = self.norm2(x + self.dropout2(self.ff(x)))

        return x


class ParameterInit:
    def reset_parameters(self: nn.Module):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)


class TransformerDecoder(nn.Module, ParameterInit):
    def __init__(self, max_len, d_model, n_layers, n_heads, d_ff, postnorm=False, norm=nn.LayerNorm,
            causal=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff, norm=norm, **kwargs)
            for _ in range(n_layers)])
        self.norm = norm(d_model) if postnorm else None
        self.merge_op = torch.logical_or if self.invert_mask else torch.logical_and
        causal_mask = self._create_causal_mask(max_len) if causal else None
        self.register_buffer('causal_mask', causal_mask, persistent=False)
        self.reset_parameters()

    @property
    def invert_mask(self):
        return self.layers[0].invert_mask

    def _create_causal_mask(self, max_len):
        ones = torch.ones(max_len, max_len, dtype=torch.bool)
        return ones.triu(diagonal=1) if self.invert_mask else ones.tril(diagonal=0)

    def forward(self, x, seq_mask=None):
        mask, is_causal = self._merge_masks(self.causal_mask, seq_mask, x.size(1))  # bnn
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def _merge_masks(self, causal_mask, seq_mask, seq_len):
        # causal_mask: NN
        # seq_mask: bn
        if causal_mask is None and seq_mask is None:
            return None, False
        if causal_mask is not None:
            causal_mask = causal_mask[:seq_len, :seq_len].unsqueeze(0)  # 1nn
            if seq_mask is None:
                return causal_mask, True
        if seq_mask is not None:
            seq_mask = seq_mask.unsqueeze(1)  # b1n
            if causal_mask is None:
                return seq_mask, False
        return self.merge_op(causal_mask, seq_mask), False


class PyTorchTransformerDecoder(nn.TransformerEncoder, ParameterInit):
    invert_mask = True

    def __init__(self, max_len, d_model, n_layers, n_heads, d_ff, prenorm=False, postnorm=False, norm=nn.LayerNorm,
            causal=False, dropout=0.1, **kwargs):
        kwargs = dict(dropout=dropout, norm_first=prenorm, batch_first=True, **kwargs)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, **kwargs)
        norm = norm(d_model) if postnorm else None
        super().__init__(layer, n_layers, norm=norm)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_len) if causal else None
        self.register_buffer('causal_mask', causal_mask)
        self.reset_parameters()

    def forward(self, x, seq_mask=None):  # noqa
        n = x.size(1)
        mask = self.causal_mask[:n, :n] if self.causal_mask is not None else None
        return super().forward(x, mask=mask, src_key_padding_mask=seq_mask, is_causal=(mask is not None))


class Transformer(nn.Module):
    def __init__(self, position=PositionalEncoding, decoder=TransformerDecoder,
            max_len=128, d_model=128, n_layers=2, n_heads=2, d_ff=256, dropout=0.1, pos_drop=True, **kwargs):
        super().__init__()
        self.position = position(max_len, d_model)
        self.dropout = nn.Dropout(dropout * pos_drop)
        self.decoder = decoder(max_len, d_model, n_layers, n_heads, d_ff, dropout=dropout, **kwargs)

    @property
    def invert_mask(self):
        return self.decoder.invert_mask

    def forward(self, x, seq_mask=None):
        x = x + self.position(x.size(-2)).unsqueeze(0)
        x = self.dropout(x)
        x = self.decoder(x, seq_mask=seq_mask)
        return x


class GenerativeTextTransformer(nn.Module):
    def __init__(self, n_vocab, max_len, d_model, pad_index=None, pad_mask=True, **kwargs):
        super().__init__()
        self.pad_index = pad_index if pad_mask else None
        self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_index)
        self.transformer = Transformer(max_len=max_len, d_model=d_model, causal=True, **kwargs)
        self.pad_cmp = torch.eq if self.transformer.invert_mask else torch.ne
        self.head = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        pad_mask = self.pad_cmp(x, self.pad_index) if self.pad_index is not None else None
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.transformer(x, seq_mask=pad_mask)
        y = self.head(x)
        return y
