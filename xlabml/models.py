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

from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm

from . import transformers


class XLabModule(L.LightningModule):
    """XLab model base"""
    def __init__(
            self,
            n_vocab, max_len, d_model, pad_index: Optional[int],
            debug: bool = False, dummy: bool = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = transformers.GenerativeTextTransformer(
            n_vocab, max_len, d_model, pad_index=pad_index, **kwargs
        ) if not dummy else nn.Linear(2, 2)
        self.pad_index = pad_index
        self.debug = debug
        if self.debug:
            self.example_input_array = torch.zeros((1, max_len), dtype=torch.long)

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def loss(self, logits, targets):
        ignore_index = self.pad_index if self.pad_index is not None else -1
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)

    def configure_optimizers(self):
        # optimizers and lr schedulers are defined in configuration, this is only a default
        return optim.Adam(self.parameters(), lr=3e-4)

    def _accuracy(self, logits: torch.Tensor, targets: torch.Tensor):
        indices = logits.argmax(dim=2)
        if self.pad_index is not None:
            mask = targets != self.pad_index
            indices = indices[mask]
            targets = targets[mask]
        correct = indices == targets
        # ignore division by zero, e.g. targets must contain non-ignored elements
        return correct.sum() / correct.numel()

    def _step(self, batch, name, sync_dist=False):
        x, targets = batch[:2]
        mask = batch[2] if len(batch) > 2 else None
        logits = self(x, mask=mask)
        loss = self.loss(logits, targets)
        accuracy = self._accuracy(logits.detach(), targets)
        log_data = {f'{name}_loss': loss, f'{name}_accuracy': accuracy}
        self.log_dict(log_data, prog_bar=True, sync_dist=sync_dist)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val', sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test', sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def on_before_backward(self, loss):
        if self.debug and torch.isnan(loss) or torch.isinf(loss):
            warnings.warn('Loss is NaN or Inf')

    def on_after_backward(self):
        if self.debug \
                and isinstance(plugin := self.trainer.precision_plugin, MixedPrecision) \
                and (scaler := plugin.scaler) is not None:
            scale = scaler.get_scale()
            self.log('grad_scale', scale)

    def on_before_optimizer_step(self, optimizer):
        if self.debug:
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)


class XLabModel(XLabModule):
    """XLab model"""
    def __init__(
            self,
            n_vocab: int, max_len: int = 128, d_model: int = 128, pad_index: Optional[int] = None,
            position: type[nn.Module] = transformers.PositionalEncoding,
            n_layers: int = 2, n_heads: int = 2, d_ff: int = 256, dropout: float = 0.1, pos_drop: bool = True,
            prenorm: bool = False, postnorm: bool = False, norm: type[nn.Module] = nn.LayerNorm,
            attention: type[nn.Module] = transformers.MultiHeadSelfAttention, activation: nn.Module = nn.ReLU(),
            block_drop: bool = True, attn_drop: bool = True, ff_drop: bool = True,
            debug: bool = False, dummy: bool = False,
    ):
        super().__init__(
            n_vocab, max_len, d_model, pad_index=pad_index,
            position=position, decoder=transformers.TransformerDecoder,
            n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout, pos_drop=pos_drop,
            prenorm=prenorm, postnorm=postnorm, norm=norm,
            attention=attention, activation=activation,
            block_drop=block_drop, attn_drop=attn_drop, ff_drop=ff_drop,
            debug=debug, dummy=dummy,
        )


class XLabPyTorchModel(XLabModule):
    """XLab PyTorch model"""
    def __init__(
            self,
            n_vocab: int, max_len: int = 128, d_model: int = 128, pad_index: Optional[int] = None,
            position: type[nn.Module] = transformers.PositionalEncoding,
            n_layers: int = 2, n_heads: int = 2, d_ff: int = 256, dropout: float = 0.1, pos_drop: bool = True,
            prenorm: bool = False, postnorm: bool = False, norm: type[nn.Module] = nn.LayerNorm,
            activation: nn.Module = nn.ReLU(),
            debug: bool = False, dummy: bool = False,
    ):
        super().__init__(
            n_vocab, max_len, d_model, pad_index=pad_index,
            position=position, decoder=transformers.PyTorchTransformerDecoder,
            n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout, pos_drop=pos_drop,
            prenorm=prenorm, postnorm=postnorm, norm=norm,
            activation=activation,
            debug=debug, dummy=dummy,
        )
