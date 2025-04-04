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

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from . import transformers


class XLabModule(L.LightningModule, ABC):
    model: transformers.GenerativeTextTransformer

    _show_shape_info = False

    @abstractmethod
    def __init__(self, pad_index: Optional[int]):
        super().__init__()
        self.save_hyperparameters()
        self.pad_index = pad_index

    def forward(self, x):
        return self.model(x)

    def loss(self, logits, targets):
        ignore_index = self.pad_index if self.pad_index is not None else -1
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index)

    def configure_optimizers(self):
        # optimizers and lr schedulers are defined in configuration, this is only a default
        optimizer = optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def _step(self, batch, name, sync_dist=False):
        x, targets = batch
        logits = self(x)
        loss = self.loss(logits, targets)
        correct = (logits.detach().argmax(dim=2) == targets).sum().item()
        accuracy = correct / targets.numel()
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
        y = self(x)
        return y


class XLabModel(XLabModule):
    """XLab model"""
    def __init__(
            self,
            n_vocab: int, max_len: int = 128, d_model: int = 128, pad_index: Optional[int] = None,
            position: type[nn.Module] = transformers.PositionalEncoding,
            n_layers: int = 2, n_heads: int = 2, d_ff: int = 256, dropout: float = 0.1,
            prenorm: bool = False, postnorm: bool = False, norm: type[nn.Module] = nn.LayerNorm,
            activation: nn.Module = nn.ReLU(),
            attn_drop: bool = True, ff_drop: bool = True,
    ):
        super().__init__(pad_index)
        self.model = transformers.GenerativeTextTransformer(
            n_vocab, max_len, d_model, pad_index=pad_index, pad_mask=False,
            position=position, decoder=transformers.TransformerDecoder,
            n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            prenorm=prenorm, postnorm=postnorm, norm=norm,
            activation=activation,
            attn_drop=attn_drop, ff_drop=ff_drop,
        )
        if self._show_shape_info:
            self.example_input_array = torch.zeros((1, max_len), dtype=torch.long)


class XLabPyTorchModel(XLabModule):
    """XLab PyTorch model"""
    def __init__(
            self,
            n_vocab: int, max_len: int = 128, d_model: int = 128, pad_index: Optional[int] = None,
            position: type[nn.Module] = transformers.PositionalEncoding,
            n_layers: int = 2, n_heads: int = 2, d_ff: int = 256, dropout: float = 0.1,
            prenorm: bool = False, postnorm: bool = False, norm: type[nn.Module] = nn.LayerNorm,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__(pad_index)
        self.model = transformers.GenerativeTextTransformer(
            n_vocab, max_len, d_model, pad_index=pad_index, pad_mask=False,
            position=position, decoder=transformers.PyTorchTransformerDecoder,
            n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
            prenorm=prenorm, postnorm=postnorm, norm=norm,
            activation=activation,
        )
        if self._show_shape_info:
            self.example_input_array = torch.zeros((1, max_len), dtype=torch.long)
