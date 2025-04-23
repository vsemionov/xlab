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

import math
from typing import Union

import torch.optim.lr_scheduler as lr_scheduler


class CosineWarmupScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, max_iters: Union[int, float], warmup_iters: Union[int, float] = 0.1):
        super().__init__(optimizer)
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters if isinstance(warmup_iters, int) else int(warmup_iters * max_iters)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch = min(epoch, self.max_iters - 1)
        if epoch < self.warmup_iters:
            return (epoch + 1) / (self.warmup_iters + 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)))
