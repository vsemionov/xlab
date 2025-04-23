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

import sys
import math
from typing import Union

import torch.optim.lr_scheduler as lr_scheduler


__all__ = ['XLabLRScheduler']


class XLabLRScheduler(lr_scheduler.LRScheduler):
    def __init__(self, *args, config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or {}


class CosineWarmupLR(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, total_iters: Union[int, float], warmup_iters: Union[int, float] = 0.1):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters if isinstance(warmup_iters, int) else int(warmup_iters * total_iters)
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        epoch = min(epoch, self.total_iters - 1)
        if epoch < self.warmup_iters:
            return (epoch + 1) / (self.warmup_iters + 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters)))


# Dynamically create subclasses of lr_scheduler.LRScheduler, accepting Lightning's additional configuration,
# and add them to the module namespace, so that they can be used in the configuration file.
# This is a workaround for Lightning CLI's incomplete support for LR scheduler configuration.
for module in [lr_scheduler, sys.modules[__name__]]:
    for var in vars(module).copy().values():
        if type(var) is type and issubclass(var, lr_scheduler.LRScheduler) \
                and var is not lr_scheduler.LRScheduler and not issubclass(var, XLabLRScheduler):
            name = f'XLab{var.__name__}'
            cls = type(name, (XLabLRScheduler, var), {})
            vars()[name] = cls
            __all__.append(name)
