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

import collections
from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def sample(
        model: nn.Module,
        x: torch.Tensor,
        output_length: int,
        block_size: Optional[int],
        eos_class: Optional[int] = None,
        exclude_classes: Optional[list[int]] = None,
        temperature: float = 1,
        top_k: Optional[float] = None,
        top_p: Optional[float] = None,
        generator: Optional[torch.Generator] = None
):
    """
    Sampling for autoregressive models in PyTorch.
    See `How to generate text <https://huggingface.co/blog/how-to-generate>`_.

    Args:
        model (Module): The model to predict output classes. It is expected to output unnormalized logits.
        x (Tensor): Input class indices (1-d, int64).
        output_length (int): The maximum number of predictions to generate.
        block_size (int): The maximum sequence length that the model accepts. ``None`` denotes unlimited.
        eos_class (int): Index of the end-of-sequence class. Not returned in output. ``None`` denotes unavailable.
        exclude_classes (list[int]): Indices of classes to exclude from results (e.g. padding and unknown).
        temperature (float): A divisor for the logits to flatten (if < 1) or emphasize (if > 1) class probabilities.
        top_k (float): The number of most likely classes, from which to sample each next class. Set to ``1`` for greedy search.
        top_p (float): The minimum probability of the set of classes to sample from (aka "nucleus sampling" or "dynamic top-k").
            Can be combined with ``top_k``.
        generator (Generator): A pseudorandom number generator for sampling.

    Returns:
        Tensor of output class indices (1-d, int64).
    """

    model.eval()
    seq = x

    for _ in range(output_length):
        inputs = seq[-block_size:].unsqueeze(0)
        logits = model.forward(inputs).squeeze(0)[-1]
        if exclude_classes:
            logits[exclude_classes] = float('-inf')
        logits = logits / temperature
        probas = logits.softmax(dim=-1)

        if top_k:
            probas, indices = probas.topk(top_k)
        else:
            indices = torch.arange(probas.size(-1))

        if top_p:
            sorted_probas, sorted_indices = probas.sort()  # ascending sort simplifies the following
            cumprobas = sorted_probas.cumsum(-1)
            nucleus_size = cumprobas.size(-1) - torch.sum(cumprobas <= (1 - top_p))
            nucleus_indices = sorted_indices[-nucleus_size:]
            probas = sorted_probas[-nucleus_size:]
            indices = indices[nucleus_indices]

        index = probas.multinomial(1, generator=generator)
        if index == eos_class:
            break
        seq = torch.cat([seq, indices[index]])

    return seq[x.size(-1):]


@torch.no_grad()
def beam_search(
        model: nn.Module,
        x: torch.Tensor,
        beam_width: int,
        output_length: int,
        block_size: Optional[int],
        eos_class: Optional[int] = None,
        exclude_classes: Optional[list[int]] = None,
        length_penalty: float = 0
):
    """
    Beam search for PyTorch.
    See `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_.

    Args:
        model (Module): The model to predict output classes. It is expected to output unnormalized logits.
        x (Tensor): Input class indices (1-d, int64).
        beam_width (int): The number of branches (output combinations) to track while searching.
        output_length (int): The maximum number of predictions to generate.
        block_size (int): The maximum sequence length that the model accepts. ``None`` denotes unlimited.
        eos_class (int): Index of the end-of-sequence class. Not returned in output. ``None`` denotes unavailable.
        exclude_classes (list[int]): Indices of classes to exclude from results (e.g. padding and unknown).
        length_penalty (float): The exponent of the output length in the score divisor (`score = score / length ** length_penalty`).
            Use positive values to promote longer outputs, and negative values for shorter outputs.

    Returns:
        Tensor of output class indices (1-d, int64).
    """

    Node = collections.namedtuple('Node', ['path', 'proba', 'score'])
    model.eval()

    empty = torch.tensor([], dtype=torch.int64, device=model.device)
    root = Node(empty, 0.0, 0.0)
    nodes = branches = [root]
    leaves = []

    for level in range(output_length):
        candidates = []
        score_divisor = (level + 1) ** length_penalty

        # early stopping if we won't find a branch better than the current best leaf
        best_score = max(leaf.score for leaf in leaves) if leaves else float('-inf')
        early_stopping_divisor = score_divisor if length_penalty <= 0 else output_length ** length_penalty
        branches = [branch for branch in branches if branch.proba / early_stopping_divisor >= best_score]

        for branch in branches:
            _x, _path = x, branch.path
            if block_size:
                _path = branch.path[-block_size:]
                _x = x[max(x.size(0) + _path.size(0) - block_size, 0):]
            inputs = torch.cat([_x, _path]).unsqueeze(0)

            logits = model(inputs).squeeze(0)[-1]
            if exclude_classes:
                logits[exclude_classes] = float('-inf')
            probas = logits.log_softmax(0)
            probas, indices = probas.topk(beam_width)
            probas += branch.proba
            scores = probas / score_divisor
            cand = [Node(torch.cat([branch.path, indices[i:i+1]]), proba, score)
                    for i, (proba, score) in enumerate(zip(probas, scores))]
            candidates.extend(cand)

        candidates += leaves
        candidates = sorted(candidates, key=lambda node: node.score, reverse=True)
        nodes = candidates[:beam_width]
        leaves = [node for node in nodes if node.path[-1] == eos_class]
        branches = set(nodes) - set(leaves)
        if not branches:
            break

    node = max(nodes, key=lambda node: (node.path[-1] == eos_class, node.score))
    output = node.path
    if output[-1] == eos_class:
        output = output[:-1]
    return output
