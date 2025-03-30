import collections
from typing import Optional, List

import torch
import torch.nn as nn


@torch.no_grad()
def beam_search(
        model: nn.Module,
        x: torch.Tensor,
        beam_width: int,
        output_length: int,
        block_size: Optional[int],
        eos_class: Optional[int] = None,
        exclude_classes: Optional[List[int]] = None,
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
