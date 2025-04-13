import numpy as np

from .datasets import TokenDataset


def compute_stats(dataset: TokenDataset, sample_size: int = 10_000):
    np.random.seed(42)
    sample_indices = np.random.permutation(min(sample_size, len(dataset)))
    sample = dataset[sample_indices]
    lengths = np.array([len(indices) for indices in sample])
    return {
        'size_est': round(np.sum(lengths) * len(dataset) / len(sample)),
        'length_mean': np.mean(lengths),
        'length_median': np.median(lengths),
        'length_std': np.std(lengths),
    }
