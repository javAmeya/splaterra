import torch


def compute_percentile_threshold(values, ratio):
    """
    Value below which the bottom `ratio` fraction of `values` falls. Used by
    CityGaussian V2's contribution-based trimming to turn a prune ratio into
    a concrete cutoff (pseudocode's `compute_percentile`).
    """
    k = max(1, min(values.numel(), int(values.numel() * ratio)))
    return torch.kthvalue(values, k).values