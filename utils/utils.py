import math
import warnings

import numpy as np
import torch


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_aux_modules_dict(optimizer, scheduler, suffix=""):
    """
    Wrap optimizer and scheduler in a dict to save and load checkpoints
    """
    aux_modules = {}
    for module_type, module in zip(["optimizer", "scheduler"], [optimizer, scheduler]):
        if module is not None:
            name = f"{module_type}_{suffix}" if suffix else module_type
            aux_modules[name] = module
    return aux_modules


@torch.no_grad()
def compute_sets_distance(feats1, feats2, metric="cosine_distance", mean=True):
    """Compute the average distance between pairs of samples coming from two sets"""
    feats1 = torch.from_numpy(feats1)
    feats2 = torch.from_numpy(feats2)

    if metric == "cosine_distance":

        def set_distance_fn(set1, set2):
            norms1 = set1.norm(dim=1, keepdim=True)
            norms2 = set2.norm(dim=1, keepdim=True)
            set1_normalized = set1 / norms1
            set2_normalized = set2 / norms2
            return 1 - (torch.matmul(set1_normalized, set2_normalized.T))

    elif metric == "euclidean_distance":

        def set_distance_fn(set1, set2):
            return torch.cdist(set1, set2)

    else:
        raise NotImplementedError(f"Unknown distace metric {metric}")

    all_distances = set_distance_fn(feats1, feats2)
    if mean:
        all_distances = all_distances.mean()
    return all_distances.numpy()


def get_base_job_name(args):
    base_name = f"{args.network}_{args.model}"

    if args.checkpoint_path:
        base_name += f"_{args.checkpoint_path.replace('/', '_')}"

    base_name += f"_{args.dataset}_{args.support}_{args.test}"

    if not args.data_order == -1:
        base_name += f"_do{args.data_order}"

    base_name += f"_{args.evaluator}"

    return base_name
