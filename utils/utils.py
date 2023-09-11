import math 
import torch
import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

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
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def clean_ckpt(ckpt, model):
    new_dict = {}
    model_dict = model.state_dict()
    for k in ckpt.keys():
        new_k = k
        if k not in model_dict:
            if k.startswith("base_model"):
                new_k = k.replace("base_model.", "")
            ### NF Hybrid models
            elif k.startswith("encoder"):
                new_k = k.replace("encoder", "base_model")
            elif k.startswith("flow_module"):
                new_k = k.replace("flow_module", "nf")
            ###
        new_dict[new_k] = ckpt[k]
    return new_dict


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


def interpolate_ckpts(zeroshot_ckpt, finetuned_ckpt, alpha):
    # make sure checkpoints are compatible
    assert set(zeroshot_ckpt.keys()) == set(finetuned_ckpt.keys())
    # interpolate between all weights in the checkpoints
    new_state_dict = {
        k: (1-alpha) * zeroshot_ckpt[k] + alpha * finetuned_ckpt[k] for k in zeroshot_ckpt.keys()
    }
    return new_state_dict
