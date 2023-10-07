import math 
import torch
import warnings
import numpy as np
import os

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


@torch.no_grad()
def compute_sets_distance(feats1, feats2, metric="cosine_distance", mean=True):
    """Compute the average distance between pairs of samples coming from two sets"""
    feats1 = torch.from_numpy(feats1)
    feats2 = torch.from_numpy(feats2)

    if metric == "cosine_distance":
        def set_distance_fn(set1, set2):
            norms1 = set1.norm(dim=1, keepdim=True)
            norms2 = set2.norm(dim=1, keepdim=True)
            set1_normalized = set1/norms1
            set2_normalized = set2/norms2
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

    if args.suffix:
        base_name += f"_{args.suffix}"

    return base_name


def compute_R2(feats, labels, metric="cosine_distance", return_stats=False):
    """Compute the R2 metric from https://papers.nips.cc/paper/2021/hash/f0bf4a2da952528910047c31b6c2e951-Abstract.html

    Parameters: 
    feats (np.ndarray): tensor of shape (BS x features_len)
    labels (np.ndarray): tensor of labels (BS)

    Returns: 
    R2 (int): R2 metric
    """

    label_set = np.unique(labels)
    cls_compactness = np.zeros(len(label_set))

    for lbl in label_set:
        mask = labels == lbl 

        cls_feats = feats[mask]
        compactness = compute_sets_distance(cls_feats, cls_feats, metric=metric)
        cls_compactness[lbl] = compactness
        #print(f"CLS {lbl}. Compactness: {compactness:.4f}")

    d_within = cls_compactness.mean()

    d_total = 0
    for lbl1 in label_set:
        mask1 = labels == lbl1
        cls_feats1 = feats[mask1]

        for lbl2 in label_set: 
            mask2 = labels == lbl2
            cls_feats2 = feats[mask2]

            cls_pair_distance = compute_sets_distance(cls_feats1, cls_feats2, metric=metric)
            d_total += cls_pair_distance
            #print(f"CLS {lbl1} vs CLS {lbl2}. Similarity: {cls_pair_distance:.4f}")

    d_total /= (len(label_set)*len(label_set))

    res = 1 - (d_within/d_total)
    res = res.item()
    #print(f"d_within: {d_within:.4f}, d_total: {d_total:.4f}")

    if return_stats:
        return res, d_total, d_within
    
    return res


def compute_ranking_index(feats, labels, metric="euclidean_distance"):
    # compute pairwise distances
    all_distances = compute_sets_distance(feats, feats, metric, mean=False)
    np.fill_diagonal(all_distances, -1) # "mask" distance of samples with themselves -> this will result in rank 0

    n_samples = len(feats)
    assert n_samples == len(labels) == all_distances.shape[0] == all_distances.shape[1]

    # compute all ranks for each sample (i.e. for each row)
    all_ranks = np.empty((n_samples, n_samples), dtype=int)
    arange = np.arange(n_samples)
    all_ranks[arange.reshape(-1, 1), all_distances.argsort(axis=1)] = arange

    # ensure that for each sample the rank 0 corresponds to the sample itself
    assert np.all(all_ranks.diagonal() == 0)

    # consider only the samples from the same class
    mask = (labels.reshape(-1, 1) == labels.reshape(1, -1))
    np.fill_diagonal(mask, False) # mask samples from themselves

    accum = 0
    for ranks_i, mask_i in zip(all_ranks, mask):
        sorted_ranks = np.sort(ranks_i[mask_i])
        arange = np.arange(1, len(sorted_ranks) + 1)
        accum += 1 - np.mean(arange / sorted_ranks)
    
    rank_index = 1 - accum / (n_samples - 1)
    return rank_index


def plot_tsne(args, support_feats, support_lbls, test_feats, test_lbls, centroids=None, centroids_lbls=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from models.evaluators.common import prepare_ood_labels
    cm = mpl.colormaps['jet']
    mpl.rcParams['figure.dpi']=1000
    mpl.rc("savefig", dpi=1000)

    print("Plotting and saving t-SNE projections")

    # prepare dir for saving images 
    job_name = get_base_job_name(args)
    output_dir = os.path.join("t-SNE_plots", job_name)
    os.makedirs(output_dir, exist_ok=True)

    if centroids is not None or centroids_lbls is not None:
        assert centroids is not None and centroids_lbls is not None
        support_feats = np.concatenate((centroids, support_feats), axis=0)

    all_feats = np.concatenate((support_feats, test_feats), axis=0)

    feats_embedded = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(all_feats)

    ID_label_set = np.unique(support_lbls)
    ID_colors = (cm(ID_label_set/ID_label_set.max()))[:,:3]

    ood_labels = prepare_ood_labels(ID_label_set, test_lbls)

    if centroids is not None:
        first_support_idx = len(centroids)
        centroids_feats_embedded = feats_embedded[:first_support_idx]
    else:
        first_support_idx = 0
    support_feats_embedded = feats_embedded[first_support_idx:len(support_feats)]
    test_feats_embedded = feats_embedded[len(support_feats):]
    known_test_feats_embedded = test_feats_embedded[ood_labels == 1]
    unknown_test_feats_embedded = test_feats_embedded[ood_labels == 0]

    # first plot, only support set
    fig, ax = plt.subplots()
    alpha = None if centroids is None else 0.2  # None is plt default value
    for ID_lbl in ID_label_set: 
        # train 
        mask = (support_lbls == ID_lbl)
        class_feats = support_feats_embedded[mask]
        color = ID_colors[ID_lbl]
        ax.scatter(class_feats[:,0], class_feats[:,1], c=np.expand_dims(color,axis=0), edgecolors='grey', alpha=alpha)
    fig.savefig(os.path.join(output_dir, "support_feats.png"))
        
    # plot centroids, if using k-means
    if centroids is not None:
        for ID_lbl in ID_label_set: 
            mask = (centroids_lbls == ID_lbl)
            class_feats = centroids_feats_embedded[mask]
            color = ID_colors[ID_lbl]
            ax.scatter(class_feats[:,0], class_feats[:,1], c=np.expand_dims(color,axis=0), edgecolors='grey', marker="^")
        fig.savefig(os.path.join(output_dir, "support_feats_centroids.png"))

    # add known test feats
    for ID_lbl in ID_label_set: 
        # train 
        mask = (test_lbls[ood_labels==1] == ID_lbl)
        class_feats = known_test_feats_embedded[mask]
        color = ID_colors[ID_lbl]
        ax.scatter(class_feats[:,0], class_feats[:,1], c=np.expand_dims(color,axis=0), marker="*")
    fig.savefig(os.path.join(output_dir, "support_test_known_feats.png"))

    # now add unknown feats
    ax.scatter(unknown_test_feats_embedded[:,0], unknown_test_feats_embedded[:,1], c="black", edgecolors='grey', label="OOD", alpha=0.5)
    fig.savefig(os.path.join(output_dir, "support_test_known_unknown_feats.png"))
