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
        if k not in model_dict and k.startswith("base_model"):
            new_k = k.replace("base_model.","")
            new_dict[new_k] = ckpt[k]
        else:
            new_dict[k] = ckpt[k]
    return new_dict

@torch.no_grad()
def compute_sets_distance(feats1, feats2, metric="cosine_distance"):
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
    return all_distances.mean().numpy()

def get_base_job_name(args):
    base_name = f"{args.network}_{args.model}"

    if args.checkpoint_path:
        base_name += f"_{args.checkpoint_path.replace('/', '_')}"

    base_name += f"_{args.dataset}_{args.support}_{args.test}"

    if not args.data_order == -1:
        base_name += f"_do{args.data_order}"
    
    base_name += f"_{args.evaluator}"

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

def plot_tsne(args, support_feats, support_lbls, test_feats, test_lbls):
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

    all_feats = np.concatenate((support_feats, test_feats), axis=0)

    feats_embedded = TSNE(n_components=2, init='pca', learning_rate='auto').fit_transform(all_feats)

    ID_label_set = np.unique(support_lbls)
    ID_colors = (cm(ID_label_set/ID_label_set.max()))[:,:3]

    ood_labels = prepare_ood_labels(ID_label_set, test_lbls)

    support_feats_embedded = feats_embedded[:len(support_feats)]
    test_feats_embedded = feats_embedded[len(support_feats):]
    known_test_feats_embedded = test_feats_embedded[ood_labels == 1]
    unknown_test_feats_embedded = test_feats_embedded[ood_labels == 0]

    # first plot, only support set 
    fig, ax = plt.subplots()
    for ID_lbl in ID_label_set: 
        # train 
        mask = (support_lbls == ID_lbl)
        class_feats = support_feats_embedded[mask]
        color = ID_colors[ID_lbl]
        
        ax.scatter(class_feats[:,0], class_feats[:,1], c=np.expand_dims(color,axis=0), edgecolors='grey')

    fig.savefig(os.path.join(output_dir, "support_feats.png"))

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
