import torch 
import faiss
import numpy as np
from tqdm import tqdm

from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy
from models.common import normalize_feats
from utils.utils import compute_R2, plot_tsne

def get_euclidean_normality_scores(test_features, prototypes):
    # normality scores computed as the opposite of the euclidean distance w.r.t. the nearest prototype
    print("Computing euclidean distances from prototypes")
    distances = []
    for cls in tqdm(range(prototypes.shape[0])):
        distances.append(torch.from_numpy(np.linalg.norm(prototypes[cls]-test_features, axis=1)))

    test_dists = torch.stack(distances,1)

    test_scores, test_predictions = test_dists.min(dim=1)
    
    return 1 - test_scores, test_predictions

def get_cos_sim_normality_scores(test_features, prototypes):
    print("Computing cosine similarity distances from prototypes")

    similarities = []
    for cls in tqdm(range(prototypes.shape[0])):
        similarity = (test_features * prototypes[cls]).sum(axis=1)
        similarities.append(torch.from_numpy(similarity))

    test_similarities = torch.stack(similarities,1)
    assert test_similarities.max() <= 1 and test_similarities.min() >= -1, "Range for cosine similarities is not correct!"
    test_scores, test_predictions = test_similarities.max(dim=1)

    return test_scores, test_predictions

def compute_prototypes(train_feats, train_lbls, normalize=False):

    classes = np.unique(train_lbls)
    prototypes = np.zeros((len(classes),train_feats.shape[1]))

    for idx, cl in enumerate(tqdm(classes)):
        prt = train_feats[train_lbls == cl].mean(axis=0)
        prototypes[idx] = prt

    if normalize:
        prototypes = normalize_feats(prototypes)

    return prototypes

def compute_centroids(train_feats, train_lbls, n_clusters):
    from sklearn.cluster import KMeans
    print(f"Computing {n_clusters} centroids")
    classes = np.unique(train_lbls)
    centroids = []
    centroids_lbls = []
    for cl in tqdm(classes):
        feats = train_feats[train_lbls == cl]
        n_samples = feats.shape[0]
        if n_clusters < n_samples:
            cl_centroids = KMeans(n_clusters, n_init="auto").fit(feats).cluster_centers_
            centroids.append(cl_centroids)
            centroids_lbls.append(np.full(n_clusters, cl))
        else:
            print(f"WARNING: the number of centroids ({n_clusters}) is greater or equal than the one of samples ({n_samples})")
            centroids.append(feats)
            centroids_lbls.append(np.full(n_samples, cl))
    centroids = np.concatenate(centroids, axis=0)
    centroids_lbls = np.concatenate(centroids_lbls, axis=0)
    return centroids, centroids_lbls

@torch.no_grad()
def prototypes_distance_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False, cosine_sim=False): 
    # first we extract features for both source and target data

    print("Prototypes distance evaluator")
    train_logits, train_feats, train_lbls = run_model(args, model, train_loader, device, contrastive=contrastive_head, support=True)
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, contrastive=contrastive_head, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    prototypes = compute_prototypes(train_feats, train_lbls, normalize=cosine_sim)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    if cosine_sim:
        test_normality_scores, test_predictions = get_cos_sim_normality_scores(test_feats, prototypes)
    else:
        test_normality_scores, test_predictions = get_euclidean_normality_scores(test_feats, prototypes)

    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(test_predictions[known_mask], test_lbls[known_mask])

    metrics = calc_ood_metrics(test_normality_scores, torch.from_numpy(ood_labels))
    metrics["cs_acc"] = cs_acc

    return metrics 

def ratio_NN_unknown(D, test_feats, ood_lbls, cosine_sim=False):

    if cosine_sim:
        test_index = faiss.IndexFlatIP(test_feats.shape[1])
    else:
        test_index = faiss.IndexFlatL2(test_feats.shape[1])

    test_index.add(test_feats)
    D_test, test_NN_ids = test_index.search(test_feats, 2)
    test_NN_ids = test_NN_ids[:,1]
    D_test = D_test[:,1]
    cat_Ds = np.concatenate((D, np.expand_dims(D_test, 1)), axis=1)
    if cosine_sim:
        min_dist_support_test = cat_Ds.argmax(1)
    else:
        min_dist_support_test = cat_Ds.argmin(1)
    num_unk = np.sum(np.logical_and((ood_lbls == 0), (min_dist_support_test==1)))
    ratio = num_unk/len(test_feats)
    return ratio

@torch.no_grad()
def knn_distance_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False, K=50, k_means=-1, normalize=False, cosine_sim=False): 
    # implements ICML 2022: https://proceedings.mlr.press/v162/sun22d.html
    # first we extract features for both source and target data
    print(f"Running KNN distance evaluator with K={K}")
    train_logits, train_feats, train_lbls = run_model(args, model, train_loader, device, contrastive=contrastive_head, support=True)
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, contrastive=contrastive_head, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    if normalize:
        train_feats = normalize_feats(train_feats)
        test_feats = normalize_feats(test_feats)

    if k_means > 0:
        train_feats, train_lbls = compute_centroids(train_feats, train_lbls, n_clusters=k_means)

    if args.enable_TSNE:
        plot_tsne(args, support_feats=train_feats, support_lbls=train_lbls, test_feats=test_feats, test_lbls=test_lbls)

    if cosine_sim: 
        # returns neighbours with decreasing similarity (nearest to farthest) 
        index = faiss.IndexFlatIP(train_feats.shape[1])
    else:
        # returns neighbours with increasing distance (nearest to farthest) 
        index = faiss.IndexFlatL2(train_feats.shape[1])

    index.add(train_feats)
    D, train_NN_ids = index.search(test_feats, K)
    train_NN_lbls = train_lbls[train_NN_ids]
    test_predictions = np.array([np.bincount(pred_NN_lbls).argmax() for pred_NN_lbls in train_NN_lbls])

    if args.enable_ratio_NN_unknown:
        ratio = ratio_NN_unknown(D, test_feats, ood_labels, cosine_sim)

    test_normality_scores = D[:,-1]
    if not cosine_sim:
        # (inverted) distance from Kth nearest neighbour is the normality score 
        test_normality_scores *= -1


    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(test_predictions[known_mask], test_lbls[known_mask])
    metrics = calc_ood_metrics(test_normality_scores, ood_labels)
    metrics["cs_acc"] = cs_acc
    if not args.disable_R2:
        r2_metric = compute_R2(train_feats, train_lbls, metric='cosine_distance' if normalize else 'euclidean_distance')
        metrics["support_R2"] = r2_metric

    if args.enable_ratio_NN_unknown:
        metrics["ratio_NN_unknown"] = ratio

    return metrics 

@torch.no_grad()
def knn_ood_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False, K=50, k_means=-1): 
    # implements ICML 2022: https://proceedings.mlr.press/v162/sun22d.html
    # similar to standard knn evaluator, but apply normalize before L2 distance

    return knn_distance_evaluator(args, train_loader, test_loader, device, model, contrastive_head=contrastive_head, K=K, k_means=k_means, normalize=True)

