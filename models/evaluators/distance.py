import torch 
import faiss

from models.evaluators.common import normalize, prepare_ood_labels, calc_ood_metrics, run_model

def get_euclidean_normality_scores(test_features, prototypes):
    # normality scores computed as the opposite of the euclidean distance w.r.t. the nearest prototype
    print("Computing euclidean distances from prototypes")
    distances = []
    for cls in range(prototypes.shape[0]):
        distances.append(torch.norm(prototypes[cls]-test_features, dim=1))

    test_dists = torch.stack(distances,1)

    test_scores, test_predictions = test_dists.min(dim=1)
    
    return 1 - test_scores

def get_cos_sim_normality_scores(test_features, prototypes):
    print("Computing cosine similarity distances from prototypes")

    similarities = []
    for cls in range(prototypes.shape[0]):
        similarity = (test_features * prototypes[cls]).sum(dim=1)
        similarities.append(similarity)

    test_similarities = torch.stack(similarities,1)
    assert test_similarities.max() <= 1 and test_similarities.min() >= -1, "Range for cosine similarities is not correct!"
    test_scores, test_predictions = test_similarities.max(dim=1)

    return test_scores

def compute_prototypes(train_feats, train_lbls, contrastive = False):

    classes = torch.unique(train_lbls)
    prototypes = torch.zeros((len(classes),train_feats.shape[1]))

    for idx, cl in enumerate(classes):
        prt = train_feats[train_lbls == cl].mean(dim=0)
        prototypes[idx] = prt
    
    if contrastive:
        prototypes = normalize(prototypes)

    return prototypes

@torch.no_grad()
def prototypes_distance_evaluator(train_loader, test_loader, device, model, contrastive_head=None): 
    # first we extract features for both source and target data
    train_logits, train_feats, train_lbls = run_model(model, train_loader, device, contrastive_head=contrastive_head)
    test_logits, test_feats, test_lbls = run_model(model, test_loader, device, contrastive_head=contrastive_head)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(train_lbls)
    prototypes = compute_prototypes(train_feats, train_lbls, contrastive=contrastive_head is not None)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    if contrastive_head is not None:
        test_normality_scores = get_cos_sim_normality_scores(test_feats, prototypes)
    else:
        test_normality_scores = get_euclidean_normality_scores(test_feats, prototypes)

    metrics = calc_ood_metrics(test_normality_scores, ood_labels)

    return metrics 

@torch.no_grad()
def knn_distance_evaluator(train_loader, test_loader, device, model, contrastive_head=None, K=50): 
    # implements ICML 2022: https://proceedings.mlr.press/v162/sun22d.html
    # first we extract features for both source and target data
    print(f"Running KNN distance evaluator with K={K}")
    train_logits, train_feats, train_lbls = run_model(model, train_loader, device, contrastive_head=contrastive_head)
    test_logits, test_feats, test_lbls = run_model(model, test_loader, device, contrastive_head=contrastive_head)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    train_feats = normalize(train_feats)
    test_feats = normalize(test_feats)

    index = faiss.IndexFlatL2(train_feats.shape[1])
    index.add(train_feats.numpy())
    D, _ = index.search(test_feats.numpy(), K)

    # (inverted) distance from Kth nearest neighbour is the normality score 
    test_normality_scores = -D[:,-1]

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(test_normality_scores, ood_labels)

    return metrics 
