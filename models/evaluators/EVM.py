import torch 
import numpy as np
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy
from utils.dist_utils import get_max_cpu_count
from utils.utils import compute_R2
import scipy

@torch.no_grad()
def EVM_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False, normalize=False): 
    # this evaluator defines an Extreme Value Classifier and use it to estimate the normality of a given test sample 
    # ref paper: http://doi.org/10.1109/TPAMI.2017.2707495
    workers = get_max_cpu_count()
    print(f"Running EVM evaluator with {workers} worker(s)")
    import EVM

    # first we extract features for both support and test data
    _, train_feats, train_lbls = run_model(args, model, train_loader, device, contrastive=contrastive_head, support=True)
    _, test_feats, test_lbls = run_model(args, model, test_loader, device, contrastive=contrastive_head, support=False)

    r2_metric = compute_R2(train_feats, train_lbls, metric='cosine_distance' if normalize else 'euclidean_distance')

    # we need to divide train sample by class 
    known_labels = np.unique(train_lbls)
    known_labels.sort()
    train_classes = [train_feats[train_lbls == lbl] for lbl in known_labels]
    # create and train the classifier 
    mevm = EVM.MultipleEVM(tailsize=len(train_feats), distance_function=scipy.spatial.distance.cosine if normalize else scipy.spatial.distance.euclidean)
    mevm.train(train_classes, parallel=workers)
    
    # estimate probabilities for test data
    pred_prob, indices = mevm.max_probabilities(test_feats, parallel=workers)
    pred_prob = np.array(pred_prob)
    cs_preds = np.stack(indices)[:,0]
    cs_preds[pred_prob==0] = len(known_labels)

    # known labels have 1 for known samples and 0 for unknown ones
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    # closed set accuracy
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_preds[known_mask], test_lbls[known_mask])

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(pred_prob, ood_labels)
    metrics["cs_acc"] = cs_acc
    metrics["support_R2"] = r2_metric

    return metrics 


