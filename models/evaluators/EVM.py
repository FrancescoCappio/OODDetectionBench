import torch 
import numpy as np
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy

@torch.no_grad()
def EVM_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False): 
    # this evaluator defines an Extreme Value Classifier and use it to estimate the normality of a given test sample 
    # ref paper: http://doi.org/10.1109/TPAMI.2017.2707495
    print("Running EVM evaluator")
    import EVM
    import scipy

    # first we extract features for both support and test data
    train_logits, train_feats, train_lbls = run_model(args, model, train_loader, device, contrastive=contrastive_head, support=True)
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, contrastive=contrastive_head, support=False)

    # we need to divide train sample by class 
    known_labels = np.unique(train_lbls)
    known_labels.sort()
    train_classes = [train_feats[train_lbls == lbl] for lbl in known_labels]
    # create and train the classifier 
    mevm = EVM.MultipleEVM(tailsize=len(train_feats), distance_function=scipy.spatial.distance.euclidean)
    mevm.train(train_classes)
    
    # estimate probabilities for test data
    pred_prob, indices = mevm.max_probabilities(test_feats)
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

    return metrics 


