import torch 
from sklearn.linear_model import LogisticRegression
import numpy as np
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy

@torch.no_grad()
def linear_probe_evaluator(args, train_loader, test_loader, device, model, contrastive_head=False): 
    # this evaluator trains a logistic regression model on top of the frozen source features 
    # MSP is then applied to compute test normality scores 
    print("Running linear probe evaluator")

    # first we extract features for both source and target data
    train_logits, train_feats, train_lbls = run_model(args, model, train_loader, device, contrastive=contrastive_head, support=True)
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, contrastive=contrastive_head, support=False)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, solver='liblinear')
    classifier.fit(train_feats, train_lbls)
    # Evaluate on the target features
    pred_prob = classifier.predict_proba(test_feats) # get the softmax probability for each class
    # Get the maximum softmax probability
    max_pred_prob = np.amax(pred_prob, axis=1)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    # closed set predictions
    cs_preds = np.argmax(pred_prob, axis=1)
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_preds[known_mask], test_lbls[known_mask])

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(max_pred_prob, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics 


