import torch 
from sklearn.linear_model import LogisticRegression
import numpy as np
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model

@torch.no_grad()
def linear_probe_evaluator(train_loader, test_loader, device, model, contrastive_head=None): 
    # this evaluator trains a logistic regression model on top of the frozen source features 
    # MSP is then applied to compute test normality scores 
    print("Running linear probe evaluator")

    # first we extract features for both source and target data
    train_logits, train_feats, train_lbls = run_model(model, train_loader, device, contrastive_head=contrastive_head)
    test_logits, test_feats, test_lbls = run_model(model, test_loader, device, contrastive_head=contrastive_head)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, solver='liblinear')
    classifier.fit(train_feats.numpy(), train_lbls.numpy())
    # Evaluate on the target features
    pred_prob = classifier.predict_proba(test_feats.numpy()) # get the softmax probability for each class
    # Get the maximum softmax probability
    max_pred_prob = np.amax(pred_prob, axis=1)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(max_pred_prob, ood_labels)

    return metrics 


