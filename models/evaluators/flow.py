import numpy as np
import torch

from models.evaluators.common import calc_ood_metrics, closed_set_accuracy, prepare_ood_labels, run_model


@torch.no_grad()
def flow_evaluator(args, train_loader, test_loader, device, model):
    train_lbls = np.array(train_loader.dataset.labels)
    test_logits, test_lls, test_lbls = run_model(args, model, test_loader, device, flow=True, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    cs_predictions = test_logits.argmax(axis=1)
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_predictions[known_mask], test_lbls[known_mask])

    metrics = calc_ood_metrics(test_lls, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics
