import numpy as np
import torch

from models.evaluators.common import (
    calc_ood_metrics,
    closed_set_accuracy,
    prepare_ood_labels,
    run_model,
)


@torch.no_grad()
def open_hybrid_evaluator(args, train_loader, test_loader, device, model, s=0):
    _, train_lls, train_lbls = run_model(
        args, model, train_loader, device, open_hybrid=True, support=True
    )
    test_logits, test_lls, test_lbls = run_model(
        args, model, test_loader, device, open_hybrid=True, support=False
    )

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    known_mask = ood_labels == 1
    # cs_acc = closed_set_accuracy(cs_predictions[known_mask], test_lbls[known_mask])
    cs_acc = -1 # TODO
    metrics = calc_ood_metrics(test_lls, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics
