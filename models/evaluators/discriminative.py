import numpy as np
import torch
from tqdm import tqdm

from models.evaluators.common import calc_ood_metrics, closed_set_accuracy, prepare_ood_labels, run_model


def np_softmax(logits, axis=1):
    exp_scores = np.exp(logits)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


@torch.no_grad()
def MSP_evaluator(args, train_loader, test_loader, device, model, disable_softmax=False):
    # implements ICLR 2017 https://openreview.net/forum?id=Hkg4TI9xl

    # first we extract features for target data
    train_lbls = train_loader.dataset.labels
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(torch.tensor(train_lbls))
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    # closed set predictions
    cs_preds = test_logits.argmax(axis=1)

    # with softmax disabled we have MLS
    if not disable_softmax:
        test_logits = np_softmax(test_logits)

    normality_scores = test_logits[np.arange(len(cs_preds)), cs_preds]
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_preds[known_mask], test_lbls[known_mask])

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics


@torch.no_grad()
def _estimate_react_thres(args, model, loader, device, id_percentile=0.9):
    """
    Estimate the threshold to be used for react.
    Strategy: choose threshold which allows to keep id_percentile% of
    activations in in distribution data (source https://openreview.net/pdf?id=IBVBtz_sRSm).

    Args:
        model: base network
        loader: in distribution data loader (some kind of validation set)
        device: torch device
        id_percentile: percent of in distribution activations that we want to keep

    Returns:
        threshold: float value indicating the computed threshold
    """

    _, feats, _ = run_model(args, model, loader, device, support=True)

    feats = feats.flatten()
    feats = np.percentile(feats, id_percentile * 100)  # thres, use same var name to reduce memory usage

    return feats


@torch.no_grad()
def _iterate_data_react(model, loader, device, threshold=1, energy_temper=1):
    confs = []
    gt_list = []

    if hasattr(model, "fc"):
        fc = model.fc
    elif hasattr(model, "base_model") and hasattr(model.base_model, "fc"):
        fc = model.base_model.fc
    else:
        raise NotImplementedError("Don't know how to access fc")

    for batch in tqdm(loader):
        images, labels = batch
        images = images.to(device)
        gt_list.append(labels)

        # we perform forward on modules separately so that we can access penultimate layer
        _, feats = model(images)
        # apply react
        x = feats.clip(max=threshold)
        logits = fc(x)

        # apply energy
        conf = energy_temper * torch.logsumexp(logits / energy_temper, dim=1)

        confs.append(conf.cpu())

    return torch.cat(confs), torch.cat(gt_list)


def react_evaluator(args, train_loader, test_loader, device, model):
    # implements neurips 2021: https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html

    train_lbls = train_loader.dataset.labels
    # ood labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    threshold = _estimate_react_thres(args, model, train_loader, device)
    normality_scores, test_labels = _iterate_data_react(model, test_loader, device, threshold=threshold)
    ood_labels = prepare_ood_labels(known_labels, test_labels)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

    return metrics
