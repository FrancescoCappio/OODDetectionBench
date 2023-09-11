import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy

def ash_b_flattened(x, percentile=65):
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    v, i = torch.topk(x, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    x.zero_().scatter_(dim=1, index=i, src=fill)
    return x

@torch.no_grad()
def _iterate_data_ASH(model, loader, device, energy_temper=1):
    confs = []
    gt_list = []

    for batch in tqdm(loader):
        images, labels = batch 
        images = images.to(device)
        gt_list.append(labels)

        # we perform forward on modules separately so that we can access penultimate layer
        _, feats = model(images)

        # we need unflattened features but we have flattened ones
        #import ipdb; ipdb.set_trace() 
        # apply ash
        feats = ash_b_flattened(feats)
        #feats = torch.flatten(feats,1)

        logits = model.fc(feats)

        # apply energy 
        conf = energy_temper * torch.logsumexp(logits / energy_temper, dim=1)

        confs.append(conf.cpu())

    return torch.cat(confs), torch.cat(gt_list)

def ASH_evaluator(args, train_loader, test_loader, device, model):
    # implements ICLR 2023: https://openreview.net/forum?id=ndYXTEL6cZz
    # original code taken from: https://github.com/andrijazz/ash
    
    train_lbls = train_loader.dataset.labels
    # ood labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    normality_scores, test_labels = _iterate_data_ASH(model, test_loader, device)
    ood_labels = prepare_ood_labels(known_labels, test_labels)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

    return metrics 
