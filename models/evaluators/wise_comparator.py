from itertools import combinations

import torch
import numpy as np
from models.evaluators.common import prepare_ood_labels, run_model
from models.common import normalize_feats
from utils.cka import CKA, CudaCKA


@torch.no_grad()
def wise_comparator(
    args, train_loader, test_loader, device, model, ckpts, contrastive_head=False, normalize=False
):
    keys = ["zeroshot", "wise", "ft"]

    train_lbls = train_loader.dataset.labels
    known_lbls = np.unique(train_lbls)

    ood_lbls = None
    feats_dict = {}
    for k in keys:
        print(f"Running {k} model")
        ckpt, ckpt_device = ckpts[k]
        model.to(ckpt_device)
        model.load_state_dict(ckpt)
        model.to(device)
        _, train_feats, _ = run_model(
            args, model, train_loader, device, contrastive=contrastive_head, support=True
        )
        _, test_feats, test_lbls = run_model(
            args, model, test_loader, device, contrastive=contrastive_head, support=False
        )
        if normalize:
            train_feats = normalize_feats(train_feats)
            test_feats = normalize_feats(test_feats)
        if ood_lbls is None:
            ood_lbls = prepare_ood_labels(known_lbls, test_lbls)
        feats_dict[k] = np.concatenate((train_feats, test_feats))

    cka = CudaCKA(device) if device == "cuda" or device.type == "cuda" else CKA()
    
    metrics = {}

    n_train = len(train_lbls)
    n_test = len(ood_lbls)
    masks = {
        "support": np.concatenate((np.full(n_train, True), np.full(n_test, False))),
        "test_global": np.concatenate((np.full(n_train, False), np.full(n_test, True))),
        "test_ood": np.concatenate((np.full(n_train, False), ood_lbls == 0)),
        "test_id": np.concatenate((np.full(n_train, False), ood_lbls == 1)),
    }
    for k1, k2 in combinations(keys, r=2):
        for mode, mask in masks.items():
            feats = (torch.from_numpy(feats_dict[k][mask]).to(device) for k in (k1, k2))
            metrics[f"{k1}_{k2}_{mode}"] = cka.linear_CKA(*feats)

    return metrics
