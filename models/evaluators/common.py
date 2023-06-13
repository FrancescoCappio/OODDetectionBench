import torch 
from ood_metrics import calc_metrics
from tqdm import tqdm 
import numpy as np
from utils.dist_utils import all_gather

def prepare_ood_labels(known_labels, test_labels):
    # 0 means OOD
    # 1 means ID
    ood_labels = torch.zeros_like(test_labels)
    ood_labels[torch.any(test_labels.reshape(-1,1).expand(-1,len(known_labels)) == known_labels, dim=1)] = 1
    return ood_labels

@torch.no_grad()
def run_model(args, model, loader, device, contrastive=False):

    feats_list = []
    logits_list = []
    gt_list = []

    for images, target in tqdm(loader):
        images = images.to(device)
        if contrastive:
            out, contrastive_feats = model(images, contrastive=True)
            feats = contrastive_feats
        else:
            out, feats = model(images)

        feats_list.append(feats.cpu())
        gt_list.append(target)
        logits_list.append(out.cpu())

    logits_list, feats_list, gt_list = torch.cat(logits_list), torch.cat(feats_list), torch.cat(gt_list)

    if args.distributed and args.n_gpus > 1: 
        all_logits = all_gather(logits_list)
        all_feats = all_gather(feats_list)
        all_gt = all_gather(gt_list)

        logits_list = torch.cat([l for l in all_logits])
        feats_list = torch.cat([l for l in all_feats])
        gt_list = torch.cat([l for l in all_gt])

    return logits_list, feats_list, gt_list

def calc_ood_metrics(test_normality_scores, ood_labels):
    num_known = (ood_labels == 1).sum()

    if num_known == len(test_normality_scores):
        print("This dataset contains only known data")
        return {"auroc": -1, "fpr_at_95_tpr": -1}

    return calc_metrics(test_normality_scores, ood_labels)

def closed_set_accuracy(closed_set_preds, closed_set_lbls):

    assert len(closed_set_preds) == len(closed_set_lbls), "Numbers of closed set predictions and labels do not match"

    if not isinstance(closed_set_preds, np.ndarray):
        closed_set_preds = closed_set_preds.cpu().numpy()
    if not isinstance(closed_set_lbls, np.ndarray):
        closed_set_lbls = closed_set_lbls.cpu().numpy()

    correct = (closed_set_preds == closed_set_lbls).sum()

    acc = correct/len(closed_set_lbls)

    return acc
