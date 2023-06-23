import os
import torch 
from ood_metrics import calc_metrics
from tqdm import tqdm 
import numpy as np
from utils.dist_utils import all_gather

def prepare_ood_labels(known_labels, test_labels):
    # 0 means OOD
    # 1 means ID

    if isinstance(known_labels, torch.Tensor) and isinstance(test_labels, torch.Tensor):
        ood_labels = torch.zeros_like(test_labels)
        ood_labels[torch.any(test_labels.reshape(-1,1).expand(-1,len(known_labels)) == known_labels, dim=1)] = 1
    elif isinstance(known_labels, np.ndarray) and isinstance(test_labels, np.ndarray):
        ood_labels = np.zeros_like(test_labels)
        ood_labels[np.any(test_labels.reshape(-1,1) == known_labels, axis=1)] = 1
    else:
        raise NotImplementedError("Unknown type for labels")
    return ood_labels

def get_disk_mm(args, ds_size, support=True):

    if not os.path.isdir("cache"):
        os.makedirs("cache")

    split = args.support if support else args.test
    base_name = f"cache/{args.network}_{args.model}_{args.dataset}_{split}"

    if args.checkpoint_path:
        base_name += "_" + args.checkpoint_path.replace("/", "_")

    feats_cache_file = f"{base_name}_feats.mmap"
    logits_cache_file = f"{base_name}_logits.mmap"
    gts_cache_file = f"{base_name}_gts.mmap"

    if not os.path.isfile(feats_cache_file):
        # we should create it 
        feats_np = np.memmap(feats_cache_file, dtype=np.float32, mode='w+', shape=(ds_size, args.output_num))
        logits_np = np.memmap(logits_cache_file, dtype=np.float32, mode='w+', shape=(ds_size, args.n_known_classes))
        gts_np = np.memmap(gts_cache_file, dtype=int, mode='w+', shape=(ds_size,))
        load = True
    else:
        feats_np = np.memmap(feats_cache_file, dtype=np.float32, mode='r', shape=(ds_size, args.output_num))
        logits_np = np.memmap(logits_cache_file, dtype=np.float32, mode='r', shape=(ds_size, args.n_known_classes))
        gts_np = np.memmap(gts_cache_file, dtype=int, mode='r', shape=(ds_size,))
        load = False 

    return feats_np, logits_np, gts_np, load 

@torch.no_grad()
def run_model(args, model, loader, device, contrastive=False, support=True):

    assert (not args.on_disk or not args.distributed), "Cannot execute distributed eval with memory mapped vectors"

    ds_size = len(loader.dataset)
    batch_size = loader.batch_size
    if args.on_disk:
        feats_np, logits_np, gts_np, load = get_disk_mm(args, ds_size=ds_size, support=support)
    else: 
        feats_np = np.zeros(dtype=np.float32, shape=(ds_size, args.output_num))
        logits_np = np.zeros(dtype=np.float32, shape=(ds_size, args.n_known_classes))
        gts_np = np.zeros(dtype=int, shape=(ds_size,))
        load = True

    if load:
        for batch_idx, (images, target) in enumerate(tqdm(loader)):
            images = images.to(device)
            this_batch_size = len(images)
            if contrastive:
                out, contrastive_feats = model(images, contrastive=True)
                feats = contrastive_feats
            else:
                out, feats = model(images)

            pos_start = batch_idx*batch_size
            pos_end = batch_idx*batch_size+this_batch_size
            feats_np[pos_start:pos_end] = feats.cpu().numpy()
            logits_np[pos_start:pos_end] = out.cpu().numpy()
            gts_np[pos_start:pos_end] = target.numpy()


    if args.distributed and args.n_gpus > 1: 
        assert not args.on_disk, "Cannot execute distributed eval with memory mapped vectors"
        all_logits = all_gather(logits_np)
        all_feats = all_gather(feats_np)
        all_gt = all_gather(gts_np)

        logits_list = np.concatenate([l for l in all_logits])
        feats_list = np.concatenate([l for l in all_feats])
        gt_list = np.concatenate([l for l in all_gt])

    else:
        logits_list, feats_list, gt_list = logits_np, feats_np, gts_np

    return logits_list, feats_list, gt_list

def calc_ood_metrics(test_normality_scores, ood_labels):
    num_known = (ood_labels == 1).sum()
    num_unknown = len(test_normality_scores) - num_known

    if num_unknown == 0:
        print("This dataset contains only known data")
        metrics = {"auroc": -1, "fpr_at_95_tpr": -1}
    
    else:
        metrics = calc_metrics(test_normality_scores, ood_labels)

    metrics["num_known"] = num_known
    metrics["num_unknown"] = num_unknown

    return metrics

def closed_set_accuracy(closed_set_preds, closed_set_lbls):

    assert len(closed_set_preds) == len(closed_set_lbls), "Numbers of closed set predictions and labels do not match"

    if not isinstance(closed_set_preds, np.ndarray):
        closed_set_preds = closed_set_preds.cpu().numpy()
    if not isinstance(closed_set_lbls, np.ndarray):
        closed_set_lbls = closed_set_lbls.cpu().numpy()

    correct = (closed_set_preds == closed_set_lbls).sum()

    acc = correct/len(closed_set_lbls)

    return acc
