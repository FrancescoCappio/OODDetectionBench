import torch 
from ood_metrics import calc_metrics as calc_ood_metrics
from tqdm import tqdm 

def prepare_ood_labels(known_labels, test_labels):
    # 0 means OOD
    # 1 means ID
    ood_labels = torch.zeros_like(test_labels)
    ood_labels[torch.any(test_labels.reshape(-1,1).expand(-1,len(known_labels)) == known_labels, dim=1)] = 1
    return ood_labels

@torch.no_grad()
def run_model(model, loader, device, contrastive=False):

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

    return  torch.cat(logits_list), torch.cat(feats_list), torch.cat(gt_list)

