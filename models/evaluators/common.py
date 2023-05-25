import torch 
from ood_metrics import calc_metrics as calc_ood_metrics
from tqdm import tqdm 

def normalize(feats):
    return feats/feats.norm(dim=1,keepdim=True).expand(-1,feats.shape[1])

def prepare_ood_labels(known_labels, test_labels):
    # 0 means OOD
    # 1 means ID
    ood_labels = torch.zeros_like(test_labels)
    ood_labels[torch.any(test_labels.reshape(-1,1).expand(-1,len(known_labels)) == known_labels, dim=1)] = 1
    return ood_labels

@torch.no_grad()
def run_model(model, loader, device, contrastive_head=None):

    feats_list = []
    logits_list = []
    gt_list = []

    for images, target in tqdm(loader):
        images = images.to(device)
        out, feats = model(images)
        if contrastive_head is not None:
            feats = normalize(contrastive_head(feats))

        feats_list.append(feats.cpu())
        gt_list.append(target)
        logits_list.append(out.cpu())

    return  torch.cat(logits_list), torch.cat(feats_list), torch.cat(gt_list)

