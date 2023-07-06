import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from models.evaluators.common import run_model, prepare_ood_labels, calc_ood_metrics, closed_set_accuracy

@torch.no_grad()
def MSP_evaluator(args, train_loader, test_loader, device, model): 
    # implements ICLR 2017 https://openreview.net/forum?id=Hkg4TI9xl

    # first we extract features for target data
    train_lbls = train_loader.dataset.labels
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(torch.tensor(train_lbls))
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    # closed set predictions
    cs_preds = test_logits.argmax(axis=1)
    normality_scores = test_logits[np.arange(len(cs_preds)),cs_preds]
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_preds[known_mask], test_lbls[known_mask])

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics 

def _iterate_data_odin(model, data_loader, device):
    # values *verified* from original paper (should be validated on val OOD data)
    epsilon = 0.001
    temper = 1000
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    gt_list = []
    softmax_fn = nn.Softmax(dim=1)
    for images, labels in tqdm(data_loader):

        gt_list.append(labels)
        images = images.to(device)
        images.requires_grad = True
        outputs, _ = model(images)

        maxIndexTemp = outputs.argmax(dim=1)
        outputs = outputs / temper

        labels = maxIndexTemp
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        model.zero_grad()

        with torch.no_grad():
        # Adding small perturbations to images
            tempInputs = torch.add(images, gradient, alpha=-epsilon)
            outputs, _ = model(tempInputs)
            outputs = outputs / temper

            # Calculating the confidence after adding perturbations
            confidences = softmax_fn(outputs)
            
            # apply max take, value not intex
            confs.append(confidences.max(dim=1)[0].cpu())

    return torch.cat(confs), torch.cat(gt_list)

def ODIN_evaluator(train_loader, test_loader, device, model):
    # implements ICLR 2018: https://openreview.net/forum?id=H1VGkIxRZ
    
    train_lbls = train_loader.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    normality_scores, test_labels = _iterate_data_odin(model, test_loader, device)
    ood_labels = prepare_ood_labels(known_labels, test_labels)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

    return metrics 

def _iterate_data_energy(model, data_loader, device):
    # value verified from original paper
    temper = 1

    confs = []
    gt_list = []
    for batch in tqdm(data_loader):
        images, labels = batch
        gt_list.append(labels)
        with torch.no_grad():

            images = images.to(device)

            # compute output, measure accuracy and record loss.
            logits, feats = model(images)

            conf = temper * torch.logsumexp(logits / temper, dim=1)

            confs.append(conf.cpu())
    return torch.cat(confs), torch.cat(gt_list)

def energy_evaluator(train_loader, test_loader, device, model):
    # implements neurips 2020: https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html
    
    train_lbls = train_loader.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    normality_scores, test_labels = _iterate_data_energy(model, test_loader, device)
    ood_labels = prepare_ood_labels(known_labels, test_labels)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

    return metrics 

def _iterate_data_gradnorm(model, data_loader, device):
    # value verified from original paper
    temperature = 1

    confs = []
    gt_list = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    # gradnorm does not support batch size larger than 1
    # as grads on the cls layer have to be computed for each sample separately
    for images, labels in tqdm(data_loader.dataset):
        images = images.unsqueeze(0).to(device)
        gt_list.append(labels)

        model.zero_grad()
        outputs, _ = model(images)

        targets = torch.ones((images.shape[0], outputs.shape[1])).to(device)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu()
        confs.append(layer_grad_norm.item())

    return torch.tensor(confs), torch.tensor(gt_list)

def gradnorm_evaluator(train_loader, test_loader, device, model):
    # implements neurips 2021: https://papers.nips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html
    
    train_lbls = train_loader.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    normality_scores, test_labels = _iterate_data_gradnorm(model, test_loader, device)
    ood_labels = prepare_ood_labels(known_labels, test_labels)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_labels) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

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

    id_activations = []

    _, feats, _ = run_model(args, model, loader, device, support=True)

    feats = feats.flatten()
    feats = np.percentile(feats, id_percentile * 100) # thres, use same var name to reduce memory usage
     
    return feats

@torch.no_grad()
def _iterate_data_react(model, loader, device, threshold=1, energy_temper=1):
    confs = []
    gt_list = []

    for batch in tqdm(loader):
        images, labels = batch 
        images = images.to(device)
        gt_list.append(labels)

        # we perform forward on modules separately so that we can access penultimate layer
        _, feats = model(images)
        # apply react
        x = feats.clip(max=threshold)
        logits = model.fc(x)

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
