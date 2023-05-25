import torch
import torch.nn.functional as F
import numpy as np
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics

from tqdm import tqdm
import random

@torch.no_grad()
def gram_evaluator(train_loader, val_loader, test_loader, device, model, finetuned=True):
    #implements ICML 2020 https://proceedings.mlr.press/v119/sastry20a.html

    train_lbls = train_loader.dataset.dataset.labels
    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(torch.tensor(train_lbls))

    normality_scores, test_lbls = iterate_data_gram(model, train_loader, val_loader, test_loader, device, len(known_labels), finetuned)

    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = calc_ood_metrics(normality_scores, ood_labels)

    return metrics 


@torch.no_grad()
def iterate_data_gram(model, train_loader, val_loader, test_loader, device, n_known_classes, finetuned):
    selected_powers = range(1,6)

    detector = Detector(model, n_known_classes, finetuned, device)
    detector.compute_minmaxs(train_loader, POWERS=selected_powers)
    val_deviations, val_labels, val_predictions, val_confidences = detector.compute_deviations(val_loader, POWERS=selected_powers)
    test_deviations, test_labels, test_predictions, test_confidences = detector.compute_deviations(test_loader, POWERS=selected_powers)

    # rescale test deviations by confidence 
    test_deviations = test_deviations/test_confidences.reshape(-1,1).expand(test_deviations.shape)

    # higher deviations mean lower normality scores
    gram_scores = -detect(val_deviations, test_deviations).cpu()

    return gram_scores, test_labels

def detect(val_deviations, test_deviations, verbose=True, normalize=True):
    """
    compare deviations for validation data and test data to obtain normality scores. 
    """

    validation = val_deviations

    t95 = validation.mean(axis=0) + 10 ** -7
    import math
    import sys
    t95[t95 == math.inf] = 0
    t95[t95 == math.nan] = 0
    #t95[t95 >= sys.float_info.max] = 0

    if not normalize:
        t95 = np.ones_like(t95)
    ood_deviations = (test_deviations / t95[np.newaxis, :]).sum(axis=1)

    return ood_deviations

class Detector:
    def __init__(self, model, n_known_classes, finetuned, device):
        self.all_test_deviations = None
        self.mins = {}
        self.maxs = {}

        self.model = model
        self.finetuned = finetuned
        self.n_known_classes = n_known_classes
        self.device = device

        self.classes = range(n_known_classes)

    @torch.no_grad()
    def compute_minmaxs(self, source_loader, POWERS=[10]):
        '''
        for each known class, get the samples 
        that are predicted to be part of that class (or use GT if model not finetuned),
        get the features at various layers,
        compute the gram matrix for each power, get min and max 
        '''

        correct = 0

        # for each class, for each feature layer, for each power get min and max
        self.mins = {}
        self.maxs = {}

        print("Extracting min and max values from train data")

        for batch in tqdm(source_loader):
            images, labels = batch 
            images = images.to(self.device)

            logits, f_list = gram_feature_list(self.model, images)
            pred = logits.argmax(dim=1).cpu()

            correct += (labels == pred).sum()
            # only at the beginning we prepare the data structures
            if len(self.mins) == 0:
                self.mins = float("Inf")*torch.ones((self.n_known_classes,len(f_list),len(POWERS))).to(self.device)
                self.maxs = (-float("Inf"))*torch.ones((self.n_known_classes,len(f_list),len(POWERS))).to(self.device)

            # if the model has not been finetuned the predictions are not reliable, use GT
            if not self.finetuned:
                pred = labels

            for l_id, f_l in enumerate(f_list):
                for p_id, p in enumerate(POWERS):
                    gp = G_p(f_l, p)
                    mins = gp.min(dim=1)[0]
                    maxs = gp.max(dim=1)[0]

                    for pred_cls, f_min, f_max in zip(pred, mins, maxs):
                        pred_cls = pred_cls.item()
                        self.mins[pred_cls,l_id,p_id] = torch.min(self.mins[pred_cls,l_id,p_id], f_min)
                        self.maxs[pred_cls,l_id,p_id] = torch.max(self.maxs[pred_cls,l_id,p_id], f_max)
        print(f"Train accuracy: {100*(correct/len(source_loader.dataset)):.2f}")

    @torch.no_grad()
    def compute_deviations(self, data_loader, POWERS=[10]):

        deviations = []
        predictions = []
        confidences = []
        gt_labels = []

        for batch in tqdm(data_loader):
            images, labels = batch 
            images = images.to(self.device)

            # for each image the deviation is computed by comparing its layer-wise output 
            # with min-max stats of the corresponding label 
            # we obtain a deviation value for each feature layer 

            logits, f_list = gram_feature_list(self.model, images)
            conf, pred = logits.max(dim=1)

            predictions.append(pred)
            confidences.append(conf)
            gt_labels.append(labels)

            dev = torch.zeros((len(images),len(f_list))).to(self.device)
            for l_id, f_l in enumerate(f_list):
                for p_id, p in enumerate(POWERS):
                    gp_batch = G_p(f_l, p)

                    mins = self.mins[pred,l_id,p_id].reshape(-1,1).expand(gp_batch.shape)
                    maxs = self.maxs[pred,l_id,p_id].reshape(-1,1).expand(gp_batch.shape)
                    dev[:,l_id] += (F.relu(mins - gp_batch) / torch.abs(mins + 10 ** -6)).sum(dim=1)
                    dev[:,l_id] += (F.relu(gp_batch - maxs) / torch.abs(maxs + 10 ** -6)).sum(dim=1)

            deviations.append(dev)
        return torch.cat(deviations), torch.cat(gt_labels), torch.cat(predictions), torch.cat(confidences)

def gram_feature_list(model, x):

    def forward_block(blocks, tensor):
        feats_list = []

        for block in blocks:
            identity = tensor

            t = block.conv1(tensor)
            out = block.bn1(t)
            out = block.relu(out)
            feats_list.extend([t,out])

            t = block.conv2(out)
            out = block.bn2(t)
            out = block.relu(out)
            feats_list.extend([t,out])

            t = block.conv3(out)
            out = block.bn3(t)
            feats_list.extend([t,out])

            if block.downsample is not None:
                identity = block.downsample(x)

            out += identity
            feats_list.append(identity)
            tensor = block.relu(out)
            feats_list.append(tensor)

        return tensor, feats_list

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    feats_list = []
    x, lf = forward_block(model.layer1, x)
    feats_list.extend(lf)
    x, lf = forward_block(model.layer2, x)
    feats_list.extend(lf)
    x, lf = forward_block(model.layer3, x)
    feats_list.extend(lf)
    x, lf = forward_block(model.layer4, x)
    feats_list.extend(lf)

    x = model.avgpool(x)
    x = x.view(x.size(0), -1)

    logits = model.fc(x)

    return logits, feats_list

def G_p(ob, p):
    # compute gram matrix power p
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp
