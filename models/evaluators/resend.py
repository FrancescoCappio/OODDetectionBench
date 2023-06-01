import torch
import numpy as np
from tqdm import tqdm
from models.evaluators.common import run_model, prepare_ood_labels, calc_ood_metrics
from models.evaluators.distance import compute_prototypes

def resend_evaluator(train_loader, test_loader, device, model, batch_size=32):
    train_logits, train_feats, train_lbls = run_model(model, train_loader, device)
    test_logits, test_feats, test_lbls = run_model(model, test_loader, device)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = torch.unique(train_lbls)
    n_known_classes = len(known_labels)
    prototypes = compute_prototypes(train_feats, train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")
    #TODO: implement memory checks
    prototypes = prototypes.to(device)
    test_feats = test_feats.to(device)
    predictions = torch.zeros((len(test_feats), len(prototypes)))

    nchunks = int(np.ceil(prototypes.shape[0]/batch_size))

    with torch.no_grad():
        for test_id, feat in enumerate(tqdm(test_feats)):
            chunks = prototypes.chunk(nchunks, dim=0)
            pos = 0

            for idx in range(nchunks): 
                chunk_protos = chunks[idx]
                chunk_size = len(chunk_protos)

                aggregated_batch = torch.cat((feat.expand(chunk_size, -1), chunk_protos), 1)
                out = - model.cls_rel(aggregated_batch)
                predictions[test_id, pos:pos+chunk_size] = out.squeeze().cpu()
                pos += chunk_size

    MSP_normality_scores, _ = torch.nn.functional.softmax(predictions, dim=1).max(dim=1)
    metrics = calc_ood_metrics(MSP_normality_scores, ood_labels)

    return metrics 


    



