import torch
import numpy as np
from tqdm import tqdm
from models.evaluators.common import run_model, prepare_ood_labels, calc_ood_metrics, closed_set_accuracy
from models.evaluators.distance import compute_prototypes

def resend_evaluator(args, train_loader, test_loader, device, model, batch_size=32):
    train_logits, train_feats, train_lbls = run_model(args, model, train_loader, device, support=True)
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, support=False)

    # known labels have 1 for known samples and 0 for unknown ones
    known_labels = np.unique(train_lbls)
    n_known_classes = len(known_labels)
    prototypes = compute_prototypes(train_feats, train_lbls)
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")
    #TODO: implement memory checks
    prototypes = torch.from_numpy(prototypes).float()
    test_feats = torch.from_numpy(test_feats).float()
    prototypes = prototypes.to(device)
    test_feats = test_feats.to(device)
    predictions = torch.zeros((len(test_feats), len(prototypes)))

    nchunks = int(np.ceil(prototypes.shape[0]/batch_size))

    rel_module = model.module.cls_rel if args.distributed else model.cls_rel
    with torch.no_grad():
        for test_id, feat in enumerate(tqdm(test_feats)):
            chunks = prototypes.chunk(nchunks, dim=0)
            pos = 0

            for idx in range(nchunks): 
                chunk_protos = chunks[idx]
                chunk_size = len(chunk_protos)

                aggregated_batch = torch.cat((feat.expand(chunk_size, -1), chunk_protos), 1)
                out = - rel_module(aggregated_batch)
                predictions[test_id, pos:pos+chunk_size] = out.squeeze().cpu()
                pos += chunk_size

    MSP_normality_scores, cs_predictions = torch.nn.functional.softmax(predictions, dim=1).max(dim=1)
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(cs_predictions[known_mask], test_lbls[known_mask])
    metrics = calc_ood_metrics(MSP_normality_scores, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics 


    



