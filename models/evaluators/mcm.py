import torch
import clip
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model, closed_set_accuracy
from models.common import normalize_feats
from utils.log_utils import CompProfiler
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def MCM_evaluator(args, train_loader, test_loader, model, clip_model, device, known_class_names): 
    print("Running MCM evaluator with known class names:")
    print(known_class_names)
    # implements NeurIPS 2022: https://openreview.net/forum?id=KnCS9390Va
    # value verified from original paper
    temperature = 1

    # first we extract features for test data
    test_logits, test_feats, test_lbls = run_model(args, model, test_loader, device, support=False)
    test_feats = normalize_feats(test_feats)

    # known labels have 1 for known samples and 0 for unknown ones
    train_lbls = train_loader.dataset.labels
    known_labels = torch.unique(torch.tensor(train_lbls))
    ood_labels = prepare_ood_labels(known_labels.numpy(), test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    # prepare "concept prototypes" by using known class names and text encoder
    # prompt taken from original implementation 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in known_class_names]).to(device)
    
    concept_prototypes = clip_model.to(device).encode_text(text_inputs)
    concept_prototypes = normalize_feats(concept_prototypes)
    test_feats = torch.from_numpy(test_feats).to(device).half()

    # compute the cos sim
    if args.profile:
        profiler = CompProfiler()
        test_feats = test_feats.cpu().numpy()
        concept_prototypes = concept_prototypes.cpu().numpy()
        cos_sim_logits = np.zeros((len(test_feats),len(concept_prototypes)))
        for test_id, test_f in enumerate(tqdm(test_feats)):
            for proto_id, proto in enumerate(concept_prototypes):
                profiler.start()
                dist = (test_f*proto).sum()
                profiler.end()
                cos_sim_logits[test_id][proto_id] = dist
        print(profiler)
        cos_sim_logits = torch.from_numpy(cos_sim_logits)
    else:
        cos_sim_logits = torch.matmul(test_feats, concept_prototypes.T)

    # apply softmax with temperature
    probs = cos_sim_logits.type(torch.float).softmax(dim=-1).cpu()

    max_pred_prob, pred_cls = probs.max(dim=1)
    known_mask = ood_labels == 1
    cs_acc = closed_set_accuracy(pred_cls[known_mask], test_lbls[known_mask])

    metrics = calc_ood_metrics(max_pred_prob, ood_labels)
    metrics["cs_acc"] = cs_acc

    return metrics 


