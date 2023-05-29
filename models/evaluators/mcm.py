import torch
import clip
from models.evaluators.common import prepare_ood_labels, calc_ood_metrics, run_model
from models.common import normalize_feats

@torch.no_grad()
def MCM_evaluator(train_loader, test_loader, model, clip_model, device, known_class_names): 
    print("Running MCM evaluator with known class names:")
    print(known_class_names)
    # implements NeurIPS 2022: https://openreview.net/forum?id=KnCS9390Va
    # value verified from original paper
    temperature = 1

    # first we extract features for test data
    test_logits, test_feats, test_lbls = run_model(model, test_loader, device)
    test_feats = normalize_feats(test_feats)

    # known labels have 1 for known samples and 0 for unknown ones
    train_lbls = train_loader.dataset.labels
    known_labels = torch.unique(torch.tensor(train_lbls))
    ood_labels = prepare_ood_labels(known_labels, test_lbls)

    print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    # prepare "concept prototypes" by using known class names and text encoder
    # prompt taken from original implementation 
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in known_class_names]).to(device)
    
    concept_prototypes = clip_model.to(device).encode_text(text_inputs)
    concept_prototypes = normalize_feats(concept_prototypes).cpu()

    # compute the cos sim
    cos_sim_logits = torch.matmul(test_feats, concept_prototypes.T)

    # apply softmax with temperature
    probs = cos_sim_logits.type(torch.float).softmax(dim=-1)

    max_pred_prob, pred_cls = probs.max(dim=1)

    metrics = calc_ood_metrics(max_pred_prob, ood_labels)

    return metrics 


