import argparse
from os.path import join

import torch
from torch import nn, optim
import os
import sys
from models.evaluators.common import run_model, prepare_ood_labels, closed_set_accuracy

from models.resnet import get_resnet
from models.data_helper import get_eval_dataloader, get_train_dataloader, split_train_loader, check_data_consistency
from utils.log_utils import count_parameters

from models.evaluators import *

def get_args():
    parser = argparse.ArgumentParser("OODDetectionBench")

    parser.add_argument("--local_rank", type=int)  # automatically passed by torch.distributed.launch

    parser.add_argument("--path_dataset", type=str, default='~/data', help="Base data path")

    parser.add_argument("--dataset", default="ImageNet", help="Dataset name",
                        choices=['DTD', 'DomainNet_Real', 'DomainNet_Painting', 'DomainNet_Sketch', 'Places', 
                                 'PACS_DG', 'PACS_SS_DG', 'imagenet_ood', 'imagenet_ood_small', 'DomainNet_DGv2',
                                 'MCM_benchmarks', 'PatternNet', 'SUN', 'ImageNet1k'])
    parser.add_argument("--source",
                        help="PACS_DG: no_ArtPainting, no_Cartoon, no_Photo, no_Sketch | PACS_SS_DG: Source")
    parser.add_argument("--target",
                        help="PACS_DG: ArtPainting, Cartoon, Photo, Sketch | PACS_SS_DG: ArtPainting, Cartoon, Photo")

    # model parameters
    parser.add_argument("--network", type=str, default="resnet101", choices=["resnet101", "vit", "resend", 
                                                                             "resnetv2_101x3"])
    parser.add_argument("--model", type=str, default="CE", choices=["CE", "simclr", "supclr", "cutmix", "CSI", "supCSI", "clip", "DINO", 
                                                                    "resend", "DINOv2", "BiT", "CE-IM22k", "random_init"])
    parser.add_argument("--evaluator", type=str, help="Strategy to compute normality scores", default="prototypes_distance",
                        choices=["prototypes_distance", "MSP", "ODIN", "energy", "gradnorm", "mahalanobis", "gram", "knn_distance",
                                 "linear_probe", "MCM", "knn_ood", "resend"])

    # evaluators-specific parameters 
    parser.add_argument("--NNK", help="K value to use for Knn distance evaluator", type=int, default=1)
    parser.add_argument("--disable_contrastive_head", action='store_true', default=False, help="Do not use contrastive head for distance-based evaluators")

    # data params
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training data loader")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for test data loaders")
    parser.add_argument("--few_shot", type=int, default=-1, help="Number of training samples for each class, -1 means use whole dataset")

    # finetuning params 
    parser.add_argument("--iterations", type=int, default=100, help="Number of finetuning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate (fixed) for finetuning")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Train only cls head during finetuning")

    # run params 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # checkpoint evaluation
    parser.add_argument("--only_eval", action='store_true', default=False,
                        help="If you want only to evaluate a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to pretrained checkpoint")

    # output_folder for checkpoints
    parser.add_argument("--save_ckpt", action='store_true', default=False, help="Should save the training output checkpoint to --output_dir?")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Location for output checkpoints")

    args = parser.parse_args()
    args.path_dataset = os.path.expanduser(args.path_dataset)

    return args

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.args.device = device

        # load the support and test set dataloaders 
        self.target_loader, self.source_loader_test, known_class_names, n_known_classes = get_eval_dataloader(self.args)
        self.known_class_names = known_class_names

        if self.args.evaluator == "gram":
            # this method needs a split of source data to be used as validation set 
            self.source_loader_test, self.source_loader_val = split_train_loader(self.source_loader_test, args.seed)

        self.n_known_classes = n_known_classes
        self.contrastive_enabled = False
        print(f"Model: {self.args.model}, Backbone: {self.args.network}")

        ckpt = torch.load(self.args.checkpoint_path) if self.args.checkpoint_path else None

        # setup the network and OOD model 
        if self.args.network == "resnet101":
            if self.args.model in ["CE", "cutmix"]:
                if self.args.model == "cutmix" and self.args.only_eval:
                    assert ckpt is not None, "Cannot perform eval without a pretrained model"

                self.model, self.output_num = get_resnet(self.args.network, n_known_classes=self.n_known_classes)

                # if ckpt fc size does not match current size discard it
                if ckpt is not None: 
                    old_size = ckpt["fc.bias"].shape
                    if not old_size[0] == self.n_known_classes:
                        del ckpt["fc.weight"]
                        del ckpt["fc.bias"]
            elif self.args.model == "random_init":
                self.model, self.output_num = get_resnet(self.args.network, n_known_classes=self.n_known_classes, random_init=True)

            elif self.args.model in ["simclr", "supclr", "CSI", "supCSI"]:
                self.contrastive_enabled = not args.disable_contrastive_head
                if self.args.only_eval:
                    assert ckpt is not None, "Cannot perform eval without a pretrained model"

                contrastive_type = "simclr" if self.args.model in ["simclr", "supclr"] else "CSI"

                from models.common import WrapperWithContrastiveHead
                base_model, self.output_num = get_resnet(self.args.network, n_known_classes=self.n_known_classes)
                self.model = WrapperWithContrastiveHead(base_model, out_dim=self.output_num, contrastive_type=contrastive_type)
                
                # if ckpt fc size does not match current size discard it
                if ckpt is not None: 
                    old_size = ckpt["base_model.fc.bias"].shape
                    if not old_size == self.n_known_classes:
                        del ckpt["base_model.fc.weight"]
                        del ckpt["base_model.fc.bias"]

            elif self.args.model == "clip":
                import clip 

                model, preprocess = clip.load("RN101", self.device)
                self.clip_preprocessor = preprocess
                # substitute preprocess with CLIP's one
                self.substitute_val_preprocessor(preprocess)
                self.clip_model = model

                # the model has no fc by default, so it does not support closed set finetuning
                from models.common import WrapperWithFC
                self.output_num = 512
                self.model = WrapperWithFC(model.visual, self.output_num, self.n_known_classes, half_precision=True)
            else:
                raise NotImplementedError(f"Model {self.args.model} is not supported with network {self.args.network}")

        elif self.args.network == "vit":
            if self.args.model in ["CE", "DINO"]:

                import timm
                # we set num_classes to 0 in order to obtain pooled feats
                model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=0)
                self.output_num = 768

                if self.args.model == "DINO":
                    # if we didn't need the contrastive head we could use the model from huggingface:
                    # https://huggingface.co/facebook/dino-vitb16
                    self.contrastive_enabled = not args.disable_contrastive_head
                    from models.common import WrapperWithContrastiveHead
                    self.model = WrapperWithContrastiveHead(model, out_dim=self.output_num, contrastive_type="DINO", 
                                                            add_cls_head=True, n_classes=self.n_known_classes)
                else:
                    from models.common import WrapperWithFC
                    model = WrapperWithFC(model, self.output_num, self.n_known_classes)
                    self.model = model 

            elif self.args.model == "clip":
                # ViT-B/16
                import clip 

                model, preprocess = clip.load("ViT-L/14", self.device)
                self.clip_model = model
                # the model has no fc by default, so it does not support closed set finetuning
                from models.common import WrapperWithFC
                self.output_num = 768
                self.model = WrapperWithFC(model.visual, self.output_num, self.n_known_classes, half_precision=True)

            elif self.args.model == "DINOv2":
                dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                from models.common import WrapperWithFC
                self.output_num = 1024
                self.model = WrapperWithFC(dinov2_vitb14, self.output_num, self.n_known_classes)

            elif self.args.model == "CE-IM22k":
                # https://huggingface.co/google/vit-large-patch16-224-in21k
                from transformers import ViTModel
                model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
                self.output_num = 1024
                from models.common import WrapperWithFC
                self.model = WrapperWithFC(model, self.output_num, self.n_known_classes, base_output_map=lambda x: x["pooler_output"])
            else:
                raise NotImplementedError(f"Model {self.args.model} is not supported with network {self.args.network}")
        
        elif self.args.network == "resnetv2_101x3":
            assert self.args.model == "BiT", f"The network {self.args.network} supports only BiT model"

            # we set num_classes to 0 in order to obtain pooled feats
            import timm 
            from models.common import WrapperWithFC
            # https://huggingface.co/timm/resnetv2_101x3_bit.goog_in21k
            model = timm.create_model('resnetv2_101x3_bit.goog_in21k', pretrained=True, num_classes=0)
            self.output_num = 6144

            self.model = WrapperWithFC(model, self.output_num, self.n_known_classes)

        elif self.args.network == "resend":
            assert self.args.only_eval and ckpt is not None, "Cannot perform eval without a pretrained model"

            from models.resend import ReSeND

            self.model = ReSeND()
            self.output_num = self.model.output_num
        else:
            raise NotImplementedError(f"Network {self.args.network} not implemented")

        if ckpt is not None: 
            print(f"Loading checkpoint {self.args.checkpoint_path}")
            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
            print(f"Missing keys: {missing}, unexpected keys: {unexpected}")

        self.to_device(self.device)

        print("Number of parameters: ", count_parameters(self.model))

    def to_device(self, device):
        self.model = self.model.to(device)

    def to_eval(self):
        self.model.eval()

    def to_train(self):
        self.model.train()

    def do_final_eval(self):
        self.to_eval()

        # when using the contrastive head and a model trained with cosine-similarity-based loss we need to use cosine
        # similarity based scores instead of L2 distance based ones
        cosine_similarity = self.args.model in ["simclr", "supclr", "CSI", "supCSI"] and self.contrastive_enabled

        if self.args.evaluator == "prototypes_distance":
            metrics = prototypes_distance_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_enabled,
                                                    cosine_sim=cosine_similarity)
        elif self.args.evaluator == "MSP":
            metrics = MSP_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "ODIN":
            metrics = ODIN_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "energy":
            metrics = energy_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "gradnorm":
            metrics = gradnorm_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "mahalanobis":
            metrics = mahalanobis_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "gram":
            if self.args.only_eval:
                print("The gram evaluator exploits cls predictions on train data to estimate statistics that are later used for computing normality scores.")
                print("If the model is not finetuned the statistics will not be much relevant")
            metrics = gram_evaluator(train_loader=self.source_loader_test, val_loader=self.source_loader_val, 
                                    test_loader=self.target_loader, device=self.device, model=self.model, finetuned=not self.args.only_eval)

        elif self.args.evaluator == "knn_ood":
            metrics = knn_ood_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                        device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK)

        elif self.args.evaluator == "knn_distance":
            metrics = knn_distance_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK,
                                            cosine_sim=cosine_similarity)

        elif self.args.evaluator == "linear_probe":
            metrics = linear_probe_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled)

        elif self.args.evaluator == "MCM":
            assert self.args.model == "clip", "MCM evaluator supports only clip based models!"
            metrics = MCM_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, clip_model=self.clip_model, known_class_names=self.known_class_names)

        elif self.args.evaluator == "resend":
            metrics = resend_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        else:
            raise NotImplementedError(f"Unknown evaluator {self.args.evaluator}")

        if "cs_acc" in metrics:
            print(f"Closed set accuracy: {metrics['cs_acc']:.4f}")

        auroc = metrics["auroc"]
        fpr_auroc = metrics["fpr_at_95_tpr"]
        print(f"Auroc,FPR95: {auroc:.4f},{fpr_auroc:.4f}")

    def do_train(self):
        # prepare data 
        
        train_loader = get_train_dataloader(self.args)

        if self.args.evaluator == "gram":
            # this method needs a split of source data to be used as validation set 
            train_loader, _ = split_train_loader(train_loader, self.args.seed)

        check_data_consistency(train_loader, self.source_loader_test)

        self.to_train()

        # prepare optimizer 
        optim_params = self.model.parameters()
        if self.args.freeze_backbone:
            if hasattr(self.model, "fc"):
                optim_params = self.model.fc.parameters()
            elif hasattr(self.model, "base_model") and hasattr(self.model.base_model, "fc"):
                optim_params = self.model.base_model.fc.parameters()
            else:
                raise NotImplementedError("Don't know how to access fc")

        optimizer = optim.SGD(optim_params, weight_decay=.0005, momentum=.9, lr=self.args.learning_rate)
        # loss function 
        loss_fn = nn.CrossEntropyLoss()

        train_iter = iter(train_loader)
        log_period = 10
        avg_loss = 0
        print("Start training")
        for it in range(self.args.iterations):

            try: 
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images, labels = batch 
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs, feats = self.model(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if (it+1) % log_period == 0:
                _,preds = outputs.max(dim=1)
                train_acc = ((preds == labels).sum()/len(preds)).cpu().item()
                print(f"Iterations: {it+1:6d}/{self.args.iterations}\t Loss: {avg_loss / log_period:6.4f} \t Acc: {train_acc:6.4f}")
                avg_loss = 0

        # save final checkpoint 
        if self.args.save_ckpt:
            os.makedirs(self.args.output_dir)
            torch.save(self.model.state_dict(), join(self.args.output_dir,"model_last.pth"))

    @torch.no_grad()
    def do_periodic_eval(self):
        self.to_eval()
        correct = 0
        count = 0
        for batch in self.target_loader:
            images, labels = batch 
            images = images.to(self.device)
            outputs, _ = self.model(images)
            _, preds = outputs.max(dim=1)
            mask = labels < self.n_known_classes
            correct += (preds[mask].cpu() == labels[mask]).sum()
            count += len(labels[mask])

        print(f"Test acc {correct/count:.4f}")

        self.to_train()

def main():
    args = get_args()

    print("###############################################################################")
    print("######################### OOD Detection Benchmark #############################")
    print("###############################################################################")
    ### Set torch device ###
    if torch.cuda.is_available():
        if not hasattr(args, 'local_rank') or args.local_rank is None:
            args.distributed = False
            args.n_gpus = 0
        else:
            torch.cuda.set_device(args.local_rank)
            args.distributed = True
        device = torch.device("cuda")
    else:
        print("WARNING. Running in CPU mode")
        args.distributed = False
        device = torch.device("cpu")

    assert not args.distributed, "This code does not support distributed execution!"

    if not args.only_eval:
        if args.save_ckpt:
            assert not os.path.exists(args.output_dir), "Output dir {ckpt_dir} already exists, stopping to avoid overwriting"

    trainer = Trainer(args, device)
    if not args.only_eval:
        trainer.do_train()

    trainer.do_final_eval()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
