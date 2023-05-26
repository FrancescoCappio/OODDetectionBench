import argparse
from os.path import join
from numpy import test

import torch
from torch import nn, optim
from tqdm import tqdm
import os

from models.resnet import get_resnet
from models.data_helper import get_eval_dataloader, get_train_dataloader, split_train_loader, check_data_consistency
from utils.ckpt_utils import sanitize_ckpt, sanitize_simclr_ckpt, sanitize_CSI_ckpt
from utils.log_utils import count_parameters

from models.evaluators import *

def get_args():
    parser = argparse.ArgumentParser("OODDetectionBench")

    parser.add_argument("--local_rank", type=int)  # automatically passed by torch.distributed.launch

    parser.add_argument("--path_dataset", type=str, default='~/data', help="Base data path")

    parser.add_argument("--dataset", default="ImageNet", help="Dataset name",
                        choices=['DTD', 'DomainNet_Real', 'DomainNet_Painting', 'DomainNet_Sketch', 'Places', 
                                 'PACS_DG', 'PACS_SS_DG', 'imagenet_ood', 'imagenet_ood_small', 'DomainNet_DG',
                                 'MCM_benchmarks'])
    parser.add_argument("--source",
                        help="PACS_DG: no_ArtPainting, no_Cartoon, no_Photo, no_Sketch | PACS_SS_DG: Source")
    parser.add_argument("--target",
                        help="PACS_DG: ArtPainting, Cartoon, Photo, Sketch | PACS_SS_DG: ArtPainting, Cartoon, Photo")

    # model parameters
    parser.add_argument("--network", type=str, default="resnet101", choices=["resnet101", "vit"])
    parser.add_argument("--model", type=str, default="CE", choices=["CE", "simclr", "supclr", "cutmix", "CSI", "supCSI", "clip"])
    parser.add_argument("--evaluator", type=str, help="Strategy to compute normality scores", default="prototypes_distance",
                        choices=["prototypes_distance", "MSP", "ODIN", "energy", "gradnorm", "mahalanobis", "gram", "knn_distance",
                                 "linear_probe", "MCM"])

    # evaluators-specific parameters 
    parser.add_argument("--NNK", help="K value to use for Knn distance evaluator", type=int, default=1)

    # data params
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training data loader")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for test data loaders")
    parser.add_argument("--few_shot", type=int, default=-1, help="Number of training samples for each class, -1 means use whole dataset")

    # finetuning params 
    parser.add_argument("--iterations", type=int, default=100, help="Number of finetuning iterations")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate (fixed) for finetuning")

    # run params 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # checkpoint evaluation
    parser.add_argument("--only_eval", action='store_true', default=False,
                        help="If you want only to evaluate a checkpoint")
    parser.add_argument("--checkpoint_path", help="Path to pretrained checkpoint")

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
        self.contrastive_head = None
        print(f"Model: {self.args.model}, Backbone: {self.args.network}")

        # setup the network and OOD model 
        if self.args.network == "resnet101":
            if self.args.model == "CE":
                self.model, self.output_num = get_resnet(self.args.network, self.device, n_known_classes=self.n_known_classes)

            elif self.args.model in ["simclr", "supclr"]:
                ckpt_path = self.args.checkpoint_path
                ckpt = torch.load(ckpt_path)
                print('checkpoint loaded')
                self.model, self.output_num = get_resnet(self.args.network, self.device, ckpt=sanitize_simclr_ckpt(ckpt["resnet"])["backbone"], n_known_classes=self.n_known_classes)
                from models.common import SimCLRContrastiveHead
                self.contrastive_head = SimCLRContrastiveHead(channels_in=self.output_num)
                self.contrastive_head.load_state_dict(ckpt['head'],strict=True)
                self.contrastive_head = self.contrastive_head.to(device)

            elif self.args.model in ["CSI", "supCSI"]:
                ckpt_path = self.args.checkpoint_path
                ckpt = sanitize_CSI_ckpt(torch.load(ckpt_path))
                print('checkpoint loaded')
                self.model, self.output_num = get_resnet(self.args.network, self.device, ckpt=ckpt['backbone'], n_known_classes=self.n_known_classes)
                from models.common import CSIContrastiveHead
                self.contrastive_head = CSIContrastiveHead(channels_in=self.output_num)
                self.contrastive_head.load_state_dict(ckpt['head'],strict=True)
                self.contrastive_head = self.contrastive_head.to(device)

            elif self.args.model == "cutmix":
                ckpt = torch.load(args.checkpoint_path)
                print('checkpoint loaded')
                ckpt = sanitize_ckpt(ckpt['state_dict'])
                self.model, self.output_num = get_resnet(self.args.network, self.device, ckpt=ckpt, n_known_classes=self.n_known_classes)
            
            elif self.args.model == "clip":
                import clip 
                import types

                model, preprocess = clip.load("RN101", device)
                self.clip_preprocessor = preprocess
                # substitute preprocess with CLIP's one
                self.substitute_val_preprocessor(preprocess)
                self.clip_model = model

                # the model has no fc by default, so it does not support closed set finetuning
                from models.common import WrapperWithFC
                self.output_num = 512
                self.model = WrapperWithFC(model.visual, self.output_num, self.n_known_classes, half_precision=True)
                self.model = self.model.to(device)
            else:
                raise NotImplementedError(f"Model {self.args.model} is not supported with network {self.args.network}")

        elif self.args.network == "vit":
            if self.args.model == "CE":

                import timm
                import types
                model = timm.create_model("deit_base_patch16_224",pretrained=True)

                def my_forward(self, x):
                    net_feats = self.forward_features(x)
                    feats = self.forward_head(net_feats,pre_logits=True)
                    x = self.forward_head(net_feats)
                    return x, feats

                model.forward = types.MethodType(my_forward, model)
                self.model = model.to(device)
                self.output_num = 768

            elif self.args.model == "clip":
                # ViT-L/14
                import clip 
                import types

                model, preprocess = clip.load("ViT-L/14", device)
                self.clip_preprocessor = preprocess
                # substitute preprocess with CLIP's one
                self.substitute_val_preprocessor(preprocess)
                self.clip_model = model
                # the model has no fc by default, so it does not support closed set finetuning
                from models.common import WrapperWithFC
                self.output_num = 768
                self.model = WrapperWithFC(model.visual, self.output_num, self.n_known_classes, half_precision=True)
                self.model = self.model.to(device)
            else:
                raise NotImplementedError(f"Model {self.args.model} is not supported with network {self.args.network}")
        else:
            raise NotImplementedError(f"Network {self.args.network} not implemented")

        print("Number of parameters: ", count_parameters(self.model))

    def substitute_val_preprocessor(self, preprocess):
        self.target_loader.dataset._image_transformer = preprocess
        if hasattr(self, "source_loader_val"):
            self.source_loader_test.dataset.dataset._image_transformer = preprocess
            self.source_loader_val.dataset.dataset._image_transformer = preprocess
        else:
            self.source_loader_test.dataset._image_transformer = preprocess

    def to_device(self, device):
        self.model = self.model.to(device)

    def to_eval(self):
        self.model.eval()
        if self.contrastive_head is not None:
            self.contrastive_head.eval()

    def to_train(self):
        self.model.train()
        if self.contrastive_head is not None:
            self.contrastive_head.train()

    def do_final_eval(self):
        self.to_eval()

        if self.args.evaluator == "prototypes_distance":
            metrics = prototypes_distance_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_head)
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

        elif self.args.evaluator == "knn_distance":
            metrics = knn_distance_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_head, K=self.args.NNK)

        elif self.args.evaluator == "linear_probe":
            metrics = linear_probe_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_head)

        elif self.args.evaluator == "MCM":
            assert self.args.model == "clip", "MCM evaluator supports only clip based models!"
            metrics = MCM_evaluator(train_loader=self.source_loader_test, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, clip_model=self.clip_model, known_class_names=self.known_class_names)

        else:
            raise NotImplementedError(f"Unknown evaluator {self.args.evaluator}")

        
        auroc = metrics["auroc"]
        fpr_auroc = metrics["fpr_at_95_tpr"]
        print(f"Auroc,FPR95: {auroc:.4f},{fpr_auroc:.4f}")

    def do_train(self):
        # prepare data 
        
        train_loader = get_train_dataloader(self.args)
        check_data_consistency(train_loader, self.source_loader_test)

        if self.args.model == "clip":
            # clip needs its own preprocessing pipeline
            train_loader.dataset._image_transformer = self.clip_preprocessor

        if self.args.evaluator == "gram":
            # this method needs a split of source data to be used as validation set 
            train_loader, _ = split_train_loader(train_loader, self.args.seed)

        self.to_train()

        # prepare optimizer 
        optimizer = optim.SGD(self.model.parameters(), weight_decay=.0005, momentum=.9, lr=self.args.learning_rate)
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
            outputs, feats = self.model(images)

            loss = loss_fn(outputs, labels.to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if (it+1) % log_period == 0:
                print(f"Iterations: {it+1:6d}/{self.args.iterations}\t Loss: {avg_loss / log_period:6.4f}")
                avg_loss = 0

        # save final checkpoint 
        if self.args.save_ckpt:
            os.makedirs(self.args.output_dir)
            torch.save(self.model.state_dict(), join(self.args.output_dir,"model_last.pth"))


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
