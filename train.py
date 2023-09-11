import argparse
import os
import sys
from os.path import join
from os import environ

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from models.resnet import get_resnet
from models.data_helper import get_eval_dataloader, get_train_dataloader, split_train_loader, check_data_consistency
from utils.log_utils import count_parameters, LogUnbuffered
from utils.optim import LinearWarmupCosineAnnealingLR
from utils.utils import clean_ckpt
from utils.dist_utils import is_main_process
from models.evaluators import *

try:
    import wandb 
except: 
    pass

def get_args():
    parser = argparse.ArgumentParser("OODDetectionBench")

    parser.add_argument("--local_rank", type=int)  # automatically passed by torch.distributed.launch

    parser.add_argument("--path_dataset", type=str, default='~/data', help="Base data path")

    parser.add_argument("--dataset", default="ImageNet", help="Dataset name",
                        choices=['DTD', 'DomainNet_Real', 'DomainNet_Painting', 'DomainNet_Sketch', 'Places', 
                                 'PACS_DG', 'PACS_SS_DG', 'imagenet_ood', 'imagenet_ood_small', 'DomainNet_DGv2',
                                 'MCM_benchmarks', 'PatternNet', 'SUN', 'ImageNet1k', "Stanford_Cars"])
    parser.add_argument("--support",
                        help="support split name")
    parser.add_argument("--test",
                        help="test split name")
    parser.add_argument("--data_order", type=int, default=-1, help="Which data order to use if more than one is available")

    # model parameters
    parser.add_argument("--network", type=str, default="resnet101", choices=["resnet101", "vit", "resend", 
                                                                             "resnetv2_101x3"])
    parser.add_argument("--model", type=str, default="CE", choices=["CE", "simclr", "supclr", "cutmix", "CSI", "supCSI", "clip", "DINO", 
                                                                    "resend", "DINOv2", "BiT", "CE-IM22k", "random_init", "CE-IM21k"])
    parser.add_argument("--evaluator", type=str, help="Strategy to compute normality scores", default="prototypes_distance",
                        choices=["prototypes_distance", "MSP", "MLS", "ODIN", "energy", "gradnorm", "mahalanobis", "gram", "knn_distance",
                                 "linear_probe", "MCM", "knn_ood", "resend", "react", "EVM", "EVM_norm", "ASH"])

    # evaluators-specific parameters 
    parser.add_argument("--NNK", help="K value to use for Knn distance evaluator", type=int, default=1)
    parser.add_argument("--disable_contrastive_head", action='store_true', default=False, help="Do not use contrastive head for distance-based evaluators")
    parser.add_argument("--disable_R2", action='store_true', default=False, help="Disable R2 computation for a slight speed up in evals")

    # data params
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training data loader")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for test data loaders")
    parser.add_argument("--few_shot", type=int, default=-1, help="Number of training samples for each class, -1 means use whole dataset")
    parser.add_argument("--rand_augment", action="store_true", default=False, help="Use RandAugment strategy for data augmentation")

    # finetuning params 
    parser.add_argument("--iterations", type=int, default=100, help="Number of finetuning iterations")
    parser.add_argument("--epochs", type=int, default=-1, help="Use value >= 0 if you want to specify training length in terms of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for finetuning, automatically multiplied by world size")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of lr warmup iterations")
    parser.add_argument("--warmup_epochs", type=int, default=-1, help="Number of lr warmup epochs, to use when train len specified with epochs")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Train only cls head during finetuning")
    parser.add_argument("--clip_grad", default=-1, type=float, help="If > 0 used as clip grad value")
    parser.add_argument("--resume", default=False, action='store_true', help="Resume from last ckpt")

    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing for loss computation")

    # run params 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # checkpoint evaluation
    parser.add_argument("--only_eval", action='store_true', default=False,
                        help="If you want only to evaluate a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to pretrained checkpoint")

    # output_folder for checkpoints
    parser.add_argument("--save_ckpt", action='store_true', default=False, help="Should save the training output checkpoint to --output_dir?")
    parser.add_argument("--output_dir", type=str, default="", help="Location for output checkpoints")
    parser.add_argument("--debug", action='store_true', default=False, help="Run in debug mode, disable file logger")

    # save run on wandb 
    parser.add_argument("--wandb", action='store_true', default=False, help="Save this run on wandb")

    # performance options 
    parser.add_argument("--on_disk", action='store_true', default=False, help="Save/Recover extracted features on the disk. To be used for really large ID datasets (e.g. ImageNet)")

    args = parser.parse_args()
    args.path_dataset = os.path.expanduser(args.path_dataset)

    return args

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.args.device = device

        # load the support and test set dataloaders 
        self.target_loader, self.support_test_loader, known_class_names, n_known_classes = get_eval_dataloader(self.args)
        self.known_class_names = known_class_names

        if self.args.evaluator == "gram":
            # this method needs a split of support data to be used as validation set 
            self.support_test_loader, self.support_val_loader = split_train_loader(self.support_test_loader, args.seed)

        self.n_known_classes = n_known_classes
        self.contrastive_enabled = False
        print(f"Model: {self.args.model}, Backbone: {self.args.network}")

        ckpt = torch.load(self.args.checkpoint_path, map_location=device) if self.args.checkpoint_path else None

        # setup the network and OOD model 
        if self.args.network == "resnet101":
            if self.args.model in ["CE", "cutmix"]:
                if self.args.model == "cutmix" and self.args.only_eval:
                    assert ckpt is not None, "Cannot perform eval without a pretrained model"

                self.model, self.output_num = get_resnet(self.args.network, n_known_classes=self.n_known_classes)

                # if ckpt fc size does not match current size discard it
                if ckpt is not None: 
                    old_size = ckpt["fc.bias"].shape[0]
                    if not old_size == self.n_known_classes:
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
                if not args.disable_contrastive_head:
                    self.output_num = self.model.contrastive_out_dim
                
                # if ckpt fc size does not match current size discard it
                if ckpt is not None: 
                    old_size = ckpt["base_model.fc.bias"].shape[0]
                    if not old_size == self.n_known_classes:
                        del ckpt["base_model.fc.weight"]
                        del ckpt["base_model.fc.bias"]

            elif self.args.model == "clip":
                import clip 

                model, preprocess = clip.load("RN101", self.device)
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
                self.output_num = 768

                if self.args.model == "DINO":
                    # we set num_classes to 0 in order to obtain pooled feats
                    model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=0)

                    # if we didn't need the contrastive head we could use the model from huggingface:
                    # https://huggingface.co/facebook/dino-vitb16
                    self.contrastive_enabled = not args.disable_contrastive_head
                    from models.common import WrapperWithContrastiveHead
                    self.model = WrapperWithContrastiveHead(model, out_dim=self.output_num, contrastive_type="DINO", 
                                                            add_cls_head=True, n_classes=self.n_known_classes)
                    if not args.disable_contrastive_head:
                        self.output_num = self.model.contrastive_out_dim

                else:
                    import types 

                    # we set num_classes to 0 in order to obtain pooled feats
                    model = timm.create_model("deit_base_patch16_224", pretrained=True)
                    model.fc = model.head

                    if not self.n_known_classes == 1000:
                        model.fc = nn.Linear(in_features=768, out_features=self.n_known_classes)

                    def my_forward(self, x):
                        feats = self.forward_head(self.forward_features(x), pre_logits=True)
                        logits = self.fc(feats)
                        return logits, feats

                    model.forward = types.MethodType(my_forward, model)
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

            elif self.args.model == "CE-IM22k" or self.args.model == "CE-IM21k":
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

            self.model = ReSeND(n_known_classes=self.n_known_classes)
            self.output_num = self.model.output_num
        else:
            raise NotImplementedError(f"Network {self.args.network} not implemented")

        if ckpt is not None: 
            print(f"Loading checkpoint {self.args.checkpoint_path}")
            missing, unexpected = self.model.load_state_dict(clean_ckpt(ckpt, self.model), strict=False)
            print(f"Missing keys: {missing}, unexpected keys: {unexpected}")

        self.to_device(self.device)
        self.args.output_num = self.output_num
        self.args.n_known_classes = self.n_known_classes

        if self.args.distributed:
            self.model = DDP(self.model, find_unused_parameters=True)

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

        args = self.args

        if self.args.evaluator == "prototypes_distance":
            metrics = prototypes_distance_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_enabled,
                                                    cosine_sim=cosine_similarity)
        elif self.args.evaluator == "MSP":
            metrics = MSP_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)
        elif self.args.evaluator == "MLS":
            metrics = MLS_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)
        elif self.args.evaluator == "ODIN":
            metrics = ODIN_evaluator(train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "energy":
            metrics = energy_evaluator(train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "gradnorm":
            metrics = gradnorm_evaluator(train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "react":
            metrics = react_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "mahalanobis":
            metrics = mahalanobis_evaluator(train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "gram":
            if self.args.only_eval:
                print("The gram evaluator exploits cls predictions on train data to estimate statistics that are later used for computing normality scores.")
                print("If the model is not finetuned the statistics will not be much relevant")
            metrics = gram_evaluator(train_loader=self.support_test_loader, val_loader=self.support_val_loader, 
                                    test_loader=self.target_loader, device=self.device, model=self.model, finetuned=not self.args.only_eval)

        elif self.args.evaluator == "knn_ood":
            metrics = knn_ood_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                        device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK)

        elif self.args.evaluator == "knn_distance":
            metrics = knn_distance_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK,
                                            cosine_sim=cosine_similarity)

        elif self.args.evaluator == "linear_probe":
            metrics = linear_probe_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled)

        elif self.args.evaluator == "EVM":
            metrics = EVM_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled)

        elif self.args.evaluator == "EVM_norm":
            metrics = EVM_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, normalize=True)

        elif self.args.evaluator == "MCM":
            assert self.args.model == "clip", "MCM evaluator supports only clip based models!"
            metrics = MCM_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, clip_model=self.clip_model, known_class_names=self.known_class_names)

        elif self.args.evaluator == "resend":
            metrics = resend_evaluator(args,train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "ASH":
            metrics = ASH_evaluator(args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        else:
            raise NotImplementedError(f"Unknown evaluator {self.args.evaluator}")

        if "cs_acc" in metrics:
            print(f"Closed set accuracy: {metrics['cs_acc']:.4f}")
        if "support_R2" in metrics:
            print(f"Support set R2 score: {metrics['support_R2']:.4f}")

        auroc = metrics["auroc"]
        fpr_auroc = metrics["fpr_at_95_tpr"]

        if self.args.wandb and is_main_process(self.args):
            wandb.log(metrics)

        print(f"Auroc,FPR95: {auroc:.4f},{fpr_auroc:.4f}")

    def save_ckpt(self, iteration, optimizer=None, scheduler=None):
        if self.args.save_ckpt and (not self.args.distributed or self.args.global_rank == 0):
            save_model = self.model.module if self.args.distributed else self.model
            ckpt_path = join(self.args.output_dir, "model_last.pth")
            torch.save(save_model.state_dict(), ckpt_path)
            print(f"Saved model to {ckpt_path}")

            if optimizer is not None: 
                aux_data = {}
                aux_data['optimizer'] = optimizer.state_dict()
                aux_data['scheduler'] = scheduler.state_dict()
                aux_data['last_iter'] = iteration
                aux_data_path = join(self.args.output_dir, "aux_data_last.pth")
                torch.save(aux_data, aux_data_path)

    def resume(self, optimizer, scheduler):
        ckpt_path = join(self.args.output_dir, "model_last.pth")
        aux_data_path = join(self.args.output_dir, "aux_data_last.pth")
        if not (os.path.isfile(ckpt_path) and os.path.isfile(aux_data_path)):
            return False, None, None, None

        ckpt_state_dict = torch.load(ckpt_path, map_location=self.device)
        if self.args.distributed:
            self.model.module.load_state_dict(ckpt_state_dict, strict=True)
        else:
            self.model.load_state_dict(ckpt_state_dict, strict=True)
        del ckpt_state_dict

        aux_data = torch.load(aux_data_path, map_location=self.device)
        optimizer.load_state_dict(aux_data["optimizer"])
        scheduler.load_state_dict(aux_data["scheduler"])
        last_iter = aux_data['last_iter']
        return True, optimizer, scheduler, last_iter

    def do_train(self):
        # prepare data 
        
        train_loader = get_train_dataloader(self.args)

        if self.args.evaluator == "gram":
            # this method needs a split of support data to be used as validation set 
            train_loader, _ = split_train_loader(train_loader, self.args.seed)

        check_data_consistency(train_loader, self.support_test_loader)

        if self.args.epochs > 0: 
            iters_per_epoch = len(train_loader)
            self.args.iterations = self.args.epochs * iters_per_epoch

            if self.args.warmup_epochs > 0:
                self.args.warmup_iters = self.args.warmup_epochs * iters_per_epoch

        if self.args.distributed:
            self.args.learning_rate *= self.args.n_gpus
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
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_iters, max_epochs=self.args.iterations)

        if self.args.resume:
            resume_possible, resumed_optim, resumed_sched, resume_it = self.resume(optimizer, scheduler)
            if not resume_possible:
                print("Cannot resume, ckpt does not exist or is not complete")
                start_it = 0
            else:
                optimizer = resumed_optim
                scheduler = resumed_sched
                start_it = resume_it + 1
                print(f"Training resumed from it {start_it}")
        else:
            start_it = 0

        # loss function 
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        train_iter = iter(train_loader)
        log_period = 10
        ckpt_period = 500
        avg_loss = 0
        print(f"Start training, lr={self.args.learning_rate}, start_iter={start_it}, iterations={self.args.iterations}, warmup_iters={self.args.warmup_iters}")
        for it in range(start_it, self.args.iterations):

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
            if self.args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)
            optimizer.step()
            scheduler.step()

            avg_loss += loss.item()
            if (it+1) % log_period == 0:
                _,preds = outputs.max(dim=1)
                train_acc = ((preds == labels).sum()/len(preds)).cpu().item()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iterations: {it+1:6d}/{self.args.iterations}\t LR: {current_lr:.6f}\t Loss: {avg_loss / log_period:6.4f} \t Acc: {train_acc:6.4f}")
                if self.args.wandb and is_main_process(self.args):

                    wandb.log({"lr": current_lr, "loss": avg_loss/log_period, "acc": train_acc}, step=it)

                avg_loss = 0

            if (it+1) % ckpt_period == 0:
                self.save_ckpt(it, optimizer, scheduler)

        # save final checkpoint 
        if self.args.save_ckpt and (not self.args.distributed or self.args.global_rank == 0):
            self.save_ckpt(self.args.iterations)

        optimizer.zero_grad()

@record
def main():
    args = get_args()

    if args.save_ckpt:
        assert args.output_dir, "You need to specify an output dir if you want the model to be saved"

    ### Set torch device ###
    if torch.cuda.is_available():
        if hasattr(args, 'local_rank') and not args.local_rank is None:
            assert False, "Please use torchrun for distributed execution"

        if "LOCAL_RANK" in environ:
            args.local_rank = int(environ["LOCAL_RANK"])
            args.distributed = True
            torch.cuda.set_device(args.local_rank)
        else:
            args.distributed = False
            args.n_gpus = 1

        device = torch.device("cuda")
    else:
        print("WARNING. Running in CPU mode")
        args.distributed = False
        device = torch.device("cpu")

    if args.distributed:
        dist.init_process_group('nccl')
        args.n_gpus = dist.get_world_size()
        args.global_rank = int(environ['RANK'])
        print("Process rank", args.global_rank, "starting")

    if args.output_dir and is_main_process(args):
        if not args.resume:
            assert not os.path.exists(args.output_dir), f"Output dir {args.output_dir} already exists, stopping to avoid overwriting"
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        stdout_file = join(args.output_dir, "stdout.txt")
        stderr_file = join(args.output_dir, "stderr.txt")
    else:
        stdout_file, stderr_file = None, None

    # print on both log file and stdout
    if not args.debug:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        sys.stdout = LogUnbuffered(args, orig_stdout, stdout_file)
        sys.stderr = LogUnbuffered(args, orig_stderr, stderr_file)

    if args.wandb and is_main_process(args):
        run_name=f"{args.network}_{args.model}_{args.evaluator}_{args.dataset}_{args.support}_{args.test}"
        if not args.data_order == -1:
            run_name += f"_{args.data_order}"

        wandb.init(
                project="OODDetectionFramework",
                config=vars(args),
                name=run_name)

    print("###############################################################################")
    print("######################### OOD Detection Benchmark #############################")
    print("###############################################################################")
    
    if args.evaluator in ["gram", "ODIN", "energy", "gradnorm", "mahalanobis", "react"]:
        assert not args.distributed, f"{args.evaluator} evaluator does not support distributed execution!"

    trainer = Trainer(args, device)
    if not args.only_eval:
        trainer.do_train()

    trainer.do_final_eval()

    if not args.debug:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
