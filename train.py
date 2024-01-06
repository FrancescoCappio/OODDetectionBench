import argparse
import os
import sys
from os import environ
from os.path import join

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from models.data_helper import check_data_consistency, get_eval_dataloader, get_train_dataloader, split_train_loader
from models.evaluators import *
from models.models_helper import get_model
from utils.dist_utils import is_main_process
from utils.log_utils import LogUnbuffered, count_parameters, gen_train_log_msg
from utils.optim import LinearWarmupCosineAnnealingLR
from utils.utils import get_aux_modules_dict

try:
    import wandb 
except: 
    pass

def get_args():
    parser = argparse.ArgumentParser("OODDetectionBench")

    parser.add_argument("--local_rank", type=int)  # automatically passed by torch.distributed.launch

    parser.add_argument("--path_dataset", type=str, default=None, help="Base data path")

    parser.add_argument("--dataset", help="Dataset name",
                        choices=["domainnet", "dtd", "patternnet", "stanford_cars", "sun", "mcm_bench"])
    parser.add_argument("--support", help="support split name")
    parser.add_argument("--test", help="test split name")
    parser.add_argument("--data_order", type=int, default=-1, help="Which data order to use if more than one is available")

    # model parameters
    parser.add_argument("--network", type=str, default="resnet101", choices=["resnet101", "vit", "resend", "resnetv2_101x3"])
    parser.add_argument("--model", type=str, default="CE", choices=["CE", "simclr", "supclr", "CSI", "supCSI", "clip", "DINO", 
                                                                    "resend", "DINOv2", "BiT", "CE-IM22k", "CE-IM21k", "random_init"])
    parser.add_argument("--evaluator", type=str, help="Strategy to compute normality scores", default="prototypes_distance",
                        choices=["prototypes_distance", "MSP", "MLS", "ODIN", "energy", "gradnorm", "mahalanobis", "gram", "knn_distance",
                                 "linear_probe", "MCM", "knn_ood", "resend", "react", "flow", "EVM", "EVM_norm", "ASH"])

    # evaluators-specific parameters 
    parser.add_argument("--NNK", help="K value to use for Knn distance evaluator", type=int, default=1)
    parser.add_argument("--k_means", type=int, default=-1, help="Number of centroids for Knn distance evaluator (if any)")
    parser.add_argument("--disable_contrastive_head", action='store_true', default=False, help="Do not use contrastive head for distance-based evaluators")
    parser.add_argument("--disable_R2", action='store_true', default=False, help="Disable R2 computation for a slight speed up in evals")
    parser.add_argument("--enable_TSNE", action='store_true', default=False, help="Plot and save t-SNE representations of extracted features")
    parser.add_argument("--enable_ratio_NN_unknown", action='store_true', default=False, help="Compute ratio of test OOD samples whose NN is another OOD sample")
    parser.add_argument("--enable_ranking_index", action='store_true', default=False, help="Compute ranking index")

    # data params
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training data loader")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for test data loaders")
    parser.add_argument("--few_shot", type=int, default=-1, help="Number of training samples for each class, -1 means use whole dataset")
    parser.add_argument("--rand_augment", action="store_true", default=False, help="Use RandAugment strategy for data augmentation")

    # finetuning params 
    parser.add_argument("--iterations", type=int, default=100, help="Number of finetuning iterations")
    parser.add_argument("--epochs", type=int, default=-1, help="Use value >= 0 if you want to specify training length in terms of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for finetuning, automatically multiplied by world size")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of lr warmup iterations")
    parser.add_argument("--warmup_epochs", type=int, default=-1, help="Number of lr warmup epochs, to use when train len specified with epochs")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Train only cls head during finetuning")
    parser.add_argument("--clip_grad", default=-1, type=float, help="If > 0 used as clip grad value")
    parser.add_argument("--resume", default=False, action='store_true', help="Resume from last ckpt")
    parser.add_argument("--label_smoothing", type=float, default=0, help="Label smoothing for loss computation")

    # NF
    parser.add_argument("--nf_head", action="store_true", default=False)
    parser.add_argument("--nf_lr_mult", type=float, default=1, help="LR multiplier for flow head optimizer")
    parser.add_argument("--nf_clip_grad", default=-1, type=float, help="If > 0 used as clip grad value (for NF head)")

    # run params 
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")

    # checkpoint evaluation
    parser.add_argument("--only_eval", action='store_true', default=False,
                        help="If you want only to evaluate a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to pretrained checkpoint")

    # WiSE-FT
    parser.add_argument("--wise_ft_alpha", type=float, default=1, help="Alpha value for WiSE-FT interpolation")
    parser.add_argument("--wise_ft_zs_ckpt_path", type=str, default="",
                        help="Path to zeroshot checkpoint for WiSE-FT interpolation (optional)")

    # output_folder for checkpoints
    parser.add_argument("--save_ckpt", action='store_true', default=False, help="Should save the training output checkpoint to --output_dir?")
    parser.add_argument("--output_dir", type=str, default="", help="Location for output checkpoints")
    parser.add_argument("--debug", action='store_true', default=False, help="Run in debug mode, disable file logger")
    parser.add_argument("--print_args", action='store_true', default=False, help="Print args to stdout")

    # save run on wandb 
    parser.add_argument("--wandb", action='store_true', default=False, help="Save this run on wandb")
    parser.add_argument("--suffix", type=str, default="", help="Additional suffix for the run name on wandb")

    # performance options 
    parser.add_argument("--on_disk", action='store_true', default=False, help="Save/Recover extracted features on the disk. To be used for really large ID datasets (e.g. ImageNet)")
    
    args = parser.parse_args()

    return args

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = self.args.device = device

        ### Data ###

        # load the support and test set dataloaders 
        self.target_loader, self.support_test_loader, self.known_class_names, self.n_known_classes = get_eval_dataloader(self.args)
        if self.args.evaluator == "gram":
            # this method needs a split of support data to be used as validation set 
            self.support_test_loader, self.support_val_loader = split_train_loader(self.support_test_loader, args.seed)
        self.args.n_known_classes = self.n_known_classes

        ### Model ###

        print(f"Model: {self.args.model}, Backbone: {self.args.network}")

        # setup the network and OOD model 
        self.model, self.output_num, self.contrastive_enabled = get_model(self.args)
        if args.model == "clip":
            self.clip_model = self.model
        self.args.output_num = self.output_num

        self.to_device(self.device)

        self.raw_model = self.model # maintain access to the "raw" internal module in case of DDP
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

        if self.args.evaluator == "prototypes_distance":
            metrics = prototypes_distance_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, contrastive_head=self.contrastive_enabled,
                                                    cosine_sim=cosine_similarity)
        elif self.args.evaluator == "MSP":
            metrics = MSP_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)
        elif self.args.evaluator == "MLS":
            metrics = MLS_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
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
            metrics = react_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
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
            metrics = knn_ood_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                        device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK,
                                        k_means=self.args.k_means)

        elif self.args.evaluator == "knn_distance":
            metrics = knn_distance_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, K=self.args.NNK,
                                            k_means=self.args.k_means, cosine_sim=cosine_similarity)

        elif self.args.evaluator == "linear_probe":
            metrics = linear_probe_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled)

        elif self.args.evaluator == "EVM":
            metrics = EVM_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled)

        elif self.args.evaluator == "EVM_norm":
            metrics = EVM_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                            device=self.device, model=self.model, contrastive_head=self.contrastive_enabled, normalize=True)

        elif self.args.evaluator == "MCM":
            assert self.args.model == "clip", "MCM evaluator supports only clip based models!"
            metrics = MCM_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model, clip_model=self.clip_model, known_class_names=self.known_class_names)

        elif self.args.evaluator == "resend":
            metrics = resend_evaluator(self.args,train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "ASH":
            metrics = ASH_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader,
                                                    device=self.device, model=self.model)

        elif self.args.evaluator == "flow":
            metrics = flow_evaluator(self.args, train_loader=self.support_test_loader, test_loader=self.target_loader, device=self.device,
                                            model=self.model)

        else:
            raise NotImplementedError(f"Unknown evaluator {self.args.evaluator}")

        optional_metrics = [
            ("cs_acc", "Closed set accuracy"),
            ("support_R2", "Support set R2 score"),
            ("id_ood_R2", "Test ID-OOD R2 score"),
            ("ratio_NN_unknown", "Ratio NN unknown"),
            ("ranking_index", "Ranking index"),
            ("avg_dist", "Avg dist"),
            ("avg_dist_id", "Avg dist ID"),
            ("avg_dist_ood", "Avg dist OOD"),
        ]

        for id, name in optional_metrics:
            if id in metrics:
                print(f"{name}: {metrics[id]:.4f}")

        auroc = metrics["auroc"]
        fpr_auroc = metrics["fpr_at_95_tpr"]
        print(f"Auroc,FPR95: {auroc:.4f},{fpr_auroc:.4f}")

        if self.args.wandb and is_main_process(self.args):
            wandb.log(metrics)

    def save_ckpt(self, aux_modules=None, iteration=None):
        if self.args.save_ckpt and (not self.args.distributed or self.args.global_rank == 0):
            ckpt_path = join(self.args.output_dir, "model_last.pth")
            torch.save(self.raw_model.state_dict(), ckpt_path)
            print(f"Saved model to {ckpt_path}")

            if aux_modules:
                assert iteration is not None, "Specify iteration when saving additional modules"
                aux_data = {module_name: module.state_dict() for module_name, module in aux_modules.items()}
                aux_data["last_iter"] = iteration
                aux_data_path = join(self.args.output_dir, "aux_data_last.pth")
                torch.save(aux_data, aux_data_path)

    def resume(self, aux_modules):
        ckpt_path = join(self.args.output_dir, "model_last.pth")
        aux_data_path = join(self.args.output_dir, "aux_data_last.pth")
        if not (os.path.isfile(ckpt_path) and os.path.isfile(aux_data_path)):
            return False, None

        ckpt_state_dict = torch.load(ckpt_path, map_location=self.device)
        self.raw_model.load_state_dict(ckpt_state_dict, strict=True)
        del ckpt_state_dict

        aux_data = torch.load(aux_data_path, map_location=self.device)
        for module_name, module in aux_modules.items():
            module.load_state_dict(aux_data[module_name])
        last_iter = aux_data["last_iter"]
        return True, last_iter

    def do_train(self):
        self.to_train()

        ### Prepare data ###
        
        train_loader = get_train_dataloader(self.args)

        if self.args.evaluator == "gram":
            # this method needs a split of support data to be used as validation set 
            train_loader, _ = split_train_loader(train_loader, self.args.seed)

        check_data_consistency(train_loader, self.support_test_loader)

        ### Adjust n_iters and lr ###

        if self.args.epochs > 0: 
            iters_per_epoch = len(train_loader)
            self.args.iterations = self.args.epochs * iters_per_epoch
            if self.args.warmup_epochs > 0:
                self.args.warmup_iters = self.args.warmup_epochs * iters_per_epoch

        if self.args.distributed:
            self.args.learning_rate *= self.args.n_gpus

        ### Prepare loss, optimizer and scheduler ###

        # loss function 
        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        def get_optim_sched(params, lr):
            optim_ = optim.Adam(params, lr=lr, weight_decay=1e-5)
            sched_ = LinearWarmupCosineAnnealingLR(optim_, warmup_epochs=self.args.warmup_iters, max_epochs=self.args.iterations)
            return optim_, sched_

        if self.args.freeze_backbone:
            if hasattr(self.raw_model, "fc"):
                optim_params = self.raw_model.fc.parameters()
            elif hasattr(self.raw_model, "base_model") and hasattr(self.raw_model.base_model, "fc"):
                optim_params = self.raw_model.base_model.fc.parameters()
            else:
                raise NotImplementedError("Don't know how to access fc")
        elif self.args.nf_head:
            optim_params = self.raw_model.cls_parameters()
        else:
            optim_params = self.model.parameters()

        optimizer, scheduler = get_optim_sched(optim_params, self.args.learning_rate)
        if self.args.nf_head:
            optimizer_nf, scheduler_nf = get_optim_sched(self.raw_model.nf.parameters(), self.args.learning_rate * self.args.nf_lr_mult)

        ### Resuming ###

        start_it = 0

        if self.args.resume:
            aux_modules = get_aux_modules_dict(optimizer, scheduler)
            if self.args.nf_head:
                aux_modules.update(get_aux_modules_dict(optimizer_nf, scheduler_nf, suffix="nf"))
            resume_possible, resume_it = self.resume(aux_modules)
            if resume_possible:
                start_it = resume_it + 1
                print(f"Training resumed from it {start_it}")
            else:
                print("Cannot resume, ckpt does not exist or is not complete")

        ### Training stats ###

        log_period = 10
        ckpt_period = 500
        running_loss = 0
        if self.args.nf_head:
            running_loss_nf = 0

        # batch iterator
        train_iter = iter(train_loader)

        print(f"Start training, lr={self.args.learning_rate}, start_iter={start_it}, iterations={self.args.iterations}, warmup_iters={self.args.warmup_iters}")

        ### Training loop ###

        for it in range(start_it, self.args.iterations):
            try: 
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            images, labels = batch 
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs, _ = self.model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            if self.args.clip_grad > 0:
                params = self.raw_model.cls_parameters() if self.args.nf_head else self.model.parameters()
                nn.utils.clip_grad_value_(params, self.args.clip_grad)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            # flow
            if self.args.nf_head:
                ll, _ = self.model(images, classify=False, flow=True, backbone_grad=False)
                loss_nf = -torch.mean(ll) / self.output_num
                optimizer_nf.zero_grad()
                loss_nf.backward()
                if self.args.nf_clip_grad > 0:
                    nn.utils.clip_grad_value_(self.raw_model.nf.parameters(), self.args.nf_clip_grad)
                optimizer_nf.step()
                scheduler_nf.step()
                running_loss_nf += loss_nf.item()

            # logging
            if (it + 1) % log_period == 0:
                _, preds = outputs.max(dim=1)
                train_acc = ((preds == labels).sum() / len(preds)).item()
                optims_dict = {"": optimizer}
                losses_dict = {"": running_loss}
                if self.args.nf_head:
                    optims_dict["nf"] = optimizer_nf
                    losses_dict["nf"] = running_loss_nf
                log_msg, wandb_log_data = gen_train_log_msg(it, self.args.iterations, optims_dict, losses_dict, train_acc, log_period)
                print(log_msg)
                if self.args.wandb and is_main_process(self.args):
                    wandb.log(wandb_log_data, step=it)
                running_loss = 0
                if self.args.nf_head:
                    running_loss_nf = 0

            # ckpt saving
            if (it + 1) % ckpt_period == 0:
                aux_modules = get_aux_modules_dict(optimizer, scheduler)
                if self.args.nf_head:
                    aux_modules.update(get_aux_modules_dict(optimizer_nf, scheduler_nf, suffix="nf"))
                self.save_ckpt(aux_modules, it)

        # save final checkpoint 
        self.save_ckpt()


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
        if args.suffix:
            run_name += f"_{args.suffix}"

        wandb.init(
            project="OODDetectionFramework",
            config=vars(args),
            name=run_name,
        )
    
    if args.print_args:
        print(args)

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
