import torch
from torch import nn

from .resnet import get_resnet


def _clean_ckpt(ckpt, model):
    new_dict = {}
    model_dict = model.state_dict()
    for k in ckpt.keys():
        new_k = k
        if k not in model_dict:
            if k.startswith("base_model"):
                new_k = k.replace("base_model.", "")
            ### NF Hybrid models
            elif k.startswith("encoder"):
                new_k = k.replace("encoder", "base_model")
            elif k.startswith("flow_module"):
                new_k = k.replace("flow_module", "nf")
            ###
        new_dict[new_k] = ckpt[k]
    return new_dict


def _interpolate_ckpts(zeroshot_ckpt, finetuned_ckpt, alpha):
    # make sure checkpoints are compatible
    assert set(zeroshot_ckpt.keys()) == set(finetuned_ckpt.keys())
    # interpolate between all weights in the checkpoints
    new_state_dict = {
        k: (1 - alpha) * zeroshot_ckpt[k] + alpha * finetuned_ckpt[k] for k in zeroshot_ckpt.keys()
    }
    return new_state_dict


def _load_ckpt(model, ckpt, alpha, ckpt_device):
    assert alpha >= 0 and alpha <= 1, "wise_ft_alpha must be a value betweeen 0 and 1"
    if alpha == 1:
        missing, unexpected = model.load_state_dict(_clean_ckpt(ckpt, model), strict=False)
        print(f"Missing keys: {missing}, unexpected keys: {unexpected}")
    elif alpha > 0:
        print(f"Interpolating weights with alpha={alpha}")
        model.to(ckpt_device)
        ckpt = _interpolate_ckpts(model.state_dict(), ckpt, alpha)
        model.load_state_dict(ckpt)


def get_model(args, n_known_classes, device, ckpt):
    model = None
    output_num = None
    contrastive_enabled = False

    if args.network == "resnet101":
        if args.model in ["CE", "cutmix"]:
            if args.model == "cutmix" and args.only_eval:
                assert ckpt is not None, "Cannot perform eval without a pretrained model"

            model, output_num = get_resnet(args.network, n_known_classes=n_known_classes)

            # if ckpt fc size does not match current size discard it
            if ckpt is not None:
                old_size = ckpt["fc.bias"].shape[0]
                if not old_size == n_known_classes:
                    del ckpt["fc.weight"]
                    del ckpt["fc.bias"]
        elif args.model == "random_init":
            model, output_num = get_resnet(
                args.network, n_known_classes=n_known_classes, random_init=True
            )

        elif args.model in ["simclr", "supclr", "CSI", "supCSI"]:
            contrastive_enabled = not args.disable_contrastive_head
            if args.only_eval:
                assert ckpt is not None, "Cannot perform eval without a pretrained model"

            contrastive_type = "simclr" if args.model in ["simclr", "supclr"] else "CSI"

            from models.common import WrapperWithContrastiveHead

            base_model, output_num = get_resnet(args.network, n_known_classes=n_known_classes)
            model = WrapperWithContrastiveHead(
                base_model, out_dim=output_num, contrastive_type=contrastive_type
            )
            if not args.disable_contrastive_head:
                output_num = model.contrastive_out_dim

            # if ckpt fc size does not match current size discard it
            if ckpt is not None:
                old_size = ckpt["base_model.fc.bias"].shape[0]
                if not old_size == n_known_classes:
                    del ckpt["base_model.fc.weight"]
                    del ckpt["base_model.fc.bias"]

        elif args.model == "clip":
            import clip

            model, preprocess = clip.load("RN101", device)

            # the model has no fc by default, so it does not support closed set finetuning
            from models.common import WrapperWithFC

            output_num = 512
            model = WrapperWithFC(model.visual, output_num, n_known_classes, half_precision=True)

        else:
            raise NotImplementedError(
                f"Model {args.model} is not supported with network {args.network}"
            )

    elif args.network == "vit":
        if args.model in ["CE", "DINO"]:
            import timm

            output_num = 768

            if args.model == "DINO":
                if not args.checkpoint_path:
                    raise AssertionError("Specify ckpt for DINO")

                # we set num_classes to 0 in order to obtain pooled feats
                model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=0)

                if args.nf_head:
                    from models.common import WrapperWithNF

                    model = WrapperWithNF(model, output_num, n_known_classes)
                else:
                    # if we didn't need the contrastive head we could use the model from huggingface:
                    # https://huggingface.co/facebook/dino-vitb16
                    contrastive_enabled = not args.disable_contrastive_head
                    from models.common import WrapperWithContrastiveHead

                    model = WrapperWithContrastiveHead(
                        model,
                        out_dim=output_num,
                        contrastive_type="DINO",
                        add_cls_head=True,
                        n_classes=n_known_classes,
                    )
                    if not args.disable_contrastive_head:
                        output_num = model.contrastive_out_dim

            else:
                import types

                # we set num_classes to 0 in order to obtain pooled feats
                model = timm.create_model("deit_base_patch16_224", pretrained=True)
                model.fc = model.head

                if not n_known_classes == 1000:
                    model.fc = nn.Linear(in_features=768, out_features=n_known_classes)

                def my_forward(self, x):
                    feats = self.forward_head(self.forward_features(x), pre_logits=True)
                    logits = self.fc(feats)
                    return logits, feats

                model.forward = types.MethodType(my_forward, model)

                if args.nf_head:
                    from models.common import WrapperWithNF

                    model = WrapperWithNF(model, output_num, add_cls_head=False)
                else:
                    model = model

        elif args.model == "clip":
            # ViT-B/16
            import clip

            model, preprocess = clip.load("ViT-L/14", device)

            # the model has no fc by default, so it does not support closed set finetuning
            from models.common import WrapperWithFC

            output_num = 768
            model = WrapperWithFC(model.visual, output_num, n_known_classes, half_precision=True)

        elif args.model == "DINOv2":
            from models.common import WrapperWithFC, WrapperWithNF

            wrapper = WrapperWithNF if args.nf_head else WrapperWithFC
            dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            output_num = 1024
            model = wrapper(dinov2_vitb14, output_num, n_known_classes)

        elif args.model == "CE-IM22k" or args.model == "CE-IM21k":
            # https://huggingface.co/google/vit-large-patch16-224-in21k
            from transformers import ViTModel

            model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
            output_num = 1024
            from models.common import WrapperWithFC

            model = WrapperWithFC(
                model,
                output_num,
                n_known_classes,
                base_output_map=lambda x: x["pooler_output"],
            )
        else:
            raise NotImplementedError(
                f"Model {args.model} is not supported with network {args.network}"
            )

    elif args.network == "resnetv2_101x3":
        assert args.model == "BiT", f"The network {args.network} supports only BiT model"

        # we set num_classes to 0 in order to obtain pooled feats
        import timm

        from models.common import WrapperWithFC, WrapperWithNF

        wrapper = WrapperWithNF if args.nf_head else WrapperWithFC
        # https://huggingface.co/timm/resnetv2_101x3_bit.goog_in21k
        model = timm.create_model("resnetv2_101x3_bit.goog_in21k", pretrained=True, num_classes=0)
        output_num = 6144
        model = wrapper(model, output_num, n_known_classes)

    elif args.network == "resend":
        assert args.only_eval and ckpt is not None, "Cannot perform eval without a pretrained model"

        from models.resend import ReSeND

        model = ReSeND(n_known_classes=n_known_classes)
        output_num = model.output_num
    else:
        raise NotImplementedError(f"Network {args.network} not implemented")

    if ckpt is not None and args.wise_ft_alpha > 0:
        print(f"Loading checkpoint {args.checkpoint_path}")
        _load_ckpt(model, ckpt, args.wise_ft_alpha, device)

    return model, output_num, contrastive_enabled
