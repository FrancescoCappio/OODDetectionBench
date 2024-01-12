# Out of distribution detection benchmark 

This repo contains a framework to execute a number of published and unpublished out of distribution detection methods on both a [novel benchmark introduced by this work](https://ooddb.github.io/) and on the more established benchmark from [MOS](https://arxiv.org/abs/2105.01879). 

The implemented OOD detection methods can be divided into 2 groups:
 - methods requiring a finetuning on closed set data;
 - finetuning-free methods, which simply compare support data (the closed set samples the others use for finetuning) and test data to provide normality scores for the latter. 

Therefore, the included code can be used to either finetune a pretrained model on a specific
OOD detection task, before evaluating it, or to directly evaluate a pretrained model.

## Dependencies

The dependencies are listed in the `requirements.txt` file.

We downloaded the public models for resnet101 versions of SimCLR, SupCon, CSI, SupCSI from
the original repositories and converted them to make them compatible with our framework. 
Converted models can be found
[here](https://drive.google.com/file/d/1w41RjKaOx5tbOcb3AleAWAOTzNxEw9ap/view?usp=sharing).
Downloaded models should be put in the `pretrained_models` directory. 

## Data

### [Our benchmark](https://ooddb.github.io/)

For our benchmark, refer to the [package setup guide](https://github.com/ooddb/OODDB#setup).

**NOTE:** in this case the `path_dataset` argument is ignored and sorts no effect.

### [MOS benchmark](https://arxiv.org/abs/2105.01879)

In order to evaluate a model on this benchmark you will need the following datasets, organized as shown:
- ImageNet-1K:
  - train split: `ILSVRC/Data/CLS-LOC/train/nXXXXXXXX/nXXXXXXXX_<img_id>.JPEG`
  - val split: `ILSVRC/Data/CLS-LOC/val/nXXXXXXXX/ILSVRC2012_val_<img_id>.JPEG`
- iNaturalist: `iNaturalist/images/<img_id>.jpg`
- SUN: `SUN/images/sun_<img_id>.jpg`
- Places: `Places/images/<cls_first_letter>_<cls_name>_<img_id>.jpg`
- DTD (Texture): `dtd/images/<cls_name>/<cls_name>_<img_id>.jpg`

For the specific file paths utilized, refer to the txt file lists under `data/txt_lists/mos_bench`.

The datasets' location on disk can be specified at runtime through the `path_dataset` argument (by default `~/data` is used).


## Supported methods

### Included architectures and models

The pretrained models considered in this work are:
- ImageNet-1K pretrainings
  - ReSeND
  - ResNet-101, CE 
  - ResNet-101, SimCLR
  - ResNet-101, SupCon
  - ResNet-101, CSI
  - ResNet-101, SupCSI
  - ViT-B, CE
  - ViT-B, DINO
- Larger pretrainings
  - Vit-L, CE (ImageNet-21K)
  - ResNetV2-101x3 BiT, CE (ImageNet-21K)
  - ViT-L, CLIP (CLIP)
  - ViT-L, DINOv2 (LVD-142M)

### Included evaluation methods

The supported evaluation methods are:
- Finetuning-free
  - Prototypes-based distance
  - *k*-NN (both with and without feature normalization)
  - Mahalanobis
  - ReSeND (ReSeND only)
  - MCM (CLIP only)
- Finetuning only
  - MSP
  - ReAct
  - ASH
  - Normalizing flow

## Running the code

To **reproduce the paper's main results**, you can simply run the bash scripts contained in the `scripts` folder (in case of `wise_ft.sh` you may want to adjust the path pointing to your finetuned models, refer to the first line of the script).
Below we report some extra details regarding the supported execution arguments.

In order to evaluate a model, the generic command to run is the following:

```
python train.py --only_eval \
  --network <network> --model <model> --checkpoint_path <ckpt_path> \
  --evaluator <evaluator> \
  --dataset <dataset> --support <support> --test <test> --data_order <order>
```

### Model selection

To select the pretrained model, choose the appropriate `network` and `model` (and, if specified, `checkpoint_path`) values.
The following combinations are supported:
- ReSeND: `--network resend --model resend --checkpoint_path "pretrained_models/resend.pth"`
- ResNet-101, CE (ImageNet-1K): `--network resnet101 --model CE` 
- ResNet-101, SimCLR (ImageNet-1K):`--network resnet101 --model simclr --checkpoint_path "pretrained_models/simclr_r101.pth"`
- ResNet-101, SupCon (ImageNet-1K): `--network resnet101 --model supclr --checkpoint_path "pretrained_models/supclr_r101.pth"`
- ResNet-101, CSI (ImageNet-1K): `--network resnet101 --model CSI --checkpoint_path "pretrained_models/CSI_r101.pth"`
- ResNet-101, SupCSI (ImageNet-1K): `--network resnet101 --model supCSI --checkpoint_path "pretrained_models/supCSI_r101.pth"`
- ViT-B, CE (ImageNet-1K): `--network vit --model CE`
- ViT-B, DINO (ImageNet-1K): `--network vit --model DINO --checkpoint_path "pretrained_models/DINO_vitb.pth"`
- ViT-L, CE (ImageNet-21K): `--network vit --model CE-IN21k`
- BiT ResNetV2-101x3, CE (ImageNet-21K): `--network resnetv2_101x3 --model BiT`
- ViT-L, CLIP: `--network vit --model clip`
- ViT-L, DINOv2 (LVD-142M): `--network vit --model DINOv2`

### Evaluator selection

To select the evaluation method, choose the appropriate `evaluator` value:
- Prototypes distance: `--evaluator prototypes_distance`
- *k*-NN: `--evaluator knn_distance`
- *k*-NN with normalized features: `--evaluator knn_ood`
- Mahalanobis: `--evaluator mahalanobis`
- ReSeND (ReSeND only): `--evaluator resend`
- MCM (CLIP only): `--evaluator MCM`
- MSP (finetuning only): `--evaluator MSP`
- ReAct (finetuning only): `--evaluator react`
- ASH (finetuning only): `--evaluator ASH`
- Normalizing flow (finetuning only, requires `--nf_head`): `--evaluator flow`

### Benchmark track selection

- To run [our benchmark](https://ooddb.github.io/), choose a `dataset` value among `domainnet`, `dtd`, `patternnet`, `stanford_cars`, `sun`. As for the `support` and `test` arguments, specify a valid `split` value as described in the [documentation](https://github.com/ooddb/OODDB#usage). For this track, `data_order` must assume a value between `0` and `2` (inclusive - for more details, refer to the documentation).
- To evaluate a model on the benchmark from [MOS](https://arxiv.org/abs/2105.01879), choose `--dataset mos_bench` and `--support imagenet`. As for the `test` argument, you can select one among `inaturalist`, `sun`, `places`, `dtd` (texture). For this track, the `data_order` arg is unused and it must be left at its default `-1` value.

### Finetuning

To perform a finetuning operation, omit the `only_eval` argument (in case of the `flow` evaluator, you must also pass the `--nf_head` argument, since an additional flow head must be trained). By default the finetuning operation will cover 25 epochs (5 warmup ones and 20 cosine annealing ones, as specified in the paper).

## Citation 

If you find this code useful, please cite our paper: 

```
@inproceedings{cappio2022relationalreasoning,
  title={Semantic Novelty Detection via Relational Reasoning},
  author={Francesco Cappio Borlino, Silvia Bucci, Tatiana Tommasi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
} 
```
