# ReSeND competitors results 

This repo contains the code to replicate the results of ReSeND's competitors of Table 1, 2 and 3. 

Additional competitors and baseline have been added later. 

## Dependencies

The dependencies are listed in the `requirements.txt` file.

## ReSeND's paper results replication

Competitors involved:

 - Cross Entropy - ResNet, which refers to a resnet101 standard pretrained on ImageNet1k via Cross
   Entropy. The pretrained model is automatically downloaded from torchvision models;
 - Cross Entropy - ViT, which refers to a Vit_Base standard model pretrained via DeiT strategy on
   ImageNet1k via Cross Entropy. The pretrained model is automatically downloaded from timm library;
 - CutMix, ResNet - which is simply a resnet101 pretrained with CutMix augmentation on ImageNet1k
   via Cross Entropy. The pretrained model can be found in the [CutMix original
   repo](https://github.com/clovaai/CutMix-PyTorch)
 - SimCLR, ResNet - which is a standard resnet101 pretrained on ImageNet1k in a self-supervised
   fashion via SimCLR v2. The pretrained model can be downloaded from the original repo by Google;
 - SupCLR, ResNet - which is a standard resnet101 pretrained on ImageNet1k using the SupCLR learning
   objective. The pretrained model can be downloaded from the original repo by Google.
 - CSI and supCSI are the self-supervised and supervised variant of a model trained following
   https://arxiv.org/abs/2007.08176 on ImageNet1k. 

The pretrained models for CutMix, SimCLR and SupCLR, CSI, supCSI should be downloaded and put in the
`pretrained_models` directory. 

### Table 1

The benchmarks involved in the table are: 

 - "Texture", called `DTD` in the code
 - "Real", called `DomainNet_Real` in the code
 - "Sketch", called `DomainNet_Sketch` in the code
 - "Painting", called `DomainNet_Painting` in the code. 


#### CE - ResNet

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model CE --only_eval; done
```

#### CE - Vit

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network vit --model CE --only_eval; done
```

#### CutMix - ResNet

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done
```

#### SimCLR - ResNet

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done
```

#### SupCLT - ResNet 

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth; done
```

#### CSI - ResNet

```python 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth; done
```

#### SupCSI - ResNet 

```python
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth; done
```

### Table 2 Top (PACS Single-Source)

The benchmarks involved in the table are based on PACS with a single-source single-target domain
shift:

 - for all columns the domain adopted as source is "Photo", called `Source` in the code;
 - the other three domains (`ArtPainting`, `Sketch`, `Cartoon`) are used in turn as target.

#### CE - ResNet

```python
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model CE --only_eval; done
```

#### CE - Vit

```python
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network vit --model CE --only_eval; done
```

#### Cutmix - ResNet

```python 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done
```

#### SimCLR - ResNet

```python 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done
```

#### SupCLR - ResNet 

```python 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth; done
```

N.B. Results may differ significantly from original paper, since we did not use the contrastive head
at the time of the original paper. 

#### CSI - ResNet

```python
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth; done
```

#### SupCSI - ResNet

```python
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth; done
```

### Table 2 Bottom (PACS Multi-Source)

The benchmarks involved in the table are based on PACS with a multi-source single-target domain
shift. The domain used as target for each table column is one of the four domains (`ArtPainting`, `Sketch`, `Cartoon`, `Photo`). 
The source in each case contains images from all the other three domains and for simplicity in the
code it is referred as `no_{target}`, e.g.: `no_ArtPainting` for when `ArtPainting` is the target
domain. 

#### CE - ResNet 

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model CE --only_eval; done
```

#### CE - Vit

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network vit --model CE --only_eval; done
```

#### Cutmix - ResNet

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done
```

#### SimCLR - ResNet

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done
```

#### SupCLR - ResNet 

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth ; done
```

N.B. Results may differ significantly from original paper, since we did not use the contrastive head
at the time of the original paper. 

#### CSI - ResNet

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth ; done
```

#### SupCSI - ResNet

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth ; done
```

### Table 3

Differently from the others, this table is designed to compare ReSeND with methods designed explicitly for OOD detection. 
Given that no finetuning on support data is required for ReSeND while OOD detection methods need it
we allow them to finetune for a small number (e.g. 100) of iterations before testing them. 

The benchmark used in the Table is PACS Multi-Source, the same of Table 2 Bottom. 

The methods involved are: 

 - MSP: https://openreview.net/forum?id=Hkg4TI9xl 
 - ODIN: https://openreview.net/forum?id=H1VGkIxRZ
 - Energy: https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html 
 - GradNorm: https://papers.nips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html
 - OODFormer: https://www.bmvc2021-virtualconference.com/conference/papers/paper_1391.html
 - Mahalanobis: https://papers.nips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html
 - Gram: https://proceedings.mlr.press/v119/sastry20a.html

The last two appear two times in the Table, as they are executed both with and without the budget
limited finetuning. Their inclusion was requested by the reviewers and thus they were added in the
rebuttal phase. 
Their implementation has been heavily modified from the initial one, thus the performance may change
significantly. Moreover Gram does not really make much sense without the finetuning, as this method
exploits the network output predictions (as MSP and others do) and thus the predictions are not
related to the task at hand if the network is used without finetuning. 

#### Replicating methods requiring finetuning 

##### MSP 

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator MSP; done
```

##### ODIN 

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator ODIN; done
```

##### Energy 

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator energy; done
```

##### GradNorm

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator gradnorm; done
```

##### OODFormer

```python 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network vit --model CE --evaluator MSP; done
```

##### Mahalanobis 

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator mahalanobis; done
```

##### Gram 

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator gram; done
```

#### Replicating methods not requiring a finetuning 

##### Mahalanobis

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --only_eval --evaluator mahalanobis; done
```

##### Gram

```python
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --only_eval --evaluator gram; done
```

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
