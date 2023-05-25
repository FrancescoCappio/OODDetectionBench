#!/bin/bash 

## Table 1

echo "Table 1"

### Results replication

#### CE - ResNet

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model CE --only_eval; done


#### CE - Vit

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network vit --model CE --only_eval; done


#### CutMix - ResNet

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done


#### SimCLR - ResNet

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done


#### SupCLT - ResNet 

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth; done


#### CSI - ResNet

 
for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth; done


#### SupCSI - ResNet 


for ds in DTD DomainNet_Real DomainNet_Sketch DomainNet_Painting ; do python train.py --dataset "$ds" --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth; done


## Table 2 Top (PACS Single-Source)

echo "Table 2 Top - PACS Single-Source"

### Results replication 

#### CE - ResNet


for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model CE --only_eval; done


#### CE - Vit


for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network vit --model CE --only_eval; done


#### Cutmix - ResNet

 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done


#### SimCLR - ResNet

 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done


#### SupCLR - ResNet 

 
for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth; done


#### CSI - ResNet


for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth; done


#### SupCSI - ResNet


for d in ArtPainting Sketch Cartoon; do python train.py --dataset PACS_SS_DG --source Source --target "$d"  --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth; done


## Table 2 Bottom (PACS Multi-Source)

echo "Table 2 Bottom - PACS Multi-Source"

### Results replication

#### CE - ResNet 

 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model CE --only_eval; done


#### CE - Vit


for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network vit --model CE --only_eval; done


#### Cutmix - ResNet

 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model cutmix --only_eval --checkpoint_path pretrained_models/cutmix_r101.pth; done


#### SimCLR - ResNet

 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model simclr --only_eval --checkpoint_path pretrained_models/simclr_r101.pth; done


#### SupCLR - ResNet 

 
for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model supclr --only_eval --checkpoint_path pretrained_models/supclr_r101.pth ; done


#### CSI - ResNet


for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model CSI --only_eval --checkpoint_path pretrained_models/CSI_r101.pth ; done


#### SupCSI - ResNet


for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d"  --network resnet101 --model supCSI --only_eval --checkpoint_path pretrained_models/supCSI_r101.pth ; done


## Table 3

echo "Table 3"

##### MSP 

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator MSP; done

##### ODIN 

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator ODIN; done

##### Energy 

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator energy; done

##### GradNorm

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator gradnorm; done

##### OODFormer

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network vit --model CE --evaluator MSP; done

##### Mahalanobis 

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator mahalanobis; done

##### Gram 

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --evaluator gram; done

#### Replicating methods not requiring a finetuning 

##### Mahalanobis

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --only_eval --evaluator mahalanobis; done

##### Gram

for d in ArtPainting Sketch Cartoon Photo; do python train.py --dataset PACS_DG --source no_"$d" --target "$d" --network resnet101 --model CE --only_eval --evaluator gram; done

