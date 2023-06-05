#!/bin/bash 

src_domain="$1"
tgt_domain="$2"

echo "Running with source $src_domain and target $tgt_domain"


#### CE - ResNet
 
python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval

#### CE - Vit

python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval

#### CutMix - SimCLR - SupCLR - SupCSI

for model in cutmix simclr supclr CSI supCSI; do python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model"  --only_eval --checkpoint_path pretrained_models/"$model"_r101.pth; done

#### Mahalanobis on CE 

python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator mahalanobis 

#### KNN on CE 

python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator knn_distance

#### MCM 

python train.py --dataset DomainNet_OOD --source "$src_domain" --target "$tgt_domain" --network resnet101 --model clip --only_eval --evaluator MCM
