#!/bin/bash 

dataset="$1"
src_domain="$2"
tgt_domain="$3"
data_order="$4"

echo "Running eval on $dataset with source $src_domain and target $tgt_domain"

echo "ReSeND"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resend --model resend --only_eval --evaluator resend --checkpoint_path pretrained_models/resend.pth --data_order "$data_order"

echo "Res101 CE KNN OOD"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator knn_ood --data_order "$data_order"

echo "Res101 CE Mahalanobis"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator mahalanobis --data_order "$data_order"

echo "Res101 CE KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator knn_distance --data_order "$data_order"

echo "Res101 CE prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator prototypes_distance --data_order "$data_order"

for model in simclr supclr CSI supCSI
do
    echo "Res101 $model KNN"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator knn_distance --disable_contrastive_head --data_order "$data_order"

    echo "Res101 $model prototypes"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator prototypes_distance --disable_contrastive_head --data_order "$data_order"

done

echo "ViT-B CE KNN OOD"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator knn_ood  --data_order "$data_order"

echo "ViT-B CE Mahalanobis"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator mahalanobis --data_order "$data_order"

echo "ViT-B CE KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator knn_distance --data_order "$data_order"

echo "ViT-B CE prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator prototypes_distance --data_order "$data_order"

echo "ViT-B DINO KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator knn_distance --disable_contrastive_head --data_order "$data_order"

echo "ViT-B DINO prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator prototypes_distance --disable_contrastive_head --data_order "$data_order"

echo "Vit-L CE ImageNet22k KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE-IM22k --only_eval --evaluator knn_distance --data_order "$data_order"

echo "Vit-L CE ImageNet22k prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE-IM22k --only_eval --evaluator prototypes_distance --data_order "$data_order"

echo "BiT resnetv2_101x3 KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnetv2_101x3 --model BiT --only_eval --evaluator knn_distance --data_order "$data_order"

echo "BiT resnetv2_101x3 prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnetv2_101x3 --model BiT --only_eval --evaluator prototypes_distance --data_order "$data_order"

echo "ViT-L CLIP MCM"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model clip --only_eval --evaluator MCM --data_order "$data_order"

echo "Vit-L DINOv2 KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINOv2 --only_eval --evaluator knn_distance --data_order "$data_order"

echo "Vit-L DINOv2 prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINOv2 --only_eval --evaluator prototypes_distance --data_order "$data_order"

>>>>>>> fb2eef2ffa8d5f3d41ec4a103f198be6622f6701
