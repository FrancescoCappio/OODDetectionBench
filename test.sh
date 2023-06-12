#!/bin/bash 

out=$(python train.py --dataset DTD --source in_distribution --target out_distribution --network resnet101 --model CE --only_eval | tail -n 1)

if [ "$out" = "Auroc,FPR95: 0.6721,0.8965" ]; then
    echo "OK"
else
    echo "KO!!" 
    echo $out
fi

out=$(python train.py --dataset PACS_DG --source no_Sketch --target Sketch --network resnet101 --model simclr --checkpoint_path pretrained_models/simclr_r101.pth --only_eval | tail -n 1)
if [ "$out" = "Auroc,FPR95: 0.9326,0.6625" ]; then
    echo "OK"
else
    echo "KO!!" 
    echo $out
fi

out=$(python train.py --dataset PACS_DG --source no_ArtPainting --target ArtPainting --network resnet101 --model CE --few_shot 5 --evaluator knn_ood --only_eval --seed 13 | tail -n 1)

if [ "$out" = "Auroc,FPR95: 0.6720,0.9488" ]; then
    echo "OK"
else
    echo "KO!!" 
    echo $out
fi

