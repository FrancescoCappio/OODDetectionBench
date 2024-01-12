run_eval () {
    python train.py --only_eval \
        --network "$network" --model "$model" \
        --checkpoint_path="$ckpt_path" \
        --evaluator "$evaluator" \
        --dataset "$dataset" --support "$support" --test "$tgt_dataset"
}

run_eval_ft () {
    python train.py \
        --network "$network" --model "$model" \
        --checkpoint_path="$ckpt_path" \
        --evaluator "$evaluator" \
        --dataset "$dataset" --support "$support" --test "$tgt_dataset"
}

run_all_eval () {
    network="$1"
    model="$2"
    evaluator="$3"
    ckpt_path="$4"
    local finetuning="$5"

    dataset="mos_bench"
    support="imagenet"

    echo -e "### Evaluating $network-$model with $evaluator evaluator ###\n"

    local tgt_datasets="inaturalist sun places dtd"

    for tgt_dataset in $tgt_datasets
    do
        if [ "$finetuning" = "yes" ]
        then
            run_eval_ft
        else
            run_eval
        fi
        echo -e "\n##########\n"
    done
}

##############################

########################
### Fine-tuning-free ###

# BiT ResNetV2-101x3 CE (IN-21K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnetv2_101x3 BiT "$evaluator" "" "no"
done

# ViT-L CLIP
for evaluator in MCM
do
    run_all_eval vit clip "$evaluator" "" "no"
done

# ViT-L DINOv2 (LVD-142M)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval vit DINOv2 "$evaluator" "" "no"
done

###################
### Fine-tuning ###

evaluators="MSP react knn_ood ASH flow"

# ViT-B CE (IN-1K)
for evaluator in $evaluators
do
    run_all_eval vit CE "$evaluator" "" "yes"
done

# ViT-B DINO (IN-1K)
for evaluator in $evaluators
do
    run_all_eval vit DINO "$evaluator" "pretrained_models/DINO_vitb.pth" "yes"
done

# BiT ResNetV2-101x3 CE (IN-21K)
for evaluator in $evaluators
do
    run_all_eval resnetv2_101x3 BiT "$evaluator" "" "yes"
done

# ViT-L DINOv2 (LVD-142M)
for evaluator in $evaluators
do
    run_all_eval vit DINOv2 "$evaluator" "" "yes"
done
