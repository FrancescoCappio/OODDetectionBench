run_eval () {
    python train.py --only_eval \
        --network "$network" --model "$model" \
        --checkpoint_path="$ckpt_path" \
        --evaluator "$evaluator" \
        --dataset "$dataset" --support "$support" --test "$test" --data_order "$data_order"
    echo -e "\n##########\n"
}

run_all_eval () {
    network="$1"
    model="$2"
    evaluator="$3"
    ckpt_path="$4"

    echo -e "### Evaluating $network-$model with $evaluator evaluator ###\n"

    local intra_datasets="dtd patternnet stanford_cars sun"
    local dn_domains="clipart infograph painting quickdraw real sketch"

    for data_order in {0..2}
    do

        ### Intra domain ###
        echo -e "Intra-Domain\n"

        support="train"
        test="test"
        for dataset in $intra_datasets
        do
            run_eval
        done

        ### Intra domain (DomainNet) ###

        dataset="domainnet"

        local domain
        for domain in $dn_domains
        do
            support="${domain}_train"
            test="${domain}_test"
            run_eval
        done

        ### Cross domain SS ###
        echo -e "Cross-Domain SS\n"

        local src_domain
        local tgt_domain
        for src_domain in $dn_domains
        do
            for tgt_domain in $dn_domains
            do
                if [ "$src_domain" = "$tgt_domain" ]; then continue; fi
                support="${src_domain}_train"
                test="${tgt_domain}_test"
                run_eval
            done
        done

        ### Cross domain MS ###
        echo -e "Cross-Domain MS\n"

        local domain
        for domain in $dn_domains
        do
            support="no_${domain}_train"
            test="${domain}_test"
            run_eval
        done
    done
}

##############################

# ReSeND (IN-1K)
for evaluator in knn_ood mahalanobis knn_distance prototypes_distance
do
    run_all_eval resend resend resend "pretrained_models/resend.pth"
done

# ReSeND-H (IN-1K)
# for evaluator in knn_ood mahalanobis knn_distance prototypes_distance
# do
#     run_all_eval resend resend resend "pretrained_models/resend_h.pth"
# done

# ResNet-101 CE (IN-1K)
for evaluator in knn_ood mahalanobis knn_distance prototypes_distance
do
    run_all_eval resnet101 CE "$evaluator" ""
done

# ResNet-101 SimCLR (IN-1K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnet101 simclr "$evaluator" "pretrained_models/simclr_r101.pth"
done

# ResNet-101 SupCon (IN-1K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnet101 supclr "$evaluator" "pretrained_models/supclr_r101.pth"
done

# ResNet-101 CSI (IN-1K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnet101 CSI "$evaluator" "pretrained_models/CSI_r101.pth"
done

# ResNet-101 SupCSI (IN-1K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnet101 supCSI "$evaluator" "pretrained_models/supCSI_r101.pth"
done

# ViT-B CE (IN-1K)
for evaluator in knn_ood mahalanobis knn_distance prototypes_distance
do
    run_all_eval vit CE "$evaluator" ""
done

# ViT-B DINO (IN-1K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval vit DINO "$evaluator" "pretrained_models/DINO_vitb.pth"
done

# ViT-L CE (IN-21K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval vit CE-IN21k "$evaluator" ""
done

# BiT ResNetV2-101x3 CE (IN-21K)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval resnetv2_101x3 BiT "$evaluator" ""
done

# ViT-L CLIP
for evaluator in MCM
do
    run_all_eval vit clip "$evaluator" ""
done

# ViT-L DINOv2 (LVD-142M)
for evaluator in knn_distance prototypes_distance
do
    run_all_eval vit DINOv2 "$evaluator" ""
done
