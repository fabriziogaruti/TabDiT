#!/bin/bash


# 1. Unconditional Generation

# 1.1 Rossmann Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/rossmann/uncond_s0" \
    --output_dir="./discriminator_results/rossmann/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="rossmann"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/rossmann/uncond_s0" \
    --output_dir="./discriminator_results/rossmann/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false

# 1.2 Airbnb Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/airbnb/uncond_s0" \
    --output_dir="./discriminator_results/airbnb/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="airbnb"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/airbnb/uncond_s0" \
    --output_dir="./discriminator_results/airbnb/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false

# 1.3 PKDD’99 Financial Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/pkdd99/uncond_s0" \
    --output_dir="./discriminator_results/pkdd99/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="pkdd99"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/pkdd99/uncond_s0" \
    --output_dir="./discriminator_results/pkdd99/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false

# 1.4 Age2 Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/age2/uncond_s0" \
    --output_dir="./discriminator_results/age2/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="age2"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/age2/uncond_s0" \
    --output_dir="./discriminator_results/age2/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false

# 1.5 Age1 Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/age1/uncond_s0" \
    --output_dir="./discriminator_results/age1/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="age1"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/age1/uncond_s0" \
    --output_dir="./discriminator_results/age1/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false

# 1.6 Leaving Dataset
# MLD-TS
python discriminator_MLD_TS.py \
    --data_root="./generated_data/leaving/uncond_s0" \
    --output_dir="./discriminator_results/leaving/uncond_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="leaving"
# LD-SR
python discriminator_LD_SR.py \
    --data_root="./generated_data/leaving/uncond_s0" \
    --output_dir="./discriminator_results/leaving/uncond_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false




# 2. Conditional Generation


# 2.1 Rossmann Dataset

# MLD-TS (child gt-cond)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/rossmann/child_gt_s0" \
    --output_dir="./discriminator_results/rossmann/child_gt_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="rossmann"
# MLD-TS (child)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="rossmann"
# MLD-TS (merged)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=true \
    --dataset_name_to_use="rossmann"

# LD-SR (child gt-cond)
python discriminator_LD_SR.py \
    --data_root="./generated_data/rossmann/child_gt_s0" \
    --output_dir="./discriminator_results/rossmann/child_gt_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (child)
python discriminator_LD_SR.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (merged)
python discriminator_LD_SR.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=true

# MLE-TS (original)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=false \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="rossmann"
# MLE-TS (child)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/rossmann/merged_s0" \
    --output_dir="./discriminator_results/rossmann/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="rossmann"
# MLE-TS (child gt-cond)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/rossmann/child_gt_s0" \
    --output_dir="./discriminator_results/rossmann/child_gt_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="rossmann"



# 2.2 Airbnb Dataset

# MLD-TS (child gt-cond)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/airbnb/child_gt_s0" \
    --output_dir="./discriminator_results/airbnb/child_gt_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="airbnb"
# MLD-TS (child)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="airbnb"
# MLD-TS (merged)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=true \
    --dataset_name_to_use="airbnb"

# LD-SR (child gt-cond)
python discriminator_LD_SR.py \
    --data_root="./generated_data/airbnb/child_gt_s0" \
    --output_dir="./discriminator_results/airbnb/child_gt_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (child)
python discriminator_LD_SR.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (merged)
python discriminator_LD_SR.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=true

# MLE-TS (original)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=false \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="airbnb"
# MLE-TS (child)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/airbnb/merged_s0" \
    --output_dir="./discriminator_results/airbnb/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="airbnb"
# MLE-TS (child gt-cond)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/airbnb/child_gt_s0" \
    --output_dir="./discriminator_results/airbnb/child_gt_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="airbnb"



# 2.3 Age2 Dataset

# MLD-TS (child gt-cond)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/age2/child_gt_s0" \
    --output_dir="./discriminator_results/age2/child_gt_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="age2"
# MLD-TS (child)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="age2"
# MLD-TS (merged)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=true \
    --dataset_name_to_use="age2"

# LD-SR (child gt-cond)
python discriminator_LD_SR.py \
    --data_root="./generated_data/age2/child_gt_s0" \
    --output_dir="./discriminator_results/age2/child_gt_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (child)
python discriminator_LD_SR.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (merged)
python discriminator_LD_SR.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=true

# MLE-TS (original)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=false \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="age2"
# MLE-TS (child)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/age2/merged_s0" \
    --output_dir="./discriminator_results/age2/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="age2"
# MLE-TS (child gt-cond)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/age2/child_gt_s0" \
    --output_dir="./discriminator_results/age2/child_gt_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="age2"



# 2.4 PKDD’99 Financial Dataset

# MLD-TS (child gt-cond)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/pkdd99/child_gt_s0" \
    --output_dir="./discriminator_results/pkdd99/child_gt_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="pkdd99"
# MLD-TS (child)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=false \
    --dataset_name_to_use="pkdd99"
# MLD-TS (merged)
python discriminator_MLD_TS.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_multi_CB" \
    --seq_len_max=-1 \
    --categorical_data_usage="ordinal" \
    --use_parent_table=true \
    --dataset_name_to_use="pkdd99"

# LD-SR (child gt-cond)
python discriminator_LD_SR.py \
    --data_root="./generated_data/pkdd99/child_gt_s0" \
    --output_dir="./discriminator_results/pkdd99/child_gt_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (child)
python discriminator_LD_SR.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=false
# LD-SR (merged)
python discriminator_LD_SR.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_LD_SR" \
    --use_parent_table=true

# MLE-TS (original)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=false \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="pkdd99"
# MLE-TS (child)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/pkdd99/merged_s0" \
    --output_dir="./discriminator_results/pkdd99/merged_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="pkdd99"
# MLE-TS (child gt-cond)
python discriminator_MLE_TS.py \
    --data_root="./generated_data/pkdd99/child_gt_s0" \
    --output_dir="./discriminator_results/pkdd99/child_gt_s0" \
    --experiment_name="discriminator_MLE_TS" \
    --seq_len_max=-1 \
    --use_generated_data_efficacy=true \
    --categorical_data_usage="ordinal" \
    --dataset_name_to_use="pkdd99"

