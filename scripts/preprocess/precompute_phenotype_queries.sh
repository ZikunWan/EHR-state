#!/bin/bash
set -euo pipefail

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 python ./preprocess/precompute_phenotype_queries.py \
    --stage discover \
    --num_discovery_workers 32 \
    --dataset mimic_iv eicu ehrshot \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train/next_token_prediction.csv" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --eicu_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --ehrshot_root_dir "/data/zikun_workspace/input/tables/ehrshot" \
    --ehrshot_sample_info_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_train.csv" \
    --phenotype_spec_output_path "/data/zikun_workspace/.cache/phenotype_metric_learning/phenotype_query_specs.json" \
    --min_phenotype_occurrence 50 \
    --max_auto_phenotypes 256 \
    --phenotype_statistics latest first mean max min \
    --phenotype_time_windows "full::" "first24h:0:24" "first48h:0:48" \
    --phenotype_category_regex "^measurement$"

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 torchrun --standalone --nproc_per_node=8 \
    ./preprocess/precompute_phenotype_queries.py \
    --stage encode \
    --phenotype_spec_output_path "/data/zikun_workspace/.cache/phenotype_metric_learning/phenotype_query_specs.json" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/phenotype_metric_learning/knowledge_query_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --query_embedding_batch_size 256
