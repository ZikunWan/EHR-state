#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 deepspeed --num_gpus=8 ./pretraining/phenotype_metric_learning.py \
    --deepspeed "./ds_config_zero2.json" \
    --dataset mimic_iv eicu ehrshot \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train/next_token_prediction.csv" \
    --val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/val/next_token_prediction.csv" \
    --table_text_embedding "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --eicu_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json" \
    --eicu_table_text_embedding "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
    --ehrshot_root_dir "/data/zikun_workspace/input/tables/ehrshot" \
    --ehrshot_sample_info_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_train.csv" \
    --ehrshot_val_sample_info_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_val.csv" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
    --phenotype_spec_path "/data/zikun_workspace/.cache/phenotype_metric_learning/phenotype_query_specs.json" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/phenotype_metric_learning/knowledge_query_embeddings.pt" \
    --precomputed_queries_only true \
    --preprocessed_input_dir "/data/zikun_workspace/.cache/phenotype_metric_learning/inputs" \
    --preprocessed_inputs_only true \
    --max_table_len 4096 \
    --min_table_rows 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --min_lr_ratio 0.1 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --huber_delta 1.0 \
    --projection_loss_weight 1.0 \
    --transe_loss_weight 0.0 \
    --relation_l2_weight 0.0 \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --save_total_limit 1 \
    --wandb_project "Phenotype_Metric_Learning" \
    --run_name "phenotype_metric_learning" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/phenotype_metric_learning"
