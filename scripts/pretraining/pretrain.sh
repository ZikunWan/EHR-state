#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 deepspeed --log_level warning --num_gpus=8 ./pretraining/pretrain.py \
    --deepspeed "./ds_config_zero2.json" \
    --dataset mimic_iv eicu \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_raw_dir "/data/EHR_data_public/eicu-crd/2.0" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --task_train_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/train" \
    --task_val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/val" \
    --eicu_task_train_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_train.json" \
    --eicu_task_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
    --pretraining_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/train/next_token_prediction.csv" \
    --pretraining_val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/val/next_token_prediction.csv" \
    --eicu_pretraining_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_pretraining_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json" \
    --tte_index_dir "/data/zikun_workspace/input/tasks/time_to_event" \
    --table_embedding_cache "/data/zikun_workspace/input/cache/embeddings/merged_table_embeddings.pt" \
    --task_query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt" \
    --phenotype_query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/pretraining/phenotype_query_knowledge_embeddings.pt" \
    --diagnosis_embedding_cache "/data/zikun_workspace/input/cache/embeddings/mimic_iv/diagnosis_text_embeddings.pt" \
    --phenotype_pair_count_path "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_pair_counts.csv" \
    --runtime_index_path "/data/zikun_workspace/input/cache/pretraining/runtime_index.sqlite" \
    --runtime_index_num_workers 96 \
    --type_vocab_file "/data/zikun_workspace/code/data/type_vocab.json" \
    --max_table_len 4096 \
    --ntp_stride 3000 \
    --max_eval_samples 10000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --ddp_find_unused_parameters true \
    --dataloader_drop_last false \
    --dataloader_num_workers 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 60 \
    --weight_decay 0.01 \
    --ntp_ratio 5 \
    --pml_ratio 3 \
    --sft_ratio 2 \
    --ntp_time_loss_weight 0.1 \
    --pml_huber_delta 1.0 \
    --num_train_epochs 1 \
    --logging_steps 3 \
    --save_steps 30 \
    --eval_strategy "steps" \
    --eval_steps 150 \
    --save_total_limit 1 \
    --bf16 true \
    --report_to "wandb" \
    --wandb_project "Joint_Pretraining" \
    --run_name "550M_pretrain" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/550M" \
    "$@"
