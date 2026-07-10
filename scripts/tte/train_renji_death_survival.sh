#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

deepspeed --include localhost:4,5,6,7 train/tte/train_renji_survival.py \
    --deepspeed "ds_config_zero2.json" \
    --survival_task death \
    --data_dir "/data/zikun_workspace/input/tables/renji/raw" \
    --output_dir "/data/zikun_workspace/checkpoints/renji/death_survival" \
    --run_name "renji_death_survival" \
    --patient_subset_path "data/patients.json" \
    --tte_index_dir "/data/zikun_workspace/input/tasks/time_to_event/renji" \
    --max_table_len 4096 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --monitor_fraction 0.1 \
    --monitor_seed 42 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --early_stopping_patience 10 \
    --load_best_model_at_end true \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --warmup_steps 100 \
    --bf16 true \
    --dataloader_num_workers 32 \
    --report_to wandb \
    --query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/query_candidate/renji_death_survival_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_metric_learning"
