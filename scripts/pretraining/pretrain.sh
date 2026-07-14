#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../common/silent_info.sh"

TORCH_NCCL_DESYNC_DEBUG=1 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 deepspeed --num_gpus=8 ./pretraining/pretrain.py \
    --deepspeed "./ds_config_zero2.json" \
    --dataset mimic_iv eicu \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --ehrshot_root_dir "/data/zikun_workspace/input/tables/ehrshot" \
    --table_text_embedding "/data/zikun_workspace/input/cache/embeddings/mimic_iv/text_embeddings.pt" \
    --eicu_table_text_embedding "/data/zikun_workspace/input/cache/embeddings/eicu/text_embeddings.pt" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/input/cache/embeddings/ehrshot/text_embeddings.pt" \
    --merged_table_embedding_cache "/data/zikun_workspace/input/cache/embeddings/merged_table_embeddings.pt" \
    --phenotype_spec_path "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_query_specs.json" \
    --pretraining_input_dir "/data/zikun_workspace/input/cache/pretraining/ehr_encoder/inputs" \
    --task_query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/pretraining/task_query_knowledge_embeddings.pt" \
    --phenotype_query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/pretraining/phenotype_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 64 \
    --query_embedding_batch_size 256 \
    --max_table_len 16384 \
    --min_table_rows 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --activation_checkpointing false \
    --grad_cache true \
    --grad_cache_micro_batch_size 12 \
    --grad_cache_embedding_micro_batch_size 32 \
    --length_grouped_batching true \
    --length_bucket_count 128 \
    --dataloader_drop_last false \
    --text_embedding_on_gpu true \
    --dataloader_num_workers 0 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --min_lr_ratio 0.1 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --ntp_loss_weight 1.0 \
    --task_loss_weight 1.0 \
    --metric_loss_weight 1.0 \
    --ntp_time_loss_weight 0.1 \
    --huber_delta 1.0 \
    --projection_loss_weight 1.0 \
    --transe_loss_weight 0.0 \
    --relation_l2_weight 0.0 \
    --min_pair_delta 0.0 \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_strategy "no" \
    --eval_steps 5000 \
    --save_total_limit 1 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better false \
    --early_stopping_patience 20 \
    --bf16 true \
    --report_to "wandb" \
    --wandb_project "Joint_Pretraining" \
    --run_name "1B_pretrain" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/1B" \
    --resume_from_checkpoint "/data/zikun_workspace/checkpoints/pretraining/1B/checkpoint-11500"
