#!/bin/bash
source "$(dirname "$0")/../common/silent_info.sh"

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 deepspeed --num_gpus=8 ./pretraining/contrastive_learning.py \
    --deepspeed "./ds_config_zero2.json" \
    --dataset mimic_iv eicu ehrshot \
    --sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train/next_token_prediction.csv" \
    --val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/val/next_token_prediction.csv" \
    --table_text_embedding "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
    --markdown_embedding_path "/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_free_text/embeddings.pt" \
    --eicu_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json" \
    --eicu_table_text_embedding "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
    --eicu_markdown_embedding_path "/data/zikun_workspace/eicu-crd/processed/pretraining_markdown_embeddings/embeddings.pt" \
    --ehrshot_sample_info_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_train.csv" \
    --ehrshot_val_sample_info_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_val.csv" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
    --ehrshot_markdown_embedding_path "/data/zikun_workspace/input/tables/ehrshot/pretraining_markdown_embeddings/embeddings.pt" \
    --markdown_candidate_count 1024 \
    --max_table_len 4096 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --eval_retrieval_device cuda \
    --eval_retrieval_chunk_size 1024 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --num_train_epochs 5 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --save_total_limit 1 \
    --wandb_project "Contrastive_Learning" \
    --run_name "contrastive_learning" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/contrastive_learning"
