#!/bin/bash
source "$(dirname "$0")/../common/silent_info.sh"

deepspeed --num_gpus=8 ./pretraining/next_token_prediction.py \
    --deepspeed "./ds_config_zero2.json" \
    --dataset mimic_iv eicu ehrshot \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --train_info_path \
        "/data/zikun_workspace/input/tasks/classification/mimic_iv/index/train/next_token_prediction.csv" \
    --table_text_embedding "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --eicu_train_info_path \
        "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_table_text_embedding "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
    --ehrshot_root_dir "/data/zikun_workspace/input/tables/ehrshot" \
    --ehrshot_train_info_path \
        "/data/zikun_workspace/input/tables/ehrshot/pretraining_index/sample_info_train.csv" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
    --max_table_len 16384 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 32 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 1 \
    --bf16 true \
    --report_to "wandb" \
    --wandb_project "Next_Token_Prediction" \
    --run_name "next_token_prediction" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/next_token_prediction"
