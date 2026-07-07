#!/bin/bash
source "$(dirname "$0")/../common/silent_info.sh"

if [ "$CACHE_ONLY" = "true" ]; then
    rm -rf "/data/zikun_workspace/input/knowledge/cache/triples_cache"
    python ./pretraining/knowledge_encoder.py \
        --model_name_or_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --concept_path "/data/zikun_workspace/input/knowledge/omop/CONCEPT.csv" \
        --concept_relationship_path "/data/zikun_workspace/input/knowledge/omop/CONCEPT_RELATIONSHIP.csv" \
        --triple_cache "/data/zikun_workspace/input/knowledge/cache/triples_cache" \
        --kg_max_triples "None" \
        --kg_eval_ratio 0.01 \
        --kg_build_workers 8 \
        --kg_build_chunksize 2000000 \
        --output_dir "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder" \
        --report_to "none" \
        --cache_only
else
    deepspeed --num_gpus=8 ./pretraining/knowledge_encoder.py \
        --deepspeed "./ds_config_zero2.json" \
        --model_name_or_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --concept_path "/data/zikun_workspace/input/knowledge/omop/CONCEPT.csv" \
        --concept_relationship_path "/data/zikun_workspace/input/knowledge/omop/CONCEPT_RELATIONSHIP.csv" \
        --triple_cache "/data/zikun_workspace/input/knowledge/cache/triples_cache" \
        --kg_max_triples "None" \
        --kg_eval_ratio 0.01 \
        --kg_num_negatives 4 \
        --kg_margin 1.0 \
        --kg_distance_p 2 \
        --kg_relation_reg 1e-4 \
        --max_length 128 \
        --batch_size 128 \
        --epochs 5 \
        --learning_rate 2e-5 \
        --min_lr 1e-6 \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --num_workers 16 \
        --bf16 \
        --logging_steps 50 \
        --eval_steps 5000 \
        --save_steps 1000 \
        --save_total_limit 1 \
        --report_to "wandb" \
        --wandb_project "knowledge_encoder" \
        --wandb_run_name "knowledge_encoder_pretrain" \
        --output_dir "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder" \
        --resume_from_checkpoint "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/checkpoint-92000"
fi
