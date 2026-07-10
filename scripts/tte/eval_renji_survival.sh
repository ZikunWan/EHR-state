#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python test/tte/test_renji_survival.py \
    --data_dir "/data/zikun_workspace/input/tables/renji/raw" \
    --embedding_cache "/data/zikun_workspace/input/cache/embeddings/renji/text_embeddings.pt" \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/renji/tacrolimus_survival" \
    --patient_subset_path "data/patients.json" \
    --tte_index_dir "/data/zikun_workspace/input/tasks/time_to_event/renji" \
    --split test \
    --type_vocab_file "data/type_vocab.json" \
    --query_embedding_cache "/data/zikun_workspace/input/cache/query_embeddings/query_candidate/renji_survival_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --max_table_len 4096 \
    --batch_size 128
