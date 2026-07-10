#!/bin/bash
set -euo pipefail

MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-900}"
MEMORY_POLL_SECONDS="${MEMORY_POLL_SECONDS:-1}"
SUPERVISION_WRITE_BUFFER_SIZE="${SUPERVISION_WRITE_BUFFER_SIZE:-8192}"

collect_process_tree() {
    local root_pid="$1"
    local queue=("$root_pid")
    local all_pids=()
    local pid
    local children

    while ((${#queue[@]} > 0)); do
        pid="${queue[0]}"
        queue=("${queue[@]:1}")
        [[ -d "/proc/${pid}" ]] || continue
        all_pids+=("$pid")
        children="$(pgrep -P "$pid" || true)"
        if [[ -n "$children" ]]; then
            while read -r child_pid; do
                [[ -n "$child_pid" ]] && queue+=("$child_pid")
            done <<< "$children"
        fi
    done
    printf '%s\n' "${all_pids[@]}"
}

process_tree_rss_kb() {
    local rss_kb=0
    local pid
    local value
    for pid in "$@"; do
        [[ -r "/proc/${pid}/status" ]] || continue
        value="$(awk '/VmRSS:/ {print $2}' "/proc/${pid}/status" 2>/dev/null || true)"
        [[ -n "$value" ]] && rss_kb=$((rss_kb + value))
    done
    echo "$rss_kb"
}

terminate_process_tree() {
    local root_pid="$1"
    mapfile -t pids < <(collect_process_tree "$root_pid")
    if ((${#pids[@]} > 0)); then
        kill -TERM "${pids[@]}" 2>/dev/null || true
        sleep 10
    fi
    mapfile -t pids < <(collect_process_tree "$root_pid")
    if ((${#pids[@]} > 0)); then
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
}

memory_watchdog() {
    local root_pid="$1"
    local limit_kb="$2"
    local poll_seconds="$3"
    local rss_kb
    local rss_gb

    while kill -0 "$root_pid" 2>/dev/null; do
        mapfile -t pids < <(collect_process_tree "$root_pid")
        rss_kb="$(process_tree_rss_kb "${pids[@]}")"
        if ((rss_kb > limit_kb)); then
            rss_gb="$(awk -v kb="$rss_kb" 'BEGIN {printf "%.1f", kb/1024/1024}')"
            echo "Memory watchdog: RSS ${rss_gb} GB exceeded limit ${MEMORY_LIMIT_GB} GB; terminating build process tree." >&2
            terminate_process_tree "$root_pid"
            return 0
        fi
        sleep "$poll_seconds"
    done
}

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 python ./preprocess/build_unified_pretrain_cache.py \
    --dataset mimic_iv eicu ehrshot \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --ehrshot_root_dir "/data/zikun_workspace/input/tables/ehrshot" \
    --table_text_embedding "/data/zikun_workspace/input/cache/embeddings/mimic_iv/text_embeddings.pt" \
    --eicu_table_text_embedding "/data/zikun_workspace/input/cache/embeddings/eicu/text_embeddings.pt" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/input/cache/embeddings/ehrshot/text_embeddings.pt" \
    --task_train_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/train" \
    --task_val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/val" \
    --eicu_task_train_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_train.json" \
    --eicu_task_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
    --ehrshot_task_train_sample_info_path "/data/zikun_workspace/input/cache/ehrshot/classification_sample_info/ehrshot_train.csv" \
    --ehrshot_task_val_sample_info_path "/data/zikun_workspace/input/cache/ehrshot/classification_sample_info/ehrshot_val.csv" \
    --pretraining_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/train/next_token_prediction.csv" \
    --pretraining_val_sample_info_path "/data/zikun_workspace/input/tasks/classification/mimic_iv/val/next_token_prediction.csv" \
    --eicu_pretraining_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_pretraining_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json" \
    --ehrshot_pretraining_sample_info_path "/data/zikun_workspace/input/tasks/pretraining/ehrshot/indices/sample_info_train.csv" \
    --ehrshot_pretraining_val_sample_info_path "/data/zikun_workspace/input/tasks/pretraining/ehrshot/indices/sample_info_val.csv" \
    --include_pretraining_context true \
    --tte_index_dir "/data/zikun_workspace/input/tasks/time_to_event" \
    --phenotype_spec_path "/data/zikun_workspace/input/cache/pretraining/phenotype_metric_learning/phenotype_query_specs.json" \
    --output_dir "/data/zikun_workspace/input/cache/pretraining/ehr_encoder/inputs" \
    --run_id "pretraining_cache_v5_sorted_text_embeddings" \
    --resume true \
    --min_table_rows 2 \
    --part_size 2048 \
    --num_workers 24 \
    --worker_torch_threads 1 \
    --worker_max_tasks_per_child 0 \
    --supervision_write_buffer_size "${SUPERVISION_WRITE_BUFFER_SIZE}" &

build_pid="$!"
watchdog_pid=""
if ((MEMORY_LIMIT_GB > 0)); then
    memory_limit_kb=$((MEMORY_LIMIT_GB * 1024 * 1024))
    echo "Memory watchdog enabled: ${MEMORY_LIMIT_GB} GB limit, poll every ${MEMORY_POLL_SECONDS}s."
    memory_watchdog "$build_pid" "$memory_limit_kb" "$MEMORY_POLL_SECONDS" &
    watchdog_pid="$!"
fi

set +e
wait "$build_pid"
status="$?"
set -e

if [[ -n "$watchdog_pid" ]]; then
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true
fi

exit "$status"
