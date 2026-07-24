#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="/data/zikun_workspace/checkpoints/pretraining/550M_mha_ntp"

cd "${PROJECT_ROOT}"
bash "${SCRIPT_DIR}/pretrain.sh" \
    --objectives ntp \
    --max_table_len 4096 \
    --run_name "550M_mha_ntp" \
    --output_dir "${OUTPUT_DIR}"
