#!/bin/bash
set -euo pipefail

python preprocess/Renji/generate_tte_task_index.py \
    --root-dir "/data/zikun_workspace/input/tables/renji/raw" \
    --output-dir "/data/zikun_workspace/input/tasks/time_to_event/renji/indices" \
    --death-horizon-days 1825 \
    --tasks death tacrolimus
