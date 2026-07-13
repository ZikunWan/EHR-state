#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

tasks=(
  "pds.sh:severe_outcome"
  "pds.sh:adverse_event_next_visit"
  "renji.sh:ALB"
  "renji.sh:ALP"
  "renji.sh:CR"
  "renji.sh:Glucose"
  "renji.sh:HB"
  "renji.sh:INR"
  "renji.sh:N_Percent"
  "renji.sh:PLT"
  "renji.sh:PT"
  "renji.sh:TP"
  "renji.sh:Uric_Acid"
  "renji.sh:WBC"
  "renji_tte.sh:tacrolimus_abnormal"
  "renji_tte.sh:death"
)

gpu_ids=(${GPU_IDS:-0 1 2 3 4 5 6 7})
free_gpus=("${gpu_ids[@]}")
next_task=0
failure=0
declare -A pid_to_gpu=()

while ((next_task < ${#tasks[@]} || ${#pid_to_gpu[@]} > 0)); do
  while ((next_task < ${#tasks[@]} && ${#free_gpus[@]} > 0)); do
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS=: read -r script task <<< "${tasks[next_task]}"
    echo "===== Starting zero-shot ${script} task=${task} on GPU ${gpu} ====="
    GPU_IDS="${gpu}" bash "${script_dir}/${script}" "${task}" &
    pid=$!
    pid_to_gpu["${pid}"]="${gpu}"
    ((next_task += 1))
  done

  finished_pid=""
  for pid in "${!pid_to_gpu[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null \
      || [[ "$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')" == Z* ]]; then
      if ! wait "${pid}"; then
        failure=1
      fi
      free_gpus+=("${pid_to_gpu[${pid}]}")
      unset "pid_to_gpu[${pid}]"
      finished_pid="${pid}"
      break
    fi
  done
  if [[ -z "${finished_pid}" && ${#pid_to_gpu[@]} -gt 0 ]]; then
    sleep 0.5
  fi
done

if ((failure)); then
  exit 1
fi
echo "===== All zero-shot classification and TTE tasks completed ====="
