#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

force=false
if [[ "${1:-}" == "--force" ]]; then
  force=true
  shift
fi
if (($#)); then
  echo "Usage: $0 [--force]" >&2
  exit 2
fi

tasks=(
  "ehrshot.sh:all"
)

result_complete() {
  local script="$1" task="$2" input output
  if [[ "${script}" == "ehrshot.sh" && "${task}" == "all" ]]; then
    local classification_task
    for classification_task in \
      guo_los guo_readmission guo_icu \
      lab_anemia lab_hyperkalemia lab_hyponatremia lab_hypoglycemia lab_thrombocytopenia \
      new_acutemi new_celiac new_hyperlipidemia new_hypertension new_lupus new_pancan; do
      result_complete "ehrshot.sh" "${classification_task}" || return 1
    done
    return 0
  fi

  input="/data/zikun_workspace/input/tasks/classification/ehrshot/test/${task}.csv"
  output="/data/zikun_workspace/checkpoints/classification/ehrshot/${task}/zero_shot/test_results_${task}.csv"
  [[ -s "${input}" && -s "${output}" ]] || return 1

  [[ "$(wc -l < "${input}")" -eq "$(wc -l < "${output}")" ]]
}

failure=0
for spec in "${tasks[@]}"; do
  IFS=: read -r script task <<< "${spec}"
  if [[ "${force}" == false ]] && result_complete "${script}" "${task}"; then
    echo "===== Skipping completed ${script} task=${task} ====="
    continue
  fi
  echo "===== Starting 8-GPU zero-shot ${script} task=${task} ====="
  if ! bash "${script_dir}/${script}" "${task}"; then
    echo "===== FAILED ${script} task=${task} =====" >&2
    failure=1
  fi
done

if ((failure)); then
  exit 1
fi
echo "===== All zero-shot classification and TTE tasks completed ====="
