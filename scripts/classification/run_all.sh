#!/bin/bash
set -euo pipefail

mode="${1:-train}"
use_eval_dataset="${2:-true}"
if [[ "${mode}" != "train" && "${mode}" != "eval" ]]; then
  echo "Usage: $0 [train|eval] [true|false]" >&2
  exit 2
fi
if [[ "${use_eval_dataset}" != "true" && "${use_eval_dataset}" != "false" ]]; then
  echo "Usage: $0 [train|eval] [true|false]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
scripts=(eicu.sh ehrshot.sh mimic_iv.sh mimic_iv_cdm.sh pds.sh renji.sh)

cd "${project_root}"
for script in "${scripts[@]}"; do
  echo "===== classification/${script%.sh}: ${mode} ====="
  if [[ "${script}" == "renji.sh" ]]; then
    bash "${script_dir}/${script}" "${mode}" all "${use_eval_dataset}"
  else
    bash "${script_dir}/${script}" "${mode}" "${use_eval_dataset}"
  fi
done

echo "===== All classification ${mode} tasks completed ====="
