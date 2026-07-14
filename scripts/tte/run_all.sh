#!/bin/bash
set -euo pipefail

mode="${1:-train}"
if [[ "${mode}" != "train" && "${mode}" != "eval" ]]; then
  echo "Usage: $0 [train|eval]" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
scripts=(eicu.sh ehrshot.sh ehr_bench.sh renji.sh)

cd "${project_root}"
for script in "${scripts[@]}"; do
  echo "===== tte/${script%.sh}: ${mode} ====="
  bash "${script_dir}/${script}" "${mode}"
done

echo "===== All TTE ${mode} tasks completed ====="
