#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_repo"
if [[ $# -eq 0 ]]; then
  set -- launch
fi
exec /home/zx/nas/miniconda3/envs/gia/bin/python -m evaluation.multilayer_property_runner "$@"
