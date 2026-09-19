#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec conda run --no-capture-output -n gia \
  python -m utils.run_cmds \
  --cmd-config-yaml run_yaml/slake_llava_dlg.yaml \
  --gpu-ids 7 \
  --occupy-after-run \
  --occupancy-script /home/zx/nas/gpu/train_stealth.py \
  --execute
