#!/usr/bin/env bash
set -euo pipefail

# Utility script to run linear probe training layer by layer in batch
# Usage:
#   ./run_layers_linear.sh 22 23 24 25 26 27
# Or run without arguments to default to layers 0-27

CONFIG_BASE="configs/train_config_linear.yaml"
OUT_DIR="generated_for_layer_linear"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  LAYERS=("$@")
else
  # Default layer range 0-27 (number of layers in R1-Onevision-7B model)
  LAYERS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)
fi

echo "Using base config: $CONFIG_BASE"
echo "Output configs: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

for L in "${LAYERS[@]}"; do
  NEWCFG="$OUT_DIR/train_config_linear_layer_${L}.yaml"
  echo "Generating config for layer $L -> $NEWCFG"
  "$PYTHON_BIN" -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_BASE'))
cfg.setdefault('probe_config', {})
cfg['probe_config']['layers_to_hook'] = [int($L)]
cfg['probe_config']['layer'] = int($L)
cfg['probe_config']['probe_id'] = 'R1-Onevision-7B-linear-' + str(int($L))
with open('$NEWCFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
"

  echo "Start training for layer $L"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  "$PYTHON_BIN" -m probe.train --config "$NEWCFG"
  echo "Finished layer $L"
done

echo "All layers finished."
