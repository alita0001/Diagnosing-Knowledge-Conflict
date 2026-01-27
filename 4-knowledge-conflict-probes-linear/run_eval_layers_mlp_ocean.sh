#!/usr/bin/env bash
set -euo pipefail

# Utility script to run linear probe evaluation layer by layer in batch
# Usage:
#   ./run_eval_layers_mlp_ocean.sh 22 23 24 25 26 27
# Or run without arguments to default to layers 0-27

CONFIG_BASE="configs/eval_config_mlp_ocean.yaml"
OUT_DIR="generated_for_eval_layer_mlp"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  LAYERS=("$@")
else
  # Default layer range 0-27 (number of layers in Ocean_R1_7B_Instruct model)
  LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)
fi

echo "Using base config: $CONFIG_BASE"
echo "Output configs: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

for L in "${LAYERS[@]}"; do
  NEWCFG="$OUT_DIR/ocean_eval_config_mlp_layer_${L}.yaml"
  echo "Generating eval config for layer $L -> $NEWCFG"
  "$PYTHON_BIN" -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_BASE'))
cfg.setdefault('probe_config', {})
cfg['probe_config']['probe_id'] = 'Ocean_R1_7B_Instruct-MLP-' + str(int($L))
cfg['probe_config']['layer'] = int($L)  # Set correct layer number
cfg['output_dir'] = 'value_head_probes/Ocean_R1_7B_InstructMLP-' + str(int($L)) + '/evaluation_results'
with open('$NEWCFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
"

  echo "Start evaluation for layer $L"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  "$PYTHON_BIN" -c "
import sys
sys.path.append('.')
from probe.evaluate import main
from probe.config import EvaluationConfig
from utils.file_utils import load_yaml

# Load configuration from YAML
config_dict = load_yaml('$NEWCFG')
eval_config = EvaluationConfig(**config_dict)

# Run evaluation
main(eval_config)
"
  echo "Finished layer $L"
done

echo "All layers evaluation finished."
