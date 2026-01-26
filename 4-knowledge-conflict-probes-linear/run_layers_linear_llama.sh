#!/usr/bin/env bash
set -euo pipefail

# 批量逐层运行线性探针训练脚本的小工具
# 用法:
#   ./run_layers_linear.sh 22 23 24 25 26 27
# 或者不带参数则默认尝试 0-27 层

CONFIG_BASE="configs/train_config_linear_llama.yaml"
OUT_DIR="llama_generated_for_layer_linear"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  LAYERS=("$@")
else
  # 默认层范围 0-31（Llama-3.2V-11B-cot 模型的层数）
  LAYERS=(3 8 13 18 23 28 33 38)
  # LAYERS=(37 39)
fi

echo "Using base config: $CONFIG_BASE"
echo "Output configs: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

for L in "${LAYERS[@]}"; do
  NEWCFG="$OUT_DIR/llama_train_config_linear_layer_${L}.yaml"
  echo "Generating config for layer $L -> $NEWCFG"
  "$PYTHON_BIN" -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_BASE'))
cfg.setdefault('probe_config', {})
cfg['probe_config']['layers_to_hook'] = [int($L)]
cfg['probe_config']['layer'] = int($L)
cfg['probe_config']['probe_id'] = 'Llama-3.2V-11B-cot-linear-' + str(int($L))
with open('$NEWCFG', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
"

  echo "Start training for layer $L"
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  "$PYTHON_BIN" -m probe.train --config "$NEWCFG"
  echo "Finished layer $L"
done

echo "All layers finished."
