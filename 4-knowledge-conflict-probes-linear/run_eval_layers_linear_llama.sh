#!/usr/bin/env bash
set -euo pipefail

# 批量逐层运行线性探针评估脚本的小工具
# 用法:
#   ./run_eval_layers_linear_llama.sh 22 23 24 25 26 27
# 或者不带参数则默认尝试 0-27 层

CONFIG_BASE="configs/eval_config_linear_llama.yaml"
OUT_DIR="generated_for_eval_layer_linear"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  LAYERS=("$@")
else
  # 默认层范围 0-27（Llama-3.2V-11B-cot 模型的层数）
  # LAYERS=(0 1 2 4 5 6 7 9 10 11 12 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39)
  LAYERS=(3 8 13 18 23 28 33 38)
fi

echo "Using base config: $CONFIG_BASE"
echo "Output configs: $OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"

for L in "${LAYERS[@]}"; do
  NEWCFG="$OUT_DIR/llama_eval_config_linear_layer_${L}.yaml"
  echo "Generating eval config for layer $L -> $NEWCFG"
  "$PYTHON_BIN" -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_BASE'))
cfg.setdefault('probe_config', {})
cfg['probe_config']['probe_id'] = 'Llama-3.2V-11B-cot-linear-' + str(int($L))
cfg['probe_config']['layer'] = int($L)  # 设置正确的层号
cfg['output_dir'] = 'value_head_probes/Llama-3.2V-11B-cot-linear-' + str(int($L)) + '/evaluation_results'
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

# 从YAML加载配置
config_dict = load_yaml('$NEWCFG')
eval_config = EvaluationConfig(**config_dict)

# 运行评估
main(eval_config)
"
  echo "Finished layer $L"
done

echo "All layers evaluation finished."
