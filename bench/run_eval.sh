#!/usr/bin/env bash
# 示例：三个学段各配一个 LoRA，各起一个 llama.cpp 服务后，逐个测硬规则、再请裁判模型打分。
#   llama-server -m base-Q4_K_M.gguf --lora runs/band0-lora.gguf --jinja --port 8090
#   llama-server -m base-Q4_K_M.gguf --lora runs/band1-lora.gguf --jinja --port 8091
#   llama-server -m base-Q4_K_M.gguf --lora runs/band2-lora.gguf --jinja --port 8092
# 用法（在仓库根目录）：export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...（裁判）；bash bench/run_eval.sh
set -euo pipefail
mkdir -p out
for b in 0 1 2; do
  N=${N:-150} python bench/persona_eval.py data/band${b}_test.jsonl out/eval_band$b.json "lora@$b=http://localhost:809$b/v1@$b"
  python bench/judge_eval.py out/eval_band$b.json
done
