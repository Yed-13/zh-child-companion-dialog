#!/usr/bin/env bash
# 三个学段各训一个 LoRA（同一个底模），每轮存检查点，可选转成 LoRA GGUF，比较后选。
# 用法（在仓库根目录）：BASE=Qwen/Qwen3-4B-Instruct-2507 BANDS="0 1 2" EPOCHS=3 bash pipeline/train_bands.sh
# 可选：LLAMA_CPP=<llama.cpp 目录>，设了就把每个检查点转成 runs/bandX_eN-lora.gguf。
set -euo pipefail
BASE=${BASE:?设底模目录或名字 BASE}
mkdir -p runs
for b in ${BANDS:-0 1 2}; do
  m=band$b
  BAND=$b SAVE_EPOCHS=1 python pipeline/train_lora.py "$BASE" data/band${b}_train.jsonl runs/$m ${EPOCHS:-3} > runs/$m.log 2>&1 || continue
  if [ -n "${LLAMA_CPP:-}" ]; then
    e=0
    for ck in $(ls -d runs/$m/checkpoint-* | sort -t- -k2 -n); do
      e=$((e+1))
      python "$LLAMA_CPP/convert_lora_to_gguf.py" --base "$BASE" --outtype f16 --outfile runs/${m}_e$e-lora.gguf "$ck" > runs/${m}_e$e-convert.log 2>&1
    done
    rm -rf runs/$m                                        # 检查点转完就删，只留 LoRA GGUF 和日志
  fi
  echo BAND_DONE $b
done
echo ALL_DONE
