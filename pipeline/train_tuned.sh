#!/usr/bin/env bash
# 调过参数的一组：学习率 1e-4、dropout 0.1、权重衰减 0.01、NEFTune 5、每半轮评测留最好的一份（BEST=1）、2 轮。
# 评测题固定用测试集（EVAL_FILE），保证它们不进训练。
# 用法（在仓库根目录）：BASE=Qwen/Qwen3-4B-Instruct-2507 bash pipeline/train_tuned.sh
#   MIXED=1：再训一个把三个学段混在一起的 LoRA（对照）。
set -euo pipefail
BASE=${BASE:?设底模目录或名字 BASE}
mkdir -p runs
export LR=1e-4 LORA_DROPOUT=0.1 WD=0.01 NEFT=5 BEST=1 GC=${GC:-1}   # GC=0 更快，但 32GB 显存、批大小 8、长度 640 会爆显存
for b in ${BANDS:-0 1 2}; do
  BAND=$b EVAL_FILE=data/band${b}_test.jsonl python pipeline/train_lora.py "$BASE" data/band${b}_train.jsonl runs/tuned_band$b 2 > runs/tuned_band$b.log 2>&1
  echo TUNED_DONE $b                                      # LoRA 在 runs/tuned_band$b_best
done
if [ -n "${MIXED:-}" ]; then
  python pipeline/prepare_mixed.py data runs > runs/prepare_mixed.log 2>&1
  EVAL_FILE=runs/heldout_mixed.json python pipeline/train_lora.py "$BASE" runs/train_mixed.jsonl runs/mixed 2 > runs/mixed.log 2>&1
  echo MIXED_DONE
fi
