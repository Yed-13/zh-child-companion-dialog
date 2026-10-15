#!/usr/bin/env bash
# 生成一个学段的数据：出题 → 回答并审核（跑两遍，第二遍补失败的）→ 修被丢的 → 整理成训练数据。可重跑，每步都能续。
# 用法：export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...；BAND=1 W=work/band1 bash pipeline/run_pipeline.sh
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
BAND=${BAND:?设学段 BAND=0|1|2}; export BAND
W=${W:-work/band$BAND}; mkdir -p "$W"
[ -s "$W/questions.json" ] || python "$P/gen_questions.py" "$W/questions.json" > "$W/questions.log"
for i in 1 2; do WORKERS=${WORKERS:-24} python "$P/gen_answers.py" "$W/questions.json" "$W/answers.jsonl" >> "$W/answers.log"; done
WORKERS=${WORKERS:-24} python "$P/repair.py" "$W/answers.jsonl" "$W/repaired.jsonl" >> "$W/repair.log"
python "$P/build_data.py" "$W/answers.jsonl" "$W/repaired.jsonl" "$W/band${BAND}_train.jsonl" > "$W/build.log"
cat "$W/build.log"
