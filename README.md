# zh-child-companion-dialog

中文儿童陪伴对话数据集（按学段）和 LoRA 训练、评测小工具。

English: a Chinese child–companion dialogue dataset (by school stage) plus helpers to generate such data with an LLM, fine-tune LoRA adapters, and evaluate the replies. See the English section below.

## 内容

- `data/`：16,227 条合成的单轮对话（孩子一句话，AI 学习陪伴助手一两句短回答），分三个学段：0 学前到小学二年级、1 小学三到六年级、2 初中，已划好训练/测试集。数据卡见 [data/README.md](data/README.md)。
- `pipeline/`：数据生成（出题、回答、LLM 审核、修复、规则过滤）和训练（LoRA 微调、合并、混合学段数据）脚本。
- `bench/`：评测。硬规则（字数、句数、问句数、有无英文/表情/markdown、身份、保密、不替算式报答案）和 LLM 裁判打分（correct / persona / safe / fit，1–5 分）。
- `persona/`：一份中性的示例人设和各学段的说话方式、字数上限（`bands.json`）。想换人设就换这两个文件。
- `tests/`：过滤逻辑和 LLM 客户端回退逻辑的小测试。

## 流程

```
questions  ->  answers  ->  judge  ->  repair  ->  filter  ->  train / test
gen_questions  gen_answers (回答 + 审核)  repair (带审核意见重写再审)  build_data (长度/英文/表情/算式/句数)
                                                                                |
                         data/bandX_{train,test}.jsonl  ->  train_lora.py  ->  merge_ckpt.py / LoRA GGUF
                                                                                |
                                        bench/persona_eval.py (硬规则)  ->  bench/judge_eval.py (裁判打分)
```

## 快速开始

只用数据：

```python
import json
train = [json.loads(l) for l in open("data/band1_train.jsonl", encoding="utf-8")]
print(train[0])   # {"q": ..., "a": ..., "category": ..., "band": 1}
```

检查数据：`python data/stats.py`。跑测试：`python tests/test_build.py`。

重新生成数据（任意 OpenAI 兼容接口）：

```bash
export LLM_BASE_URL=https://your-endpoint/v1 LLM_API_KEY=... LLM_MODEL=your-model
BAND=1 bash pipeline/run_pipeline.sh      # 结果在 work/band1/band1_train.jsonl
# 换人设：PERSONA=path/to/persona.txt PERSONA_BANDS=path/to/bands.json
```

训练（需要 GPU 和 `pip install -r requirements.txt`；我们用的底模是 Qwen3-4B-Instruct-2507）：

```bash
BASE=Qwen/Qwen3-4B-Instruct-2507 BANDS="0 1 2" EPOCHS=3 bash pipeline/train_bands.sh   # 每轮存检查点
BASE=Qwen/Qwen3-4B-Instruct-2507 bash pipeline/train_tuned.sh                          # 调过参数的一组，留评测损失最低的
# 训练时按学段补系统提示（persona/example_persona.txt）；推理时请用同一份提示
```

可调环境变量：`LR`、`LORA_R`、`LORA_DROPOUT`、`WD`、`NEFT`、`GC`、`EVAL_FILE`、`BEST`、`SAVE_EPOCHS`（说明在 `pipeline/train_lora.py` 开头）。

评测：先把模型（底模加 LoRA）放在任意 OpenAI 兼容的聊天接口后面，比如 llama.cpp 的 `llama-server`，再

```bash
python bench/persona_eval.py data/band1_test.jsonl out/eval_band1.json "mine=http://localhost:8090/v1@1"
LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... python bench/judge_eval.py out/eval_band1.json
```

硬规则按模型原始输出算，不做后处理。完整示例见 `bench/run_eval.sh`。

## 局限

- 数据全是 LLM 合成的单轮对话，没有真实孩子的话，也没有个人信息；知识类回答没有逐条人工核对，可能有错。
- 回答是统一的“短、鼓励、承认是 AI、不保密、不替算算式”风格，不一定适合所有场景。
- 部分类别（保密、不适合的话题等）包含故意设计的、带轻微挑衅的孩子话语，用来训练拒绝和引导的回答。
- `persona/example_persona.txt` 是按数据遵循的规则重写的中性示例，数据本身不是用它生成的；用它训练和推理时前后要一致。
- 评测的裁判模型是另一个 LLM，打分只能作参考，不能替代人工审核和真实使用中的安全评估。

## 许可

- 代码：Apache-2.0，见 [LICENSE](LICENSE)。
- 数据（`data/`）：CC BY 4.0，见 [data/LICENSE](data/LICENSE)。引用方式见数据卡。

---

## English

**What it is.** A Chinese dataset of synthetic single-turn dialogues between a child and an AI study companion, in three school stages (band 0: preschool to grade 2, 1: grades 3–6, 2: middle school), 16,227 rows with train/test splits (`data/`), plus scripts to generate such data with any OpenAI-compatible LLM endpoint (`pipeline/`), fine-tune LoRA adapters (`pipeline/train_lora.py`), and evaluate replies with hard rules and an LLM judge (`bench/`).

**Pipeline.** LLM-generated child utterances per category → persona-conditioned answers → LLM judge against 9 rules → rewrite or repair failures → rule filter (length, English, emoji, arithmetic, sentence count, question count) → train/test.

**Quickstart.** `export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...; BAND=1 bash pipeline/run_pipeline.sh` to generate data; `BASE=<base model> bash pipeline/train_bands.sh` to train one LoRA per band; `python bench/persona_eval.py data/band1_test.jsonl out/eval.json "name=http://localhost:8090/v1@1"` then `python bench/judge_eval.py out/eval.json` to evaluate. Only the training scripts need extra packages (`requirements.txt`).

**Limitations.** Fully synthetic and single-turn; knowledge answers are LLM-generated and not human-verified; a fixed reply style (short, encouraging, always admits being an AI, never promises secrecy); some categories deliberately contain mildly provocative child utterances. The example persona is a neutral rewrite of the rules the data follows, not the exact prompt used for generation.

**Licenses.** Code: Apache-2.0. Data: CC BY 4.0 (citation: Zehao Li, 2026).
