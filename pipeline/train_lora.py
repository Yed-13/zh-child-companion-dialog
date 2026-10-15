"""LoRA 微调对话模型（我们用的底模：Qwen3-4B-Instruct-2507）。
用法：BAND=1 python train_lora.py <基础模型目录或名字> <data.jsonl> <输出目录> [轮数]
data.jsonl 每行 {"q": ..., "a": ..., "band": 学段}，可再带 "system"；没有 system 就按该行的 band（没有就用环境变量 BAND）补人设的系统提示（persona.py）。只在回答上算损失。
环境变量 SAVE_EPOCHS=1：每轮存一份 LoRA 检查点（输出目录/checkpoint-步数），不在最后合并，用 merge_ckpt.py 逐个合并比较。
可调（环境变量）：LR 学习率（默认 2e-4）、LORA_R 秩（16）、LORA_DROPOUT（0.05）、WD 权重衰减（0）、NEFT 噪声强度（不加）；
EVAL_FILE=评测题（.json 列表或 .jsonl，如 data/band1_test.jsonl）：用这批题评测，并保证它们不进训练（几个方案比较时评测题一样）；
GC=0：不开梯度检查点（显存够时快约四分之一）；BEST=1：每半轮评测一次，训练完把评测损失最低的那份 LoRA 存到 输出目录_best。"""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, transformers
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import persona
base, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = float(sys.argv[4]) if len(sys.argv) > 4 else 2
tok = AutoTokenizer.from_pretrained(base)


def load_rows(path):
    """.json（列表）或 .jsonl；缺 system 的按学段补上。"""
    if path.endswith(".json"):
        rows = json.load(open(path, encoding="utf-8"))
    else:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r if r.get("system") else {**r, "system": persona.system_prompt(r.get("band", os.environ.get("BAND", 1)))} for r in rows]


rows = load_rows(data)
random.seed(0); random.shuffle(rows)
def to_pc(r):           # 先用模型自带对话模板排好“提问”，回答接在后面（Qwen3 混合模型关思考）
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["q"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {"prompt": prompt, "completion": r["a"] + "<|im_end|>\n"}
if os.environ.get("EVAL_FILE"):
    held = load_rows(os.environ["EVAL_FILE"])
    seen = {(r["system"], r["q"]) for r in held}
    rows = [r for r in rows if (r["system"], r["q"]) not in seen]
else:
    n_eval = max(50, len(rows) // 20)
    held, rows = rows[:n_eval], rows[n_eval:]
train, evals = Dataset.from_list([to_pc(r) for r in rows]), Dataset.from_list([to_pc(r) for r in held])
json.dump(held, open(f"{out}_heldout.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("训练", len(rows), "评测", len(held), flush=True)
best = bool(os.environ.get("BEST"))
steps_half = max(10, len(rows) // 16 // 2)                 # 半轮多少步（每步 8×2 条）
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
WARMUP = {"warmup_ratio": 0.05} if transformers.__version__ < "5" else {"warmup_steps": 0.05}   # 5.x 起 warmup_steps 小于 1 当比例
sched = ({"eval_strategy": "steps", "eval_steps": steps_half, "save_strategy": "steps", "save_steps": steps_half,
          "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False, "save_total_limit": 2}
         if best else {"eval_strategy": "epoch", "save_strategy": "epoch" if os.environ.get("SAVE_EPOCHS") else "no"})
extra = {"neftune_noise_alpha": float(os.environ["NEFT"])} if os.environ.get("NEFT") else {}
cfg = SFTConfig(output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=8, gradient_accumulation_steps=2,
                learning_rate=float(os.environ.get("LR", 2e-4)), weight_decay=float(os.environ.get("WD", 0)), lr_scheduler_type="cosine",
                **WARMUP, logging_steps=20, **sched, **extra,
                save_only_model=not best, bf16=True, max_length=640, gradient_checkpointing=os.environ.get("GC", "1") != "0", completion_only_loss=True, report_to=[])
R = int(os.environ.get("LORA_R", 16))
lora = LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=float(os.environ.get("LORA_DROPOUT", 0.05)), task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                  "in_proj_qkv", "in_proj_z", "out_proj"])   # 后三个是 Qwen3.5 线性注意力层的投影，Qwen3 没有
tr = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=evals, processing_class=tok, peft_config=lora)
tr.train()
evals = [{"epoch": round(h["epoch"], 2), "eval_loss": h["eval_loss"]} for h in tr.state.log_history if "eval_loss" in h]
json.dump(evals, open(f"{out}_evals.json", "w"), indent=0)
print("EVALS", json.dumps(evals), flush=True)
if best:                                                   # load_best_model_at_end 已把最好的那份载回来
    tr.model.save_pretrained(f"{out}_best")
    print("BEST", tr.state.best_metric, tr.state.best_model_checkpoint, flush=True)
if os.environ.get("SAVE_EPOCHS"):
    print("TRAIN_DONE（检查点在", out, "）", flush=True)
    sys.exit(0)
merged = tr.model.merge_and_unload()                    # 合并成完整模型
merged.save_pretrained(f"{out}_merged", safe_serialization=True); tok.save_pretrained(f"{out}_merged")
print("TRAIN_DONE", json.dumps(tr.state.log_history[-3:], ensure_ascii=False), flush=True)
