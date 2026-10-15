"""把一个 LoRA 检查点合并进基础模型，存成完整模型。
用法：python merge_ckpt.py <基础模型目录或名字> <检查点目录> <输出目录>"""
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, ckpt, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("MERGED", out, flush=True)
