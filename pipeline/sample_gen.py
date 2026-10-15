"""用合并后的模型回答测试集里的问题，看效果。
用法：BAND=1 python sample_gen.py 模型目录 data/band1_test.jsonl 条数"""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForCausalLM
import persona
path, held, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
system = persona.system_prompt(os.environ.get("BAND", 1))
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(l) for l in open(held, encoding="utf-8")][:n]
for r in rows:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["q"]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
    a = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"问：{r['q']}\n答：{a}\n参考：{r['a']}\n", flush=True)
