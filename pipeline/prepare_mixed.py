"""把三个学段合成一份训练数据（对照实验：一个 LoRA 学三个学段）。
各学段的测试集合起来当评测题、不进训练；某个学段太大可以抽样（默认学段 1 抽 5000 条，免得占太多）。
每条补上自己学段的系统提示（persona.py），所以输出可以直接给 train_lora.py。
用法：python prepare_mixed.py [数据目录=data] [输出目录=.]   输出 train_mixed.jsonl、heldout_mixed.json"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persona

data = sys.argv[1] if len(sys.argv) > 1 else "data"
out = sys.argv[2] if len(sys.argv) > 2 else "."
CAP = {0: None, 1: 5000, 2: None}
random.seed(0)
held, train = [], []
for b in (0, 1, 2):
    sysmsg = persona.system_prompt(b)
    load = lambda name: [{**json.loads(l), "system": sysmsg} for l in open(f"{data}/band{b}_{name}.jsonl", encoding="utf-8")]
    h, rows = load("test"), load("train")
    seen = {r["q"] for r in h}
    rows = [r for r in rows if r["q"] not in seen]
    if CAP[b] and len(rows) > CAP[b]:
        rows = random.sample(rows, CAP[b])
    held += h
    train += rows
    print("学段", b, "训练", len(rows), "评测", len(h))
random.shuffle(train)
with open(f"{out}/train_mixed.jsonl", "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(held, open(f"{out}/heldout_mixed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("合计 训练", len(train), "评测", len(held))
