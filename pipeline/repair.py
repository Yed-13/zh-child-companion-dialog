"""第四步：修被审核丢掉的回答：告诉老师上次哪里不合格，重写后再审（这次审核开深度思考）。可中断续跑。
用法：BAND=0|1|2 [WORKERS=6] python repair.py answers.jsonl repaired.jsonl"""
import concurrent.futures as cf, json, os, re, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client, rules

SRC, OUT = sys.argv[1], sys.argv[2]
R = rules.load(rules.band_from_env())
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["q"] for l in open(OUT, encoding="utf-8")}
last = {}
for l in open(SRC, encoding="utf-8"):                 # 同一问以最后一次为准（重答过的）
    r = json.loads(l)
    last[r["q"]] = r
rows = [r for r in last.values() if r["status"] == "drop" and r["q"] not in done]
FIX = ("孩子说：{q}\n你上次的回答：{a}\n审核意见：{p}\n\n请按人设和规则重新回答。" + f"闲聊不超过{R.chat_max}字，讲知识不超过{R.know_max}字，"
       f"知识必须是课本上正确的说法，只输出{R.name}要说的话。")
lock, stats = threading.Lock(), {"ok": 0, "drop": 0}


def work(r):
    know = any(k in r["cat"] for k in R.know)
    problem = r.get("problem") or "太长，或者不够像这个人设"
    a, _ = llm_client.chat([{"role": "system", "content": R.teacher},
                            {"role": "user", "content": FIX.format(q=r["q"], a=r["draft"], p=problem)}],
                           think=know, max_tokens=1500, temperature=0.7)
    a = a.strip()
    j, _ = llm_client.chat([{"role": "user", "content": R.judge.format(q=r["q"], a=a)}], think=True, max_tokens=2000, temperature=0.2)
    m = re.search(r"\{.*\}", j, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
    except ValueError:
        v = {}
    fixed = (v.get("fixed") or "").strip()
    final, status = (a, "ok") if v.get("ok") else ((fixed, "fixed") if fixed else ("", "drop"))
    rec = {"cat": r["cat"], "q": r["q"], "draft": a, "a": final, "status": status, "problem": v.get("problem", "")}
    with lock:
        stats["ok" if status != "drop" else "drop"] += 1
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = stats["ok"] + stats["drop"]
        if n % 100 == 0:
            print(f"{n}/{len(rows)}  救回 {stats['ok']}  仍丢弃 {stats['drop']}", flush=True)


def safe(r):
    try:
        work(r)
    except Exception as e:
        print("跳过（下次再试）：", r["q"][:20], e, flush=True)


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        list(ex.map(safe, rows))
    print("REPAIR_DONE", stats, flush=True)
