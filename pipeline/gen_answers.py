"""第二步：模型以人设身份回答；第三步：逐条审核，不合格的改写或丢弃。可中断续跑。
用法：BAND=0|1|2 [WORKERS=6] python gen_answers.py questions.json answers.jsonl"""
import concurrent.futures as cf, json, os, re, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client, rules

QFILE, OUT = sys.argv[1], sys.argv[2]
R = rules.load(rules.band_from_env())
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        r = json.loads(l)
        if r["status"] != "drop" or r.get("problem"):     # 审核没返回结果（problem 空）的丢弃不算，重答
            done.add(r["q"])
rows = [r for r in json.load(open(QFILE, encoding="utf-8")) if r["q"] not in done]
lock, stats = threading.Lock(), {"tok": 0, "ok": 0, "fixed": 0, "drop": 0}


def work(r):
    know = any(k in r["cat"] for k in R.know)             # 讲知识的类别开思考，闲聊不开
    a, u1 = llm_client.chat([{"role": "system", "content": R.teacher}, {"role": "user", "content": r["q"]}],
                            think=know, max_tokens=1500, temperature=0.8)
    a = a.strip()
    j, u2 = llm_client.chat([{"role": "user", "content": R.judge.format(q=r["q"], a=a)}], think=know, max_tokens=1500, temperature=0.2)
    m = re.search(r"\{.*\}", j, re.S)
    try:
        v = json.loads(m.group(0)) if m else {"ok": False, "fixed": ""}
    except ValueError:
        v = {"ok": False, "fixed": ""}
    fixed = (v.get("fixed") or "").strip()
    final, status = (a, "ok") if v.get("ok") else (fixed, "fixed" if fixed else "drop")
    rec = {"cat": r["cat"], "q": r["q"], "draft": a, "a": final, "status": status, "problem": v.get("problem", "")}
    with lock:
        stats["tok"] += u1.get("total_tokens", 0) + u2.get("total_tokens", 0); stats[status] += 1
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = sum(stats[k] for k in ("ok", "fixed", "drop"))
        if n % 200 == 0:
            print(f"{n}/{len(rows)}  合格 {stats['ok']}  改写 {stats['fixed']}  丢弃 {stats['drop']}  tokens {stats['tok']}", flush=True)


def safe(r):
    try:
        work(r)
    except Exception as e:                                # 单条失败不影响其他，下次运行补上
        print("跳过（下次再试）：", r["q"][:20], e, flush=True)


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        list(ex.map(safe, rows))
    print("ANSWERS_DONE", stats, flush=True)
