"""第五步：把合格回答整理成训练数据。修过的替换被丢掉的；去掉带表情、英文、算式或过长的。
用法：BAND=0|1|2 python build_data.py answers.jsonl repaired.jsonl out.jsonl   （repaired.jsonl 可以是不存在的文件）
输出每行 {"q", "a", "category", "band"}，不含系统提示（训练时按学段补，见 persona.py）。"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules

NUM = "[\\d零一二两三四五六七八九十百]"
MATH_RE = re.compile(f"{NUM}+\\s*[加减乘除+\\-×÷]\\s*{NUM}+")     # 只拦算式（模型会算错）；“内角和等于180度”这种讲解不拦
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
SLACK_CHAT, SLACK_KNOW = 5, 10                                    # 比审核上限多容忍几个字


def build(src, fix, band):
    """返回 (rows, 各种被去掉的原因计数)。"""
    R = rules.load(band)
    best = {}
    for path in (src, fix):                                   # 后读的修补结果覆盖先前的丢弃
        if not os.path.exists(path):
            continue
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["status"] in ("ok", "fixed") and r["a"].strip():
                best[r["q"]] = r
            else:
                best.setdefault(r["q"], None)
    why, rows = collections.Counter(), []
    for q, r in best.items():
        if r is None:
            why["审核没过"] += 1; continue
        a = re.sub(r"\s+", "", r["a"]).strip("\"“”")
        limit = R.know_max + SLACK_KNOW if any(k in r["cat"] for k in R.know) else R.chat_max + SLACK_CHAT
        if re.search(r"[A-Za-z]{2,}", a): why["有英文"] += 1; continue
        if EMOJI_RE.search(a): why["有表情"] += 1; continue
        if MATH_RE.search(a): why["有算式"] += 1; continue
        if len(a) > limit: why["太长"] += 1; continue
        if len(re.findall(r"[。！？!?]", a.rstrip("。！？!?"))) > 2: why["超过三句"] += 1; continue
        if len(re.findall(r"[？?]", a)) > 1: why["问句太多"] += 1; continue
        a = re.sub("^" + re.escape(R.name) + "[：:]", "", a)         # 开头带了“名字：”的去掉
        rows.append({"q": q, "a": a, "category": r["cat"], "band": band})
    return rows, why


def main():
    src, fix, out = sys.argv[1:4]
    rows, why = build(src, fix, rules.band_from_env())
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("可用", len(rows), "去掉", dict(why))
    print("各类", collections.Counter(r["category"][:12] for r in rows).most_common())


if __name__ == "__main__":
    main()
