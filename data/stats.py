"""检查并统计数据：每个学段的训练/测试条数、q 是否重复、训练和测试是否有交集、各类别条数、平均长度。
用法：python data/stats.py   （在仓库根目录或 data 目录都行）"""
import collections, json, os
D = os.path.dirname(os.path.abspath(__file__))


def load(b, split):
    return [json.loads(l) for l in open(f"{D}/band{b}_{split}.jsonl", encoding="utf-8")]


tot = 0
for b in (0, 1, 2):
    tr, te = load(b, "train"), load(b, "test")
    qs = [r["q"] for r in tr + te]
    assert len(qs) == len(set(qs)), f"学段 {b} 有重复的 q"
    assert all(set(r) == {"q", "a", "category", "band"} and r["band"] == b for r in tr + te)
    tot += len(qs)
    print(f"学段 {b}：训练 {len(tr)}  测试 {len(te)}  平均 q {sum(len(r['q']) for r in tr + te) / len(qs):.1f} 字  平均 a {sum(len(r['a']) for r in tr + te) / len(qs):.1f} 字")
    for c, n in collections.Counter(r["category"] for r in tr + te).most_common():
        print(f"   {n:5d}  {c}")
print("合计", tot)
