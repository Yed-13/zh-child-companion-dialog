"""build_data 过滤逻辑和 llm_client 回退逻辑的小测试。运行：python tests/test_build.py（装了 pytest 也可以 pytest tests）"""
import http.server, json, os, sys, tempfile, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_data, llm_client, persona

MOOD = "说自己的心情：开心、难过"
ANSWERS = [   # 5 条：合格（带名字前缀）、有英文、有算式、审核没过（靠修补救回）、两个问句
    {"cat": MOOD, "q": "我今天很开心", "a": f"{persona.NAME}：真好呀！今天发生什么开心事了？", "status": "ok"},
    {"cat": MOOD, "q": "我们玩手机", "a": "好呀，我们一起玩 game 吧。", "status": "ok"},
    {"cat": "直接问算式题", "q": "三加五等于几", "a": "3加5等于8。", "status": "ok"},
    {"cat": MOOD, "q": "我有点难过", "a": "", "status": "drop", "problem": "太长"},
    {"cat": MOOD, "q": "你好吗", "a": "我很好呀？你呢？", "status": "ok"},
]
REPAIRED = [{"cat": MOOD, "q": "我有点难过", "a": "抱抱你，想说说发生了什么吗？", "status": "ok"}]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_build_filters():
    with tempfile.TemporaryDirectory() as d:
        _write(f"{d}/a.jsonl", ANSWERS); _write(f"{d}/r.jsonl", REPAIRED)
        rows, why = build_data.build(f"{d}/a.jsonl", f"{d}/r.jsonl", 1)
    assert {r["q"] for r in rows} == {"我今天很开心", "我有点难过"}
    assert next(r for r in rows if r["q"] == "我今天很开心")["a"] == "真好呀！今天发生什么开心事了？"   # 名字前缀去掉
    assert all(set(r) == {"q", "a", "category", "band"} and r["band"] == 1 for r in rows)
    assert why["有英文"] == 1 and why["有算式"] == 1 and why["问句太多"] == 1


def test_build_without_repair_file():
    with tempfile.TemporaryDirectory() as d:
        _write(f"{d}/a.jsonl", ANSWERS)
        rows, why = build_data.build(f"{d}/a.jsonl", f"{d}/missing.jsonl", 1)
    assert {r["q"] for r in rows} == {"我今天很开心"} and why["审核没过"] == 1


def test_math_regex():
    assert build_data.MATH_RE.search("3加5等于8") and build_data.MATH_RE.search("十乘二十")
    assert not build_data.MATH_RE.search("内角和等于180度")


def test_llm_client_retries_without_thinking_flags():
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append("thinking" in body)
            code, out = (400, {"error": "unknown field"}) if "thinking" in body else (200, {"choices": [{"message": {"content": "<think>x</think>好的"}}], "usage": {"total_tokens": 3}})
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("localhost", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    text, usage = llm_client.chat([{"role": "user", "content": "hi"}], base_url=f"http://localhost:{srv.server_port}/v1", api_key="")
    srv.shutdown()
    assert text == "好的" and usage["total_tokens"] == 3 and seen == [True, False]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
