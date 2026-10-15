"""OpenAI 兼容接口的最小客户端（只用标准库）。
环境变量：LLM_BASE_URL（如 https://host/v1，或直接写到 /chat/completions）、LLM_API_KEY（本地服务可不设）、LLM_MODEL。
chat() 返回 (文字, usage)。think=False 时请服务别先“思考”（带 thinking 关闭参数；服务返回 400/422 就去掉参数重试一次）。"""
import json, os, re, socket, time, urllib.error, urllib.request

NO_THINK = {"thinking": {"type": "disabled"}, "enable_thinking": False}


def _url(base):
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _post(url, body, key, timeout):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat(messages, think=False, max_tokens=2000, temperature=0.9, top_p=None, retries=6, timeout=180,
         model=None, base_url=None, api_key=None, extra=None):
    """messages 同 OpenAI 格式。extra：并进请求体的其他字段（如 chat_template_kwargs）。失败重试，退避等待。"""
    base = base_url or os.environ.get("LLM_BASE_URL")
    if not base:
        raise RuntimeError("请设置 LLM_BASE_URL（或传 base_url）")
    url, key = _url(base), api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
    body = {"model": model or os.environ.get("LLM_MODEL", "default"), "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, **(extra or {})}
    if top_p is not None:
        body["top_p"] = top_p
    send_flags = not think                                     # 还要不要带“关闭思考”参数
    last = None
    for k in range(retries):
        payload = {**body, **NO_THINK} if send_flags else body
        try:
            r = _post(url, payload, key, timeout)
            text = r["choices"][0]["message"].get("content") or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if not text:
                raise ValueError("empty")                       # 空回答（常见于思考耗尽 max_tokens）：重试
            return text, r.get("usage") or {}
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 422) and send_flags:             # 服务不认这两个参数：去掉重试一次，不算一次失败
                send_flags = False
                continue
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, KeyError, IndexError, ValueError) as e:
            last = e
        if k < retries - 1:
            time.sleep(min(60, 3 * (k + 1) ** 1.5))
    raise RuntimeError(f"LLM 调用失败：{last}")
