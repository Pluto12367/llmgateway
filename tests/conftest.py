"""测试公共夹具：临时配置文件 + 网关应用 + 虚拟 key + Mock 上游传输。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

# Mock 上游不发真实请求，但网关要求厂商 key 已配置（has_key 检查）
os.environ.setdefault("TEST_UPSTREAM_A_KEY", "test-key-a")
os.environ.setdefault("TEST_UPSTREAM_B_KEY", "test-key-b")

from llmgateway.config import GatewayConfig
from llmgateway.server import make_server
from llmgateway.store import Store

CONFIG_TEMPLATE = """\
[gateway]
monthly_budget_usd = {budget}
data_dir = "data"

[gateway.cache]
enabled = true
threshold = 0.60
max_entries = 5000
embedder = "hash"

[gateway.limits]
default_rpm = 60

[providers.a]
base_url = "http://upstream-a/v1"
api_key_env = "TEST_UPSTREAM_A_KEY"
in_usd_per_1m = 1.0
out_usd_per_1m = 2.0

[providers.b]
base_url = "http://upstream-b/v1"
api_key_env = "TEST_UPSTREAM_B_KEY"
in_usd_per_1m = 0.5
out_usd_per_1m = 1.0

[models.fast]
candidates = [
    {{ provider = "a", model = "model-a" }},
    {{ provider = "b", model = "model-b" }},
]
"""


def write_config(tmp_path: Path, budget: float = 100.0) -> Path:
    path = tmp_path / "gateway.toml"
    path.write_text(CONFIG_TEMPLATE.format(budget=budget), encoding="utf-8")
    return path


def stub_ok_response(body: dict, tag: str = "a") -> httpx.Response:
    msgs = body.get("messages", [])
    user = [m for m in msgs if m.get("role") == "user"]
    text = user[-1]["content"] if user else ""
    return httpx.Response(200, json={
        "id": f"chatcmpl-{tag}", "object": "chat.completion", "created": 0,
        "model": body.get("model", ""),
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": f"[{tag}] 回答：{text}"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })


def sse_lines(chunks: list[dict]) -> bytes:
    out = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
    return (out + "data: [DONE]\n\n").encode("utf-8")


def stub_stream_content(text: str, tag: str = "a", model: str = "model-a") -> bytes:
    return sse_lines([
        {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {"content": text[:5]}, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {"content": text[5:]}, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 0, "model": model,
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
    ])


@pytest.fixture
def config(tmp_path) -> GatewayConfig:
    return GatewayConfig.load(write_config(tmp_path))


@pytest.fixture
def store(config):
    s = Store(config.db_path)
    yield s
    s.close()


@pytest.fixture
def key_row(store) -> dict:
    return store.create_key("tester", rpm=60, monthly_budget_usd=10.0)


@pytest.fixture
def auth(key_row) -> dict:
    return {"Authorization": f"Bearer {key_row['key']}"}


def make_transport(mode: str = "ok", calls: list | None = None):
    """构造 MockTransport：mode 控制厂商 a 的行为（b 始终正常）。

    - ok        两家正常
    - down-a    a 返回 503（触发降级到 b）
    - bad-json  a 返回 200 但 body 不是 JSON
    - stream    a 对流式请求返回 SSE
    """
    def handler(request: httpx.Request) -> httpx.Response:
        tag = "a" if "upstream-a" in str(request.url) else "b"
        body = json.loads(request.content)
        if calls is not None:
            calls.append({"tag": tag, "stream": bool(body.get("stream"))})
        if tag == "a":
            if mode == "down-a":
                return httpx.Response(503, json={"error": {"message": "a down"}})
            if mode == "bad-json":
                return httpx.Response(200, content=b"not-json{{{", headers={
                    "content-type": "application/json"})
        if body.get("stream"):
            text = f"[{tag}] 流式回答：{[m for m in body.get('messages', []) if m.get('role') == 'user'][-1]['content']}"
            return httpx.Response(200, content=stub_stream_content(text, tag, body.get("model", "")),
                                  headers={"content-type": "text/event-stream"})
        return stub_ok_response(body, tag)

    return httpx.MockTransport(handler)


@pytest.fixture
def app(config, key_row):
    return make_server(config, transport=make_transport("ok"))


@pytest.fixture
def client(app):
    import asyncio

    import httpx

    async def _make():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://gw")

    c = asyncio.run(_make())
    yield c
    asyncio.run(c.aclose())
