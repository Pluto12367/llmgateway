"""HTTP API 端到端测试（走 ASGITransport + MockTransport 上游，全程离线）。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from conftest import make_transport, write_config

from llmgateway.config import GatewayConfig
from llmgateway.server import make_server
from llmgateway.store import Store


def make_app(tmp_path, mode="ok", calls=None, budget=100.0):
    cfg = GatewayConfig.load(write_config(tmp_path, budget=budget))
    app = make_server(cfg, transport=make_transport(mode, calls))
    store = Store(cfg.db_path)
    key = store.create_key("tester", rpm=60, monthly_budget_usd=10.0)
    return app, key["key"], store


def call(app, method, path, **kw):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            return await c.request(method, path, **kw)

    return asyncio.run(run())


def test_chat_nonstream_roundtrip(tmp_path):
    app, key, store = make_app(tmp_path)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "fast", "temperature": 0, "messages": [
                 {"role": "user", "content": "接口测试问题甲。"}]},
             headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["content"].startswith("[a] 回答")
    assert data["usage"]["prompt_tokens"] == 100
    assert r.headers["x-gw-provider"] == "a"
    assert r.headers["x-gw-cache"] == "MISS"

    # 第二次相同请求 → 缓存命中 + 零成本
    r2 = call(app, "POST", "/v1/chat/completions",
              json={"model": "fast", "temperature": 0, "messages": [
                  {"role": "user", "content": "接口测试问题甲。"}]},
              headers={"Authorization": f"Bearer {key}"})
    assert r2.headers["x-gw-cache"] == "HIT"
    assert float(r2.headers["x-gw-cost-usd"]) == 0.0


def test_chat_requires_auth(tmp_path):
    app, key, _ = make_app(tmp_path)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "fast", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 401
    assert "error" in r.json()


def test_chat_unknown_model(tmp_path):
    app, key, _ = make_app(tmp_path)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
             headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 404


def test_chat_failover_and_logging(tmp_path):
    calls = []
    app, key, store = make_app(tmp_path, mode="down-a", calls=calls)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "fast", "temperature": 0, "messages": [
                 {"role": "user", "content": "降级测试问题乙。"}]},
             headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200
    assert r.headers["x-gw-provider"] == "b"
    assert [c["tag"] for c in calls] == ["a", "b"]
    # 用量已记录
    rows = store.query("SELECT provider, status FROM usage")
    assert rows and rows[-1]["provider"] == "b" and rows[-1]["status"] == "ok"


def test_chat_stream_sse(tmp_path):
    app, key, store = make_app(tmp_path)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            async with c.stream(
                "POST", "/v1/chat/completions",
                json={"model": "fast", "temperature": 0, "stream": True,
                      "messages": [{"role": "user", "content": "流式接口测试丙。"}]},
                headers={"Authorization": f"Bearer {key}"},
            ) as r:
                head = dict(r.headers)
                lines = []
                async for line in r.aiter_lines():
                    lines.append(line)
                return head, lines

    head, lines = asyncio.run(run())
    assert head["x-gw-cache"] == "MISS"
    body = "\n".join(lines)
    assert body.rstrip().endswith("data: [DONE]")
    chunks = [json.loads(l[5:]) for l in lines
              if l.startswith("data: ") and not l.endswith("[DONE]")]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks
                   if c.get("choices"))
    assert text.startswith("[a] 流式回答")
    # 结束块带 usage
    last = [c for c in chunks if c["choices"][0]["finish_reason"] == "stop"][0]
    assert last["usage"]["completion_tokens"] == 20


def test_models_list(tmp_path):
    app, key, _ = make_app(tmp_path)
    r = call(app, "GET", "/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == ["fast"]


def test_healthz_and_dashboard(tmp_path):
    app, key, _ = make_app(tmp_path)
    assert call(app, "GET", "/healthz").json() == {"status": "ok"}
    r = call(app, "GET", "/dashboard")
    assert r.status_code == 200
    assert "LLMGateway" in r.text and "<html" in r.text.lower()


def test_stats_endpoint(tmp_path):
    app, key, _ = make_app(tmp_path)
    call(app, "POST", "/v1/chat/completions",
         json={"model": "fast", "temperature": 0,
               "messages": [{"role": "user", "content": "统计测试问题丁。"}]},
         headers={"Authorization": f"Bearer {key}"})
    r = call(app, "GET", "/api/stats?span=all")
    assert r.status_code == 200
    st = r.json()
    assert st["totals"]["requests"] == 1
    assert st["totals"]["ok"] == 1
    assert st["by_provider"][0]["provider"] == "a"
    assert st["totals"]["cost_usd"] > 0


def test_global_budget_blocks(tmp_path):
    # 全局预算先烧光，再发请求应 429
    app, key, store = make_app(tmp_path, budget=1.0)
    store.record_usage(key_id="someone", key_name="someone", status="ok",
                       cost_usd=2.0)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "fast", "temperature": 0,
                   "messages": [{"role": "user", "content": "预算测试问题戊。"}]},
             headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "budget"


def test_key_budget_blocks(tmp_path):
    app, key, store = make_app(tmp_path, budget=100.0)
    row = store.query("SELECT id, name FROM keys")[0]
    # 该 key 自己的月度预算是 $10，烧到 $10 触发 429
    store.record_usage(key_id=row["id"], key_name=row["name"], status="ok",
                       cost_usd=10.0)
    r = call(app, "POST", "/v1/chat/completions",
             json={"model": "fast", "temperature": 0,
                   "messages": [{"role": "user", "content": "key 预算测试。"}]},
             headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 429
    assert "已用完" in r.json()["error"]["message"]


def test_malformed_json(tmp_path):
    app, key, _ = make_app(tmp_path)
    r = call(app, "POST", "/v1/chat/completions",
             content=b"{not json",
             headers={"Authorization": f"Bearer {key}",
                      "Content-Type": "application/json"})
    assert r.status_code == 400
