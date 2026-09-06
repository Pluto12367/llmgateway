"""路由层测试：选择 / 降级 / 成本核算 / 全部失败 / 缓存集成。

工程上不引 pytest-asyncio：异步场景统一用 asyncio.run 包装。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from conftest import make_transport

from llmgateway.cache import SemanticCache
from llmgateway.embed import HashingEmbedder
from llmgateway.router import GatewayError, Router
from llmgateway.store import Store

PAYLOAD = {"messages": [
    {"role": "user", "content": "路由层测试的问题甲。"},
], "temperature": 0}


@pytest.fixture
def cache(store):
    return SemanticCache(store, embedder=HashingEmbedder(), threshold=0.60)


def test_routes_to_first_candidate(config, store, cache):
    calls = []

    async def run():
        router = Router(config, store, cache, transport=make_transport("ok", calls))
        return await router.chat("fast", PAYLOAD)

    attempt = asyncio.run(run())
    assert attempt.provider == "a"
    assert attempt.model == "model-a"
    assert attempt.status == "ok"
    assert calls[0]["tag"] == "a"
    assert not attempt.cache_hit


def test_failover_to_second_candidate(config, store, cache):
    calls = []

    async def run():
        router = Router(config, store, cache, transport=make_transport("down-a", calls))
        return await router.chat("fast", PAYLOAD)

    attempt = asyncio.run(run())
    assert attempt.provider == "b"
    assert attempt.model == "model-b"
    assert [c["tag"] for c in calls] == ["a", "b"]


def test_cost_accounting_real_usage(config, store, cache):
    async def run():
        router = Router(config, store, cache, transport=make_transport("ok"))
        return await router.chat("fast", PAYLOAD)

    attempt = asyncio.run(run())
    # stub usage: 100 in / 50 out；厂商 a 单价 $1/M in、$2/M out
    assert attempt.usage["prompt_tokens"] == 100
    assert attempt.usage["completion_tokens"] == 50
    assert attempt.cost_usd == pytest.approx(100 / 1e6 * 1.0 + 50 / 1e6 * 2.0)


def test_all_candidates_down_raises(config, store, cache):
    def both_down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    async def run():
        router = Router(config, store, cache, transport=httpx.MockTransport(both_down))
        await router.chat("fast", PAYLOAD)

    with pytest.raises(GatewayError) as ei:
        asyncio.run(run())
    assert ei.value.status == 503
    assert "已尝试" in str(ei.value)


def test_unknown_alias(config, store, cache):
    async def run():
        router = Router(config, store, cache, transport=make_transport("ok"))
        await router.chat("nope", PAYLOAD)

    with pytest.raises(GatewayError) as ei:
        asyncio.run(run())
    assert ei.value.status == 404


def test_cache_roundtrip_via_router(config, store, cache):
    async def run():
        router = Router(config, store, cache, transport=make_transport("ok"))
        first = await router.chat("fast", PAYLOAD)
        second = await router.chat("fast", dict(PAYLOAD))
        return first, second

    first, second = asyncio.run(run())
    assert not first.cache_hit
    assert second.cache_hit
    assert second.cost_usd == 0.0
    assert second.content == first.content


def test_stream_prepare_and_consume(config, store, cache):
    async def run():
        router = Router(config, store, cache, transport=make_transport("ok"))
        attempt, chunks = await router.prepare_stream(
            "fast", {**PAYLOAD, "stream": True})
        parts = []
        async for c in chunks:
            choices = c.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            if isinstance(delta.get("content"), str):
                parts.append(delta["content"])
        return attempt, "".join(parts)

    attempt, text = asyncio.run(run())
    assert attempt.provider == "a"
    assert text.startswith("[a] 流式回答")
    assert attempt.status == "ok"
    assert attempt.usage["completion_tokens"] == 20
    assert store.cache_count() == 1  # 流式内容也应入缓存


def test_stream_failover(config, store, cache):
    calls = []

    async def run():
        router = Router(config, store, cache, transport=make_transport("down-a", calls))
        attempt, chunks = await router.prepare_stream(
            "fast", {**PAYLOAD, "stream": True})
        async for _ in chunks:
            pass  # 消费完
        return attempt

    attempt = asyncio.run(run())
    assert attempt.provider == "b"
    assert [c["tag"] for c in calls] == ["a", "b"]
