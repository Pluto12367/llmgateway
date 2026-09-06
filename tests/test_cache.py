"""语义缓存测试：精确命中 / 同义命中 / 阈值以下不串 / 可缓存判定 / 流式回放。"""

from __future__ import annotations

import pytest

from llmgateway.cache import (SemanticCache, cacheable, replay_chunks,
                              serialize_messages)
from llmgateway.embed import HashingEmbedder

MSG_A = [
    {"role": "system", "content": "你是简洁的助手。"},
    {"role": "user", "content": "用一句话总结这段代码的异常处理模式。"},
]
MSG_A_SYNONYM = [
    {"role": "system", "content": "你是简洁的助手。"},
    {"role": "user", "content": "请用一句话说明这段代码是如何处理异常的。"},
]
MSG_UNRELATED = [
    {"role": "system", "content": "你是简洁的助手。"},
    {"role": "user", "content": "解释一下什么是依赖倒置原则。"},
]


def make_cache(store, threshold=0.60) -> SemanticCache:
    return SemanticCache(store, embedder=HashingEmbedder(),
                         threshold=threshold, max_entries=100)


def test_exact_hit(store):
    cache = make_cache(store)
    cache.put("fast", "a", "model-a", MSG_A, "回答A", {"completion_tokens": 5})
    hit = cache.lookup("fast", MSG_A)
    assert hit is not None
    assert hit["content"] == "回答A"


def test_synonym_hit_above_threshold(store):
    cache = make_cache(store)
    cache.put("fast", "a", "model-a", MSG_A, "回答A", {"completion_tokens": 5})
    hit = cache.lookup("fast", MSG_A_SYNONYM)
    assert hit is not None, "同义改写应在 0.60 阈值内命中"


def test_unrelated_miss(store):
    cache = make_cache(store)
    cache.put("fast", "a", "model-a", MSG_A, "回答A", {"completion_tokens": 5})
    assert cache.lookup("fast", MSG_UNRELATED) is None


def test_alias_isolation(store):
    cache = make_cache(store)
    cache.put("fast", "a", "model-a", MSG_A, "回答A", {"completion_tokens": 5})
    assert cache.lookup("reason", MSG_A) is None  # 不同别名互不可见


def test_no_embedder_exact_only(store):
    cache = SemanticCache(store, embedder=None, threshold=0.6)
    cache.put("fast", "a", "m", MSG_A, "回答A", {})
    assert cache.lookup("fast", MSG_A) is not None
    assert cache.lookup("fast", MSG_A_SYNONYM) is None  # 无嵌入只能精确命中


def test_prune_on_put(store):
    cache = make_cache(store)
    cache.max_entries = 3
    for i in range(6):
        msgs = [{"role": "user", "content": f"问题编号 {i} 独特内容"}]
        cache.put("fast", "a", "m", msgs, f"答{i}", {})
    assert store.cache_count() == 3


class TestCacheable:
    def test_temperature_zero_ok(self):
        assert cacheable({"messages": MSG_A, "temperature": 0})

    def test_temperature_missing_ok(self):
        assert cacheable({"messages": MSG_A})

    def test_nonzero_temperature_not(self):
        assert not cacheable({"messages": MSG_A, "temperature": 0.7})

    def test_tools_not(self):
        assert not cacheable({"messages": MSG_A, "tools": [{"type": "function"}]})

    def test_structured_output_not(self):
        assert not cacheable({"messages": MSG_A, "response_format": {"type": "json_object"}})

    def test_no_messages_not(self):
        assert not cacheable({"messages": []})

    def test_non_text_content_not(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert not cacheable({"messages": msgs})


def test_serialize_messages():
    text = serialize_messages(MSG_A)
    assert "system: 你是简洁的助手。" in text
    assert "user: 用一句话" in text


def test_replay_chunks_is_async_and_complete(store):
    import asyncio

    class FakeAttempt:
        alias = "fast"

    content = "这是一段用于回放的缓存内容，长度超过十六个字符以触发多段切分。"
    chunks = asyncio.run(_collect(content, FakeAttempt()))

    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    joined = "".join(c["choices"][0]["delta"].get("content", "")
                     for c in chunks[:-1])
    assert joined == content

async def _collect(content, attempt):
    out = []
    async for c in replay_chunks(content, attempt):
        out.append(c)
    return out
