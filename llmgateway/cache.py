"""语义响应缓存。

命中规则：请求须为纯文本对话（无 tools/functions、temperature 为 0 或省略），
以「序列化后的完整消息」为语义单元——先按精确哈希快速命中，再做向量余弦比对，
超过 config 阈值的最近条目即命中。缓存键包含模型别名：同一语义问题问不同别名
（不同底层模型）不会互相串答。

无嵌入器（embedder=None）时退化为「完全一致的请求命中」的精确文本缓存。
"""

from __future__ import annotations

import hashlib
import time

from .store import Store


def serialize_messages(messages: list[dict]) -> str:
    """把消息列表序列化为可哈希/可嵌入的纯文本。"""
    parts = []
    for m in messages:
        role = str(m.get("role", ""))
        content = m.get("content", "")
        parts.append(f"{role}: {content}" if isinstance(content, str) else f"{role}: <non-text>")
    return "\n".join(parts)


def content_only(messages: list[dict]) -> bool:
    """消息是否全是纯文本 content（无图片/工具等非文本块）。"""
    for m in messages:
        c = m.get("content")
        if c is None:
            continue
        if isinstance(c, str):
            continue
        return False
    return True


def cacheable(payload: dict) -> bool:
    """只缓存确定性、纯文本请求，避免缓存工具调用/采样输出。"""
    if payload.get("tools") or payload.get("functions"):
        return False
    if payload.get("response_format"):
        # 结构化输出（json schema）多数实现要求逐请求命中，这里不缓存
        return False
    temperature = payload.get("temperature")
    if temperature is not None and temperature != 0:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    return content_only(messages)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class SemanticCache:
    def __init__(self, store: Store, embedder=None, threshold: float = 0.96,
                 max_entries: int = 5000):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.max_entries = max_entries

    @property
    def enabled(self) -> bool:
        return True

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def lookup(self, alias: str, messages: list[dict]) -> dict | None:
        """返回命中条目（dict）或 None。"""
        text = serialize_messages(messages)
        h = self._hash(text)
        rows = self.store.cache_rows(alias=alias)
        # 精确命中
        for r in rows:
            if r["prompt_hash"] == h:
                return r
        # 语义命中（需要嵌入器）
        if self.embedder is None or not rows:
            return None
        vec = self.embedder.embed([text])[0]
        best: dict | None = None
        best_sim = 0.0
        for r in rows:
            sim = cosine(vec, r["vector"])
            if sim > best_sim:
                best_sim = sim
                best = r
        if best is not None and best_sim >= self.threshold:
            return best
        return None

    def put(self, alias: str, provider: str, model: str, messages: list[dict],
            content: str, usage: dict) -> None:
        text = serialize_messages(messages)
        vec: list[float] = []
        if self.embedder is not None:
            vec = self.embedder.embed([text])[0]
        self.store.cache_put(
            prompt_hash=self._hash(text),
            alias=alias,
            provider=provider,
            model=model,
            vector=vec,
            content=content,
            usage=usage,
        )
        self.store.cache_prune(self.max_entries)

    def fill_attempt(self, attempt, hit: dict) -> None:
        """把命中条目的内容/用量/来源写入 attempt。"""
        attempt.cache_hit = True
        attempt.provider = hit["provider"]
        attempt.model = hit["model"]
        attempt.content = hit["content"]
        attempt.usage = dict(hit["usage"])
        attempt.status = "ok"


async def replay_chunks(content: str, attempt):
    """把缓存内容切成 OpenAI 风格的流式 chunk 逐段异步回放。"""
    import secrets
    import time as _t

    cid = f"chatcmpl-gw-{secrets.token_hex(4)}"
    created = int(_t.time())
    step = max(16, min(64, max(1, len(content) // 8)))
    pos = 0
    while pos < len(content):
        piece = content[pos:pos + step]
        pos += step
        yield {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": attempt.alias,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
    # 结束块
    yield {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": attempt.alias,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
