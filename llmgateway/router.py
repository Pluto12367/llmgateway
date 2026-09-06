"""路由编排：别名解析 → 语义缓存 → 候选厂商逐个尝试（自动降级）→ 成本核算。

对外暴露两种入口：
- `Router.chat()` 非流式：一次调用返回完整 Attempt（内含最终响应 JSON）；
- `Router.prepare_stream()` 流式：先打开上游连接做降级决策（此时可确定 provider
  与 cache 状态，供响应头使用），返回 (Attempt, chunk 异步迭代器)。

Attempt 是请求的账本：provider/用量/成本/延迟/缓存命中/错误，服务器据此写日志。
"""

from __future__ import annotations

import time

from . import proxy
from .cache import SemanticCache, cacheable, replay_chunks
from .config import GatewayConfig
from .store import Store
from .proxy import UpstreamFailure


class GatewayError(Exception):
    """网关层错误，携带可映射为 HTTP 状态码的信息。"""

    def __init__(self, message: str, *, kind: str = "upstream", status: int = 502,
                 provider: str = "", model: str = "", error_type: str = ""):
        super().__init__(message)
        self.kind = kind  # auth | ratelimit | budget | unknown_model | upstream | client
        self.status = status
        self.provider = provider
        self.model = model
        self.error_type = error_type or ("invalid_request_error"
                                         if status < 500 else "upstream_error")


class Attempt:
    """单次请求的执行账本。"""

    def __init__(self, alias: str, messages: list | None = None, *,
                 endpoint: str = "chat", stream: bool = False):
        self.alias = alias
        self.messages = messages or []
        self.endpoint = endpoint
        self.stream = stream
        self.t0 = time.perf_counter()
        self.cache_hit = False
        self.cacheable = False
        self.provider = ""
        self.model = ""
        self.status = "ok"          # ok | error
        self.error = ""
        self.error_status = 0
        self.content = ""
        self.usage: dict | None = None
        self.usage_estimated = False
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        self.response: dict | None = None  # 非流式最终响应 JSON

    def done(self) -> None:
        self.latency_ms = max(0.0, (time.perf_counter() - self.t0) * 1000.0)


class Router:
    def __init__(self, config: GatewayConfig, store: Store,
                 cache: SemanticCache | None, transport=None):
        self.config = config
        self.store = store
        self.cache = cache
        self.transport = transport  # httpx transport（测试注入 stub；生产为 None）

    # ---------- 解析 ----------

    def resolve(self, alias: str):
        try:
            return self.config.resolve(alias)
        except Exception as e:  # ConfigError
            raise GatewayError(str(e), kind="unknown_model", status=404,
                               error_type="invalid_request_error") from e

    def _cache_enabled_for(self, attempt: Attempt, payload: dict) -> None:
        attempt.cacheable = bool(
            self.cache is not None and self.config.cache.enabled
            and cacheable(payload)
        )

    # ---------- 非流式 ----------

    async def chat(self, alias: str, payload: dict) -> Attempt:
        attempt = Attempt(alias, payload.get("messages"))
        candidates = self.resolve(alias)
        self._cache_enabled_for(attempt, payload)

        if attempt.cacheable:
            hit = self.cache.lookup(alias, attempt.messages)
            if hit is not None:
                self.cache.fill_attempt(attempt, hit)
                attempt.cost_usd = 0.0  # 缓存命中：无上游调用，零成本
                attempt.done()
                return attempt

        last: UpstreamFailure | None = None
        for cand in candidates:
            provider = self.config.provider(cand.provider)
            try:
                data, latency = await proxy.chat_nonstream(
                    provider, cand.model, payload, self.transport
                )
            except UpstreamFailure as e:
                last = e
                continue
            # 200 但返回体异常（个别厂商在 200 里塞 error）
            if not isinstance(data, dict) or ("choices" not in data and data.get("error")):
                last = UpstreamFailure(
                    f"厂商 {cand.provider} 响应格式异常",
                    retryable=True, provider=cand.provider, model=cand.model,
                )
                continue
            self._fill_success(attempt, cand.provider, cand.model, data, latency,
                               payload)
            return attempt

        raise self._exhausted(alias, candidates, last)

    # ---------- 流式 ----------

    async def prepare_stream(self, alias: str, payload: dict):
        """返回 (Attempt, chunks 异步迭代器)。连接级失败已完成降级。"""
        attempt = Attempt(alias, payload.get("messages"), stream=True)
        candidates = self.resolve(alias)
        self._cache_enabled_for(attempt, payload)

        if attempt.cacheable:
            hit = self.cache.lookup(alias, attempt.messages)
            if hit is not None:
                self.cache.fill_attempt(attempt, hit)
                attempt.cost_usd = 0.0
                return attempt, replay_chunks(hit["content"], attempt)

        last: UpstreamFailure | None = None
        for cand in candidates:
            provider = self.config.provider(cand.provider)
            try:
                handle = await proxy.open_stream(
                    provider, cand.model, payload, self.transport
                )
            except UpstreamFailure as e:
                last = e
                continue
            attempt.provider = cand.provider
            attempt.model = cand.model
            return attempt, self._consume_stream(attempt, handle, payload)

        raise self._exhausted(alias, candidates, last)

    async def _consume_stream(self, attempt: Attempt, handle,
                              payload: dict):
        """消费上游流：透传 chunk、累计内容与用量、结束落缓存/记账。"""
        parts: list[str] = []
        upstream_usage: dict | None = None
        try:
            async for chunk in handle.aiter_chunks():
                if isinstance(chunk, dict):
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        dc = delta.get("content")
                        if isinstance(dc, str):
                            parts.append(dc)
                    u = chunk.get("usage")
                    if isinstance(u, dict) and u.get("prompt_tokens") is not None:
                        upstream_usage = u
                yield chunk
        except UpstreamFailure as e:
            attempt.status = "error"
            attempt.error = e.message
            attempt.error_status = e.http_status
            attempt.done()
            await handle.aclose()
            # 让客户端收到一条可读的 SSE 错误，再正常收尾
            yield {"error": {"message": e.message, "type": "upstream_error"}}
            return
        finally:
            await handle.aclose()

        attempt.content = "".join(parts)
        if upstream_usage is not None:
            attempt.usage = upstream_usage
            attempt.usage_estimated = False
        else:
            attempt.usage, attempt.usage_estimated = proxy.extract_usage(
                payload, {}, attempt.content
            )
        provider = self.config.provider(attempt.provider)
        attempt.cost_usd = provider.cost(
            attempt.usage.get("prompt_tokens", 0),
            attempt.usage.get("completion_tokens", 0),
        )
        attempt.done()
        if attempt.cacheable and attempt.content:
            self.cache.put(attempt.alias, attempt.provider, attempt.model,
                           attempt.messages, attempt.content, attempt.usage)

    # ---------- 内部 ----------

    def _fill_success(self, attempt: Attempt, provider_name: str, model: str,
                      data: dict, latency: float, payload: dict) -> None:
        attempt.provider = provider_name
        attempt.model = model
        attempt.response = data
        choices = data.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            if isinstance(content, list):  # 罕见：多段内容数组
                content = "".join(
                    seg.get("text", "") for seg in content if isinstance(seg, dict)
                )
        attempt.content = content
        attempt.usage, attempt.usage_estimated = proxy.extract_usage(
            payload, data, content
        )
        provider = self.config.provider(provider_name)
        attempt.cost_usd = provider.cost(
            attempt.usage.get("prompt_tokens", 0),
            attempt.usage.get("completion_tokens", 0),
        )
        attempt.latency_ms = latency
        attempt.done()
        if attempt.cacheable and content:
            self.cache.put(attempt.alias, attempt.provider, attempt.model,
                           attempt.messages, content, attempt.usage)

    def _exhausted(self, alias: str, candidates, last: UpstreamFailure | None):
        tried = ", ".join(f"{c.provider}/{c.model}" for c in candidates)
        if last is None:
            msg = f"别名 {alias} 无可用候选（已尝试 {tried}）"
            return GatewayError(msg, kind="upstream", status=502)
        hint = last.body.splitlines()[0][:200] if last.body else ""
        msg = (f"所有候选厂商均失败 [已尝试 {tried}]：{last.message}"
               + (f"。详情：{hint}" if hint else ""))
        status = last.http_status if last.status else 502
        kind = "client" if status < 500 else "upstream"
        return GatewayError(msg, kind=kind, status=status,
                            provider=last.provider, model=last.model)
