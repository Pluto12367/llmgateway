"""上游 OpenAI 兼容厂商调用层：非流式 + SSE 流式透传。

失败统一抛 `UpstreamFailure`：retryable=True 表示「换下一家厂商可能成功」的
故障（网络/超时/5xx/429/401/403/404），由路由层做降级；retryable=False 是
请求本身问题（如 400 参数非法），不应换厂商重试。
"""

from __future__ import annotations

import json
import time

import httpx

from .config import Provider

RETRYABLE_STATUS = {401, 403, 404, 408, 409, 429, 500, 502, 503, 504, 529}


class UpstreamFailure(Exception):
    def __init__(self, message: str, *, status: int | None = None,
                 retryable: bool = True, provider: str = "", model: str = "",
                 body: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.body = body[:800]

    @property
    def http_status(self) -> int:
        return self.status if self.status else 502


def _headers(provider: Provider) -> dict:
    return {
        "Authorization": f"Bearer {provider.api_key()}",
        "Content-Type": "application/json",
    }


class UpstreamStream:
    """一条已打开的上游流式响应；由路由层消费并负责关闭。"""

    def __init__(self, client: httpx.AsyncClient, resp: httpx.Response,
                 provider: str, model: str, opened_ts: float):
        self._client = client
        self._resp = resp
        self.provider = provider
        self.model = model
        self.opened_ts = opened_ts

    async def aiter_chunks(self):
        """逐条 yield OpenAI 流式 chunk（dict）；[DONE] 被消费掉。"""
        async for line in self._resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            if data == "[DONE]":
                return
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            yield obj

    async def aclose(self) -> None:
        await self._resp.aclose()
        await self._client.aclose()


def _client(provider: Provider, transport=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=provider.base_url,
        timeout=httpx.Timeout(provider.timeout, connect=10.0),
        transport=transport,
    )


def _request_body(payload: dict, model: str) -> dict:
    body = dict(payload)
    body["model"] = model
    return body


async def chat_nonstream(provider: Provider, model: str, payload: dict,
                         transport=None) -> tuple[dict, float]:
    """发一次非流式对话请求，返回 (响应 JSON, 延迟毫秒)。失败抛 UpstreamFailure。"""
    started = time.perf_counter()
    if not provider.has_key():
        raise UpstreamFailure(
            f"厂商 {provider.name} 未配置 API Key（环境变量 {provider.api_key_env}）",
            retryable=True, provider=provider.name, model=model,
        )
    body = _request_body(payload, model)
    async with _client(provider, transport) as client:
        try:
            resp = await client.post("/chat/completions", json=body,
                                     headers=_headers(provider))
        except httpx.HTTPError as e:
            raise UpstreamFailure(
                f"连接厂商 {provider.name} 失败：{e.__class__.__name__}",
                retryable=True, provider=provider.name, model=model,
            ) from e
        latency = (time.perf_counter() - started) * 1000.0
        text = (await resp.aread()).decode("utf-8", "replace")
    if resp.status_code >= 400:
        retryable = resp.status_code in RETRYABLE_STATUS
        raise UpstreamFailure(
            f"厂商 {provider.name} 返回 HTTP {resp.status_code}",
            status=resp.status_code, retryable=retryable,
            provider=provider.name, model=model, body=text,
        )
    try:
        data = json.loads(text)
    except ValueError as e:
        raise UpstreamFailure(
            f"厂商 {provider.name} 返回非 JSON：{text[:200]}",
            retryable=True, provider=provider.name, model=model,
        ) from e
    return data, latency


async def open_stream(provider: Provider, model: str, payload: dict,
                      transport=None) -> UpstreamStream:
    """打开一条上游流式响应（已收到 HTTP 头）。连接/HTTP 级失败抛 UpstreamFailure。"""
    if not provider.has_key():
        raise UpstreamFailure(
            f"厂商 {provider.name} 未配置 API Key（环境变量 {provider.api_key_env}）",
            retryable=True, provider=provider.name, model=model,
        )
    body = _request_body(payload, model)
    body["stream"] = True
    client = _client(provider, transport)
    try:
        req = client.build_request("POST", "/chat/completions", json=body,
                                   headers=_headers(provider))
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        await client.aclose()
        raise UpstreamFailure(
            f"连接厂商 {provider.name} 失败：{e.__class__.__name__}",
            retryable=True, provider=provider.name, model=model,
        ) from e
    if resp.status_code >= 400:
        body_text = (await resp.aread()).decode("utf-8", "replace")
        await resp.aclose()
        await client.aclose()
        retryable = resp.status_code in RETRYABLE_STATUS
        raise UpstreamFailure(
            f"厂商 {provider.name} 返回 HTTP {resp.status_code}",
            status=resp.status_code, retryable=retryable,
            provider=provider.name, model=model, body=body_text,
        )
    return UpstreamStream(client, resp, provider.name, model, time.perf_counter())


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（中英混合，1.6 字符/token），供无 usage 的上游兜底记账。"""
    if not text:
        return 0
    return max(1, round(len(text) / 1.6))


def extract_usage(payload: dict, data: dict, content: str) -> tuple[dict, bool]:
    """提取用量。上游给了 usage 就用真实的，否则按文本长度估算并标记 estimated。"""
    usage = (data or {}).get("usage")
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }, False
    # 估算：输入侧按请求体 JSON 长度粗估
    import json as _json

    try:
        prompt_raw = _json.dumps(payload.get("messages", ""), ensure_ascii=False)
    except (TypeError, ValueError):
        prompt_raw = ""
    return {
        "prompt_tokens": estimate_tokens(prompt_raw),
        "completion_tokens": estimate_tokens(content),
    }, True
