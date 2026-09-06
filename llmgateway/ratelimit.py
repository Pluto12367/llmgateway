"""按虚拟 key 的令牌桶限流 + 月度预算（全局/按 key）授权。

限流为进程内状态（单机网关足够）；预算以 usage 表当月累计成本为准，超出即拒绝。
授权失败抛 `GatewayError`（由 FastAPI 异常处理器统一转 JSON）。
"""

from __future__ import annotations

import threading
import time

from .config import GatewayConfig
from .router import GatewayError
from .store import Store


class TokenBucket:
    """按 RPM 的令牌桶；突发允许 capacity 个，之后按 rate 匀速补充。"""

    def __init__(self, rpm: int):
        self.capacity = max(1, rpm)
        self.rate = self.capacity / 60.0
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity,
                          self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class Limiter:
    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key_id: str, rpm: int) -> bool:
        with self._lock:
            bucket = self._buckets.get(key_id)
            if bucket is None or bucket.capacity != max(1, rpm):
                bucket = TokenBucket(rpm)
                self._buckets[key_id] = bucket
            return bucket.take()


def extract_bearer(authorization: str | None) -> str | None:
    """从 Authorization 头解析 Bearer 密钥。"""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def authorize(config: GatewayConfig, store: Store, limiter: Limiter,
              authorization: str | None) -> dict:
    """校验密钥并做限流/预算检查；通过返回 key 信息，否则抛 GatewayError。"""
    secret = extract_bearer(authorization)
    if not secret:
        raise GatewayError("缺少 Authorization: Bearer sk-gw_xxx 请求头",
                           kind="auth", status=401,
                           error_type="invalid_request_error")
    row = store.verify_key(secret)
    if row is None:
        raise GatewayError("无效的网关密钥", kind="auth", status=401,
                           error_type="invalid_request_error")

    rpm = row["rpm"] or config.limits.default_rpm
    if not limiter.allow(row["id"], rpm):
        raise GatewayError(f"超出限流：该 key 每分钟最多 {rpm} 次",
                           kind="ratelimit", status=429,
                           error_type="rate_limit_error")

    # 预算检查：先按 key 自己的额度，再看全局额度
    key_budget = row["monthly_budget_usd"]
    spend_key = store.month_spend(row["id"]) if key_budget else 0.0
    if key_budget is not None and spend_key >= key_budget:
        raise GatewayError(
            f"该 key 当月预算 ${key_budget:.2f} 已用完（已用 ${spend_key:.2f}）",
            kind="budget", status=429, error_type="budget_exceeded",
        )
    global_budget = config.limits.monthly_budget_usd
    if global_budget is not None and global_budget > 0:
        spend_all = store.month_spend()
        if spend_all >= global_budget:
            raise GatewayError(
                f"网关当月总预算 ${global_budget:.2f} 已用完（已用 ${spend_all:.2f}）",
                kind="budget", status=429, error_type="budget_exceeded",
            )
    return {"key_id": row["id"], "key_name": row["name"]}
