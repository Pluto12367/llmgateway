"""令牌桶限流 + Bearer 解析 + 授权（含预算）测试。"""

from __future__ import annotations

import pytest

from conftest import write_config

from llmgateway.config import GatewayConfig
from llmgateway.ratelimit import Limiter, TokenBucket, authorize
from llmgateway.router import GatewayError
from llmgateway.store import Store


def test_token_bucket_burst():
    b = TokenBucket(rpm=2)  # 容量 2，速率 2/分钟
    assert b.take() and b.take()  # 突发耗尽
    assert not b.take()


def test_limiter_per_key():
    lim = Limiter()
    assert lim.allow("k1", 2)
    assert lim.allow("k1", 2)
    assert not lim.allow("k1", 2)  # 第 3 次（1 分钟内）拒绝
    assert lim.allow("k2", 2)  # 另一个 key 独立


def test_bearer_extraction():
    assert authorize_bearer("Bearer sk-gw_x") == "sk-gw_x"
    assert authorize_bearer("bearer sk-gw_x") == "sk-gw_x"
    assert authorize_bearer("sk-gw_x") is None
    assert authorize_bearer(None) is None


def authorize_bearer(h):
    from llmgateway.ratelimit import extract_bearer
    return extract_bearer(h)


def test_authorize_ok(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    store = Store(cfg.db_path)
    row = store.create_key("u", rpm=60)
    info = authorize(cfg, store, Limiter(), f"Bearer {row['key']}")
    assert info["key_name"] == "u"
    store.close()


def test_authorize_missing_or_bad_key(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    store = Store(cfg.db_path)
    with pytest.raises(GatewayError) as ei:
        authorize(cfg, store, Limiter(), None)
    assert ei.value.status == 401

    with pytest.raises(GatewayError) as ei:
        authorize(cfg, store, Limiter(), "Bearer sk-gw_xxxx.nothing")
    assert ei.value.status == 401
    store.close()


def test_authorize_rate_limited(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    store = Store(cfg.db_path)
    row = store.create_key("u", rpm=1)
    lim = Limiter()
    authorize(cfg, store, lim, f"Bearer {row['key']}")
    with pytest.raises(GatewayError) as ei:
        authorize(cfg, store, lim, f"Bearer {row['key']}")
    assert ei.value.status == 429
    assert ei.value.kind == "ratelimit"
    store.close()


def test_authorize_key_budget_exceeded(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    store = Store(cfg.db_path)
    row = store.create_key("u", rpm=60, monthly_budget_usd=1.0)
    store.record_usage(key_id=row["id"], key_name="u", status="ok", cost_usd=1.0)
    with pytest.raises(GatewayError) as ei:
        authorize(cfg, store, Limiter(), f"Bearer {row['key']}")
    assert ei.value.status == 429
    assert ei.value.kind == "budget"
    store.close()


def test_authorize_global_budget_exceeded(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path, budget=5.0))
    store = Store(cfg.db_path)
    row = store.create_key("u", rpm=60)  # key 自身无预算
    other = store.create_key("other")
    store.record_usage(key_id=other["id"], key_name="other", status="ok", cost_usd=6.0)
    with pytest.raises(GatewayError) as ei:
        authorize(cfg, store, Limiter(), f"Bearer {row['key']}")
    assert ei.value.kind == "budget"
    store.close()
