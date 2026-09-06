"""Store 持久层测试：key 生命周期 / 用量 / 缓存条目。"""

from __future__ import annotations

import time

from llmgateway.store import Store, hash_secret, month_start_ts


def test_key_create_verify_revoke(tmp_path):
    s = Store(tmp_path / "t.db")
    row = s.create_key("alice", rpm=30, monthly_budget_usd=5.0)
    secret = row["key"]
    assert secret.startswith("sk-gw_")

    got = s.verify_key(secret)
    assert got is not None
    assert got["name"] == "alice"
    assert got["rpm"] == 30

    # 不存在的 / 格式错误的
    assert s.verify_key("sk-gw_dead.beef") is None
    assert s.verify_key("garbage") is None

    # 吊销后立即失效
    assert s.revoke_key(row["id"]) is True
    assert s.verify_key(secret) is None
    # 重复吊销返回 False
    assert s.revoke_key(row["id"]) is False
    s.close()


def test_key_hash_not_reversible(tmp_path):
    s = Store(tmp_path / "t.db")
    secret = s.create_key("bob")["key"]
    rows = s.query("SELECT key_hash FROM keys")
    assert len(rows) == 1
    assert secret not in rows[0]["key_hash"]
    assert rows[0]["key_hash"] == hash_secret(secret)
    s.close()


def test_usage_and_month_spend(tmp_path):
    s = Store(tmp_path / "t.db")
    k1 = s.create_key("k1")
    k2 = s.create_key("k2")
    now = time.time()
    s.record_usage(ts=now, key_id=k1["id"], key_name="k1", alias="fast",
                   provider="a", model="m", status="ok",
                   prompt_tokens=100, completion_tokens=50, cost_usd=0.5,
                   latency_ms=120.0)
    s.record_usage(ts=now, key_id=k1["id"], key_name="k1", alias="fast",
                   provider="a", model="m", status="ok",
                   prompt_tokens=200, completion_tokens=100, cost_usd=1.5,
                   latency_ms=80.0, cache_hit=True)
    s.record_usage(ts=now, key_id=k2["id"], key_name="k2", alias="fast",
                   provider="b", model="m", status="error", error="boom",
                   cost_usd=0.0)

    assert s.month_spend(k1["id"]) == pytest_approx(2.0)
    assert s.month_spend(k2["id"]) == 0.0  # error 请求不计成本
    assert s.month_spend() == pytest_approx(2.0)
    assert s.month_calls() == 3
    s.close()


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_cache_put_rows_prune(tmp_path):
    s = Store(tmp_path / "t.db")
    for i in range(5):
        s.cache_put(prompt_hash=f"h{i}", alias="fast", provider="a", model="m",
                    vector=[float(i)], content=f"c{i}", usage={"completion_tokens": 10})
    assert s.cache_count() == 5
    rows = s.cache_rows(alias="fast")
    assert len(rows) == 5
    assert rows[0]["vector"] == [4.0]  # 按 created_ts DESC

    removed = s.cache_prune(3)
    assert removed == 2
    assert s.cache_count() == 3
    # 保留的是最新的
    remaining = {r["content"] for r in s.cache_rows(alias="fast")}
    assert remaining == {"c2", "c3", "c4"}
    s.close()


def test_month_start_ts(tmp_path):
    import datetime
    ts = month_start_ts(1757200000.0)  # 2026-09 中的某天
    dt = datetime.datetime.fromtimestamp(ts)
    assert (dt.day, dt.hour, dt.minute) == (1, 0, 0)
