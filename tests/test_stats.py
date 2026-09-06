"""统计聚合测试。"""

from __future__ import annotations

import time

from llmgateway.stats import collect_stats
from llmgateway.store import Store


def seed(store: Store):
    now = time.time()
    store.record_usage(ts=now - 3600, key_id="k", key_name="alice", alias="fast",
                       provider="a", model="m", status="ok", cache_hit=True,
                       prompt_tokens=100, completion_tokens=50, cost_usd=0.2,
                       latency_ms=100.0)
    store.record_usage(ts=now - 1800, key_id="k", key_name="alice", alias="fast",
                       provider="b", model="m", status="ok",
                       prompt_tokens=200, completion_tokens=100, cost_usd=0.5,
                       latency_ms=300.0)
    store.record_usage(ts=now - 60, key_id="k", key_name="alice", alias="fast",
                       provider="a", model="m", status="error", error="boom",
                       latency_ms=30.0)


def test_collect_stats_all(tmp_path):
    s = Store(tmp_path / "t.db")
    seed(s)
    st = collect_stats(s, "all")
    t = st["totals"]
    assert t["requests"] == 3
    assert t["ok"] == 2
    assert t["errors"] == 1
    assert t["cache_hits"] == 1
    assert t["cache_rate"] == 0.5
    assert t["cost_usd"] == 0.7
    assert t["prompt_tokens"] == 300
    assert t["completion_tokens"] == 150
    assert t["p50_ms"] is not None and t["p95_ms"] is not None

    provs = {p["provider"]: p for p in st["by_provider"]}
    assert provs["a"]["calls"] == 1  # 错误那次不计入 provider 成功调用
    assert provs["b"]["calls"] == 1
    assert provs["b"]["cost"] == 0.5

    keys = {k["key"]: k for k in st["by_key"]}
    assert keys["alice"]["calls"] == 2
    assert keys["alice"]["errors"] == 1

    assert len(st["recent"]) == 3
    assert st["recent"][0]["status"] == "error"  # 最新在前
    s.close()


def test_collect_stats_empty(tmp_path):
    s = Store(tmp_path / "t.db")
    st = collect_stats(s, "today")
    assert st["totals"]["requests"] == 0
    assert st["totals"]["cache_rate"] == 0.0
    assert st["totals"]["p50_ms"] is None
    assert st["hourly"] == []
    s.close()


def test_collect_stats_span_invalid(tmp_path):
    s = Store(tmp_path / "t.db")
    import pytest
    with pytest.raises(ValueError):
        collect_stats(s, "week")
    s.close()
