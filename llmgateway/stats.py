"""用量统计聚合：从 usage 表算出看板所需的各类指标。

提供 today / month / day(近24h) 三种时间窗。
"""

from __future__ import annotations

import datetime
import statistics

from .store import Store, month_start_ts


def _local_midnight(now: float) -> float:
    dt = datetime.datetime.fromtimestamp(now)
    return datetime.datetime(dt.year, dt.month, dt.day).timestamp()


def span_ts(span: str, now: float | None = None) -> float | None:
    """返回时间窗起点 epoch；'all' 返回 None。"""
    now = now if now is not None else __import__("time").time()
    if span == "today":
        return _local_midnight(now)
    if span == "month":
        return month_start_ts(now)
    if span == "day":
        return now - 86400.0
    if span == "all":
        return None
    raise ValueError(f"未知时间窗：{span}")


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1] if len(values) > 1 else values[0]


def collect_stats(store: Store, span: str = "today") -> dict:
    since = span_ts(span)
    where = "WHERE ts>=?" if since is not None else ""
    params = [since] if since is not None else []

    rows = store.query(
        f"SELECT status, cache_hit, prompt_tokens, completion_tokens, cost_usd,"
        f" latency_ms, provider, alias, key_name, error, ts FROM usage {where}",
        params,
    )
    ok_rows = [r for r in rows if r["status"] == "ok"]

    total = len(rows)
    ok = len(ok_rows)
    hits = sum(1 for r in ok_rows if r["cache_hit"])
    errors = total - ok
    cost = sum(r["cost_usd"] for r in ok_rows)
    prompt_tok = sum(r["prompt_tokens"] for r in ok_rows)
    comp_tok = sum(r["completion_tokens"] for r in ok_rows)

    # 按厂商
    by_provider: dict[str, dict] = {}
    for r in ok_rows:
        d = by_provider.setdefault(r["provider"], {"calls": 0, "cost": 0.0, "cache_hits": 0})
        d["calls"] += 1
        d["cost"] += r["cost_usd"]
        d["cache_hits"] += 1 if r["cache_hit"] else 0

    # 按 key
    by_key: dict[str, dict] = {}
    for r in ok_rows:
        d = by_key.setdefault(r["key_name"], {"calls": 0, "cost": 0.0, "errors": 0})
        d["calls"] += 1
        d["cost"] += r["cost_usd"]
    for r in rows:
        if r["status"] != "ok":
            by_key.setdefault(r["key_name"], {"calls": 0, "cost": 0.0, "errors": 0})
            by_key[r["key_name"]]["errors"] += 1

    # 近 24h 小时粒度（本地时区）
    hourly: list[dict] = []
    if total:
        now = max(r["ts"] for r in rows)
        buckets: dict[str, dict] = {}
        for r in rows:
            hour_key = datetime.datetime.fromtimestamp(r["ts"]).strftime("%m-%d %H:00")
            b = buckets.setdefault(hour_key, {"hour": hour_key, "calls": 0, "cost": 0.0})
            b["calls"] += 1
            if r["status"] == "ok":
                b["cost"] += r["cost_usd"]
        hourly = [buckets[k] for k in sorted(buckets)[-24:]]

    lat = [r["latency_ms"] for r in ok_rows if r["latency_ms"] > 0]
    return {
        "span": span,
        "totals": {
            "requests": total,
            "ok": ok,
            "errors": errors,
            "cache_hits": hits,
            "cache_rate": round(hits / ok, 4) if ok else 0.0,
            "cost_usd": round(cost, 6),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "p50_ms": round(_pct(lat, 50), 1) if lat else None,
            "p95_ms": round(_pct(lat, 95), 1) if lat else None,
        },
        "by_provider": [
            {"provider": k, **v}
            for k, v in sorted(by_provider.items(), key=lambda kv: -kv[1]["cost"])
        ],
        "by_key": [
            {"key": k, **v}
            for k, v in sorted(by_key.items(), key=lambda kv: -kv[1]["cost"])
        ],
        "hourly": hourly,
        "recent": sorted(
            (
                {
                    "ts": r["ts"],
                    "alias": r["alias"],
                    "provider": r["provider"] or "-",
                    "key": r["key_name"],
                    "status": r["status"],
                    "cache_hit": bool(r["cache_hit"]),
                    "cost": round(r["cost_usd"], 6),
                    "latency_ms": round(r["latency_ms"], 1),
                    "error": (r["error"] or "")[:120],
                }
                for r in rows
            ),
            key=lambda r: -r["ts"],
        )[:25],
    }
