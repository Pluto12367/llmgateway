"""SQLite 持久层：虚拟 key、请求用量日志、语义缓存条目。

单文件数据库，线程安全（写入加锁 + WAL）。虚拟 key 只存 sha256，原始密钥只在
创建时返回一次；用量按每请求一行累计，供预算与看板统计。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

KEY_PREFIX = "sk-gw_"
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys(
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  rpm INTEGER,
  monthly_budget_usd REAL,
  created_ts REAL NOT NULL,
  revoked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_keys_hash ON keys(key_hash);

CREATE TABLE IF NOT EXISTS usage(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  key_id TEXT NOT NULL,
  key_name TEXT NOT NULL,
  alias TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  endpoint TEXT NOT NULL DEFAULT 'chat',
  status TEXT NOT NULL,
  cache_hit INTEGER NOT NULL DEFAULT 0,
  prompt_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  latency_ms REAL NOT NULL DEFAULT 0,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
CREATE INDEX IF NOT EXISTS idx_usage_key ON usage(key_id);

CREATE TABLE IF NOT EXISTS cache(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_hash TEXT NOT NULL,
  alias TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  vector TEXT NOT NULL,
  content TEXT NOT NULL,
  usage_json TEXT NOT NULL,
  tokens INTEGER NOT NULL DEFAULT 0,
  created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_alias ON cache(alias);
CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache(prompt_hash);
"""


def month_start_ts(now: float | None = None) -> float:
    """当前自然月起点（本地时区）的 epoch 秒。"""
    import datetime

    now = now if now is not None else time.time()
    dt = datetime.datetime.fromtimestamp(now)
    start = datetime.datetime(dt.year, dt.month, 1)
    return start.timestamp()


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---------- keys ----------

    def create_key(
        self,
        name: str,
        rpm: int | None = None,
        monthly_budget_usd: float | None = None,
    ) -> dict:
        """创建虚拟 key，返回含完整密钥的字典（完整密钥仅此一次可见）。"""
        key_id = secrets.token_hex(3)
        secret = f"{KEY_PREFIX}{key_id}.{secrets.token_hex(16)}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO keys(id, name, key_hash, rpm, monthly_budget_usd,"
                " created_ts, revoked) VALUES (?,?,?,?,?,?,0)",
                (key_id, name, hash_secret(secret), rpm, monthly_budget_usd, time.time()),
            )
            self._conn.commit()
        return {"id": key_id, "name": name, "key": secret}

    def list_keys(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, rpm, monthly_budget_usd, created_ts, revoked"
                " FROM keys ORDER BY created_ts"
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_key(self, key_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE keys SET revoked = 1 WHERE id = ? AND revoked = 0", (key_id,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def verify_key(self, secret: str) -> dict | None:
        """校验 bearer 密钥；有效返回 key 行字典（含配额字段），否则 None。"""
        if not secret.startswith(KEY_PREFIX):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, rpm, monthly_budget_usd, revoked FROM keys"
                " WHERE key_hash = ?",
                (hash_secret(secret),),
            ).fetchone()
        if row is None or row["revoked"]:
            return None
        return dict(row)

    # ---------- usage ----------

    def record_usage(self, **kw) -> None:
        """写入一行请求日志（用量/成本/状态）。"""
        fields = {
            "ts": kw.get("ts", time.time()),
            "key_id": kw.get("key_id", ""),
            "key_name": kw.get("key_name", ""),
            "alias": kw.get("alias", ""),
            "provider": kw.get("provider", ""),
            "model": kw.get("model", ""),
            "endpoint": kw.get("endpoint", "chat"),
            "status": kw.get("status", "ok"),
            "cache_hit": 1 if kw.get("cache_hit") else 0,
            "prompt_tokens": int(kw.get("prompt_tokens", 0)),
            "completion_tokens": int(kw.get("completion_tokens", 0)),
            "cost_usd": float(kw.get("cost_usd", 0.0)),
            "latency_ms": float(kw.get("latency_ms", 0.0)),
            "error": (kw.get("error") or "")[:500],
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage(ts, key_id, key_name, alias, provider, model,"
                " endpoint, status, cache_hit, prompt_tokens, completion_tokens,"
                " cost_usd, latency_ms, error)"
                " VALUES (:ts,:key_id,:key_name,:alias,:provider,:model,:endpoint,"
                ":status,:cache_hit,:prompt_tokens,:completion_tokens,:cost_usd,"
                ":latency_ms,:error)",
                fields,
            )
            self._conn.commit()

    def month_spend(self, key_id: str | None = None) -> float:
        since = month_start_ts()
        sql = "SELECT COALESCE(SUM(cost_usd),0) AS s FROM usage WHERE status='ok' AND ts>=?"
        params: list = [since]
        if key_id:
            sql += " AND key_id=?"
            params.append(key_id)
        with self._lock:
            return float(self._conn.execute(sql, params).fetchone()["s"])

    def month_calls(self, key_id: str | None = None) -> int:
        since = month_start_ts()
        sql = "SELECT COUNT(*) AS n FROM usage WHERE ts>=?"
        params: list = [since]
        if key_id:
            sql += " AND key_id=?"
            params.append(key_id)
        with self._lock:
            return int(self._conn.execute(sql, params).fetchone()["n"])

    # ---------- semantic cache ----------

    def cache_put(
        self,
        prompt_hash: str,
        alias: str,
        provider: str,
        model: str,
        vector: list[float],
        content: str,
        usage: dict,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache(prompt_hash, alias, provider, model, vector,"
                " content, usage_json, tokens, created_ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    prompt_hash,
                    alias,
                    provider,
                    model,
                    json.dumps(vector),
                    content,
                    json.dumps(usage),
                    int(usage.get("completion_tokens", 0)),
                    time.time(),
                ),
            )
            self._conn.commit()

    def cache_rows(self, alias: str | None = None) -> list[dict]:
        sql = "SELECT * FROM cache"
        params: list = []
        if alias:
            sql += " WHERE alias=?"
            params.append(alias)
        sql += " ORDER BY created_ts DESC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["vector"] = json.loads(d["vector"])
                d["usage"] = json.loads(d["usage_json"])
            except (ValueError, TypeError):
                continue
            out.append(d)
        return out

    def cache_prune(self, max_entries: int) -> int:
        """删除超出上限的最旧条目，返回删除条数。"""
        if max_entries <= 0:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE id IN ("
                " SELECT id FROM cache ORDER BY created_ts DESC LIMIT -1 OFFSET ?)",
                (max_entries,),
            )
            self._conn.commit()
        return cur.rowcount

    def cache_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM cache").fetchone()["n"])

    # ---------- generic query ----------

    def query(self, sql: str, params: list | tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
