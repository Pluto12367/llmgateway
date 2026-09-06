"""配置加载与校验测试。"""

from __future__ import annotations

import textwrap

import pytest

from conftest import write_config

from llmgateway.config import ConfigError, GatewayConfig


def test_load_defaults(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.db_path == tmp_path / "data" / "llmgateway.db"
    assert cfg.limits.monthly_budget_usd == 100.0
    assert cfg.limits.default_rpm == 60
    assert cfg.cache.threshold == 0.60
    assert cfg.cache.embedder == "hash"


def test_resolve_alias_order(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    cands = cfg.resolve("fast")
    assert [(c.provider, c.model) for c in cands] == [
        ("a", "model-a"), ("b", "model-b"),
    ]


def test_resolve_unknown_alias(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    with pytest.raises(ConfigError):
        cfg.resolve("nope")


def test_missing_provider_reference(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(textwrap.dedent("""\
        [gateway]
        data_dir = "data"
        [gateway.cache]
        embedder = "hash"
        [providers.a]
        base_url = "http://x/v1"
        api_key_env = "K"
        [models.fast]
        candidates = [ { provider = "ghost", model = "m" } ]
    """), encoding="utf-8")
    with pytest.raises(ConfigError, match="ghost"):
        GatewayConfig.load(p)


def test_invalid_embedder_spec(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace('embedder = "hash"', 'embedder = "nope"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="embedder"):
        GatewayConfig.load(path)


def test_threshold_range_check(tmp_path):
    path = write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("threshold = 0.60", "threshold = 0.1")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="threshold"):
        GatewayConfig.load(path)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="找不到"):
        GatewayConfig.load(tmp_path / "nope.toml")


def test_provider_cost(tmp_path):
    cfg = GatewayConfig.load(write_config(tmp_path))
    p = cfg.provider("a")
    # 1M in + 1M out = $3.00
    assert p.cost(1_000_000, 1_000_000) == pytest.approx(3.0)
