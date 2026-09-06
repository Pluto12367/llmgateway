"""网关配置加载与校验：读取 gateway.toml。

所有模块共享的只读配置对象 `GatewayConfig`；真实密钥只经环境变量引用。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_NAME = "gateway.toml"


class ConfigError(ValueError):
    pass


@dataclass
class Provider:
    """一个上游 OpenAI 兼容厂商。"""

    name: str
    base_url: str
    api_key_env: str
    in_usd_per_1m: float
    out_usd_per_1m: float
    timeout: float = 60.0
    embed_models: list[str] = field(default_factory=list)

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "").strip()

    def has_key(self) -> bool:
        return bool(self.api_key())

    def cost(self, in_tokens: int, out_tokens: int) -> float:
        """按实际 token 用量与单价估算成本（美元）。"""
        return in_tokens / 1e6 * self.in_usd_per_1m + out_tokens / 1e6 * self.out_usd_per_1m


@dataclass
class Candidate:
    """别名解析出的一个（厂商, 模型）候选。"""

    provider: str
    model: str


@dataclass
class ModelAlias:
    name: str
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class CacheSettings:
    enabled: bool = True
    threshold: float = 0.96
    max_entries: int = 5000
    embedder: str = "auto"  # auto | fastembed | hash | none


@dataclass
class Limits:
    default_rpm: int = 60
    monthly_budget_usd: float | None = None


@dataclass
class GatewayConfig:
    path: Path
    data_dir: Path
    providers: dict[str, Provider]
    aliases: dict[str, ModelAlias]
    limits: Limits
    cache: CacheSettings
    allow_origins: list[str] = field(default_factory=lambda: ["*"])

    @property
    def db_path(self) -> Path:
        return self.data_dir / "llmgateway.db"

    def resolve(self, alias: str) -> list[Candidate]:
        """模型别名 -> 候选列表；不存在则抛 ConfigError。"""
        item = self.aliases.get(alias)
        if item is None or not item.candidates:
            raise ConfigError(f"未知模型别名：{alias}（请在 gateway.toml [models] 中定义）")
        return list(item.candidates)

    def provider(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            raise ConfigError(f"配置里不存在厂商：{name}")

    def embed_target(self, model: str) -> tuple[str, str] | None:
        """查找能承接嵌入请求的厂商：(provider_name, model)。找不到返回 None。"""
        for name, p in self.providers.items():
            if model in p.embed_models:
                return name, model
        return None

    def model_catalog(self) -> list[dict]:
        """供 GET /v1/models 使用的别名清单。"""
        return [
            {
                "id": alias.name,
                "object": "model",
                "created": 0,
                "owned_by": alias.candidates[0].provider if alias.candidates else "",
            }
            for alias in self.aliases.values()
        ]

    @classmethod
    def load(cls, path: str | Path) -> "GatewayConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise ConfigError(f"找不到配置文件：{cfg_path}")
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

        gw = raw.get("gateway", {})
        cache_raw = gw.get("cache", {})
        limits_raw = gw.get("limits", {})

        providers: dict[str, Provider] = {}
        for name, p in raw.get("providers", {}).items():
            if not isinstance(p, dict):
                raise ConfigError(f"providers.{name} 必须是表")
            base_url = str(p.get("base_url", "")).rstrip("/")
            if not base_url:
                raise ConfigError(f"providers.{name}.base_url 不能为空")
            api_key_env = str(p.get("api_key_env", ""))
            if not api_key_env:
                raise ConfigError(f"providers.{name}.api_key_env 不能为空")
            providers[name] = Provider(
                name=name,
                base_url=base_url,
                api_key_env=api_key_env,
                in_usd_per_1m=float(p.get("in_usd_per_1m", 0.0)),
                out_usd_per_1m=float(p.get("out_usd_per_1m", 0.0)),
                timeout=float(p.get("timeout", 60.0)),
                embed_models=[str(m) for m in p.get("embed_models", [])],
            )

        aliases: dict[str, ModelAlias] = {}
        for name, m in raw.get("models", {}).items():
            cands_raw = m.get("candidates", []) if isinstance(m, dict) else []
            cands: list[Candidate] = []
            for c in cands_raw:
                if not isinstance(c, dict):
                    raise ConfigError(f"models.{name} 的 candidates 元素必须是表")
                provider = str(c.get("provider", ""))
                model = str(c.get("model", ""))
                if provider not in providers:
                    raise ConfigError(
                        f"models.{name} 引用了未定义的厂商 {provider}"
                    )
                cands.append(Candidate(provider=provider, model=model))
            if not cands:
                raise ConfigError(f"models.{name} 至少需要一个 candidate")
            aliases[name] = ModelAlias(name=name, candidates=cands)

        if not aliases:
            raise ConfigError("未配置任何模型别名（[models] 段为空）")

        budget = gw.get("monthly_budget_usd")
        data_dir_raw = str(gw.get("data_dir", "data"))
        data_dir = Path(data_dir_raw)
        if not data_dir.is_absolute():
            data_dir = cfg_path.parent / data_dir

        embedder = str(cache_raw.get("embedder", "auto")).strip().lower()
        if embedder not in ("auto", "fastembed", "hash", "none"):
            raise ConfigError(f"cache.embedder 非法：{embedder}（可选 auto/fastembed/hash/none）")
        threshold = float(cache_raw.get("threshold", 0.96))
        if not 0.5 <= threshold <= 0.999:
            raise ConfigError(f"cache.threshold 应在 0.5~0.999 之间，当前 {threshold}")

        return cls(
            path=cfg_path,
            data_dir=data_dir,
            providers=providers,
            aliases=aliases,
            limits=Limits(
                default_rpm=int(limits_raw.get("default_rpm", 60)),
                monthly_budget_usd=float(budget) if budget is not None else None,
            ),
            cache=CacheSettings(
                enabled=bool(cache_raw.get("enabled", True)),
                threshold=threshold,
                max_entries=int(cache_raw.get("max_entries", 5000)),
                embedder=embedder,
            ),
            allow_origins=[str(o) for o in gw.get("allow_origins", ["*"])],
        )


DEFAULT_CONFIG_TEXT = """\
# LLMGateway 配置。运行 `llmgateway init` 生成；厂商密钥只经环境变量引用。

[gateway]
monthly_budget_usd = 50.0
data_dir = "data"

[gateway.cache]
enabled = true
threshold = 0.96
max_entries = 5000
embedder = "auto"

[gateway.limits]
default_rpm = 60

# [providers.deepseek]
# base_url = "https://api.deepseek.com/v1"
# api_key_env = "DEEPSEEK_API_KEY"
# in_usd_per_1m = 0.27
# out_usd_per_1m = 1.10

# [providers.glm]
# base_url = "https://open.bigmodel.cn/api/paas/v4"
# api_key_env = "ZHIPU_API_KEY"
# in_usd_per_1m = 0.14
# out_usd_per_1m = 0.14

# [models.fast]
# candidates = [
#     { provider = "deepseek", model = "deepseek-chat" },
#     { provider = "glm",      model = "glm-4-flash" },
# ]
"""
