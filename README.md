# LLMGateway

自托管的 **OpenAI 兼容 LLM 网关**：把多家模型厂商收在一个端点后面，自带多厂商路由与故障降级、按虚拟 key 的配额/预算/限流、语义响应缓存与实时用量成本看板。任何现有 OpenAI SDK 只需改一行 `base_url` 即可接入。

```
客户端 (OpenAI SDK, base_url=http://你的网关/v1)
        │  Authorization: Bearer sk-gw_xxx（虚拟 key）
        ▼
┌─ LLMGateway ────────────────────────────────┐
│  鉴权 → 限流(令牌桶) → 月度预算检查          │
│        │                                    │
│        ▼                                    │
│  语义缓存查询 ◀── 命中: 秒回, 成本 $0        │
│        │ 未命中                             │
│        ▼                                    │
│  路由: 按别名候选顺序尝试厂商                │
│   fast: deepseek-chat → glm-4-flash         │
│        │ 某家宕机/429/5xx → 自动降级下一家   │
│        ▼                                    │
│  透传 (流式 SSE / 非流式) + 成本核算 + 日志  │
└─────────────────────────────────────────────┘
```

## 特性

- **OpenAI 兼容**：`/v1/chat/completions`（含 SSE 流式）、`/v1/models`、`/v1/embeddings`（透传）。客户端零改造。
- **多厂商路由与故障降级**：模型别名映射到有序候选列表（厂商×模型）；网络错误、429、5xx、401 等可重试故障自动切换下一家；非流式与流式都支持降级。
- **语义响应缓存**：基于嵌入向量的请求级缓存——同义改写的问题直接复用之前的回答，**成本 $0、零上游调用**。只缓存确定性请求（temperature=0、无 tools）。命中可按 SSE 流式回放。
- **按 key 治理**：虚拟 key（`sk-gw_` 前缀，只存 sha256）、每 key 令牌桶 RPM 限流、每 key 月度预算、网关全局月度预算，超限返回 OpenAI 风格错误。
- **成本可观测**：每请求记录厂商/模型/用量/成本/延迟/缓存命中；`/dashboard` 实时看板（成本、命中率、P50/P95、按厂商与 key 拆分、请求趋势）；`/api/stats` 开放 JSON。
- **可评测**：`llmgateway eval` 在录制请求集上离线回放，量测命中率与成本节省；录制集带预期标注，可算命中判断准确率。
- **纯离线可演示**：`llmgateway demo` 一键起两个本地 stub 厂商，完整演示缓存/降级/流式，不需要任何真实 API Key。

实测数字（内置录制集 `evals/recordings.json`，hash 嵌入、阈值 0.60）：

| 指标 | 数值 |
|---|---|
| 录制请求数 | 16 |
| 缓存命中 | 10（62.5%） |
| 预期命中判断准确率 | 16/16 |
| 上游调用次数 | 16 → 6 |
| 成本节省 | 62.5% |

## 快速开始

```bash
pip install -e .

# 1. 生成配置与数据目录
llmgateway init

# 2. 编辑 gateway.toml：填厂商 base_url / key 环境变量 / 价格表，
#    并定义模型别名（见 gateway.example.toml 的完整注释示例）
export DEEPSEEK_API_KEY=sk-...     # 厂商真实密钥只经环境变量引用

# 3. 创建虚拟 key
llmgateway keys add my-app --rpm 120 --budget 5.0

# 4. 启动
llmgateway serve --port 8787
```

客户端接入（以 openai python SDK 为例，唯一要改的就是 base_url 和 key）：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="sk-gw_xxx...",        # 上一步生成的虚拟 key
)

resp = client.chat.completions.create(
    model="fast",                   # 网关里的模型别名，而非具体厂商模型
    messages=[{"role": "user", "content": "你好"}],
)
```

或直接 curl：

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer sk-gw_xxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","temperature":0,"messages":[{"role":"user","content":"你好"}]}'
# 响应头: x-gw-provider: deepseek  x-gw-cache: MISS  x-gw-cost-usd: 0.00001234
```

打开 <http://127.0.0.1:8787/dashboard> 查看实时用量看板。

## 离线演示（无需任何 API Key）

```bash
llmgateway demo --outdir demo
```

脚本会起两个本地 stub 厂商，然后走网关真实端点依次演示并断言：首次 MISS → 相同提问 HIT（$0）→ 同义改写 HIT → 无关提问 MISS → 首候选宕机自动降级第二家 → 命中时流式 SSE 回放。演示产物（配置/数据库/摘要）留在 `demo/` 目录，可直接 `cd demo && llmgateway serve` 接着体验看板。

## 语义缓存说明

- **判定**：只缓存 `temperature=0`（或缺省）、纯文本、无 tools/functions/response_format 的请求；缓存键含模型别名，不同别名不互相串答。
- **嵌入后端**（`gateway.cache.embedder`）：
  - `hash`（默认回退）：零依赖、离线、确定性的字符 n-gram 词袋嵌入。标定参考：同义改写 0.57~0.87，无关问题 ≤0.4，推荐阈值 0.60。
  - `fastembed`：本地 ONNX 跑 `BAAI/bge-small-zh-v1.5`（`pip install 'llmgateway[local]'`），语义质量更高，推荐阈值 0.92~0.96。
  - `none`：只做请求完全一致的精确缓存。
- **失效边界**：语义缓存对"同义问题、答案随时间变化"的场景（如问行情）不适用——请只把确定性任务（解释、翻译、写作模板）接到设了缓存的别名上，或直接 `cache.enabled=false`。

## 评测

```bash
llmgateway eval --config evals/gateway.eval.toml evals/recordings.json
```

评测为纯离线回放（上游由 MockTransport stub 承接，不发生真实外呼），输出命中率、预期标注准确率、上游调用次数与成本节省。录制集格式见 `evals/recordings.json`：每条含 `messages` 与可选 `expect`（hit/miss）标注，换自己的业务问题集即可得到自己的数字。

## 测试

58 项单元测试，全程离线（上游由 stub 承接，不需要任何真实 API Key）：

```bash
pip install -e '.[dev]'
pytest -q
```

CI（`.github/workflows/ci.yml`）在 Ubuntu 与 Windows 上对 Python 3.11 / 3.12 / 3.13 运行同一套测试，并额外执行离线演示冒烟（`llmgateway demo`）。

## 配置参考

完整注释示例见 [gateway.example.toml](gateway.example.toml)。核心结构：

```toml
[providers.deepseek]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"     # 真实密钥不入库
in_usd_per_1m = 0.27                 # 价格表驱动成本核算与预算
out_usd_per_1m = 1.10

[models.fast]                        # 客户端只认别名
candidates = [                       # 顺序即优先级，失败自动降级
    { provider = "deepseek", model = "deepseek-chat" },
    { provider = "glm",      model = "glm-4-flash" },
]
```

## 工程结构

```
llmgateway/
├── config.py     # gateway.toml 加载与校验（厂商/别名/预算/缓存/限流）
├── store.py      # SQLite 持久层：虚拟 key(sha256)、用量日志、缓存条目（WAL）
├── embed.py      # 嵌入后端：内置 hash 嵌入 / fastembed(可选)
├── cache.py      # 语义缓存：可缓存判定、精确+余弦命中、流式回放
├── proxy.py      # 上游调用：非流式 + SSE 透传，错误分类(可降级/不可降级)
├── router.py     # 编排：缓存→候选降级→成本核算（Attempt 账本）
├── ratelimit.py  # 令牌桶限流 + Bearer 鉴权 + 月度预算
├── stats.py      # 用量聚合（看板/统计口径）
├── server.py     # FastAPI 端点 + 自包含看板(零构建单页 HTML)
├── stub.py       # 离线 stub 上游（演示/测试用）
├── demo.py       # 一键离线演示编排
├── cli.py/.py    # CLI：init/keys/config/serve/demo/eval
└── tests/        # 58 项测试（配置/持久层/限流/缓存/路由/API/流式，全程离线）
```

## 设计取舍（非目标）

- 单机单进程：限流是进程内令牌桶，缓存/日志用本地 SQLite（WAL）。多副本部署需外置 Redis/Postgres，属刻意不做。
- 缓存按"完整消息序列"为语义单元：多轮对话里 system/prior turns 变化即视为新请求。
- 不做密钥托管加密、不做请求体改写/审计策略引擎——把规模控制在三天可交付、测试全覆盖的核心链路内。

## License

MIT
