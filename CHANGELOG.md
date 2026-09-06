# Changelog

本项目的所有重要变更都会记录在本文件。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-09-07

首个可用版本。

### 新增
- OpenAI 兼容端点：`/v1/chat/completions`（非流式 + SSE 流式）、`/v1/models`、`/v1/embeddings`（透传）、`/healthz`。
- 多厂商路由：模型别名 → 有序候选（厂商×模型），可重试故障（网络/429/5xx/401 等）自动降级，非流式与流式均支持。
- 语义响应缓存：hash / fastembed 双嵌入后端，精确 + 余弦双路命中，仅缓存确定性请求（temperature=0、无 tools），命中可流式回放，命中成本 $0。
- 虚拟 key 管理（`sk-gw_` 前缀，仅存 sha256）、按 key 令牌桶 RPM 限流、按 key 与全局月度预算。
- 请求级用量日志（厂商/模型/token/成本/延迟/缓存命中/错误）与实时看板 `/dashboard`（零构建单页 HTML，成本/命中率/P50/P95/趋势），统计 JSON `/api/stats`。
- CLI：`init` / `keys add|list|revoke` / `config` / `serve` / `demo` / `eval`。
- 一键离线演示 `llmgateway demo`：本地 stub 双厂商 + 全链路断言（缓存命中/同义改写/故障降级/流式回放），无需真实 API Key。
- 回放评测 `llmgateway eval`：内置 16 条录制集（带预期标注）实测命中率 62.5%、判断准确率 16/16、上游调用 16→6、成本节省 62.5%。
- 测试 58 项（配置/持久层/限流/缓存/路由/HTTP API/流式），全程离线可复现；GitHub Actions 多版本矩阵 CI。
