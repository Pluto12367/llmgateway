"""一键离线演示：不需要任何真实 API Key。

编排两个本地 stub「厂商」（其中一个可被切到宕机态），让网关全链路真实跑一遍：
语义缓存命中、故障自动降级、流式缓存回放，全部走网关真实 HTTP/SSE 端点。

输出目录（默认 demo/）里生成 gateway.toml + llmgateway.db，可直接 `llmgateway serve`
继续体验，或打开 http://127.0.0.1:8787/dashboard 看刚才演示产生的用量数据。
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx

from .config import GatewayConfig
from .server import make_server
from .stub import STUB_FAULTY, serve_stub_in_thread

STUB_A_PORT = 9211
STUB_B_PORT = 9212

DEMO_CONFIG = """\
# 由 `llmgateway demo` 自动生成：两个本地 stub 厂商，纯离线演示。

[gateway]
monthly_budget_usd = 50.0
data_dir = "data"
allow_origins = ["*"]

[gateway.cache]
enabled = true
# hash 嵌入的合理阈值（fastembed 语义向量才用 0.9+）
threshold = 0.60
max_entries = 2000
embedder = "hash"

[gateway.limits]
default_rpm = 120

[providers.stub-a]
base_url = "http://127.0.0.1:{pa}/v1"
api_key_env = "LLMGW_STUB_A_KEY"
in_usd_per_1m = 3.0
out_usd_per_1m = 15.0

[providers.stub-b]
base_url = "http://127.0.0.1:{pb}/v1"
api_key_env = "LLMGW_STUB_B_KEY"
in_usd_per_1m = 1.0
out_usd_per_1m = 4.0

[models.fast]
candidates = [
    {{ provider = "stub-a", model = "stub-echo" }},
    {{ provider = "stub-b", model = "stub-echo" }},
]
"""


def _write_demo_dir(outdir) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_path = outdir / "gateway.toml"
    cfg_path.write_text(
        DEMO_CONFIG.format(pa=STUB_A_PORT, pb=STUB_B_PORT), encoding="utf-8"
    )


def _print_banner(console) -> None:
    console.print()
    console.print("[bold cyan]◆ LLMGateway 离线演示[/]（无需真实 API Key）")
    console.print("  两个本地 stub 厂商 · 语义缓存 · 故障自动降级 · 流式回放")


async def _exercise(gateway_app, key_secret: str, console):
    headers = {"Authorization": f"Bearer {key_secret}"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app), base_url="http://gw"
    ) as client:
        results: list[tuple] = []

        async def ask(label, text, stream=False):
            body = {"model": "fast", "messages": [
                {"role": "system", "content": "你是简洁的助手。"},
                {"role": "user", "content": text},
            ], "temperature": 0, "stream": stream}
            text_out = ""
            if stream:
                async with client.stream("POST", "/v1/chat/completions",
                                         json=body, headers=headers) as r:
                    head = dict(r.headers)
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if line.startswith("data:") and not line.endswith("[DONE]"):
                            try:
                                chunk = json.loads(line[5:].strip())
                            except ValueError:
                                continue
                            c = chunk.get("choices") or []
                            if c and isinstance(c[0].get("delta", {}).get("content"), str):
                                text_out += c[0]["delta"]["content"]
            else:
                resp = await client.post("/v1/chat/completions", json=body,
                                         headers=headers)
                data = resp.json()
                text_out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                head = dict(resp.headers)
            results.append((
                label,
                head.get("x-gw-provider", "-"),
                head.get("x-gw-cache", "-"),
                head.get("x-gw-cost-usd", "-"),
                text_out,
            ))
            return results[-1]

        console.print("\n[1/6] 语义缓存：第一次调用（MISS）…")
        first = await ask("① 首次提问", "用一句话总结这段代码的异常处理模式。")
        console.print(f"      provider={first[1]} cache={first[2]} cost=${float(first[3]):.6f}")

        console.print("\n[2/6] 语义缓存：完全相同的提问（应命中）…")
        second = await ask("② 相同提问", "用一句话总结这段代码的异常处理模式。")
        console.print(f"      provider={second[1]} cache={second[2]} cost=${float(second[3]):.6f}")
        assert second[2] == "HIT", "相同提问应命中缓存"

        console.print("\n[3/6] 语义缓存：同义改写（应命中）…")
        third = await ask("③ 同义改写", "请用一句话说明这段代码是如何处理异常的。")
        console.print(f"      provider={third[1]} cache={third[2]} cost=${float(third[3]):.6f}")
        assert third[2] == "HIT", "同义改写应命中缓存"
        assert third[4].strip() == first[4].strip(), "缓存命中的回复应与首次一致"

        console.print("\n[4/6] 语义缓存：无关提问（应 MISS 并走 stub-a）…")
        fourth = await ask("④ 无关提问", "写一首关于秋天的五言绝句。")
        console.print(f"      provider={fourth[1]} cache={fourth[2]}")
        assert fourth[2] == "MISS"

        console.print("\n[5/6] 故障降级：把 stub-a 切到宕机态，再问新问题…")
        STUB_FAULTY["stub-a"] = True
        fifth = await ask("⑤ 降级", "解释一下什么是依赖倒置原则。")
        console.print(f"      provider={fifth[1]} cache={fifth[2]}（应从首候选 stub-a 降到 stub-b）")
        assert fifth[1] == "stub-b", "stub-a 宕机后应自动降级到 stub-b"
        STUB_FAULTY["stub-a"] = False

        console.print("\n[6/6] 流式：缓存命中时 SSE 回放 + 再跑一个未命中的流式…")
        stream_hit = await ask("⑥ 流式(命中)", "用一句话总结这段代码的异常处理模式。",
                               stream=True)
        console.print(f"      provider={stream_hit[1]} cache={stream_hit[2]}  "
                      f"回放字符数={len(stream_hit[4])}")
        assert stream_hit[2] == "HIT" and len(stream_hit[4]) > 0
        stream_miss = await ask("⑦ 流式(未命中)", "把这段话翻译成英文：今天天气很好。",
                                stream=True)
        console.print(f"      provider={stream_miss[1]} cache={stream_miss[2]}  "
                      f"流式收到字符数={len(stream_miss[4])}")
        assert stream_miss[2] == "MISS"

        # 数据汇总
        models = (await client.get("/v1/models", headers=headers)).json()
        stats = (await client.get("/api/stats", headers=headers)).json()
        console.print(f"\n模型清单 {len(models['data'])} 个别名；"
                      f"用量统计：请求 {stats['totals']['requests']}，"
                      f"成本 ${stats['totals']['cost_usd']:.4f}，"
                      f"缓存命中率 {stats['totals']['cache_rate']:.0%}")
        return stats


def run_demo(outdir, console) -> int:
    _print_banner(console)
    _write_demo_dir(outdir)
    os.environ.setdefault("LLMGW_STUB_A_KEY", "sk-stub-a-demo")
    os.environ.setdefault("LLMGW_STUB_B_KEY", "sk-stub-b-demo")

    config = GatewayConfig.load(outdir / "gateway.toml")
    app = make_server(config)

    from .store import Store

    store = Store(config.db_path)
    row = store.create_key("demo", rpm=120, monthly_budget_usd=10.0)
    key_secret = row["key"]

    server_a = serve_stub_in_thread(STUB_A_PORT, "stub-a")
    server_b = serve_stub_in_thread(STUB_B_PORT, "stub-b")
    try:
        stats = asyncio.run(_exercise(app, key_secret, console))
    finally:
        server_a.should_exit = True
        server_b.should_exit = True

    # 写一份演示产物摘要，便于后续 serve / 看板接着看
    (outdir / "demo-summary.json").write_text(
        json.dumps({"gateway_key": key_secret, "stats": stats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print("\n[green]✓ 演示完成。[/]")
    console.print(f"  演示产物在 {outdir}/：")
    console.print(f"    gateway.toml / llmgateway.db / demo-summary.json")
    console.print(f"  想看板？ cd {outdir} && llmgateway serve --port 8787"
                  f" → http://127.0.0.1:8787/dashboard")
    console.print(f"  或直接用上面生成的 key 调 /v1（见 demo-summary.json）")
    return 0
