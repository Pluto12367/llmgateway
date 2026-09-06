"""CLI 各子命令实现（init/keys/config/serve/demo/eval）。"""

from __future__ import annotations

import os
from pathlib import Path

from rich.table import Table

from .config import ConfigError, DEFAULT_CONFIG_TEXT, GatewayConfig
from .store import Store


def _load_store(config_path: str, console) -> tuple[GatewayConfig, Store]:
    config = GatewayConfig.load(config_path)
    store = Store(config.db_path)
    return config, store


# ---------- init ----------

def cmd_init(args, console) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        console.print(f"[yellow]✗ {path} 已存在。用 --force 覆盖。[/]")
        return 1
    path.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
    tmp = GatewayConfig.load(path)  # data_dir 相对路径以配置文件所在目录为基准
    Store(tmp.db_path)
    console.print(f"[green]✓[/] 已生成 {path}，数据目录 {tmp.data_dir}")
    console.print("  下一步：")
    console.print("  1. 编辑 gateway.toml 填入你的厂商（base_url/api_key_env/价格）")
    console.print("  2. 设置环境变量导出真实密钥（如 export DEEPSEEK_API_KEY=sk-...）")
    console.print("  3. llmgateway keys add my-key   # 创建虚拟 key")
    console.print("  4. llmgateway serve             # 启动网关")
    return 0


# ---------- keys ----------

def cmd_keys(args, console) -> int:
    config, store = _load_store(args.config, console)
    if args.keys_command == "add":
        row = store.create_key(args.name, rpm=args.rpm, monthly_budget_usd=args.budget)
        console.print("[green]✓ 已创建 key[/]")
        console.print("  [bold]密钥（仅此一次显示，请保存）[/]")
        console.print(f"  [cyan]{row['key']}[/]")
        console.print("  使用：Authorization: Bearer " + row["key"])
        return 0
    if args.keys_command == "list":
        rows = store.list_keys()
        if not rows:
            console.print("（暂无 key）")
            return 0
        table = Table(title="虚拟 Keys", show_lines=False)
        table.add_column("id", style="cyan")
        table.add_column("name")
        table.add_column("rpm", justify="right")
        table.add_column("budget$", justify="right")
        table.add_column("状态")
        for r in rows:
            table.add_row(
                r["id"], r["name"],
                str(r["rpm"]) if r["rpm"] else "-",
                f"{r['monthly_budget_usd']:.2f}" if r["monthly_budget_usd"] else "-",
                "revoked" if r["revoked"] else "active",
            )
        console.print(table)
        return 0
    if args.keys_command == "revoke":
        ok = store.revoke_key(args.key_id)
        console.print("[green]✓[/] 已吊销 " + args.key_id if ok
                      else f"[yellow]✗[/] 未找到或已吊销：{args.key_id}")
        return 0 if ok else 1
    return 1


# ---------- config ----------

def cmd_config(args, console) -> int:
    try:
        config = GatewayConfig.load(args.config)
    except ConfigError as e:
        console.print(f"[red]✗ 配置无效：{e}[/]")
        return 1
    console.print(f"[green]✓[/] 配置有效：{config.path}")
    console.print(f"  数据目录  {config.data_dir}")
    console.print(f"  全局月度预算  ${config.limits.monthly_budget_usd}")

    table = Table(title="厂商", show_lines=False)
    table.add_column("名称", style="cyan")
    table.add_column("base_url")
    table.add_column("key 环境变量")
    table.add_column("已配置", justify="center")
    table.add_column("$ in/1M", justify="right")
    table.add_column("$ out/1M", justify="right")
    for name, p in config.providers.items():
        table.add_row(name, p.base_url, p.api_key_env,
                      "✓" if p.has_key() else "—",
                      f"{p.in_usd_per_1m:.3f}", f"{p.out_usd_per_1m:.3f}")
    console.print(table)

    table2 = Table(title="模型别名", show_lines=False)
    table2.add_column("别名", style="green")
    table2.add_column("候选（按序降级）")
    for name, alias in config.aliases.items():
        cands = " → ".join(f"{c.provider}/{c.model}" for c in alias.candidates)
        table2.add_row(name, cands)
    console.print(table2)
    missing = [p.name for p in config.providers.values() if not p.has_key()]
    if missing:
        console.print(f"[yellow]⚠ 以下厂商缺 API Key（serve 时会被跳过）：{', '.join(missing)}[/]")
    return 0


# ---------- serve ----------

def cmd_serve(args, console) -> int:
    import uvicorn

    config = GatewayConfig.load(args.config)
    console.print(f"[green]✓[/] 网关配置加载成功：{config.path}")
    console.print(f"[green]✓[/] 缓存：{'开' if config.cache.enabled else '关'}"
                  f"（嵌入 {config.cache.embedder}，阈值 {config.cache.threshold}）")
    from .server import make_server

    app = make_server(config)
    console.print(f"[cyan]◆[/] OpenAI 兼容端点: http://{args.host}:{args.port}/v1")
    console.print(f"[cyan]◆[/] 用量看板:        http://{args.host}:{args.port}/dashboard")
    console.print(f"[cyan]◆[/] 健康检查:        http://{args.host}:{args.port}/healthz")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------- demo ----------

def cmd_demo(args, console) -> int:
    from .demo import run_demo

    return run_demo(Path(args.outdir), console)


# ---------- eval ----------

def cmd_eval(args, console) -> int:
    import asyncio
    import json

    import httpx

    from .cache import SemanticCache
    from .embed import build_embedder
    from .router import Router

    config = GatewayConfig.load(args.config)
    rec_path = Path(args.recordings)
    if not rec_path.exists():
        console.print(f"[red]✗ 找不到录制文件：{rec_path}[/]")
        return 1
    data = json.loads(rec_path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    embedder = build_embedder(data.get("embedder", "hash") if isinstance(data, dict) else "hash")
    threshold = float(data.get("threshold", config.cache.threshold)
                      if isinstance(data, dict) else config.cache.threshold)

    store = Store(config.db_path)
    cache = SemanticCache(store, embedder=embedder, threshold=threshold,
                          max_entries=config.cache.max_entries)

    # 评测是离线回放（上游由 MockTransport stub 承接，不会真实外呼），
    # 但厂商调用前有 has_key 检查，这里给缺失 key 的厂商补 dummy 值
    for p in config.providers.values():
        if not p.has_key():
            os.environ.setdefault(p.api_key_env, "eval-dummy-key")

    # stub 上游：记录调用次数，返回固定用量（120 in / 60 out），便于核算
    state = {"upstream_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["upstream_calls"] += 1
        body = json.loads(request.content)
        user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
        text = user_msgs[-1]["content"] if user_msgs else ""
        return httpx.Response(200, json={
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", ""),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": f"评测回复：{text[:40]}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
        })

    router = Router(config, store, cache, transport=httpx.MockTransport(handler))

    async def run():
        stats = {"total": 0, "cache_hits": 0, "cost": 0.0, "cost_baseline": 0.0,
                 "expect_ok": 0, "expect_total": 0}
        for case in cases:
            alias = case.get("alias", "fast")
            messages = case["messages"]
            provider_name = config.aliases[alias].candidates[0].provider
            provider = config.provider(provider_name)
            stats["total"] += 1
            stats["cost_baseline"] += provider.cost(120, 60)
            attempt = await router.chat(alias, {"messages": messages})
            if attempt.cache_hit:
                stats["cache_hits"] += 1
            stats["cost"] += attempt.cost_usd
            expect = case.get("expect")
            if expect in ("hit", "miss"):
                stats["expect_total"] += 1
                stats["expect_ok"] += int(
                    (expect == "hit") == attempt.cache_hit
                )
        return stats

    stats = asyncio.run(run())
    hit_rate = stats["cache_hits"] / stats["total"] if stats["total"] else 0.0
    saved = stats["cost_baseline"] - stats["cost"]
    table = Table(title="语义缓存回放评测（recording 回放）", show_lines=False)
    table.add_column("指标", style="cyan")
    table.add_column("数值", justify="right")
    rows = [
        ("录制请求数", str(stats["total"])),
        ("缓存命中", f"{stats['cache_hits']}（{hit_rate:.1%}）"),
        ("预期命中准确率",
         f"{stats['expect_ok']}/{stats['expect_total']}"
         if stats["expect_total"] else "-"),
        ("实际上游调用", str(state["upstream_calls"])),
        ("基准成本（无缓存）$", f"{stats['cost_baseline']:.4f}"),
        ("实测成本（有缓存）$", f"{stats['cost']:.4f}"),
        ("节省 $", f"{saved:.4f}"),
        ("节省比例", f"{saved / stats['cost_baseline']:.1%}" if stats["cost_baseline"] else "-"),
    ]
    for k, v in rows:
        table.add_row(k, str(v))
    console.print(table)
    return 0
