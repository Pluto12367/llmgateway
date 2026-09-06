"""llmgateway 命令行入口。

子命令：
  init                生成默认 gateway.toml 并初始化数据目录
  keys add/list/revoke 虚拟 key 管理
  config              校验并打印配置概览
  serve               启动网关（uvicorn）
  demo                一键离线演示：stub 上游 + 语义缓存 + 故障降级
  eval                回放评测：在录制的请求集上量测缓存命中/成本节省
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from . import __version__

console = Console(legacy_windows=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmgateway",
        description="LLMGateway — 自托管 OpenAI 兼容 LLM 网关",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="生成默认 gateway.toml 并初始化")
    p_init.add_argument("--path", default="gateway.toml", help="配置文件路径")
    p_init.add_argument("--force", action="store_true", help="覆盖已存在的文件")

    # keys
    p_keys = sub.add_parser("keys", help="虚拟 key 管理")
    ksub = p_keys.add_subparsers(dest="keys_command", required=True)
    p_add = ksub.add_parser("add", help="创建新 key")
    p_add.add_argument("name", help="key 名称（如 alice / ci-bot）")
    p_add.add_argument("--rpm", type=int, default=None, help="每分钟请求上限")
    p_add.add_argument("--budget", type=float, default=None, help="月度预算（美元）")
    p_add.add_argument("--config", default="gateway.toml")
    p_list = ksub.add_parser("list", help="列出所有 key")
    p_list.add_argument("--config", default="gateway.toml")
    p_revoke = ksub.add_parser("revoke", help="吊销 key")
    p_revoke.add_argument("key_id", help="key 的 id（list 里可见）")
    p_revoke.add_argument("--config", default="gateway.toml")

    # config
    p_cfg = sub.add_parser("config", help="校验并打印配置")
    p_cfg.add_argument("--config", default="gateway.toml")

    # serve
    p_serve = sub.add_parser("serve", help="启动网关服务")
    p_serve.add_argument("--config", default="gateway.toml")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)

    # demo
    p_demo = sub.add_parser("demo", help="一键离线演示（无需真实 API Key）")
    p_demo.add_argument("--outdir", default="demo", help="演示产物输出目录")

    # eval
    p_eval = sub.add_parser("eval", help="在录制请求集上评测语义缓存收益")
    p_eval.add_argument("--config", default="gateway.toml")
    p_eval.add_argument(
        "recordings", nargs="?", default="evals/recordings.json",
        help="录制的请求集 JSON（默认 evals/recordings.json）",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            from .cli_impl import cmd_init
            return cmd_init(args, console)
        if args.command == "keys":
            from .cli_impl import cmd_keys
            return cmd_keys(args, console)
        if args.command == "config":
            from .cli_impl import cmd_config
            return cmd_config(args, console)
        if args.command == "serve":
            from .cli_impl import cmd_serve
            return cmd_serve(args, console)
        if args.command == "demo":
            from .cli_impl import cmd_demo
            return cmd_demo(args, console)
        if args.command == "eval":
            from .cli_impl import cmd_eval
            return cmd_eval(args, console)
        parser.error(f"未知命令：{args.command}")
        return 2
    except Exception as e:  # 顶层兜底（ModuleNotFound 等）
        console.print(f"[red]✗ {e}[/]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
