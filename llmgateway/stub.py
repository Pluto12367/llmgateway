"""离线演示用 stub 上游：一个「假的 OpenAI 兼容厂商」。

行为由进程内 `STUB_FAULTY` 开关控制（按 provider tag 独立，同一进程内由 demo 编排
切换）：正常时返回确定性回显回复；`STUB_FAULTY[tag]=True` 时该厂商 /chat/completions
一律返回 503（模拟宕机），供降级演示。

仅用于本地离线演示与评测，绝不用于生产。启动：python -m llmgateway.stub --port 9211
"""

from __future__ import annotations

import argparse
import json
import secrets
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

STUB_FAULTY: dict[str, bool] = {}


def make_stub_app(provider_tag: str = "stub-a"):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if STUB_FAULTY.get(provider_tag):
            return JSONResponse(
                status_code=503,
                content={"error": {"message": f"stub provider {provider_tag} 模拟宕机",
                                   "type": "upstream_error", "param": None, "code": "stub_down"}},
            )
        body = await request.json()
        user_msgs = [m for m in body.get("messages", []) if m.get("role") == "user"]
        text = user_msgs[-1]["content"] if user_msgs else ""
        content = f"[{provider_tag} 回应] {text}（这是 stub 生成的确定性回复，用于离线演示）"
        usage = {"prompt_tokens": max(1, round(len(str(body.get('messages')))) // 3),
                 "completion_tokens": max(1, round(len(content) / 2)),
                 "total_tokens": 0}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        if body.get("stream"):
            cid = f"chatcmpl-{provider_tag}-{secrets.token_hex(4)}"
            created = int(time.time())

            async def gen():
                step = 24
                chunks = []
                for i in range(0, len(content), step):
                    chunks.append({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": body.get("model", ""),
                        "choices": [{"index": 0,
                                     "delta": {"content": content[i:i + step]},
                                     "finish_reason": None}],
                    })
                chunks.append({"id": cid, "object": "chat.completion.chunk",
                               "created": created, "model": body.get("model", ""),
                               "choices": [{"index": 0, "delta": {},
                                            "finish_reason": "stop"}],
                               "usage": usage})
                for chunk in chunks:
                    # OpenAI SSE 约定：data: <json>\n\n，结尾 [DONE]
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return {
            "id": f"chatcmpl-{provider_tag}-{secrets.token_hex(4)}",
            "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", ""),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": usage,
        }

    return app


def serve_stub_in_thread(port: int, provider_tag: str) -> "uvicorn.Server":
    """在后台线程起一个 stub 上游，返回 server 句柄（调用 shutdown 停止）。"""
    import threading

    import uvicorn

    config = uvicorn.Config(make_stub_app(provider_tag), host="127.0.0.1",
                            port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 等待端口就绪
    import socket

    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"stub 上游 {port} 启动超时")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMGateway 离线演示 stub 上游")
    parser.add_argument("--port", type=int, default=9211)
    parser.add_argument("--tag", default="stub-a")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(make_stub_app(args.tag), host="127.0.0.1", port=args.port,
                log_level="info")
