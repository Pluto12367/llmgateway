"""FastAPI 服务：OpenAI 兼容代理端点 + 看板/统计。

- POST /v1/chat/completions — 非流式与流式(SSE)都走这里
- POST /v1/embeddings       — 透传到配置里支持嵌入的厂商
- GET  /v1/models           — 网关暴露的模型别名
- GET  /healthz             — 存活探针
- GET  /dashboard           — 自包含用量看板(单页 HTML)
- GET  /api/stats           — 用量统计 JSON（可选 ?span=today|month|day|all）

所有 /v1 端点要求 `Authorization: Bearer sk-gw_xxx`（除 /v1/models 外均可匿名，
按 OpenAI 惯例模型清单不鉴权——可在配置 allow_origins 处按需收紧）。
"""

from __future__ import annotations

import json

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .cache import SemanticCache
from .config import ConfigError, GatewayConfig
from .embed import build_embedder
from .ratelimit import Limiter, authorize
from .router import GatewayError, Router
from .stats import collect_stats
from .store import Store

SSE_MEDIA = "text/event-stream"


def make_server(config: GatewayConfig, transport=None) -> FastAPI:
    store = Store(config.db_path)
    embedder = build_embedder(config.cache.embedder)
    cache = SemanticCache(
        store, embedder=embedder,
        threshold=config.cache.threshold,
        max_entries=config.cache.max_entries,
    )
    router = Router(config, store, cache, transport=transport)
    limiter = Limiter()

    app = FastAPI(title="LLMGateway", version="0.1", docs_url=None, redoc_url=None)

    # 开放跨域，方便浏览器端/看板直接调用代理
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- OpenAI 兼容端点 ----------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        key = authorize(config, store, limiter, request.headers.get("authorization"))
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise GatewayError("请求体不是合法 JSON", kind="client", status=400,
                               error_type="invalid_request_error")
        if not isinstance(body, dict):
            raise GatewayError("请求体必须是 JSON 对象", kind="client", status=400,
                               error_type="invalid_request_error")

        alias = str(body.get("model", "")).strip()
        if not alias:
            raise GatewayError("缺少 model 字段", kind="client", status=400,
                               error_type="invalid_request_error")
        stream = bool(body.get("stream", False))

        if not stream:
            attempt = await router.chat(alias, body)
            log_attempt(store, key, attempt)
            usage = (attempt.usage or {})
            data = {
                "id": f"chatcmpl-gw-{abs(hash(attempt.content)) % 10**10}",
                "object": "chat.completion",
                "created": int(attempt.t0),
                "model": alias,
                "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": attempt.content
                }, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": (usage.get("prompt_tokens", 0)
                                     + usage.get("completion_tokens", 0)),
                },
            }
            return JSONResponse(
                content=data,
                headers={"x-gw-provider": attempt.provider,
                         "x-gw-cache": "HIT" if attempt.cache_hit else "MISS",
                         "x-gw-cost-usd": f"{attempt.cost_usd:.8f}"},
            )

        # 流式：prepare_stream 已确定降级/缓存状态
        attempt, chunks = await router.prepare_stream(alias, body)

        async def sse():
            try:
                async for chunk in chunks:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                log_attempt(store, key, attempt)

        headers = {"x-gw-provider": attempt.provider,
                   "x-gw-cache": "HIT" if attempt.cache_hit else "MISS",
                   "x-gw-cost-usd": f"{attempt.cost_usd:.8f}",
                   "Cache-Control": "no-cache"}
        return StreamingResponse(sse(), media_type=SSE_MEDIA, headers=headers)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        authorize(config, store, limiter, request.headers.get("authorization"))
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise GatewayError("请求体不是合法 JSON", kind="client", status=400,
                               error_type="invalid_request_error")
        target = config.embed_target(str(body.get("model", "")))
        if target is None:
            raise GatewayError(
                "网关未配置可承接该嵌入模型的厂商（在 provider 里加 embed_models）",
                kind="client", status=404, error_type="invalid_request_error",
            )
        provider_name, model = target
        provider = config.provider(provider_name)
        from .proxy import chat_nonstream  # noqa: F401  (避免循环，仅用其客户端的写法)

        import httpx

        api_key = provider.api_key()
        if not api_key:
            raise GatewayError(f"厂商 {provider_name} 未配置 API Key",
                               kind="upstream", status=502)
        # 直接透传 /embeddings
        async with httpx.AsyncClient(
            base_url=provider.base_url, timeout=httpx.Timeout(60.0, connect=10.0)
        ) as client:
            try:
                resp = await client.post(
                    "/embeddings",
                    json={"model": model, "input": body.get("input", [])},
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                )
            except httpx.HTTPError as e:
                raise GatewayError(f"嵌入上游连接失败：{e.__class__.__name__}",
                                   kind="upstream", status=502) from e
        if resp.status_code >= 400:
            raise GatewayError(f"嵌入上游返回 HTTP {resp.status_code}：{resp.text[:200]}",
                               kind="upstream", status=502)
        return JSONResponse(resp.json())

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": config.model_catalog()}

    # ---------- 看板与统计 ----------

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/stats")
    async def api_stats(span: str = "today"):
        try:
            return collect_stats(store, span)
        except ValueError as e:
            raise GatewayError(str(e), kind="client", status=400,
                               error_type="invalid_request_error")

    @app.get("/dashboard")
    async def dashboard():
        return HTMLResponse(_dashboard_html())

    # ---------- 全局异常 ----------

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "message": str(exc),
                    "type": exc.error_type,
                    "param": None,
                    "code": exc.kind,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"网关内部错误：{exc}",
                               "type": "internal_error", "param": None, "code": None}},
        )

    return app


def log_attempt(store: Store, key: dict, attempt) -> None:
    store.record_usage(
        key_id=key["key_id"],
        key_name=key["key_name"],
        alias=attempt.alias,
        provider=attempt.provider,
        model=attempt.model,
        endpoint="chat",
        status=attempt.status,
        cache_hit=attempt.cache_hit,
        prompt_tokens=(attempt.usage or {}).get("prompt_tokens", 0),
        completion_tokens=(attempt.usage or {}).get("completion_tokens", 0),
        cost_usd=attempt.cost_usd,
        latency_ms=attempt.latency_ms,
        error=attempt.error if attempt.status != "ok" else "",
    )


def _dashboard_html() -> str:
    return """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LLMGateway · 用量看板</title>
<style>
  :root{--bg:#0f1419;--card:#1a212b;--card2:#222b37;--line:#2e3946;--text:#e7edf3;
        --mut:#8fa1b3;--acc:#3fa7f0;--ok:#34c97b;--warn:#f0b23f;--err:#f05b5b;
        --mono:"SFMono-Regular",Consolas,"Liberation Mono",monospace;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
     font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;}
  header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;
         align-items:baseline;gap:14px;flex-wrap:wrap}
  h1{font-size:17px;margin:0;font-weight:650} h1 span{color:var(--acc)}
  .spans{margin-left:auto;display:flex;gap:6px}
  .spans button{background:none;border:1px solid var(--line);color:var(--mut);
     padding:4px 12px;border-radius:14px;cursor:pointer;font-size:12px}
  .spans button.on{color:var(--acc);border-color:var(--acc)}
  main{padding:22px 26px;max-width:1100px;margin:0 auto}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .card .label{color:var(--mut);font-size:12px}
  .card .value{font-size:22px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
  .card .sub{color:var(--mut);font-size:11px;margin-top:2px}
  section{margin-top:22px}
  h2{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin:0 0 8px}
  table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);
        border:1px solid var(--line);border-radius:8px;overflow:hidden}
  th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);
        font-variant-numeric:tabular-nums;white-space:nowrap}
  th{color:var(--mut);font-weight:500;font-size:12px;background:var(--card2)}
  tr:last-child td{border-bottom:none}
  .mono{font-family:var(--mono);font-size:12px}
  .ok{color:var(--ok)} .warn{color:var(--warn)} .err{color:var(--err)} .acc{color:var(--acc)}
  .bar{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
  .hbar{height:8px;background:var(--card2);border-radius:4px;overflow:hidden;margin-top:6px}
  .hbar i{display:block;height:100%;background:var(--acc)}
  #status{color:var(--mut);font-size:12px;margin-left:auto}
</style>
</head>
<body>
<header>
  <h1>LLM<span>Gateway</span> 用量看板</h1>
  <div class="spans" id="spans"></div>
  <div id="status">加载中…</div>
</header>
<main>
  <div class="cards" id="cards"></div>
  <section><h2>请求趋势 · 近24h（成本 $）</h2><div class="bar" id="hourly"></div></section>
  <section><h2>按厂商</h2><div id="providerTable"></div></section>
  <section><h2>按虚拟 Key</h2><div id="keyTable"></div></section>
  <section><h2>最近请求</h2><div id="recentTable"></div></section>
</main>
<script>
const spans=["today","month","day","all"];
const spanNames={today:"今日",month:"本月",day:"近24h",all:"全部"};
const nf=new Intl.NumberFormat("zh-CN",{maximumFractionDigits:0});
const usd=new Intl.NumberFormat("zh-CN",{style:"currency",currency:"USD",maximumFractionDigits:4});
let cur="today",timer=null;
const btn=document.getElementById("spans");
spans.forEach(s=>{const b=document.createElement("button");b.textContent=spanNames[s];
  b.onclick=()=>{cur=s;document.querySelectorAll(".spans button").forEach(x=>x.classList.remove("on"));
    b.classList.add("on");refresh();};btn.appendChild(b);});

function esc(x){return String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function tableHtml(cols,rows,rowCls){
  if(!rows.length)return "<p style='color:var(--mut)'>暂无数据</p>";
  let h="<table><tr>"+cols.map(c=>"<th>"+c+"</th>").join("")+"</tr>";
  rows.forEach(r=>{h+="<tr class='"+(rowCls?rowCls(r):"")+"'>"+
    cols.map(c=>"<td>"+(r[c]??"-")+"</td>").join("")+"</tr>";});
  return h+"</table>";
}

async function refresh(){
  try{
    const r=await fetch("/api/stats?span="+cur);const d=await r.json();
    if(d.error)throw new Error(d.error.message);
    document.getElementById("status").textContent="更新于 "+new Date().toLocaleTimeString();
    const t=d.totals;
    const cards=[
      {label:"请求数",value:nf.format(t.requests),sub:"成功 "+nf.format(t.ok)+" · 失败 "+nf.format(t.errors)},
      {label:"成本",value:usd.format(t.cost_usd),sub:"本月含已出账"},
      {label:"缓存命中",value:(t.cache_rate*100).toFixed(1)+"%",sub:nf.format(t.cache_hits)+" 次命中"},
      {label:"P50 / P95",value:(t.p50_ms??"-")+" / "+(t.p95_ms??"-")+" ms",sub:"首字延迟"},
      {label:"Token",value:nf.format(t.prompt_tokens+t.completion_tokens),sub:"in "+nf.format(t.prompt_tokens)+" / out "+nf.format(t.completion_tokens)},
    ];
    document.getElementById("cards").innerHTML=cards.map(c=>
      "<div class='card'><div class='label'>"+c.label+"</div><div class='value'>"+c.value+"</div><div class='sub'>"+c.sub+"</div></div>").join("");

    const hb=document.getElementById("hourly");
    if(d.hourly&&d.hourly.length){
      const max=Math.max(...d.hourly.map(x=>x.cost),0.0001);
      hb.innerHTML=d.hourly.map(x=>{
        const pct=Math.max(2,Math.round(x.cost/max*100));
        return "<div style='display:flex;justify-content:space-between;font-size:12px;color:var(--mut)'>"+
          "<span>"+esc(x.hour)+"</span><span>"+x.calls+" 次 · "+usd.format(x.cost)+"</span></div>"+
          "<div class='hbar'><i style='width:"+pct+"%'></i></div>";
      }).join("");
    } else hb.innerHTML="<p style='color:var(--mut)'>暂无数据</p>";

    document.getElementById("providerTable").innerHTML=tableHtml(
      ["厂商","调用","缓存命中","成本"],
      d.by_provider.map(p=>({c0:esc(p.provider),c1:nf.format(p.calls),
        c2:(p.cache_hits/p.calls*100).toFixed(0)+"%",c3:usd.format(p.cost)})));

    document.getElementById("keyTable").innerHTML=tableHtml(
      ["Key","调用","错误","成本"],
      d.by_key.map(p=>({c0:esc(p.key),c1:nf.format(p.calls),c2:nf.format(p.errors),c3:usd.format(p.cost)})));

    const recent=d.recent||[];
    document.getElementById("recentTable").innerHTML=tableHtml(
      ["时间","别名","厂商","Key","状态","缓存","延迟","成本"],
      recent.map(r=>({c0:new Date(r.ts*1000).toLocaleTimeString(),c1:esc(r.alias),
        c2:esc(r.provider),c3:esc(r.key),c4:r.status==="ok"?"<span class='ok'>ok</span>":"<span class='err'>"+esc(r.status)+"</span>",
        c5:r.cache_hit?"<span class='ok'>HIT</span>":"miss",
        c6:(r.latency_ms!=null?r.latency_ms+" ms":"-"),c7:usd.format(r.cost)})));
  }catch(e){
    document.getElementById("status").textContent="加载失败："+esc(e.message);
  }
}
btn.querySelector("button").classList.add("on");
refresh();
setInterval(refresh,10000);
</script>
</body>
</html>
"""
