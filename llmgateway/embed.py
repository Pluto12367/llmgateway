"""语义缓存嵌入后端。

默认 `hash`：零依赖、离线、确定性的字符 n-gram 词袋嵌入，足够让同义改写命中；
可选 `fastembed`：本地 ONNX 跑 bge-small-zh-v1.5（离线、无 torch），质量更高，
对应可选依赖 `llmgateway[local]`。
"""

from __future__ import annotations

import math
import re
import unicodedata

_HASH_DIM = 256
# 常见虚词：单独成特征会稀释实词相似度，过滤后同义改写的余弦分更高
_STOP = set("的一了是在有和被把这那就都也你我他她它们与及或但而是又很之例如")


def _fnv1a(text: str) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


class HashingEmbedder:
    """无依赖的确定性嵌入：CJK 单字/双字词组 + 拉丁词，TF + L2 归一。

    刻意不做跨词组合特征：词序组合会放大措辞差异，压低同义改写的相似度。
    标定参考：同义改写 0.57~0.71，无关问题 ≤0.38，推荐阈值 0.60。
    """

    model_name = "hash:v1"
    dim = _HASH_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _tokens(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text.lower())
        # 拆出 CJK 单字与连续拉丁/数字串
        pieces = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
        toks: list[str] = []
        for i, p in enumerate(pieces):
            if len(p) == 1 and "\u4e00" <= p <= "\u9fff":
                if p not in _STOP:
                    toks.append(p)
                nxt = pieces[i + 1] if i + 1 < len(pieces) else ""
                if (len(nxt) == 1 and "\u4e00" <= nxt <= "\u9fff"
                        and p not in _STOP and nxt not in _STOP):
                    toks.append(p + nxt)  # 中文双字词组
            else:
                toks.append(p)
        return toks

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            idx = _fnv1a(tok) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class FastEmbedEmbedder:
    """fastembed 本地 ONNX 嵌入（可选依赖），模型同 codeguide 默认 bge-small-zh。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError(
                "使用 fastembed 嵌入需要：pip install 'llmgateway[local]'（即 fastembed）"
            ) from e
        self.model_name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


def build_embedder(spec: str = "auto"):
    """按 spec 构造嵌入器：none->None；auto 优先 fastembed 否则内置 hash。"""
    spec = (spec or "auto").strip().lower()
    if spec == "none":
        return None
    if spec == "hash":
        return HashingEmbedder()
    if spec == "fastembed":
        return FastEmbedEmbedder()
    if spec == "auto":
        try:
            return FastEmbedEmbedder()
        except ImportError:
            return HashingEmbedder()
    raise ValueError(f"未知嵌入器 spec：{spec}")
