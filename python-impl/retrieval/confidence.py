from math import isfinite
from typing import Any

from retrieval.schemas import RetrievalHit


class ConfidenceGatedRetriever:
    """过滤低相关性精排结果；无结果时由 RAG 直接兜底。"""

    def __init__(self, retriever: Any, min_score: float):
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score 必须在 0 到 1 之间")

        self.retriever = retriever
        self.min_score = min_score

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query.strip():
            return []

        hits = self.retriever.search(query, top_k=top_k)
        accepted = []

        for hit in hits:
            if hit.retriever != "reranker":
                raise ValueError("相关性闸门只接受精排分数")

            if not isfinite(hit.score) or not 0.0 <= hit.score <= 1.0:
                raise ValueError("精排分数必须是 0 到 1 之间的有效数值")

            if hit.score >= self.min_score:
                accepted.append(hit)

        return accepted