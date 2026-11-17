import logging
from typing import Protocol

from retrieval.fusion import reciprocal_rank_fusion
from retrieval.schemas import RetrievalHit


logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]: ...


class HybridRetriever:
    def __init__(
        self,
        bm25: Retriever,
        dense: Retriever,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        if candidate_k <= 0 or rrf_k <= 0:
            raise ValueError("candidate_k 和 rrf_k 必须大于 0")

        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query.strip():
            return []

        recall_k = max(self.candidate_k, top_k)
        retrieval_lists = []

        for name, retriever in (
            ("bm25", self.bm25),
            ("dense", self.dense),
        ):
            try:
                hits = retriever.search(query, top_k=recall_k)
                retrieval_lists.append(hits)
            except Exception:
                logger.exception("检索分支执行失败：%s", name)

        if not retrieval_lists:
            raise RuntimeError("BM25 和 Dense 检索均不可用")

        return reciprocal_rank_fusion(
            retrieval_lists,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )