from math import isfinite
from numbers import Real
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.documents import Document

from retrieval.schemas import RetrievalHit


class RerankingRetriever:
    """先召回候选，再用 BGE-Reranker 重新排序。"""

    def __init__(
        self,
        retriever: Any,
        model_path: str,
        candidate_k: int = 10,
    ):
        import torch
        from FlagEmbedding import FlagReranker

        path = Path(model_path).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"精排模型配置不存在：{path}")
        if candidate_k <= 0:
            raise ValueError("candidate_k 必须大于 0")

        torch.set_num_threads(3)
        self.retriever = retriever
        self.candidate_k = candidate_k
        self._lock = Lock()
        self._model = FlagReranker(
            str(path),
            use_fp16=False,
            devices="cpu",
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not query.strip():
            return []

        hits = self.retriever.search(
            query,
            top_k=max(self.candidate_k, top_k),
        )
        if not hits:
            return []

        pairs = [
            [query, hit.document.page_content]
            for hit in hits
        ]

        with self._lock:
            raw_scores = self._model.compute_score(
                pairs,
                batch_size=1,
                max_length=512,
                normalize=True,
            )

        if isinstance(raw_scores, Real):
            raw_scores = [raw_scores]

        scores = [float(score) for score in raw_scores]
        if len(scores) != len(hits) or not all(map(isfinite, scores)):
            raise ValueError("精排分数数量不匹配或包含无效数值")

        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievalHit(
                document=Document(
                    page_content=hit.document.page_content,
                    metadata={
                        **hit.document.metadata,
                        "rerank_score": score,
                    },
                ),
                score=score,
                retriever="reranker",
            )
            for hit, score in ranked[:top_k]
        ]